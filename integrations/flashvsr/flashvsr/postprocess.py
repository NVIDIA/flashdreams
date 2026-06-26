# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FlashVSR-backed video post-processor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from torch import Tensor

from flashdreams.infra.postprocess import (
    VideoChunk,
    VideoPostProcessor,
    VideoPostProcessorConfig,
    VideoPostProcessorSession,
    VideoSpec,
    to_bvtchw,
    to_minus_one_one,
)
from flashvsr.corrector import ColorCorrectorImplementation
from flashvsr.encoder import FlashVSREncoder

_DTypeName = Literal["bfloat16", "float16", "float32"]
_TailPolicy = Literal["replicate_pad", "drop"]


@dataclass(kw_only=True)
class FlashVSRPostProcessorConfig(VideoPostProcessorConfig):
    """Post-process generated RGB video with FlashVSR."""

    _target: type["FlashVSRPostProcessor"] = field(
        default_factory=lambda: FlashVSRPostProcessor
    )

    scale: Literal[2, 4] = 2
    """Spatial upsample factor."""

    chunk_size: Literal[8, 16] = 16
    """Steady-state FlashVSR chunk size. ``16`` favors throughput; ``8`` lowers latency."""

    sparse_ratio: float = 2.0
    """Sparse attention budget multiplier. ``2.0`` is the stable preset;
    ``1.5`` is faster with lower attention budget."""

    kv_ratio: int = 3
    """Number of prior chunks kept by FlashVSR's streaming attention cache."""

    local_range: int = 11
    """Local window radius used by FlashVSR sparse attention."""

    attention_mode: Literal["sparse", "full"] = "sparse"
    """Attention backend. Use ``"full"`` for multi-GPU context parallelism."""

    compile_network: bool = True
    """Enable ``torch.compile`` for FlashVSR components.

    Defaults to ``True`` to match the shipped FlashVSR runner presets."""

    use_cuda_graph: bool = True
    """Replay steady-state DiT execution through CUDA graphs.

    Defaults to ``True`` to match the shipped FlashVSR runner presets."""

    color_corrector_implementation: ColorCorrectorImplementation = "cuda"
    """FlashVSR decoder color-correction backend."""

    enable_sync_and_profile: bool = False
    """Record FlashVSR CUDA-event timing. Adds synchronizations."""

    dtype: _DTypeName = "bfloat16"
    """FlashVSR compute dtype."""

    device: str = "cuda"
    """Device used by the FlashVSR model."""

    tail_policy: _TailPolicy = "replicate_pad"
    """How to handle final partial chunks. ``replicate_pad`` preserves all
    frames; ``drop`` favors speed and fixed-size chunks."""


class FlashVSRPostProcessor(VideoPostProcessor[FlashVSRPostProcessorConfig]):
    """Factory for FlashVSR post-processing sessions."""

    def start(self, spec: VideoSpec) -> VideoPostProcessorSession:
        """Start a lazy FlashVSR session for one generated stream."""
        return _FlashVSRPostProcessorSession(self.config, spec)


class _FlashVSRPostProcessorSession(VideoPostProcessorSession):
    """Stateful FlashVSR stream processor with chunk coalescing."""

    def __init__(self, config: FlashVSRPostProcessorConfig, spec: VideoSpec) -> None:
        self._config = config
        self._spec = spec
        self._first_size, self._subseq_size = _chunk_mode(config.chunk_size)
        self._pipeline: Any | None = None
        self._cache: Any | None = None
        self._buffer: Tensor | None = None
        self._ar_idx = 0

    @torch.no_grad()
    def process(self, chunk: VideoChunk) -> list[VideoChunk]:
        """Buffer input frames and emit complete FlashVSR chunks."""
        bcthw = self._chunk_to_bcthw(chunk)
        self._ensure_pipeline(bcthw)
        self._append_to_buffer(bcthw)
        return self._drain_ready_chunks()

    @torch.no_grad()
    def flush(self) -> list[VideoChunk]:
        """Process or drop the final partial chunk according to config."""
        if self._buffer is None or self._buffer.shape[2] == 0:
            return []
        if self._config.tail_policy == "drop":
            self._buffer = None
            return []

        target = self._next_target_size()
        tail_frames = self._buffer.shape[2]
        clip = self._buffer
        if tail_frames < target:
            repeat = clip[:, :, -1:].expand(-1, -1, target - tail_frames, -1, -1)
            clip = torch.cat([clip, repeat], dim=2)
        self._buffer = None
        output = self._run_flashvsr_chunk(clip)[:, :, :tail_frames]
        return [_bcthw_chunk(output, source="flashvsr_tail")]

    def _chunk_to_bcthw(self, chunk: VideoChunk) -> Tensor:
        canonical = to_bvtchw(
            to_minus_one_one(chunk.tensor, value_range=chunk.value_range),
            layout=chunk.layout,
        )
        batch, views, _, channels, _, _ = canonical.shape
        if batch != 1 or views != 1:
            raise ValueError(
                "FlashVSR post-processing currently supports one stream at a time; "
                f"got batch={batch}, views={views}."
            )
        if channels != 3:
            raise ValueError(
                f"FlashVSR expects RGB chunks with 3 channels; got {channels}."
            )
        return canonical[:, 0].permute(0, 2, 1, 3, 4).contiguous()

    def _ensure_pipeline(self, bcthw: Tensor) -> None:
        if self._pipeline is not None:
            return

        _, _, _, height, width = bcthw.shape
        if height != self._spec.height or width != self._spec.width:
            raise ValueError(
                "FlashVSR postprocess stream dimensions changed from "
                f"{self._spec.height}x{self._spec.width} to {height}x{width}."
            )
        dtype = _resolve_dtype(self._config.dtype)
        pipeline_cfg = _build_flashvsr_pipeline(
            input_H=height,
            input_W=width,
            scale=self._config.scale,
            sparse_ratio=self._config.sparse_ratio,
            kv_ratio=self._config.kv_ratio,
            local_range=self._config.local_range,
            compile_network=self._config.compile_network,
            use_cuda_graph=self._config.use_cuda_graph,
            color_corrector_implementation=self._config.color_corrector_implementation,
            enable_sync_and_profile=self._config.enable_sync_and_profile,
            dtype=dtype,
            attention_mode=self._config.attention_mode,
            name="flashvsr-postprocess-v1.1",
        )
        self._pipeline = pipeline_cfg.setup().to(device=self._config.device).eval()
        self._cache = self._pipeline.initialize_cache()

    def _append_to_buffer(self, bcthw: Tensor) -> None:
        assert self._pipeline is not None
        dtype = getattr(
            getattr(self._pipeline, "diffusion_model", None), "dtype", bcthw.dtype
        )
        device = getattr(self._pipeline, "device", torch.device(self._config.device))
        bcthw = bcthw.to(device=device, dtype=dtype)
        if self._buffer is None:
            self._buffer = bcthw
        else:
            self._buffer = torch.cat([self._buffer, bcthw], dim=2)

    def _drain_ready_chunks(self) -> list[VideoChunk]:
        outputs: list[VideoChunk] = []
        while (
            self._buffer is not None
            and self._buffer.shape[2] >= self._next_target_size()
        ):
            target = self._next_target_size()
            clip = self._buffer[:, :, :target]
            self._buffer = self._buffer[:, :, target:]
            outputs.append(
                _bcthw_chunk(self._run_flashvsr_chunk(clip), source="flashvsr")
            )
        return outputs

    def _next_target_size(self) -> int:
        return self._first_size if self._ar_idx == 0 else self._subseq_size

    def _run_flashvsr_chunk(self, clip: Tensor) -> Tensor:
        assert self._pipeline is not None
        assert self._cache is not None
        try:
            return self._pipeline.generate(
                autoregressive_index=self._ar_idx,
                cache=self._cache,
                input=clip,
            )
        finally:
            self._pipeline.finalize(
                autoregressive_index=self._ar_idx, cache=self._cache
            )
            self._ar_idx += 1


def _build_flashvsr_pipeline(**kwargs: Any) -> Any:
    """Build a FlashVSR pipeline config (lazy import for preset entry points)."""
    from flashvsr.config import build_flashvsr_v1_1  # noqa: PLC0415

    return build_flashvsr_v1_1(**kwargs)


def _bcthw_chunk(tensor: Tensor, *, source: str) -> VideoChunk:
    return VideoChunk(
        tensor=tensor,
        layout="bcthw",
        value_range="minus_one_one",
        metadata={"source": source},
    )


def _chunk_modes() -> dict[int, tuple[int, int]]:
    targets = FlashVSREncoder._CHUNK_FRAME_TARGETS
    cold_for: dict[int, int] = {}
    for raw, padded in targets.items():
        if raw != padded:
            cold_for[padded] = raw
    modes = {padded: (cold_for[padded], padded) for padded in cold_for}
    return modes


def _chunk_mode(chunk_size: int) -> tuple[int, int]:
    try:
        return _chunk_modes()[chunk_size]
    except KeyError as exc:
        supported = ", ".join(str(size) for size in sorted(_chunk_modes()))
        raise ValueError(
            f"Unsupported FlashVSR postprocess chunk_size={chunk_size}. "
            f"Supported steady-state sizes: {supported}."
        ) from exc


def _resolve_dtype(dtype: _DTypeName) -> torch.dtype:
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"unsupported FlashVSR dtype: {dtype}")


POSTPROCESS_PRESET_FLASHVSR_V1_1_SPARSE_2_0 = FlashVSRPostProcessorConfig(
    sparse_ratio=2.0,
    attention_mode="sparse",
)
"""Default FlashVSR post-processing preset (stable sparse attention)."""

POSTPROCESS_PRESET_FLASHVSR_V1_1_SPARSE_1_5 = FlashVSRPostProcessorConfig(
    sparse_ratio=1.5,
    attention_mode="sparse",
)
"""Faster FlashVSR post-processing preset (lower sparse attention budget)."""

POSTPROCESS_PRESET_FLASHVSR_V1_1_FULL_ATTN = FlashVSRPostProcessorConfig(
    attention_mode="full",
)
"""FlashVSR post-processing preset with dense full attention (multi-GPU)."""
