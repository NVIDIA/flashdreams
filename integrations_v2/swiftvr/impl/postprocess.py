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

"""SwiftVR-backed video post-processor."""

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
)
from swiftvr.impl.pipeline import SwiftVRPipeline

_DTypeName = Literal["bfloat16", "float16", "float32"]
_SWIFTVR_REVISION = "743ed2530c550764905400f38eb6cc41af5abc80"


class SwiftVRStream:
    """Per-video driver for an isolated SwiftVR pipeline cache."""

    def __init__(
        self,
        pipeline: SwiftVRPipeline,
        *,
        output_height: int,
        output_width: int,
        overlap: int,
    ) -> None:
        self.pipeline = pipeline
        self.output_height = output_height
        self.output_width = output_width
        self.cache = pipeline.initialize_cache(
            output_height=output_height,
            output_width=output_width,
            overlap=overlap,
        )
        self.autoregressive_index = 0

    @torch.inference_mode()
    def step(self, frames_uint8: Tensor) -> Tensor | None:
        """Process ``[T,H,W,3]`` uint8 frames."""
        output = self.pipeline.generate(
            self.autoregressive_index,
            self.cache,
            frames_uint8,
        )
        self.pipeline.finalize(self.autoregressive_index, self.cache)
        self.autoregressive_index += 1
        return output

    @torch.inference_mode()
    def flush(self) -> Tensor | None:
        """Flush the final encoder temporal group."""
        return self.pipeline.flush(self.cache)


@dataclass(kw_only=True)
class SwiftVRPostProcessorConfig(VideoPostProcessorConfig):
    """Configure causal SwiftVR video restoration."""

    _target: type["SwiftVRPostProcessor"] = field(
        default_factory=lambda: SwiftVRPostProcessor
    )
    checkpoint: str = "H-oliday/SwiftVR"
    """Hugging Face repository or local checkpoint directory."""

    revision: str | None = _SWIFTVR_REVISION
    """Immutable Hugging Face revision, or ``None`` to follow the repository head."""

    scale: int = 4
    """Spatial output scale."""

    chunk_size: int = 24
    """Input frames processed per steady-state SwiftVR call."""

    dit_overlap: int = 0
    """Latent frames blended across DiT chunks; zero is the fastest path."""

    attention_window: tuple[int, int] = (16, 16)
    """Spatial latent-token window used by shifted-window attention."""

    compile_blocks: bool = False
    """Compile transformer blocks. Disabled by default to avoid long startup."""

    prewarm: bool = True
    """Warm model kernels before the first measured rollout chunk."""

    dtype: _DTypeName = "bfloat16"
    """Model compute dtype."""

    device: str = "cuda"
    """Model execution device."""

    _processor: "SwiftVRPostProcessor | None" = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.scale <= 0:
            raise ValueError(f"SwiftVR scale must be positive, got {self.scale}.")
        if self.chunk_size <= 0 or self.chunk_size % 4:
            raise ValueError(
                "SwiftVR chunk_size must be a positive multiple of 4, got "
                f"{self.chunk_size}."
            )
        if self.dit_overlap < 0:
            raise ValueError(
                f"SwiftVR dit_overlap must be non-negative, got {self.dit_overlap}."
            )

    def output_spec(self, input_spec: VideoSpec) -> VideoSpec:
        """Return the exact spatially upscaled output contract."""
        return VideoSpec(
            height=input_spec.height * self.scale,
            width=input_spec.width * self.scale,
            fps=input_spec.fps,
            channels=3,
        )

    def setup(self, **kwargs: Any) -> "SwiftVRPostProcessor":
        """Reuse one resident processor across replacement sessions."""
        if kwargs:
            raise TypeError("SwiftVRPostProcessorConfig.setup() accepts no overrides.")
        if self._processor is None:
            self._processor = SwiftVRPostProcessor(self)
        return self._processor

    def validate_execution(self, *, world_size: int) -> None:
        """Reject unsupported multi-rank post-processing."""
        if world_size != 1:
            raise ValueError("SwiftVR post-processing currently supports one GPU.")


class SwiftVRPostProcessor(VideoPostProcessor[SwiftVRPostProcessorConfig]):
    """Resident SwiftVR model factory for isolated stream sessions."""

    def __init__(self, config: SwiftVRPostProcessorConfig) -> None:
        super().__init__(config)
        self._pipeline: SwiftVRPipeline | None = None
        self._warmed_specs: set[VideoSpec] = set()

    def start(self, spec: VideoSpec) -> VideoPostProcessorSession:
        """Start one causal video stream."""
        if spec.channels != 3:
            raise ValueError(
                f"SwiftVR expects RGB input, got {spec.channels} channels."
            )
        return _SwiftVRPostProcessorSession(self, spec)

    def pipeline(self) -> SwiftVRPipeline:
        """Load the heavyweight model once and keep it resident."""
        if self._pipeline is None:
            self._pipeline = _load_swiftvr_pipeline(self.config)
        return self._pipeline

    def prepare(self, spec: VideoSpec) -> None:
        """Load and optionally warm the model once for this input shape."""
        pipeline = self.pipeline()
        if not self.config.prewarm or spec in self._warmed_specs:
            return
        output = self.config.output_spec(spec)
        stream = SwiftVRStream(
            pipeline,
            output_height=output.height,
            output_width=output.width,
            overlap=self.config.dit_overlap,
        )
        frames = torch.zeros(
            (self.config.chunk_size, spec.height, spec.width, 3),
            device=pipeline.device,
            dtype=torch.uint8,
        )
        stream.step(frames)
        stream.step(frames)
        stream.step(frames[:1])
        stream.flush()
        if pipeline.device.type == "cuda":
            torch.cuda.synchronize(pipeline.device)
        self._warmed_specs.add(spec)


class _SwiftVRPostProcessorSession(VideoPostProcessorSession):
    """Chunk adapter that preserves arbitrary input frame counts."""

    def __init__(self, processor: SwiftVRPostProcessor, spec: VideoSpec) -> None:
        self._processor = processor
        self._spec = spec
        self._stream: SwiftVRStream | None = None
        self._buffer: Tensor | None = None
        self._last_frame: Tensor | None = None
        self._metadata_spans: list[tuple[int, dict[str, Any]]] = []
        self._input_frames = 0
        self._output_frames = 0
        self._closed = False

    def prepare(self) -> None:
        """Preload and prewarm the persistent model before timed processing."""
        self._processor.prepare(self._spec)
        self._ensure_stream()

    @torch.inference_mode()
    def process(self, chunk: VideoChunk) -> list[VideoChunk]:
        """Buffer input and emit complete SwiftVR chunks."""
        if self._closed:
            raise RuntimeError("cannot process SwiftVR after flush()")
        frames = self._to_uint8_thwc(chunk)
        self._append(frames, chunk.metadata)
        outputs: list[VideoChunk] = []
        while (
            self._buffer is not None
            and self._buffer.shape[0] >= self._processor.config.chunk_size
        ):
            size = self._processor.config.chunk_size
            current = self._buffer[:size]
            self._buffer = self._buffer[size:]
            restored = self._ensure_stream().step(current)
            if restored is not None and restored.shape[1]:
                outputs.append(self._output_chunk(restored, source="swiftvr"))
        return outputs

    @torch.inference_mode()
    def flush(self) -> list[VideoChunk]:
        """Pad the causal tail internally while emitting exactly one frame per input."""
        if self._closed:
            return []
        self._closed = True
        remaining = self._input_frames - self._output_frames
        if remaining == 0:
            return []
        assert self._last_frame is not None
        pending = self._buffer
        if pending is None:
            pending = self._last_frame[:0]
        padding = (1 - pending.shape[0]) % 4
        if pending.shape[0] + padding == 0:
            padding = 1
        if padding:
            pending = torch.cat(
                [pending, self._last_frame.expand(padding, -1, -1, -1)], dim=0
            )
        stream = self._ensure_stream()
        outputs = [
            output
            for output in (stream.step(pending), stream.flush())
            if output is not None
        ]
        if not outputs:
            raise RuntimeError(
                "SwiftVR produced no output while flushing buffered frames."
            )
        restored = torch.cat(outputs, dim=1)
        if restored.shape[1] < remaining:
            raise RuntimeError(
                f"SwiftVR emitted {restored.shape[1]} tail frames; expected {remaining}."
            )
        return [self._output_chunk(restored[:, :remaining], source="swiftvr_tail")]

    def _ensure_stream(self) -> SwiftVRStream:
        if self._stream is None:
            output = self._processor.config.output_spec(self._spec)
            self._stream = SwiftVRStream(
                self._processor.pipeline(),
                output_height=output.height,
                output_width=output.width,
                overlap=self._processor.config.dit_overlap,
            )
        return self._stream

    def _to_uint8_thwc(self, chunk: VideoChunk) -> Tensor:
        canonical = to_bvtchw(chunk.tensor, layout=chunk.layout)
        batch, views, _, channels, height, width = canonical.shape
        if batch != 1 or views != 1:
            raise ValueError(
                "SwiftVR supports one stream at a time; "
                f"got batch={batch}, views={views}."
            )
        if channels != 3:
            raise ValueError(f"SwiftVR expects RGB input, got {channels} channels.")
        if canonical.shape[2] == 0:
            raise ValueError("SwiftVR input chunks must contain at least one frame.")
        if (height, width) != (self._spec.height, self._spec.width):
            raise ValueError(
                "SwiftVR input size changed from "
                f"{self._spec.height}x{self._spec.width} to {height}x{width}."
            )
        frames = canonical[0, 0].permute(0, 2, 3, 1)
        return frames.add(1).mul(127.5).round_().clamp_(0, 255).to(torch.uint8)

    def _append(self, frames: Tensor, metadata: dict[str, Any]) -> None:
        self._last_frame = frames[-1:]
        self._input_frames += frames.shape[0]
        self._metadata_spans.append((frames.shape[0], dict(metadata)))
        self._buffer = (
            frames if self._buffer is None else torch.cat([self._buffer, frames])
        )

    def _output_chunk(self, restored: Tensor, *, source: str) -> VideoChunk:
        frame_count = restored.shape[1]
        self._output_frames += frame_count
        return VideoChunk(
            tensor=restored.mul(2).sub_(1).permute(0, 2, 1, 3, 4).contiguous(),
            layout="bcthw",
            metadata=self._consume_metadata(frame_count, source=source),
        )

    def _consume_metadata(self, frames: int, *, source: str) -> dict[str, Any]:
        remaining = frames
        inputs: list[dict[str, Any]] = []
        while remaining and self._metadata_spans:
            span_frames, metadata = self._metadata_spans.pop(0)
            inputs.append(metadata)
            consumed = min(remaining, span_frames)
            remaining -= consumed
            if consumed != span_frames:
                self._metadata_spans.insert(0, (span_frames - consumed, metadata))
        if remaining:
            raise RuntimeError(
                f"missing SwiftVR metadata for {remaining} output frames"
            )
        return {"source": source, "input_chunks": tuple(inputs)}


def _load_swiftvr_pipeline(config: SwiftVRPostProcessorConfig) -> SwiftVRPipeline:
    return SwiftVRPipeline.from_pretrained(
        config.checkpoint,
        revision=config.revision,
        device=config.device,
        dtype=_resolve_dtype(config.dtype),
        attention_window=config.attention_window,
        compile_blocks=config.compile_blocks,
        chunk_size=config.chunk_size,
    )


def _resolve_dtype(name: _DTypeName) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


POSTPROCESS_PRESET_SWIFTVR_4X = SwiftVRPostProcessorConfig()
"""Default upstream-compatible SwiftVR 4x post-processing preset."""

POSTPROCESS_PRESET_SWIFTVR_2X = SwiftVRPostProcessorConfig(scale=2, chunk_size=8)
"""SwiftVR 2x preset with an 8-frame streaming chunk."""


__all__ = [
    "POSTPROCESS_PRESET_SWIFTVR_2X",
    "POSTPROCESS_PRESET_SWIFTVR_4X",
    "SwiftVRPostProcessor",
    "SwiftVRPostProcessorConfig",
    "SwiftVRStream",
]
