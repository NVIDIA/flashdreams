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

"""FlashVSR streaming video super-resolution runner."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import mediapy as media
import numpy as np
import torch
from einops import rearrange
from loguru import logger

from flashdreams.core.distributed import init as init_distributed
from flashdreams.infra.config import derive_config
from flashdreams.infra.runner import Runner, RunnerConfig, _is_torchrun_env
from flashvsr.encoder import FlashVSREncoder
from flashvsr.pipeline import (
    FlashVSRPipeline,
    FlashVSRPipelineCache,
    FlashVSRPipelineConfig,
)
from flashvsr.transformer import FlashVSRTransformerConfig

__all__ = [
    "FlashVSRRunnerConfig",
    "FlashVSRRunner",
]


## Chunk planning helpers


def _chunk_modes() -> dict[int, tuple[int, int]]:
    """Return supported runner chunk modes from the encoder contract.

    :data:`FlashVSREncoder._CHUNK_FRAME_TARGETS` maps each accepted raw frame
    count to the padded frame count consumed by the projector. Runner modes are
    keyed by the steady-state size and store the corresponding cold-start size:
    ``{steady_size: (cold_size, steady_size)}``.

    Returns:
        Mapping such as ``{8: (5, 8), 16: (13, 16)}``.
    """
    targets = FlashVSREncoder._CHUNK_FRAME_TARGETS
    cold_for: dict[int, int] = {}
    for raw, padded in targets.items():
        if raw != padded:
            assert padded not in cold_for or cold_for[padded] == raw, (
                f"Multiple cold-start sizes map to padded {padded}: "
                f"{cold_for.get(padded)} and {raw}"
            )
            cold_for[padded] = raw
    return {padded: (cold_for.get(padded, padded), padded) for padded in cold_for}


_CHUNK_MODES: dict[int, tuple[int, int]] = _chunk_modes()


def _build_chunks(
    total_frames: int, first_size: int, subseq_size: int
) -> list[tuple[int, int]]:
    """Build contiguous ``(start, size)`` chunks for one rollout.

    The first chunk uses the cold-start size, which the encoder pad-left
    replicates to the matching steady-state length. Subsequent chunks use the
    steady-state size. Any trailing partial chunk is dropped because the
    encoder accepts only the fixed FlashVSR chunk sizes.

    Args:
        total_frames: Number of frames available in the input video.
        first_size: Raw frame count for the cold-start chunk.
        subseq_size: Raw frame count for each steady-state chunk.

    Returns:
        Start offsets and raw frame counts to pass to
        :meth:`FlashVSRPipeline.generate`.
    """
    chunks: list[tuple[int, int]] = []
    pos = 0
    first = True
    while pos < total_frames:
        target = first_size if first else subseq_size
        size = min(target, total_frames - pos)
        if size < target:
            logger.warning(
                f"Trailing chunk has {size} frames (need {target}); "
                f"truncating video to {pos} frames."
            )
            break
        chunks.append((pos, size))
        pos += size
        first = False
    return chunks


def _resolve_target_and_topk_ratio(
    *, input_H: int, input_W: int, scale: int, sparse_ratio: float
) -> tuple[int, int, float]:
    """Resolve cropped target dimensions and FlashVSR ``topk_ratio``.

    The encoder bicubic-upsamples to ``(input_H * scale, input_W * scale)``
    and center-crops each axis down to the largest 128-multiple. The sparse
    attention budget follows upstream FlashVSR:
    ``topk_ratio = sparse_ratio * 768 * 1280 / (target_H * target_W)``.

    Args:
        input_H: Low-resolution input height after any runner-side crop.
        input_W: Low-resolution input width after any runner-side crop.
        scale: Pixel upsample factor.
        sparse_ratio: User-facing sparse-attention budget multiplier.

    Returns:
        ``(target_H, target_W, topk_ratio)`` for the per-video pipeline config.

    Raises:
        AssertionError: Either scaled axis is too small to fit one
            128-multiple target.
    """
    target_H = ((input_H * scale) // 128) * 128
    target_W = ((input_W * scale) // 128) * 128
    assert target_H > 0 and target_W > 0, (
        f"Input {input_H}x{input_W} at scale={scale} is too small to crop "
        f"to a 128-multiple (need H*scale and W*scale to each be at "
        f"least 128)."
    )
    topk_ratio = sparse_ratio * 768 * 1280 / (target_H * target_W)
    return target_H, target_W, topk_ratio


def _probe_input_fps(path: Path) -> float:
    """Probe the input video's frame rate.

    Args:
        path: Video path readable by ``mediapy``.

    Returns:
        Probed frame rate, or ``30.0`` if metadata probing fails.
    """
    try:
        # ``mediapy`` ships without type stubs so ty cannot see the
        # ``VideoMetadata.from_path`` classmethod (added in mediapy 1.1).
        meta = media.VideoMetadata.from_path(str(path))  # ty: ignore[unresolved-attribute]
        return float(meta.fps)
    except Exception:
        logger.warning(f"Could not probe fps for {path}; defaulting to 30.0.")
        return 30.0


## Runner config


@dataclass(kw_only=True)
class FlashVSRRunnerConfig(RunnerConfig):
    """Runner config for the FlashVSR streaming video super-resolution pipeline."""

    _target: type = field(default_factory=lambda: FlashVSRRunner)

    input_path: Path = Path()
    """Low-resolution input video path. Must be readable by ``mediapy``."""

    chunk_size: Literal[8, 16] = 16
    """Steady-state frames per AR step; ``8`` uses one DiT iteration and
    ``16`` uses two. The matching cold-start size is derived from the encoder
    chunk table."""

    output_fps: float | None = None
    """Output frame rate; ``None`` uses the input fps, falling back to ``30.0``."""

    crop_region: Literal["none", "bottom_half", "top_half"] = "none"
    """Input crop before upsampling. ``bottom_half`` keeps Alpadreams RGB
    frames below the HDMap visualization."""

    sparse_ratio: float = 2.0
    """Sparse-attention budget multiplier used to re-derive
    ``transformer.topk_ratio`` from each video's post-crop target area."""


def _ensure_mgpu_config_supported(
    config: FlashVSRRunnerConfig, world_size: int
) -> None:
    """Reject FlashVSR configs that cannot run with multiple GPUs.

    Args:
        config: Runner config about to be used.
        world_size: Distributed world size; ``1`` means single-GPU or CPU.

    Raises:
        ValueError: ``world_size > 1`` with sparse ``block_sparse_attn``.
    """
    if world_size <= 1:
        return

    pipeline_cfg = config.pipeline
    assert isinstance(pipeline_cfg, FlashVSRPipelineConfig)
    transformer_cfg = pipeline_cfg.diffusion_model.transformer
    assert isinstance(transformer_cfg, FlashVSRTransformerConfig)
    if transformer_cfg.attention_mode == "full":
        return

    raise ValueError(
        "FlashVSR multi-GPU execution is supported only by the "
        "flashvsr-v1.1-full-attn preset (attention_mode='full'). Sparse "
        "FlashVSR presets use block_sparse_attn, which is not "
        "context-parallel aware and does not support multi-GPU execution. "
        f"Got runner_name={config.runner_name!r}, "
        f"attention_mode={transformer_cfg.attention_mode!r}."
    )


## Runner


class FlashVSRRunner(Runner[FlashVSRRunnerConfig, FlashVSRPipeline]):
    """FlashVSR streaming video super-resolution driver.

    Unlike the base :class:`Runner`, this driver builds its pipeline in
    :meth:`run` after reading the input video. The shipped pipeline literals are
    scaffolds for every non-resolution knob; the runner fills in
    ``encoder.input_H``, ``encoder.input_W``, and ``transformer.topk_ratio`` for
    each video before calling ``setup()``.
    """

    config: FlashVSRRunnerConfig

    def __init__(self, config: FlashVSRRunnerConfig) -> None:
        # Mirror ``Runner.__init__`` until pipeline construction; this runner
        # needs video dimensions before ``config.pipeline.setup()`` is valid.
        if _is_torchrun_env() and not torch.distributed.is_initialized():
            init_distributed()

        if torch.distributed.is_initialized():
            self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            self.world_size = torch.distributed.get_world_size()
            self.global_rank = torch.distributed.get_rank()
            self._device: str = f"cuda:{self.local_rank}"
        else:
            self.local_rank = 0
            self.world_size = 1
            self.global_rank = 0
            self._device = config.device
        self.is_rank_zero = self.global_rank == 0

        effective_config = config
        base_seed = config.pipeline.diffusion_model.seed
        if (
            config.offset_seed_by_global_rank
            and base_seed is not None
            and self.global_rank != 0
        ):
            effective_config = derive_config(
                config,
                pipeline=dict(
                    diffusion_model=dict(seed=base_seed + self.global_rank),
                ),
            )
        self.config = effective_config
        _ensure_mgpu_config_supported(self.config, self.world_size)
        # ``self.pipeline`` intentionally unset until :meth:`run` reads
        # the input video and derives the per-video pipeline config.

    def _load_input_video(self) -> tuple[torch.Tensor, int, int, float]:
        """Read the input video at native dimensions.

        Cropping is applied before tensor conversion so the returned ``H`` and
        ``W`` match the dimensions used to derive the per-video pipeline config.
        Device and dtype placement are deferred until :meth:`run`, after the
        pipeline has been constructed.

        Returns:
            ``(video_t, H, W, fps)`` where ``video_t`` is
            ``[1, C, T, H, W]`` in ``[-1, 1]`` on CPU/float32, and ``H`` /
            ``W`` are post-crop low-resolution pixel dimensions.
        """
        config = self.config
        path = config.input_path
        assert path != Path() and path.is_file(), (
            f"--input-path must point at an existing video file; got: {path!r}"
        )

        if self.is_rank_zero:
            logger.info(f"Reading {path} ...")
        video_np = media.read_video(str(path))[..., :3]  # uint8 [T, H, W, C]
        T, H, W, _ = video_np.shape

        if config.crop_region != "none":
            H_half = H // 2
            if config.crop_region == "bottom_half":
                video_np = video_np[:, H - H_half :, :, :]
            else:
                video_np = video_np[:, :H_half, :, :]
            H = video_np.shape[1]
            if self.is_rank_zero:
                logger.info(f"Cropped to {config.crop_region}: now {H}x{W}.")

        fps = config.output_fps
        if fps is None:
            fps = _probe_input_fps(path)

        # bf16/cuda placement is deferred to :meth:`run` once the
        # pipeline (and therefore the target device + dtype) exists.
        video_t = torch.from_numpy(video_np.astype(np.float32)) / 127.5 - 1.0
        video_t = rearrange(video_t, "T H W C -> 1 C T H W")
        return video_t, H, W, fps

    def _initialize_cache(self) -> FlashVSRPipelineCache:
        """Build a fresh FlashVSR pipeline cache.

        The configured prompt tensor is loaded during pipeline construction, so
        the runner does not pass an explicit ``prompt_tensor`` here.

        Returns:
            Fresh cache for one video rollout.
        """
        return self.pipeline.initialize_cache()

    def run(self) -> None:
        """Drive the FlashVSR rollout end-to-end.

        Reads the input video, builds the pipeline at the post-crop native
        ``(H, W)``, loops over chunked ``generate`` / ``finalize`` calls, and
        writes the assembled RGB video on rank zero.
        """
        config = self.config
        # Narrow the inherited pipeline field once so the FlashVSR-specific
        # encoder scale and transformer top-k fields are visible below.
        pipeline_cfg = config.pipeline
        assert isinstance(pipeline_cfg, FlashVSRPipelineConfig)
        transformer_cfg = pipeline_cfg.diffusion_model.transformer
        assert isinstance(transformer_cfg, FlashVSRTransformerConfig)

        first_size, subseq_size = _CHUNK_MODES[config.chunk_size]
        video_t_cpu, H, W, fps = self._load_input_video()
        total_frames = video_t_cpu.shape[2]

        # Build the pipeline for this video's post-crop dimensions. The encoder
        # handles 128-multiple target cropping; top-k must be recomputed from
        # that cropped target area to match upstream FlashVSR.
        target_H, target_W, topk_ratio = _resolve_target_and_topk_ratio(
            input_H=H,
            input_W=W,
            scale=pipeline_cfg.encoder.scale,
            sparse_ratio=config.sparse_ratio,
        )
        if self.is_rank_zero:
            mode_note = (
                f"topk_ratio={topk_ratio:.6f}"
                if transformer_cfg.attention_mode == "sparse"
                else "full attention (topk ignored)"
            )
            logger.info(
                f"Building FlashVSR pipeline at input_H={H}, input_W={W} "
                f"(post-crop target {target_H}x{target_W}, {mode_note}) ..."
            )
        pipeline_config = derive_config(
            pipeline_cfg,
            encoder=dict(input_H=H, input_W=W),
            diffusion_model=dict(
                transformer=dict(topk_ratio=topk_ratio),
            ),
        )
        self.pipeline = pipeline_config.setup().to(device=self._device).eval()

        dtype = self.pipeline.diffusion_model.dtype
        video_t = video_t_cpu.to(device=self.pipeline.device, dtype=dtype)
        del video_t_cpu

        chunks = _build_chunks(total_frames, first_size, subseq_size)
        assert chunks, (
            f"Input video too short ({total_frames} frames); need at "
            f"least {first_size} for chunk_size={config.chunk_size}."
        )
        usable_frames = sum(s for _, s in chunks)
        if usable_frames < total_frames:
            video_t = video_t[:, :, :usable_frames]
            if self.is_rank_zero:
                logger.info(f"Using first {usable_frames} of {total_frames} frames.")

        cache = self._initialize_cache()

        chunks_out: list[torch.Tensor] = []
        stats_history: list[dict[str, float]] = []
        for chunk_idx, (start, size) in enumerate(chunks):
            clip = video_t[:, :, start : start + size]
            video_chunk = self.pipeline.generate(
                autoregressive_index=chunk_idx,
                cache=cache,
                input=clip,
            )
            stats = self.pipeline.finalize(autoregressive_index=chunk_idx, cache=cache)
            if stats is not None:
                # Report throughput against the visible output frames for this
                # chunk. ``fps`` is reserved for the output video's frame rate.
                chunk_frames = int(video_chunk.shape[2])
                chunk_total_ms = stats["total_ms"]
                chunk_fps = (
                    chunk_frames / chunk_total_ms * 1000.0
                    if chunk_total_ms > 0
                    else 0.0
                )
                stats_history.append(
                    {
                        "autoregressive_index": chunk_idx,
                        **stats,
                        "frames": chunk_frames,
                        "fps": chunk_fps,
                    }
                )
            chunks_out.append(video_chunk.cpu())

        # [1, 3, T_out, target_H, target_W] in [-1, 1] -> [T_out, H, W, 3].
        generated = torch.cat(chunks_out, dim=2)
        if not self.is_rank_zero:
            return

        config.output_dir.mkdir(parents=True, exist_ok=True)
        video_path = config.output_dir / f"{config.runner_name}.mp4"
        canvas = rearrange(generated, "1 c t h w -> t h w c")
        arr = ((canvas.float().numpy() + 1.0) / 2.0 * 255).clip(0, 255).astype("uint8")
        media.write_video(str(video_path), arr, fps=fps)

        logger.info(
            f"[{config.runner_name}] wrote video {tuple(generated.shape)} "
            f"-> {video_path.resolve()}"
        )

        if stats_history:
            stats_path = config.output_dir / f"stats_{config.runner_name}.json"
            stats_path.write_text(json.dumps(stats_history, indent=2))
            logger.info(
                f"[{config.runner_name}] wrote per-AR-step stats -> "
                f"{stats_path.resolve()}"
            )
