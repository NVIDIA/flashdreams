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

"""FlashVSR streaming video super-resolution runner.

Pipeline construction is deferred from ``Runner.__init__`` to
:meth:`FlashVSRRunner.run` so the encoder's ``input_H`` / ``input_W``
can be set to the input video's native pixel dims. The
``PIPELINE_FLASHVSR_V1_1_SPARSE_*`` literals in :mod:`flashvsr.config`
therefore act as a scaffold supplying every non-resolution knob; the
runner overrides ``encoder.input_H`` / ``encoder.input_W`` per video
via ``derive_config`` before calling ``setup()``, and also re-derives
``diffusion_model.transformer.topk_ratio`` from the per-video
post-crop target dims so it matches upstream FlashVSR's
``sparse_ratio * 768*1280 / (th*tw)`` formula (the placeholder value
baked at builder time is stale for every non-scaffold input).
"""

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

__all__ = [
    "FlashVSRRunnerConfig",
    "FlashVSRRunner",
]


def _chunk_modes() -> dict[int, tuple[int, int]]:
    """``{steady_size: (cold_size, steady_size)}`` derived from the encoder.

    Single-source-of-truth: invert :data:`FlashVSREncoder._CHUNK_FRAME_TARGETS`
    (``{raw -> padded}``) into a ``{padded -> raw_cold}`` map and pair each
    ``padded`` with itself as the steady size. Currently yields
    ``{8: (5, 8), 16: (13, 16)}`` -- the legacy
    ``_CHUNK_TARGET = {5: 8, 13: 16, 8: 8, 16: 16}`` table.
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
    """Return list of (start, size) pairs for each AR step.

    The first chunk is ``first_size`` frames (cold-start; pad-left
    replicated by the encoder); subsequent chunks are ``subseq_size`` each.
    A trailing partial chunk is dropped with a warning.
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
    """Post-crop target dims + matching ``topk_ratio`` for one input.

    Mirrors the encoder's bicubic-then-128-multiple-crop rule (see
    :class:`flashvsr.encoder.FlashVSREncoder`) plus upstream's per-input
    ``topk_ratio = sparse_ratio * 768*1280 / (th*tw)`` formula (every
    upstream inference script under ``examples/WanVSR/`` uses this
    exact constant; see e.g. ``infer_flashvsr_v1.1_tiny.py:222``). Kept
    as a free function so the test suite can exercise the math without
    spinning up an actual rollout.

    Returns ``(target_H, target_W, topk_ratio)``. Raises
    :class:`AssertionError` if either axis is too small to fit one
    128-multiple post-scale.
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
    """Best-effort fps probe; falls back to 30.0 on failure."""
    try:
        # ``mediapy`` ships without type stubs so ty cannot see the
        # ``VideoMetadata.from_path`` classmethod (added in mediapy 1.1).
        meta = media.VideoMetadata.from_path(str(path))  # ty: ignore[unresolved-attribute]
        return float(meta.fps)
    except Exception:
        logger.warning(f"Could not probe fps for {path}; defaulting to 30.0.")
        return 30.0


@dataclass(kw_only=True)
class FlashVSRRunnerConfig(RunnerConfig):
    """Runner config for the FlashVSR streaming video super-resolution pipeline."""

    _target: type = field(default_factory=lambda: FlashVSRRunner)

    input_path: Path = Path()
    """Path to a low-resolution input video readable by ``mediapy``
    (``.mp4`` and friends). Required; pass via ``--input-path <path>``.
    The pipeline encoder's ``input_H`` / ``input_W`` are auto-set to
    the video's native dimensions; non-128-aligned upres dims are
    handled by the encoder via a symmetric crop (see
    :class:`flashvsr.encoder.FlashVSREncoderConfig`)."""

    chunk_size: Literal[8, 16] = 16
    """Steady-state frames per AR step (cold-start uses ``chunk_size - 3``).
    ``16`` (default) packs two DiT iters per ``pipeline.generate()`` call
    (first=13, subseq=16). ``8`` runs one DiT iter per call (first=5,
    subseq=8) and roughly halves per-chunk peak VRAM at the cost of
    more boundary stitching overhead."""

    output_fps: float | None = None
    """Output frame rate. ``None`` (default) falls back to the input
    video's probed fps (30.0 if probing fails)."""

    crop_region: Literal["none", "bottom_half", "top_half"] = "none"
    """Crop input frames before upsampling. Use ``bottom_half`` to drop
    the HDMap visualization stacked on top of Alpadreams outputs and
    upscale only the generated RGB."""

    sparse_ratio: float = 2.0
    """Block-sparse attention budget multiplier used to re-derive
    ``transformer.topk_ratio`` from the per-video post-crop target dims
    inside :meth:`FlashVSRRunner.run` (mirrors upstream's
    ``topk_ratio = sparse_ratio * 768*1280 / (th*tw)`` at every call
    site in ``examples/WanVSR/infer_flashvsr_v1.1_tiny.py`` and
    siblings). The shipped runners (``flashvsr-v1.1-sparse-ratio-1.5``,
    ``flashvsr-v1.1-sparse-ratio-2.0``) set this to match their slug;
    callers building a runner config by hand should set it to the
    ``sparse_ratio`` they passed into :func:`build_flashvsr_v1_1`."""


class FlashVSRRunner(Runner[FlashVSRRunnerConfig, FlashVSRPipeline]):
    """FlashVSR streaming video super-resolution driver.

    Overrides :meth:`Runner.__init__` to skip the parent's eager
    ``config.pipeline.setup()`` call: the encoder's ``input_H`` /
    ``input_W`` must track the input video's native pixel dims, so the
    pipeline can only be built inside :meth:`run` once the video has
    been read.
    """

    config: FlashVSRRunnerConfig

    def __init__(self, config: FlashVSRRunnerConfig) -> None:
        # Mirrors :meth:`flashdreams.infra.runner.Runner.__init__` minus
        # the final ``self.pipeline = config.pipeline.setup()`` step.
        # The pipeline is built in :meth:`run` once the input video's
        # native ``(H, W)`` is known.
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
        # ``self.pipeline`` intentionally unset until :meth:`run` reads
        # the input video and derives the per-video pipeline config.

    def _load_input_video(self) -> tuple[torch.Tensor, int, int, float]:
        """Read the input video at native dims (no resize).

        Returns ``(video_t, H, W, fps)`` where ``video_t`` is
        ``[1, C, T, H, W]`` in ``[-1, 1]`` on CPU/float32, and ``(H, W)``
        is the (post-crop) native pixel size. Conversion to the pipeline
        device + dtype happens in :meth:`run` after the pipeline has
        been built (we don't have ``self.pipeline.device`` yet here).
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
        """Build a fresh per-rollout cache.

        FlashVSR loads its frozen UMT5 prompt embedding from
        ``config.pipeline.prompt_path`` at construction time, so the cache
        seed is empty here."""
        return self.pipeline.initialize_cache()

    def run(self) -> None:
        """Drive the FlashVSR rollout end-to-end.

        Read the input video, build the pipeline at the video's native
        ``(H, W)``, then loop the ``generate`` / ``finalize`` pair over
        the per-AR-step chunks.
        """
        config = self.config
        # Narrow from the inherited ``RunnerConfig.pipeline:
        # StreamInferencePipelineConfig`` to the FlashVSR-specific subclass
        # so subsequent ``encoder.scale`` / ``transformer.topk_ratio``
        # lookups type-check, and so ``derive_config`` (generic in the
        # base type) returns ``FlashVSRPipelineConfig`` directly without
        # a cast.
        pipeline_cfg = config.pipeline
        assert isinstance(pipeline_cfg, FlashVSRPipelineConfig)

        first_size, subseq_size = _CHUNK_MODES[config.chunk_size]
        video_t_cpu, H, W, fps = self._load_input_video()
        total_frames = video_t_cpu.shape[2]

        # Build the pipeline at the input video's native (H, W). The
        # encoder's bicubic-then-128-multiple-crop handles non-aligned
        # dims internally (see :class:`flashvsr.encoder.FlashVSREncoder`);
        # we mirror upstream's per-input ``topk_ratio = sparse_ratio *
        # 768*1280 / (th*tw)`` (with ``(th, tw)`` the **post-crop**
        # target) by overriding ``transformer.topk_ratio`` here, since
        # the placeholder value baked at builder time used the recipe's
        # scaffold ``input_H=704, input_W=1280`` and is stale for every
        # other input.
        target_H, target_W, topk_ratio = _resolve_target_and_topk_ratio(
            input_H=H,
            input_W=W,
            scale=pipeline_cfg.encoder.scale,
            sparse_ratio=config.sparse_ratio,
        )
        if self.is_rank_zero:
            logger.info(
                f"Building FlashVSR pipeline at input_H={H}, input_W={W} "
                f"(post-crop target {target_H}x{target_W}, "
                f"topk_ratio={topk_ratio:.6f}) ..."
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
                # Per-chunk throughput. ``video_chunk.shape[2]`` is the
                # post-trim output frame count (cold-start drops
                # ``frames_to_trim`` frames so the cold + steady chunks
                # both report their visible frame counts here);
                # ``total_ms`` is the sum of pipeline.finalize's stage
                # events, so this is the per-chunk realised FPS.
                #
                # Named ``chunk_fps`` (not ``fps``) so we don't shadow
                # the outer ``fps`` that ``media.write_video`` consumes
                # below.
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
