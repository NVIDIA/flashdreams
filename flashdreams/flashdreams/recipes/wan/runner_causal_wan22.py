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

"""Streaming causal Wan 2.2 runner (FastVideo distilled T2V) for ``flashdreams-run``."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from einops import rearrange
from loguru import logger

from flashdreams.configs.registry import register_runner
from flashdreams.infra.runner import Runner, RunnerConfig
from flashdreams.recipes.wan.config.causal_wan22 import (
    FASTVIDEO_T2V,
    WAN_VAE_SPATIAL_COMPRESSION,
)
from flashdreams.recipes.wan.pipeline import WanInferencePipeline
from flashdreams.recipes.wan.runner_causal_wan21 import (
    DEFAULT_PIXEL_HEIGHT,
    DEFAULT_PIXEL_WIDTH,
    DEFAULT_PROMPT,
)


@dataclass(kw_only=True)
class CausalWan22RunnerConfig(RunnerConfig):
    """Runner config for the streaming causal Wan 2.2 T2V variants."""

    _target: type = field(default_factory=lambda: CausalWan22Runner)

    prompt: str = DEFAULT_PROMPT
    """Text prompt. Falls back to :attr:`prompt_path` when empty."""

    prompt_path: Path | None = None
    """Optional path to a ``.txt`` whose first line is the prompt; wins
    over :attr:`prompt` when set."""

    total_blocks: int = 60
    """Number of AR chunks to generate before terminating the rollout."""

    pixel_height: int = DEFAULT_PIXEL_HEIGHT
    """Output video pixel height. Must divide
    ``WAN_VAE_SPATIAL_COMPRESSION`` cleanly."""

    pixel_width: int = DEFAULT_PIXEL_WIDTH
    """Output video pixel width. Same divisibility rule as
    :attr:`pixel_height`."""

    fps: int = 16
    """Output video frame rate. Wan 2.2's training fps."""


class CausalWan22Runner(Runner[CausalWan22RunnerConfig, WanInferencePipeline]):
    """Streaming causal Wan 2.2 T2V driver (FastVideo distilled MoE)."""

    config: CausalWan22RunnerConfig

    def _resolve_prompt(self) -> str:
        cfg = self.config
        if cfg.prompt_path is not None:
            text = cfg.prompt_path.read_text().splitlines()
            assert text, f"prompt file {cfg.prompt_path} is empty"
            return text[0].strip()
        assert cfg.prompt, (
            "either --prompt or --prompt_path must be set "
            "(both empty resolved to no text input)."
        )
        return cfg.prompt

    def _initialize_cache(self) -> Any:
        cfg = self.config
        prompt = self._resolve_prompt()
        latent_h = cfg.pixel_height // WAN_VAE_SPATIAL_COMPRESSION
        latent_w = cfg.pixel_width // WAN_VAE_SPATIAL_COMPRESSION
        return self.pipeline.initialize_cache(
            text=[prompt], image=None, height=latent_h, width=latent_w
        )

    def run(self) -> None:
        """Drive the AR rollout for ``total_blocks`` chunks and write outputs."""
        cfg = self.config
        assert cfg.pixel_height % WAN_VAE_SPATIAL_COMPRESSION == 0, (
            f"pixel_height={cfg.pixel_height} must divide "
            f"{WAN_VAE_SPATIAL_COMPRESSION}."
        )
        assert cfg.pixel_width % WAN_VAE_SPATIAL_COMPRESSION == 0, (
            f"pixel_width={cfg.pixel_width} must divide {WAN_VAE_SPATIAL_COMPRESSION}."
        )

        cache = self._initialize_cache()

        torch.cuda.synchronize()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        chunks: list[torch.Tensor] = []
        stats_history: list[dict[str, float]] = []
        for i in range(cfg.total_blocks):
            num_frames = self.pipeline.get_num_output_frames(i)
            if self.is_rank_zero:
                logger.info(
                    f"[{cfg.runner_name}] AR step {i}/{cfg.total_blocks}, "
                    f"num_frames={num_frames}"
                )
            video_chunk = self.pipeline.generate(autoregressive_index=i, cache=cache)
            stats = self.pipeline.finalize(autoregressive_index=i, cache=cache)
            if stats is not None:
                stats_history.append({"autoregressive_index": i, **stats})
            chunks.append(video_chunk.cpu())

        generated = torch.cat(chunks, dim=1)  # [B=1, T, C, H, W]
        if not self.is_rank_zero:
            return

        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        video_path = cfg.output_dir / f"{cfg.runner_name}.mp4"
        canvas = rearrange(generated, "1 t c h w -> t h w c")
        _write_video(canvas, video_path, fps=cfg.fps)
        logger.info(
            f"[{cfg.runner_name}] wrote video {tuple(generated.shape)} "
            f"-> {video_path.resolve()}"
        )

        if stats_history:
            stats_path = cfg.output_dir / f"stats_{cfg.runner_name}.json"
            stats_path.write_text(json.dumps(stats_history, indent=2))
            logger.info(
                f"[{cfg.runner_name}] wrote per-AR-step stats -> {stats_path.resolve()}"
            )


CAUSAL_WAN22_RUNNERS: dict[str, RunnerConfig] = {
    FASTVIDEO_T2V.recipe_name: CausalWan22RunnerConfig(
        runner_name=FASTVIDEO_T2V.recipe_name,
        description="FastVideo distilled CausalWan 2.2 14B MoE T2V (8-step schedule).",
        pipeline=FASTVIDEO_T2V,
    ),
}
"""All shipped streaming causal Wan 2.2 runners, keyed by ``runner_name``."""

for _name, _cfg in CAUSAL_WAN22_RUNNERS.items():
    register_runner(_name, _cfg, source="builtin")


__all__ = [
    "CAUSAL_WAN22_RUNNERS",
    "CausalWan22Runner",
    "CausalWan22RunnerConfig",
]


## I/O helpers (lazy-imported under the ``runners`` install extras).


def _write_video(canvas: torch.Tensor, path: Path, *, fps: int) -> None:
    """Save a ``[T, H, W, C]`` ``[-1, 1]`` tensor as an MP4."""
    try:
        import mediapy as media  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - import-time gate
        raise ImportError(
            "Writing the output video needs mediapy. Install the runner "
            "extras: pip install 'flashdreams[runners]'."
        ) from exc

    arr = (canvas.float().numpy() + 1.0) / 2.0
    arr = (arr * 255).clip(0, 255).astype("uint8")
    media.write_video(str(path), arr, fps=fps)
