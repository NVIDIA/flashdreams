# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LTX-Video T2V streaming runner."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import mediapy as media
import torch
from einops import rearrange
from loguru import logger

from flashdreams.infra.runner import Runner, RunnerConfig
from ltx_video.cache import LTXPipelineCache
from ltx_video.pipeline import LTXVideoStreamingPipeline

__all__ = ["LTXVideoT2VRunnerConfig", "LTXVideoT2VRunner"]

DEFAULT_PROMPT = (
    "A coastal road at dusk, waves breaking on rocks, cinematic wide shot"
)


@dataclass(kw_only=True)
class LTXVideoT2VRunnerConfig(RunnerConfig):
    """Runner config for LTX-Video T2V streaming."""

    _target: type["LTXVideoT2VRunner"] = field(
        default_factory=lambda: LTXVideoT2VRunner
    )

    prompt: str | Path = DEFAULT_PROMPT
    total_blocks: int = 7
    pixel_height: int = 512
    pixel_width: int = 768
    fps: int = 24


class LTXVideoT2VRunner(Runner[LTXVideoT2VRunnerConfig, LTXVideoStreamingPipeline]):
    """Drive LTX-Video autoregressive streaming and write outputs."""

    config: LTXVideoT2VRunnerConfig

    def _resolve_prompt(self) -> str:
        value = self.config.prompt
        if isinstance(value, Path):
            lines = [ln.strip() for ln in value.read_text().splitlines() if ln.strip()]
            assert lines, f"prompt file {value} has no non-empty lines"
            return lines[0]
        assert value, "--prompt must be a non-empty string or a path to a .txt file"
        return str(value)

    def run(self) -> None:
        config = self.config
        prompt = self._resolve_prompt()
        cache: LTXPipelineCache = self.pipeline.initialize_cache(text=[prompt])

        chunks: list[torch.Tensor] = []
        stats_history: list[dict[str, float | int]] = []
        for i in range(config.total_blocks):
            video_chunk = self.pipeline.generate(
                i,
                cache,
                width=config.pixel_width,
                height=config.pixel_height,
            )
            stats = self.pipeline.finalize(i, cache)
            if stats is not None:
                stats_history.append({"autoregressive_index": i, **stats})
            chunks.append(video_chunk.cpu())

        generated = torch.cat(chunks, dim=0)
        if not self.is_rank_zero:
            return

        config.output_dir.mkdir(parents=True, exist_ok=True)
        video_path = config.output_dir / f"{config.runner_name}.mp4"
        canvas = rearrange(generated, "t c h w -> t h w c")
        arr = (canvas.float().numpy() + 1.0) / 2.0
        arr = (arr * 255).clip(0, 255).astype("uint8")
        media.write_video(str(video_path), arr, fps=config.fps)

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
