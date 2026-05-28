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

"""Self-Forcing Wan 2.1 streaming T2V runner class."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import mediapy as media
import torch
from einops import rearrange
from loguru import logger

from flashdreams.infra.decoder import StreamingVideoDecoder
from flashdreams.infra.runner import Runner, RunnerConfig
from flashdreams.recipes.wan import (
    WanInferencePipeline,
    WanInferencePipelineCache,
)

__all__ = [
    "SelfForcingT2VRunnerConfig",
    "SelfForcingT2VRunner",
]


DEFAULT_T2V_PROMPT = (
    "A stylish woman strolls down a bustling Tokyo street, the warm glow of "
    "neon lights and animated city signs casting vibrant reflections. She "
    "wears a sleek black leather jacket paired with a flowing red dress and "
    "black boots, her black purse slung over her shoulder. Sunglasses perched "
    "on her nose and a bold red lipstick add to her confident, casual "
    "demeanor. The street is damp and reflective, creating a mirror-like "
    "effect that enhances the colorful lights and shadows. Pedestrians move "
    "about, adding to the lively atmosphere. The scene is captured in a "
    "dynamic medium shot with the woman walking slightly to one side, "
    "highlighting her graceful strides."
)


@dataclass(kw_only=True)
class SelfForcingT2VRunnerConfig(RunnerConfig):
    """Runner config for the Self-Forcing T2V variants."""

    _target: type["SelfForcingT2VRunner"] = field(
        default_factory=lambda: SelfForcingT2VRunner
    )

    prompt: str | Path = DEFAULT_T2V_PROMPT
    """Either an inline text prompt (--prompt "...") or a path to a
    txt file whose first line is read as the prompt (--prompt prompt.txt)."""

    total_blocks: int = 60
    """Number of autoregressive chunks to generate before terminating the rollout."""

    pixel_height: int = 480
    """Output video pixel height."""

    pixel_width: int = 832
    """Output video pixel width."""

    fps: int = 16
    """Output video frame rate."""


class SelfForcingT2VRunner(Runner[SelfForcingT2VRunnerConfig, WanInferencePipeline]):
    """Self-Forcing Wan 2.1 streaming T2V driver."""

    config: SelfForcingT2VRunnerConfig

    def _resolve_prompt(self) -> str:
        """Resolve config.prompt.

        A Path reads its first non-empty line, a str is used as-is.
        """
        value = self.config.prompt
        if isinstance(value, Path):
            lines = [ln.strip() for ln in value.read_text().splitlines() if ln.strip()]
            assert lines, f"prompt file {value} has no non-empty lines"
            return lines[0]
        assert value, "--prompt must be a non-empty string or a path to a .txt file"
        return value

    def _initialize_cache(self) -> WanInferencePipelineCache:
        """Initialize the autoregressive cache."""
        config = self.config
        prompt = self._resolve_prompt()

        assert isinstance(self.pipeline.decoder, StreamingVideoDecoder)
        spatial_compression_ratio = self.pipeline.decoder.spatial_compression_ratio
        assert config.pixel_height % spatial_compression_ratio == 0, (
            f"pixel_height={self.config.pixel_height} must divide "
            f"{spatial_compression_ratio}."
        )
        assert config.pixel_width % spatial_compression_ratio == 0, (
            f"pixel_width={self.config.pixel_width} must divide {spatial_compression_ratio}."
        )
        latent_h = config.pixel_height // spatial_compression_ratio
        latent_w = config.pixel_width // spatial_compression_ratio

        return self.pipeline.initialize_cache(
            text=[prompt], image=None, height=latent_h, width=latent_w
        )

    def run(self) -> None:
        """Drive the autoregressive rollout and write outputs."""
        config = self.config

        # Initialize the autoregressive cache.
        cache = self._initialize_cache()

        # Generate the autoregressive chunks.
        chunks: list[torch.Tensor] = []
        stats_history: list[dict[str, float]] = []
        for i in range(config.total_blocks):
            video_chunk = self.pipeline.generate(autoregressive_index=i, cache=cache)
            stats = self.pipeline.finalize(autoregressive_index=i, cache=cache)
            if stats is not None:
                stats_history.append({"autoregressive_index": i, **stats})
            chunks.append(video_chunk.cpu())

        # Concatenate the autoregressive chunks along the time axis.
        # The result is a tensor of shape [T, C, H, W], value in [-1, 1].
        generated = torch.cat(chunks, dim=0)
        if not self.is_rank_zero:
            return

        # Write the video.
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

        # Write the perf stats.
        if stats_history:
            stats_path = config.output_dir / f"stats_{config.runner_name}.json"
            stats_path.write_text(json.dumps(stats_history, indent=2))
            logger.info(
                f"[{config.runner_name}] wrote per-AR-step stats -> {stats_path.resolve()}"
            )
