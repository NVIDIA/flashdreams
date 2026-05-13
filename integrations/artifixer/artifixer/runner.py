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

"""ArtiFixer DMD-distilled streaming T2V runner.

Text-only runner: a stripped clone of ``SelfForcingT2VRunner`` that
accepts a single text prompt. Used for smoke-testing the recipe wiring
and the ``flashdreams-run`` CLI entry point.

The full ArtiFixer-specific conditioning surface (``rgb_rendered``,
``opacity``, neighbor frames, camera matrices) is wired in through the
:class:`ArtifixerInferencePipeline.initialize_cache` / ``generate``
path; the dreamfix-side driver in
``dreamfix/model_eval/flashdreams_backend.py`` is the production entry
point that feeds those conditioning tensors. This runner stays
text-only as a lightweight harness.
"""

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
    "ArtifixerDmdT2VRunnerConfig",
    "ArtifixerDmdT2VRunner",
]


@dataclass(kw_only=True)
class ArtifixerDmdT2VRunnerConfig(RunnerConfig):
    """Runner config for the ArtiFixer DMD T2V variants."""

    _target: type = field(default_factory=lambda: ArtifixerDmdT2VRunner)

    prompt: str | Path = Path(__file__).resolve().parents[1] / "assets" / "prompt.txt"
    """Either an inline prompt (``--prompt "..."``) or a path to a ``.txt``
    whose first non-empty line is read as the prompt."""

    total_blocks: int = 3
    """Number of autoregressive chunks to generate. Matches dreamfix's
    ``num_frames=81`` / ``frames_per_block=7`` (21 latent frames / 7 = 3
    chunks for the Wan VAE temporal stride of 4)."""

    pixel_height: int = 480
    pixel_width: int = 832
    fps: int = 16


class ArtifixerDmdT2VRunner(Runner[ArtifixerDmdT2VRunnerConfig, WanInferencePipeline]):
    """ArtiFixer DMD streaming T2V driver (text-only smoke runner)."""

    config: ArtifixerDmdT2VRunnerConfig

    def _resolve_prompt(self) -> str:
        value = self.config.prompt
        if isinstance(value, Path):
            lines = [ln.strip() for ln in value.read_text().splitlines() if ln.strip()]
            assert lines, f"prompt file {value} has no non-empty lines"
            return lines[0]
        assert value, "--prompt must be a non-empty string or a path to a .txt file"
        return value

    def _initialize_cache(self) -> WanInferencePipelineCache:
        config = self.config
        prompt = self._resolve_prompt()

        assert isinstance(self.pipeline.decoder, StreamingVideoDecoder)
        spatial_compression_ratio = self.pipeline.decoder.spatial_compression_ratio
        assert config.pixel_height % spatial_compression_ratio == 0, (
            f"pixel_height={self.config.pixel_height} must divide "
            f"{spatial_compression_ratio}."
        )
        assert config.pixel_width % spatial_compression_ratio == 0, (
            f"pixel_width={self.config.pixel_width} must divide "
            f"{spatial_compression_ratio}."
        )
        latent_h = config.pixel_height // spatial_compression_ratio
        latent_w = config.pixel_width // spatial_compression_ratio

        return self.pipeline.initialize_cache(
            text=[prompt], image=None, height=latent_h, width=latent_w
        )

    def run(self) -> None:
        config = self.config

        cache = self._initialize_cache()

        chunks: list[torch.Tensor] = []
        stats_history: list[dict[str, float]] = []
        for i in range(config.total_blocks):
            video_chunk = self.pipeline.generate(autoregressive_index=i, cache=cache)
            stats = self.pipeline.finalize(autoregressive_index=i, cache=cache)
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
                f"[{config.runner_name}] wrote per-AR-step stats -> {stats_path.resolve()}"
            )
