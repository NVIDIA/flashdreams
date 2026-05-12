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

"""Non-streaming Cosmos-Predict2 T2V runner."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import mediapy as media
from einops import rearrange
from loguru import logger

from flashdreams.infra.decoder import StreamingVideoDecoder
from flashdreams.infra.runner import Runner, RunnerConfig
from flashdreams.recipes.cosmos.pipeline import (
    CosmosInferencePipeline,
    CosmosInferencePipelineCache,
)

__all__ = [
    "Cosmos2T2VRunnerConfig",
    "Cosmos2T2VRunner",
]

DEFAULT_T2V_PROMPT = (
    "A robotic arm, primarily white with black joints and cables, "
    "is shown in a clean, modern indoor setting with a white tabletop. "
    "The arm, equipped with a gripper holding a small, light green pitcher, "
    "is positioned above a clear glass containing a reddish-brown liquid and a spoon. "
    "The robotic arm is in the process of pouring a transparent liquid into the glass. "
    "To the left of the pitcher, there is an opened jar with a similar reddish-brown "
    "substance visible through its transparent body. In the background, a vase with "
    "white flowers and a brown couch are partially visible, adding to the contemporary ambiance. "
    "The lighting is bright, casting soft shadows on the table. The robotic arm's movements are "
    "smooth and controlled, demonstrating precision in its task. As the video progresses, "
    "the robotic arm completes the pour, leaving the glass half-filled with the reddish-brown liquid. "
    "The jar remains untouched throughout the sequence, and the spoon inside the glass remains stationary. "
    "The other robotic arm on the right side also stays stationary throughout the video. "
    "The final frame captures the robotic arm with the pitcher finishing the pour, with the glass now "
    "filled to a higher level, while the pitcher is slightly tilted but still held securely by the gripper."
)
"""Default demo prompt used when no ``--prompt`` is supplied."""


@dataclass(kw_only=True)
class Cosmos2T2VRunnerConfig(RunnerConfig):
    """Runner config for the Cosmos-Predict2 T2V variant."""

    _target: type = field(default_factory=lambda: Cosmos2T2VRunner)

    prompt: str | Path = DEFAULT_T2V_PROMPT
    """Either an inline text prompt (--prompt "...") or a path to a
    txt file whose first line is read as the prompt (--prompt prompt.txt).
    Defaults to ``DEFAULT_T2V_PROMPT``."""

    pixel_height: int = 720
    """Output video pixel height."""

    pixel_width: int = 1280
    """Output video pixel width."""

    fps: int = 16
    """Output video frame rate."""


class Cosmos2T2VRunner(Runner[Cosmos2T2VRunnerConfig, CosmosInferencePipeline]):
    """Cosmos-Predict2 non-streaming T2V driver."""

    config: Cosmos2T2VRunnerConfig

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

    def _initialize_cache(self) -> CosmosInferencePipelineCache:
        """Initialize the autoregressive cache for T2V."""
        config = self.config
        prompt = self._resolve_prompt()

        assert isinstance(self.pipeline.decoder, StreamingVideoDecoder)
        sp = self.pipeline.decoder.spatial_compression_ratio
        assert config.pixel_height % sp == 0, (
            f"pixel_height={config.pixel_height} must divide {sp}."
        )
        assert config.pixel_width % sp == 0, (
            f"pixel_width={config.pixel_width} must divide {sp}."
        )
        latent_h = config.pixel_height // sp
        latent_w = config.pixel_width // sp

        return self.pipeline.initialize_cache(
            text=[prompt], image=None, height=latent_h, width=latent_w
        )

    def run(self) -> None:
        """Drive the single-step rollout and write outputs."""
        config = self.config

        cache = self._initialize_cache()

        generated = self.pipeline.generate(autoregressive_index=0, cache=cache)
        stats = self.pipeline.finalize(autoregressive_index=0, cache=cache)
        if not self.is_rank_zero:
            return
        generated = generated.cpu()

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

        if stats is not None:
            stats_path = config.output_dir / f"stats_{config.runner_name}.json"
            stats_path.write_text(
                json.dumps([{"autoregressive_index": 0, **stats}], indent=2)
            )
            logger.info(
                f"[{config.runner_name}] wrote per-AR-step stats "
                f"-> {stats_path.resolve()}"
            )
