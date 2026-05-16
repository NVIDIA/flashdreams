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
import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapy as media
import torch
from einops import rearrange
from loguru import logger

from flashdreams.core.io.download import download_to_cache
from flashdreams.infra.decoder import StreamingVideoDecoder
from flashdreams.infra.runner import Runner, RunnerConfig
from flashdreams.recipes.cosmos.pipeline import (
    CosmosInferencePipeline,
    CosmosInferencePipelineCache,
)

__all__ = [
    "Cosmos2I2VRunner",
    "Cosmos2I2VRunnerConfig",
    "Cosmos2T2VRunner",
    "Cosmos2T2VRunnerConfig",
]

DEFAULT_PROMPT = (
    "A high-definition video captures the precision of robotic welding in an industrial setting. "
    "The first frame showcases a robotic arm, equipped with a welding torch, positioned over a "
    "large metal structure. The welding process is in full swing, with bright sparks and intense "
    "light illuminating the scene, creating a vivid display of blue and white hues. A significant "
    "amount of smoke billows around the welding area, partially obscuring the view but emphasizing "
    "the heat and activity. The background reveals parts of the workshop environment, including a "
    "ventilation system and various pieces of machinery, indicating a busy and functional industrial "
    "workspace. As the video progresses, the robotic arm maintains its steady position, continuing "
    "the welding process and moving to its left. The welding torch consistently emits sparks and light, "
    "and the smoke continues to rise, diffusing slightly as it moves upward. The metal surface beneath "
    "the torch shows ongoing signs of heating and melting. The scene retains its industrial ambiance, "
    "with the welding sparks and smoke dominating the visual field, underscoring the ongoing nature of "
    "the welding operation."
)
"""Default demo prompt used when no ``--prompt`` is supplied."""


DEFAULT_I2V_IMAGE_URL = "https://media.githubusercontent.com/media/nvidia-cosmos/cosmos-predict2.5/refs/heads/main/assets/base/robot_welding.jpg"

IMAGE_CACHE_DIR = (
    Path(os.path.expanduser(os.getenv("FLASHDREAMS_CACHE_DIR", "~/.cache/flashdreams")))
    / "cosmos_predict2"
)
"""User-writable cache for on-the-fly I2V first-frame downloads."""


def _resolve_image_path(image_path: str | Path) -> Path:
    """Return a local ``Path`` for ``image_path``, downloading URLs on the fly.

    ``http(s)://`` strings are atomically fetched into
    :data:`IMAGE_CACHE_DIR` and validated as decodable images before
    being published; local paths pass through unchanged.
    """
    if isinstance(image_path, Path):
        return image_path
    if not image_path.startswith(("http://", "https://")):
        return Path(image_path)

    return download_to_cache(
        image_path,
        cache_dir=IMAGE_CACHE_DIR,
        validator=lambda p: media.read_image(str(p)),
    )


@dataclass(kw_only=True)
class Cosmos2T2VRunnerConfig(RunnerConfig):
    """Runner config for the Cosmos-Predict2 T2V variant."""

    _target: type = field(default_factory=lambda: Cosmos2T2VRunner)

    prompt: str | Path = DEFAULT_PROMPT
    """Either an inline text prompt (--prompt "...") or a path to a
    txt file whose first line is read as the prompt (--prompt prompt.txt)."""

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


@dataclass(kw_only=True)
class Cosmos2I2VRunnerConfig(Cosmos2T2VRunnerConfig):
    """Runner config for the Cosmos-Predict2 I2V variant."""

    _target: type = field(default_factory=lambda: Cosmos2I2VRunner)

    image_path: str | Path = DEFAULT_I2V_IMAGE_URL
    """First-frame RGB image. Either a local path or an HTTP(S) URL."""


class Cosmos2I2VRunner(Cosmos2T2VRunner):
    """Cosmos-Predict2 non-streaming I2V driver."""

    config: Cosmos2I2VRunnerConfig

    def _initialize_cache(self) -> CosmosInferencePipelineCache:
        """Initialize the autoregressive cache for I2V (loads first frame)."""
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

        # Load + resize the first frame, then convert to [-1, 1] bf16
        # in shape [T=1, C, H, W] (matches batch_shape=()). Pin to the
        # pipeline's actual device so non-default ``--device`` selections
        # (and the auto cuda:LOCAL_RANK override under torchrun) both work.
        local_image_path = _resolve_image_path(config.image_path)
        arr = media.read_image(str(local_image_path))[..., :3]
        arr = cv2.resize(arr, (config.pixel_width, config.pixel_height))
        tensor = (
            torch.from_numpy(arr).to(device=self.pipeline.device, dtype=torch.bfloat16)
            / 127.5
            - 1.0
        )
        image = rearrange(tensor, "h w c -> 1 c h w")

        return self.pipeline.initialize_cache(text=[prompt], image=image)
