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

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
from loguru import logger

from flashdreams.infra.decoder import StreamingVideoDecoder
from flashdreams.infra.postprocess import VideoTensorLayout
from flashdreams.infra.runner import Runner, RunnerConfig
from flashdreams.infra.runner_io import (
    ensure_output_dir,
    load_first_frame_tensor,
    read_image_rgb,
    resolve_input_path,
    resolve_prompt_value,
    runner_artifact_path,
    write_runner_stats,
    write_video_tensor,
)
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


@dataclass(kw_only=True)
class Cosmos2T2VRunnerConfig(RunnerConfig):
    """Runner config for the Cosmos-Predict2 T2V variant."""

    _target: type["Cosmos2T2VRunner"] = field(default_factory=lambda: Cosmos2T2VRunner)

    prompt: str | Path = DEFAULT_PROMPT
    """Either an inline text prompt (--prompt "...") or a path to a
    txt file whose first line is read as the prompt (--prompt prompt.txt)."""

    pixel_height: int = 720
    """Output video pixel height."""

    pixel_width: int = 1280
    """Output video pixel width."""

    fps: int = 16
    """Output video frame rate."""

    postprocess_output_layout: VideoTensorLayout | None = "tchw"
    """Pipeline output layout for streaming post-processing."""


class Cosmos2T2VRunner(Runner[Cosmos2T2VRunnerConfig, CosmosInferencePipeline]):
    """Cosmos-Predict2 non-streaming T2V driver."""

    config: Cosmos2T2VRunnerConfig

    def _resolve_prompt(self) -> str:
        """Resolve config.prompt.

        A Path reads its first non-empty line, a str is used as-is.
        """
        return resolve_prompt_value(self.config.prompt)

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

        postprocess_stream = self.create_postprocess_stream(fps=config.fps)
        generated = self.pipeline.generate(autoregressive_index=0, cache=cache)
        stats = self.pipeline.finalize(autoregressive_index=0, cache=cache)
        postprocess_stream.process(generated, autoregressive_index=0)
        generated = postprocess_stream.finish()
        if generated is None:
            return

        ensure_output_dir(config.output_dir)
        video_path = runner_artifact_path(config.output_dir, config.runner_name, "mp4")
        write_video_tensor(generated, video_path, fps=config.fps, layout="tchw")

        logger.info(
            f"[{config.runner_name}] wrote video {tuple(generated.shape)} "
            f"-> {video_path.resolve()}"
        )

        if stats is not None:
            stats_path = write_runner_stats(
                config.output_dir,
                config.runner_name,
                [{"autoregressive_index": 0, **stats}],
            )
            logger.info(
                f"[{config.runner_name}] wrote per-AR-step stats "
                f"-> {stats_path.resolve()}"
            )


@dataclass(kw_only=True)
class Cosmos2I2VRunnerConfig(Cosmos2T2VRunnerConfig):
    """Runner config for the Cosmos-Predict2 I2V variant."""

    _target: type["Cosmos2I2VRunner"] = field(default_factory=lambda: Cosmos2I2VRunner)

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
        image = load_first_frame_tensor(
            resolve_input_path(
                config.image_path,
                cache_dir=IMAGE_CACHE_DIR,
                validator=read_image_rgb,
            ),
            pixel_height=config.pixel_height,
            pixel_width=config.pixel_width,
            device=self.pipeline.device,
            dtype=torch.bfloat16,
        )

        return self.pipeline.initialize_cache(text=[prompt], image=image)
