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

"""Non-streaming Wan 2.1 runner classes (T2V and I2V)."""

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
from flashdreams.recipes.wan import (
    WanInferencePipeline,
    WanInferencePipelineCache,
)

__all__ = [
    "Wan21I2VRunnerConfig",
    "Wan21I2VRunner",
    "Wan21T2VRunnerConfig",
    "Wan21T2VRunner",
]


DEFAULT_PROMPT = (
    "Summer beach vacation style, a white cat wearing sunglasses sits on "
    "a surfboard. The fluffy-furred feline gazes directly at the camera "
    "with a relaxed expression. Blurred beach scenery forms the background "
    "featuring crystal-clear waters, distant green hills, and a blue sky "
    "dotted with white clouds. The cat assumes a naturally relaxed posture, "
    "as if savoring the sea breeze and warm sunlight. A close-up shot "
    "highlights the feline's intricate details and the refreshing "
    "atmosphere of the seaside."
)

DEFAULT_I2V_IMAGE_URL = (
    "https://raw.githubusercontent.com/Wan-Video/Wan2.1/main/examples/i2v_input.JPG"
)

IMAGE_CACHE_DIR = (
    Path(os.path.expanduser(os.getenv("FLASHDREAMS_CACHE_DIR", "~/.cache/flashdreams")))
    / "wan21"
)
"""User-writable cache for on-the-fly I2V first-frame downloads."""


@dataclass(kw_only=True)
class Wan21T2VRunnerConfig(RunnerConfig):
    """Runner config for the Wan 2.1 T2V variant.

    Also serves as the base for :class:`Wan21I2VRunnerConfig`
    (I2V is T2V plus an ``image_path``).
    """

    _target: type["Wan21T2VRunner"] = field(default_factory=lambda: Wan21T2VRunner)

    prompt: str | Path = DEFAULT_PROMPT
    """Either an inline text prompt (--prompt "...") or a path to a
    txt file whose first line is read as the prompt (--prompt prompt.txt).
    Defaults to :data:`DEFAULT_PROMPT`."""

    pixel_height: int = 480
    """Output video pixel height."""

    pixel_width: int = 832
    """Output video pixel width."""

    fps: int = 16
    """Output video frame rate."""

    postprocess_output_layout: VideoTensorLayout | None = "tchw"
    """Pipeline output layout for streaming post-processing."""


@dataclass(kw_only=True)
class Wan21I2VRunnerConfig(Wan21T2VRunnerConfig):
    """Runner config for the Wan 2.1 I2V variant.

    Inherits all T2V fields (prompt, pixel_*, fps) and
    adds the first-frame image path that I2V needs at runtime.
    """

    _target: type["Wan21I2VRunner"] = field(default_factory=lambda: Wan21I2VRunner)

    image_path: str | Path = DEFAULT_I2V_IMAGE_URL
    """Path to the first-frame RGB image, or an ``http(s)://`` URL that
    will be downloaded on first use into :data:`IMAGE_CACHE_DIR`.
    Defaults to :data:`DEFAULT_I2V_IMAGE_URL`."""

    prompt: str | Path = DEFAULT_PROMPT
    """Either an inline text prompt (--prompt "...") or a path to a
    txt file whose first line is read as the prompt (--prompt prompt.txt).
    Defaults to :data:`DEFAULT_PROMPT`."""

    pixel_height: int = 832
    """Output video pixel height."""

    pixel_width: int = 480
    """Output video pixel width."""


class Wan21T2VRunner(Runner[Wan21T2VRunnerConfig, WanInferencePipeline]):
    """Wan 2.1 non-streaming T2V driver.

    Also serves as the base for :class:`Wan21I2VRunner` (I2V
    only overrides :meth:`_initialize_cache` to load the first frame;
    everything else, including :meth:`run`, is reused).
    """

    config: Wan21T2VRunnerConfig

    def _resolve_prompt(self) -> str:
        """Resolve config.prompt.

        A Path reads its first non-empty line, a str is used as-is.
        """
        return resolve_prompt_value(self.config.prompt)

    def _initialize_cache(self) -> WanInferencePipelineCache:
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

        # Initialize the autoregressive cache.
        cache = self._initialize_cache()

        # Generate the output in one AR step.
        postprocess_stream = self.create_postprocess_stream(fps=config.fps)
        generated = self.pipeline.generate(autoregressive_index=0, cache=cache)
        stats = self.pipeline.finalize(autoregressive_index=0, cache=cache)
        postprocess_stream.process(generated, autoregressive_index=0)
        generated = postprocess_stream.finish()
        if generated is None:
            return

        # Write the video.
        ensure_output_dir(config.output_dir)
        video_path = runner_artifact_path(config.output_dir, config.runner_name, "mp4")
        write_video_tensor(generated, video_path, fps=config.fps, layout="tchw")

        logger.info(
            f"[{config.runner_name}] wrote video {tuple(generated.shape)} "
            f"-> {video_path.resolve()}"
        )

        # Write the perf stats.
        if stats is not None:
            stats_path = write_runner_stats(
                config.output_dir,
                config.runner_name,
                [{"autoregressive_index": 0, **stats}],
            )
            logger.info(
                f"[{config.runner_name}] wrote per-AR-step stats -> {stats_path.resolve()}"
            )


class Wan21I2VRunner(Wan21T2VRunner):
    """Wan 2.1 non-streaming I2V driver (first-frame injection)."""

    config: Wan21I2VRunnerConfig

    def _initialize_cache(self) -> WanInferencePipelineCache:
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
