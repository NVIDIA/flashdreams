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

"""Causal-Forcing Wan 2.1 streaming runner classes (T2V and I2V)."""

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
from flashdreams.recipes.wan import (
    WanInferencePipeline,
    WanInferencePipelineCache,
)

__all__ = [
    "CausalForcingI2VRunnerConfig",
    "CausalForcingI2VRunner",
    "CausalForcingT2VRunnerConfig",
    "CausalForcingT2VRunner",
]


DEFAULT_T2V_PROMPT = (
    "A cinematic closeup and detailed portrait of a reindeer standing in a "
    "snowy forest at sunset. The lighting is gorgeous and soft, with a golden "
    "backlight creating a warm and dreamy effect. Soft bokeh and lens flares "
    "add a magical touch, enhancing the cinematic quality of the image. The "
    "reindeer has a gentle expression, its fur glistening in the fading light. "
    "The background features a serene snowy landscape with tall trees "
    "silhouetted against the orange and pink hues of the setting sun. The "
    "color grade is rich and magical, capturing the essence of a winter "
    "wonderland at twilight. A close-up shot from a slightly elevated angle."
)


DEFAULT_I2V_IMAGE_URL = (
    "https://raw.githubusercontent.com/Wan-Video/Wan2.1/main/examples/i2v_input.JPG"
)

IMAGE_CACHE_DIR = (
    Path(os.path.expanduser(os.getenv("FLASHDREAMS_CACHE_DIR", "~/.cache/flashdreams")))
    / "self_forcing"
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
class CausalForcingT2VRunnerConfig(RunnerConfig):
    """Runner config for the Causal-Forcing T2V variants.

    Also serves as the base for :class:`CausalForcingI2VRunnerConfig`
    (I2V is T2V plus an ``image_path``).
    """

    _target: type["CausalForcingT2VRunner"] = field(
        default_factory=lambda: CausalForcingT2VRunner
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


@dataclass(kw_only=True)
class CausalForcingI2VRunnerConfig(CausalForcingT2VRunnerConfig):
    """Runner config for the Causal-Forcing I2V variants.

    Inherits all T2V fields (prompt, total_blocks, pixel_*, fps) and
    adds the first-frame image path that I2V needs at runtime.
    """

    _target: type["CausalForcingI2VRunner"] = field(
        default_factory=lambda: CausalForcingI2VRunner
    )

    image_path: str | Path = DEFAULT_I2V_IMAGE_URL
    """First-frame RGB image. Either a local path or an HTTP(S) URL."""


class CausalForcingT2VRunner(
    Runner[CausalForcingT2VRunnerConfig, WanInferencePipeline]
):
    """Causal-Forcing Wan 2.1 streaming T2V driver.

    Also serves as the base for :class:`CausalForcingI2VRunner` (I2V
    only overrides :meth:`_initialize_cache` to load the first frame;
    everything else, including :meth:`run`, is reused).
    """

    config: CausalForcingT2VRunnerConfig

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


class CausalForcingI2VRunner(CausalForcingT2VRunner):
    """Causal-Forcing Wan 2.1 streaming I2V driver (mask-injection first frame).

    Inherits :meth:`run` and :meth:`_resolve_prompt` from
    :class:`CausalForcingT2VRunner`; only :meth:`_initialize_cache`
    differs (loads + encodes the first frame).
    """

    config: CausalForcingI2VRunnerConfig

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
