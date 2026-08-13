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

"""Shared prompt-conditioned video runners for ``flashdreams-run`` slugs.

Every model that generates video from a prompt drives the same rollout: build
a cache, call ``generate``/``finalize`` per autoregressive index, stream chunks
into an MP4, then write per-step stats. Integrations declare their defaults as
a :class:`VideoRunnerConfig` subclass in their ``config.py`` and point
``_target`` at one of the runners here instead of restating the rollout.

Pipelines are consumed structurally rather than by type: a pipeline must expose
``initialize_cache``, ``generate(autoregressive_index=..., cache=...)``,
``finalize(autoregressive_index=..., cache=...)``, and a ``decoder`` that is a
:class:`~flashdreams.infra.decoder.StreamingVideoDecoder`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import torch
from loguru import logger

from flashdreams.infra.decoder import StreamingVideoDecoder
from flashdreams.infra.postprocess import VideoTensorLayout
from flashdreams.infra.runner import Runner, RunnerConfig
from flashdreams.infra.runner_io import (
    load_first_frame_tensor,
    read_image_rgb,
    resolve_input_path,
    resolve_prompt_value,
    runner_artifact_path,
    write_runner_stats,
)
from flashdreams.runtime.output import OutputArtifact
from flashdreams.runtime.video_output import Mp4VideoOutputTarget

__all__ = [
    "ImageConditionedVideoRunnerConfig",
    "StreamingVideoRunner",
    "StreamingVideoRunnerConfig",
    "VideoRunner",
    "VideoRunnerConfig",
    "image_cache_dir",
]


def image_cache_dir(subdir: str) -> Path:
    """Return the user-writable cache for on-the-fly image downloads."""
    root = os.path.expanduser(
        os.getenv("FLASHDREAMS_CACHE_DIR", "~/.cache/flashdreams")
    )
    return Path(root) / subdir


@dataclass(kw_only=True)
class VideoRunnerConfig(RunnerConfig):
    """Base config for a prompt-conditioned single-step video runner."""

    _target: type["VideoRunner"] = field(default_factory=lambda: VideoRunner)

    prompt: str | Path = ""
    """Either an inline text prompt (--prompt "...") or a path to a
    txt file whose first line is read as the prompt (--prompt prompt.txt)."""

    pixel_height: int = 480
    """Output video pixel height."""

    pixel_width: int = 832
    """Output video pixel width."""

    fps: int = 16
    """Output video frame rate."""

    postprocess_output_layout: VideoTensorLayout | None = "tchw"
    """Pipeline output layout for streaming post-processing."""


@dataclass(kw_only=True)
class StreamingVideoRunnerConfig(VideoRunnerConfig):
    """Config for autoregressive models that roll out many chunks."""

    _target: type["StreamingVideoRunner"] = field(
        default_factory=lambda: StreamingVideoRunner
    )

    total_blocks: int = 60
    """Number of autoregressive chunks to generate before terminating the rollout."""


@dataclass(kw_only=True)
class ImageConditionedVideoRunnerConfig:
    """Mixin adding the first-frame image that I2V variants need at runtime.

    Inherit it alongside :class:`VideoRunnerConfig` or
    :class:`StreamingVideoRunnerConfig`; the runners pick these fields up when
    they are present so that T2V slugs keep an image-free CLI surface.
    """

    image_path: str | Path = ""
    """First-frame RGB image. Either a local path or an HTTP(S) URL."""

    image_cache_subdir: ClassVar[str] = "video"
    """Subdirectory of the FlashDreams cache for downloaded images.

    A per-model constant rather than a field, so it stays off the CLI.
    """


class VideoRunner(Runner[VideoRunnerConfig, Any]):
    """Prompt-conditioned video runner that generates one chunk."""

    config: VideoRunnerConfig

    def _step_count(self) -> int:
        return 1

    def _resolve_prompt(self) -> str:
        """Resolve ``config.prompt``.

        A Path reads its first non-empty line, a str is used as-is.
        """
        return resolve_prompt_value(self.config.prompt)

    def _latent_dimensions(self) -> tuple[int, int]:
        """Return the latent height and width for the configured pixel size."""
        config = self.config
        decoder = self.pipeline.decoder
        if not isinstance(decoder, StreamingVideoDecoder):
            raise TypeError(
                f"[{config.runner_name}] requires a StreamingVideoDecoder, "
                f"got {type(decoder).__name__}."
            )
        ratio = decoder.spatial_compression_ratio
        if config.pixel_height % ratio or config.pixel_width % ratio:
            raise ValueError(
                f"[{config.runner_name}] pixel_height={config.pixel_height} and "
                f"pixel_width={config.pixel_width} must both divide {ratio}."
            )
        return config.pixel_height // ratio, config.pixel_width // ratio

    def _conditioning_image(self) -> torch.Tensor | None:
        """Load the first-frame image when this model is image-conditioned.

        Read structurally so that T2V configs, which do not declare the mixin
        fields, keep an image-free CLI surface.
        """
        image_path = getattr(self.config, "image_path", "")
        if not image_path:
            return None
        config = self.config
        # Load + resize the first frame, then convert to [-1, 1] bf16 in shape
        # [T=1, C, H, W]. Pin to the pipeline's actual device so non-default
        # ``--device`` selections and the torchrun cuda:LOCAL_RANK override
        # both work.
        return load_first_frame_tensor(
            resolve_input_path(
                image_path,
                cache_dir=image_cache_dir(
                    getattr(config, "image_cache_subdir", "video")
                ),
                validator=read_image_rgb,
            ),
            pixel_height=config.pixel_height,
            pixel_width=config.pixel_width,
            device=self.pipeline.device,
            dtype=torch.bfloat16,
        )

    def _initialize_cache(self) -> Any:
        """Initialize the rollout cache for either T2V or I2V conditioning."""
        prompt = self._resolve_prompt()
        latent_height, latent_width = self._latent_dimensions()
        image = self._conditioning_image()
        if image is not None:
            return self.pipeline.initialize_cache(text=[prompt], image=image)
        return self.pipeline.initialize_cache(
            text=[prompt], image=None, height=latent_height, width=latent_width
        )

    def run(self) -> None:
        """Drive the rollout and write the video plus per-step stats."""
        config = self.config
        cache = self._initialize_cache()
        output_stream = self.create_video_output_stream(fps=config.fps)
        output_target = Mp4VideoOutputTarget(
            output_path=runner_artifact_path(
                config.output_dir, config.runner_name, "mp4"
            ),
            fps=config.fps,
            output_layout=output_stream.output_layout,
            enabled=self.is_rank_zero,
        )
        output_target.open()
        for index in range(self._step_count()):
            chunk = self.pipeline.generate(autoregressive_index=index, cache=cache)
            stats = self.pipeline.finalize(autoregressive_index=index, cache=cache)
            output_target.write(
                output_stream.process(chunk, autoregressive_index=index, metrics=stats)
            )
        tail = output_stream.finish()
        if tail is not None:
            output_target.write(tail)
        artifacts = output_target.close()
        if artifacts:
            self._log_artifact(artifacts[0])

    def _log_artifact(self, video_artifact: OutputArtifact) -> None:
        """Log the written video and persist per-step stats when present."""
        config = self.config
        logger.info(
            f"[{config.runner_name}] wrote video "
            f"{video_artifact.metadata['shape']} "
            f"-> {Path(video_artifact.uri).resolve()}"
        )
        stats_history = video_artifact.metadata["stats_history"]
        if not stats_history:
            return
        stats_path = write_runner_stats(
            config.output_dir, config.runner_name, list(stats_history)
        )
        logger.info(
            f"[{config.runner_name}] wrote per-AR-step stats -> {stats_path.resolve()}"
        )


class StreamingVideoRunner(VideoRunner):
    """Prompt-conditioned video runner that rolls out ``total_blocks`` chunks."""

    config: StreamingVideoRunnerConfig

    def _step_count(self) -> int:
        return self.config.total_blocks
