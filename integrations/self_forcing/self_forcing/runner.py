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

from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from flashdreams.infra.decoder import StreamingVideoDecoder
from flashdreams.infra.postprocess import VideoTensorLayout
from flashdreams.infra.runner import Runner, RunnerConfig
from flashdreams.infra.runner_io import (
    resolve_prompt_value,
    runner_artifact_path,
    write_runner_stats,
)
from flashdreams.recipes.wan import (
    WanInferencePipeline,
    WanInferencePipelineCache,
)
from flashdreams.runtime.video_output import Mp4VideoOutputTarget

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

    postprocess_output_layout: VideoTensorLayout | None = "tchw"
    """Pipeline output layout for streaming post-processing."""


class SelfForcingT2VRunner(Runner[SelfForcingT2VRunnerConfig, WanInferencePipeline]):
    """Self-Forcing Wan 2.1 streaming T2V driver."""

    config: SelfForcingT2VRunnerConfig

    def _resolve_prompt(self) -> str:
        """Resolve config.prompt.

        A Path reads its first non-empty line, a str is used as-is.
        """
        return resolve_prompt_value(self.config.prompt)

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
        output_stream = self.create_video_output_stream(fps=config.fps)
        video_path = runner_artifact_path(config.output_dir, config.runner_name, "mp4")
        output_target = Mp4VideoOutputTarget(
            output_path=video_path,
            fps=config.fps,
            output_layout=output_stream.output_layout,
            enabled=self.is_rank_zero,
        )
        output_target.open()
        for i in range(config.total_blocks):
            video_chunk = self.pipeline.generate(autoregressive_index=i, cache=cache)
            stats = self.pipeline.finalize(autoregressive_index=i, cache=cache)
            output_target.write(
                output_stream.process(
                    video_chunk,
                    autoregressive_index=i,
                    metrics=stats,
                )
            )

        tail = output_stream.finish()
        if tail is not None:
            output_target.write(tail)
        artifacts = output_target.close()
        if not artifacts:
            return
        video_artifact = artifacts[0]
        video_path = Path(video_artifact.uri)

        logger.info(
            f"[{config.runner_name}] wrote video {video_artifact.metadata['shape']} "
            f"-> {video_path.resolve()}"
        )

        # Write the perf stats.
        stats_history = video_artifact.metadata["stats_history"]
        if stats_history:
            stats_path = write_runner_stats(
                config.output_dir,
                config.runner_name,
                list(stats_history),
            )
            logger.info(
                f"[{config.runner_name}] wrote per-AR-step stats -> {stats_path.resolve()}"
            )
