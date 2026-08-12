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

"""LingBot-World camera-control I2V runner classes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import tyro
from loguru import logger

from flashdreams.infra.postprocess import VideoTensorLayout
from flashdreams.infra.runner import Runner, RunnerConfig
from flashdreams.infra.runner_io import runner_artifact_path
from flashdreams.runtime.demo import DemoSpec, Mp4OutputSpec, OutputSpec
from flashdreams.runtime.demo.replay import run_replay_demo
from lingbot.demo import LingbotDemoAdapter
from lingbot.example_data import (
    EXAMPLE_DATA_AVAILABLE_IDXS,
    EXAMPLE_DATA_PROMPT_AVAILABLE_IDXS,
    ensure_example_data_downloaded,
    example_data_dirname,
)
from lingbot.pipeline import (
    LingbotWorldInferencePipeline,
)
from lingbot.runtime import (
    LINGBOT_MODEL_ID,
    LingbotRunnerOutputTarget,
    inference_config_from_runner_config,
    replay_inputs_from_runner_config,
)

__all__ = [
    "EXAMPLE_DATA_AVAILABLE_IDXS",
    "LingbotWorldRunnerConfig",
    "LingbotWorldRunner",
    "example_data_dirname",
]


_INTRINSICS_REFERENCE_HEIGHT = 480
"""Capture-resolution height the bundled intrinsics ``.npy`` files are
expressed in; rescaled by :func:`get_Ks_transformed` so Plücker rays
land on the right pixel centers at the runner's actual frame size."""

_INTRINSICS_REFERENCE_WIDTH = 832
"""Capture-resolution width matching :data:`_INTRINSICS_REFERENCE_HEIGHT`."""


@dataclass(kw_only=True)
class LingbotWorldRunnerConfig(RunnerConfig):
    """Runner config for every shipped LingBot-World variant."""

    _target: type["LingbotWorldRunner"] = field(
        default_factory=lambda: LingbotWorldRunner
    )
    launch_capability: Annotated[str | None, tyro.conf.Suppress] = (
        "lingbot.launch:LAUNCH_CAPABILITY"
    )

    prompt: str = ""
    """Text prompt. A non-empty value wins; otherwise the runner reads
    the first line of :attr:`prompt_path`."""

    prompt_path: Path | None = None
    """Fallback ``.txt`` whose first line is read when :attr:`prompt` is
    empty. ``--example-data True`` lazy-fills it from the bundled demo."""

    image_path: Path | None = None
    """Path to the first-frame RGB image. Required at ``run()`` time."""

    pose_path: Path | None = None
    """Path to a ``.npy`` of camera-to-world matrices, shape ``[T, 4, 4]``.
    Required at ``run()`` time."""

    intrinsic_path: Path | None = None
    """Path to a ``.npy`` of camera intrinsics, shape ``[T, 4]``.
    Required at ``run()`` time."""

    total_blocks: int = 20
    """Upper bound on the number of AR chunks to generate. The loop
    exits early once the camera stream is consumed."""

    pixel_height: int = 464
    """Output video pixel height."""

    pixel_width: int = 832
    """Output video pixel width."""

    fps: int = 16
    """Output video frame rate. Lingbot was trained at 16fps."""

    postprocess_output_layout: VideoTensorLayout | None = "tchw"
    """Pipeline output layout for streaming post-processing."""

    example_data: bool = False
    """When ``True``, lazy-download bundled GitHub example assets into
    ``$FLASHDREAMS_CACHE_DIR/example_data/lingbot_world/`` and fill ``image_path`` /
    ``pose_path`` / ``intrinsic_path`` / ``prompt_path`` from the
    bundled defaults. Use for the README demo; pass explicit paths
    for production runs."""

    example_idx: int = 0
    """Example folder index under ``.../examples/``; allowed: ``0`` through ``5``."""


class LingbotWorldRunner(
    Runner[LingbotWorldRunnerConfig, LingbotWorldInferencePipeline]
):
    """Streaming camera-control I2V driver."""

    config: LingbotWorldRunnerConfig

    def _resolve_prompt(self) -> str:
        """Pick the prompt: non-empty ``--prompt`` wins, else ``--prompt-path``."""
        cfg = self.config
        if cfg.prompt:
            return cfg.prompt
        if cfg.prompt_path is None:
            if self.is_rank_zero:
                logger.warning(
                    "LingBot prompt.txt is missing; proceeding with an empty prompt."
                )
            return ""
        text = cfg.prompt_path.read_text().splitlines()
        prompt = text[0].strip() if text else ""
        if not prompt and self.is_rank_zero:
            logger.warning(
                "LingBot prompt file {} is empty; proceeding with an empty prompt.",
                cfg.prompt_path,
            )
        return prompt

    def _fill_example_data_defaults(self) -> None:
        """Lazy-download bundled assets and fill empty path defaults in-place."""
        cfg = self.config
        example_dir = ensure_example_data_downloaded(
            is_rank_zero=self.is_rank_zero,
            example_idx=cfg.example_idx,
        )
        if cfg.image_path is None:
            cfg.image_path = example_dir / "image.jpg"
        if cfg.pose_path is None:
            cfg.pose_path = example_dir / "poses.npy"
        if cfg.intrinsic_path is None:
            cfg.intrinsic_path = example_dir / "intrinsics.npy"
        if (
            not cfg.prompt
            and cfg.prompt_path is None
            and cfg.example_idx in EXAMPLE_DATA_PROMPT_AVAILABLE_IDXS
        ):
            cfg.prompt_path = example_dir / "prompt.txt"

    def run(self) -> None:
        """Drive an AR rollout through the Lingbot runtime API path."""
        cfg = self.config
        adapter = LingbotDemoAdapter()
        inference_config = inference_config_from_runner_config(
            cfg,
            device=f"cuda:{self.local_rank}" if self.world_size > 1 else cfg.device,
            pipeline=self.pipeline,
        )
        replay_inputs = replay_inputs_from_runner_config(
            cfg,
            is_rank_zero=self.is_rank_zero,
        )
        spec = DemoSpec(
            model_id=LINGBOT_MODEL_ID,
            preset_id=str(cfg.pipeline.name),
            input_mode="replay",
            output=Mp4OutputSpec(
                path=runner_artifact_path(cfg.output_dir, cfg.runner_name, "mp4"),
                fps=cfg.fps,
                output_layout=cfg.postprocess_output_layout or "tchw",
            ),
            scenario=replay_inputs,
            config=inference_config,
        )

        def _output_target_factory(
            output_spec: OutputSpec,
        ) -> LingbotRunnerOutputTarget:
            del output_spec
            return LingbotRunnerOutputTarget(
                output_stream=self.create_video_output_stream(fps=cfg.fps),
                output_dir=cfg.output_dir,
                runner_name=cfg.runner_name,
                fps=cfg.fps,
            )

        result = run_replay_demo(
            spec=spec,
            adapter=adapter,
            output_target_factory=_output_target_factory,
        )
        if result.status != "completed":
            raise RuntimeError(
                f"Lingbot runner failed with status {result.status!r}: "
                f"{result.reason or result.error or 'unknown error'}"
            )
