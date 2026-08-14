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

"""FlashDreams runner bridge for the standalone Crazy Robotaxi app."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from omnidreams.config import SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE

from flashdreams.infra.config import derive_config
from flashdreams.infra.pipeline import StreamInferencePipelineConfig
from flashdreams.infra.runner import Runner, RunnerConfig

_CRAZY_ROBOTAXI_PIPELINE = derive_config(
    SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE,
    name="crazy-robotaxi",
)
"""Registry metadata for the legacy manifest-owned model session."""


@dataclass(kw_only=True)
class CrazyRobotaxiRunnerConfig(RunnerConfig):
    """Launch configuration for the standalone legacy game host."""

    _target: type["CrazyRobotaxiRunner"] = field(
        default_factory=lambda: CrazyRobotaxiRunner
    )

    pipeline: StreamInferencePipelineConfig = field(
        default_factory=lambda: _CRAZY_ROBOTAXI_PIPELINE
    )

    scene: Path | None = None
    """Scene archive override; ``None`` uses the staged default scene."""

    manifest: Path | None = None
    """World-model manifest override; ``None`` uses the bundled default."""

    camera: str | None = None
    """Camera name override for the selected scene."""

    variant: str | None = None
    """Weather or numbered scene variant override."""

    prompt: str | None = None
    """Text-conditioning prompt override."""

    backend: Literal["raster", "omnidreams"] | None = None
    """Render backend override; ``None`` selects the production world model."""

    stream_mjpeg: str | None = None
    """Optional MJPEG bind address instead of a native window."""

    synthetic_scene: bool = False
    """Use the procedural CPU-safe scene fixture."""

    synthetic_model: bool | None = None
    """Override synthetic model construction when set."""

    taxi_seed: int | None = None
    """Optional deterministic fare-layout seed."""

    taxi_highscores: Path | None = None
    """Leaderboard CSV override."""

    taxi_alignment_diagnostics: Path | None = None
    """Optional alignment artifact output directory."""

    app_args: tuple[str, ...] = ()
    """Additional legacy application arguments parsed before typed overrides."""


class CrazyRobotaxiRunner(Runner):
    """Runner adapter that preserves the legacy application lifecycle."""

    def __init__(self, config: CrazyRobotaxiRunnerConfig) -> None:
        self.config = config

    def run(self) -> None:
        """Launch the native or MJPEG game host."""
        from crazy_robotaxi.cli import main

        argv = list(self.config.app_args)
        _append_value(argv, "--scene", self.config.scene)
        _append_value(argv, "--manifest", self.config.manifest)
        _append_value(argv, "--camera", self.config.camera)
        _append_value(argv, "--variant", self.config.variant)
        _append_value(argv, "--prompt", self.config.prompt)
        _append_value(argv, "--backend", self.config.backend)
        _append_value(argv, "--stream-mjpeg", self.config.stream_mjpeg)
        _append_value(argv, "--taxi-seed", self.config.taxi_seed)
        _append_value(argv, "--taxi-highscores", self.config.taxi_highscores)
        _append_value(
            argv,
            "--taxi-alignment-diagnostics",
            self.config.taxi_alignment_diagnostics,
        )
        if self.config.synthetic_scene:
            argv.append("--synthetic-scene")
        if self.config.synthetic_model is not None:
            argv.append(
                "--synthetic-model"
                if self.config.synthetic_model
                else "--no-synthetic-model"
            )
        main(argv)


def _append_value(argv: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        argv.extend((flag, str(value)))


CRAZY_ROBOTAXI_RUNNER = CrazyRobotaxiRunnerConfig(
    runner_name="crazy-robotaxi",
    description="Standalone Crazy Robotaxi game using the legacy OmniDreams runtime.",
    pipeline=_CRAZY_ROBOTAXI_PIPELINE,
)
"""Runner config discovered by the ``flashdreams.runner_configs`` entry point."""
