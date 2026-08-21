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

"""CPU-safe checks for the FlashDreams runner bridge."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from crazy_robotaxi.runner import (
    CRAZY_ROBOTAXI_RUNNER,
    CrazyRobotaxiRunner,
)

pytestmark = pytest.mark.ci_cpu


def test_runner_registry_metadata_uses_public_slug() -> None:
    """Keep runner and pipeline names aligned for registry discovery."""
    assert CRAZY_ROBOTAXI_RUNNER.runner_name == "crazy-robotaxi"
    assert CRAZY_ROBOTAXI_RUNNER.pipeline.name == "crazy-robotaxi"
    assert CRAZY_ROBOTAXI_RUNNER.description


def test_runner_delegates_to_standalone_legacy_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Translate typed runner fields without constructing the GPU pipeline."""
    received: list[list[str]] = []
    monkeypatch.setattr(
        "crazy_robotaxi.cli.main",
        lambda argv=None: received.append(list(argv or ())),
    )
    config = replace(
        CRAZY_ROBOTAXI_RUNNER,
        scene=Path("city.usdz"),
        world_model_manifest=Path("example_world_model_perf.yaml"),
        backend="raster",
        stream_mjpeg="127.0.0.1:8080",
        auto_start=True,
        synthetic_scene=True,
        synthetic_model=False,
        taxi_seed=7,
        taxi_highscores=Path("scores.csv"),
        app_args=("--camera", "front"),
    )

    CrazyRobotaxiRunner(config).run()

    assert received == [
        [
            "--camera",
            "front",
            "--scene",
            "city.usdz",
            "--manifest",
            "example_world_model_perf.yaml",
            "--backend",
            "raster",
            "--stream-mjpeg",
            "127.0.0.1:8080",
            "--taxi-seed",
            "7",
            "--taxi-highscores",
            "scores.csv",
            "--synthetic-scene",
            "--auto-start",
            "--no-synthetic-model",
        ]
    ]
