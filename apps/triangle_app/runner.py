# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runner configuration for the triangle application."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

import tyro
from flashdreams.infra.runner import Runner, RunnerConfig


@dataclass(kw_only=True)
class TriangleAppRunnerConfig(RunnerConfig):
    _target: type[TriangleAppRunner] = field(default_factory=lambda: TriangleAppRunner)
    launch_capability: Annotated[str | None, tyro.conf.Suppress] = (
        "triangle_app.launch:LAUNCH_CAPABILITY"
    )
    pipeline: Annotated[Any, tyro.conf.Suppress] = None
    model: str = "triangle-model"
    width: int = 640
    height: int = 360
    fps: int = 30
    total_frames: int = 180
    title: str = "FlashDreams · Triangle App"
    max_queued_chunks: int = 2
    close_timeout_s: float = 10.0
    output: Path = Path("outputs/triangle-app.mp4")


class TriangleAppRunner(Runner[TriangleAppRunnerConfig, Any]):
    def __init__(self, config: TriangleAppRunnerConfig) -> None:
        self.config = config

    def run(self) -> None:
        from .launch import launch_triangle_app

        launch_triangle_app(self.config)


RUNNER_TRIANGLE_APP = TriangleAppRunnerConfig(
    runner_name="triangle-app",
    description="Triangle application with a selectable model integration.",
    device="cpu",
)

__all__ = [
    "RUNNER_TRIANGLE_APP",
    "TriangleAppRunner",
    "TriangleAppRunnerConfig",
]
