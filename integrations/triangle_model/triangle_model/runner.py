# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runner configuration for the triangle model application."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

import tyro
from flashdreams.infra.runner import Runner, RunnerConfig


@dataclass(kw_only=True)
class TriangleModelRunnerConfig(RunnerConfig):
    _target: type[TriangleModelRunner] = field(
        default_factory=lambda: TriangleModelRunner
    )
    launch_capability: Annotated[str | None, tyro.conf.Suppress] = (
        "triangle_model.launch:LAUNCH_CAPABILITY"
    )
    pipeline: Annotated[Any, tyro.conf.Suppress] = None
    width: int = 640
    height: int = 360
    fps: int = 30
    total_frames: int = 180
    title: str = "FlashDreams · Triangle Model"
    max_queued_chunks: int = 2
    close_timeout_s: float = 10.0
    output: Path = Path("outputs/triangle-model.mp4")


class TriangleModelRunner(Runner[TriangleModelRunnerConfig, Any]):
    def __init__(self, config: TriangleModelRunnerConfig) -> None:
        self.config = config

    def run(self) -> None:
        from .launch import launch_triangle_model

        launch_triangle_model(self.config)


RUNNER_TRIANGLE_MODEL = TriangleModelRunnerConfig(
    runner_name="triangle-model",
    description="Triangle model implementation of the triangle application.",
    device="cpu",
)

__all__ = [
    "RUNNER_TRIANGLE_MODEL",
    "TriangleModelRunner",
    "TriangleModelRunnerConfig",
]
