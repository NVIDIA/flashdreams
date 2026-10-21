# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutable application configuration for the HandDreams V2 demo."""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flashdreams_controller_input.isaac_teleop_source import DEFAULT_ISAAC_TELEOP_PYTHON

from flashdreams.infra.postprocess import VideoPostprocessChainConfig

from .scenes import ScenePreview


@dataclass(frozen=True, slots=True)
class HanddreamsScenario:
    """External scene source and optional initial scene selection."""

    sample: str | None = None
    """Initial scene name; use the first discovered scene when omitted."""
    scene_path: Path | None = None
    """Scene directory containing ``first_frame.jpg`` and ``scene.json``."""
    sample_dir: Path | None = None
    """Root directory containing the available scene directories."""

    def __post_init__(self) -> None:
        if self.sample is not None and not self.sample.strip():
            raise ValueError("HandDreams sample must be non-empty")
        if self.scene_path is not None and self.sample_dir is not None:
            raise ValueError("scene_path and sample_dir are mutually exclusive")
        if self.scene_path is not None:
            path = self.scene_path.expanduser()
            if not path.is_dir():
                raise FileNotFoundError(
                    f"HandDreams scene directory does not exist: {path}"
                )
            for name in ("first_frame.jpg", "scene.json"):
                if not (path / name).is_file():
                    raise FileNotFoundError(
                        f"HandDreams scene asset does not exist: {path / name}"
                    )
            object.__setattr__(self, "scene_path", path)
        if self.sample_dir is not None:
            directory = self.sample_dir.expanduser()
            if not directory.is_dir():
                raise FileNotFoundError(
                    f"HandDreams sample directory does not exist: {directory}"
                )
            object.__setattr__(self, "sample_dir", directory)


@dataclass(frozen=True, kw_only=True, slots=True)
class HandDreamsApplicationDefaults:
    """Model integration hooks and output geometry for the reusable application."""

    video_width: int
    """Native generated frame width."""
    video_height: int
    """Native generated frame height."""
    add_model_arguments: Callable[[argparse.ArgumentParser], None]
    """Register the integration's model options on the application parser."""
    resolve_pipeline_config: Callable[[Mapping[str, Any]], Any]
    """Validate model options and return an integration-owned config."""
    inference_pipeline_factory: Callable[[Any], Any]
    """Construct the application-owned inference pipeline."""
    discover_scene_catalog: Callable[[HanddreamsScenario], tuple[str, ...]]
    """Resolve scene names accepted by the configured model."""
    scene_previews: Callable[
        [HanddreamsScenario, tuple[str, ...]], tuple[ScenePreview, ...]
    ]
    """Load presentation metadata from the selected scene source in catalog order."""
    scene_factory: Callable[[HanddreamsScenario, Any], Any]
    """Load and encode one scene using the model pipeline."""
    rollout_builder_factory: Callable[..., Any]
    """Bind a pipeline and scene to the application rollout contract."""


@dataclass(frozen=True, kw_only=True, slots=True)
class HandDreamsLaunchConfig:
    """Application arguments shared by each session it creates."""

    scenario: HanddreamsScenario
    pipeline_config: Any
    """Validated model configuration supplied by the integration."""
    postprocess: VideoPostprocessChainConfig = field(
        default_factory=VideoPostprocessChainConfig
    )
    """Optional decoded-video processor configured by the application."""
    total_steps: int = 2_147_483_647
    duration_seconds: float | None = None
    control_fps: int | None = None
    """Controller sampling rate; use the scene's model FPS when omitted."""
    synchronize_metrics: bool = False
    """Synchronize decoded CUDA output before recording presentation timings."""
    isaac_teleop_python: Path = Path(DEFAULT_ISAAC_TELEOP_PYTHON)
    """Interpreter for the isolated controller process; preserve its venv path."""
    isaac_teleop_startup_timeout_s: float = 300.0
    position_gain: float = 1.2
    grip_threshold: float = 0.0
    stale_ms: float | None = 500.0
    gripper_slew_per_action: float = 0.25
    open_claw_threshold: float = 0.1
    auto_start: bool = False
    """Skip guided setup for desktop testing; still require tracked open claws."""

    def __post_init__(self) -> None:
        for name in (
            "total_steps",
            "duration_seconds",
            "control_fps",
            "isaac_teleop_startup_timeout_s",
            "position_gain",
            "grip_threshold",
            "stale_ms",
            "gripper_slew_per_action",
            "open_claw_threshold",
        ):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if self.duration_seconds is not None and self.duration_seconds <= 0.0:
            raise ValueError("duration_seconds must be positive")
        if self.control_fps is not None and self.control_fps <= 0:
            raise ValueError("control rate must be positive")
        if self.isaac_teleop_startup_timeout_s <= 0.0:
            raise ValueError("Isaac Teleop startup timeout must be positive")
        if self.position_gain <= 0.0:
            raise ValueError("position gain must be positive")
        if not 0.0 <= self.grip_threshold <= 1.0:
            raise ValueError("grip threshold must be in [0, 1]")
        if self.stale_ms is not None and self.stale_ms <= 0.0:
            raise ValueError("stale timeout must be positive")
        if self.gripper_slew_per_action < 0.0:
            raise ValueError("gripper slew must be non-negative")
        if not 0.0 <= self.open_claw_threshold <= 1.0:
            raise ValueError("open-claw threshold must be in [0, 1]")


__all__ = [
    "HandDreamsApplicationDefaults",
    "HandDreamsLaunchConfig",
    "HanddreamsScenario",
]
