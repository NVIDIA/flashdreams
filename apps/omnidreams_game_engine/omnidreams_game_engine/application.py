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

"""Generic application-extension contracts for Interactive Drive."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from omnidreams_game_engine.config import VehicleConfig
from omnidreams_game_engine.simulation.ego_vehicle_kinematics import (
    PhysicsStepFn,
)
from omnidreams_game_engine.simulation.game_physics import GamePhysicsWorld
from omnidreams_game_engine.simulation.ground_snap import GroundSnapper
from omnidreams_game_engine.simulation.map_bounds import MapBounds
from omnidreams_game_engine.types import (
    DriverCommand,
    SceneBundle,
    TrajectoryChunk,
    VehicleState,
)


class RuntimeApplication(Protocol):
    """Application state advanced alongside one Interactive Drive rollout."""

    @property
    def is_running(self) -> bool:
        """Return whether the simulation should request another chunk."""
        ...

    def process_events(self, state: VehicleState) -> None:
        """Consume application-specific control events for the current state."""
        ...

    def advance_frames(
        self, trajectory: TrajectoryChunk, frame_interval_s: float
    ) -> ApplicationChunkUpdate:
        """Advance application state and decorate one simulated chunk."""
        ...

    def publish_boundary(self, state: VehicleState) -> None:
        """Publish application telemetry for the latest boundary state."""
        ...


@dataclass(frozen=True)
class ApplicationChunkUpdate:
    """Application-owned trajectory and frame annotations for one chunk."""

    trajectory: TrajectoryChunk
    """Trajectory decorated with application-specific conditioning actors."""

    frame_application_states: tuple[object | None, ...]
    """Opaque application state synchronized to every trajectory frame."""

    def __post_init__(self) -> None:
        """Reject frame annotations that do not match the trajectory."""
        if len(self.frame_application_states) != len(self.trajectory.timestamps_us):
            raise ValueError(
                "frame_application_states must match trajectory timestamps; got "
                f"{len(self.frame_application_states)} states for "
                f"{len(self.trajectory.timestamps_us)} timestamps"
            )


@dataclass(frozen=True)
class RolloutSpec:
    """Application-selected simulation policy for one rollout."""

    vehicle_config: VehicleConfig
    """Vehicle configuration used by kinematics and physics."""

    initial_speed_mps: float
    """Vehicle speed at rollout construction."""

    integrate_fn: Callable[
        [VehicleState, DriverCommand, float, VehicleConfig], VehicleState
    ]
    """Vehicle integration policy."""

    physics_world_factory: Callable[[SceneBundle, VehicleConfig], GamePhysicsWorld]
    """Physics-world factory for the active scene."""

    physics_step_fn: PhysicsStepFn
    """Command-aware physics stepping policy."""

    visual_flare_enabled: bool
    """Whether collision flare events are presented."""

    ground_snapper: GroundSnapper | None
    """Application-selected ground policy for this rollout."""

    capture_physics_debug: bool = False
    """Whether every simulated frame carries a PhysX collider snapshot."""

    include_initial_state_in_first_chunk: bool = False
    """Whether frame zero is the rollout's unsimulated initial state."""


class InteractiveDriveApplication(Protocol):
    """Optional application policy injected into the shared driving engine."""

    def configure_presenter(self, presenter: Any) -> None:
        """Configure application-aware presentation before a scene loads."""
        ...

    def load_scene(self, scene: SceneBundle, map_bounds: MapBounds | None) -> None:
        """Load application-specific data for ``scene``."""
        ...

    def configure_scene_presenter(self, presenter: Any, scene: SceneBundle) -> None:
        """Configure scene-dependent presentation state."""
        ...

    def rollout_spec(
        self,
        scene: SceneBundle,
        *,
        default_vehicle: VehicleConfig,
        default_visual_flare_enabled: bool,
    ) -> RolloutSpec:
        """Return the simulation policy for a new rollout."""
        ...

    def create_runtime(self, scene: SceneBundle, simulation: Any) -> RuntimeApplication:
        """Create application state for a new rollout."""
        ...
