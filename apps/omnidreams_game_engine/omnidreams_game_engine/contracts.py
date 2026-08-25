# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Narrow dependency-injection contracts for model-thread games."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from omnidreams_game_engine.types import (
    ConditionBatch,
    DriverCommand,
    DynamicActorTrajectory,
    SceneDefinition,
    TrajectoryChunk,
    VehicleState,
)


class SimulationWorld(Protocol):
    """Mutable vehicle/physics simulation owned by the model thread."""

    @property
    def current_state(self) -> VehicleState:
        """Return the current boundary state."""
        ...

    def pose_chunk(
        self,
        *,
        commands: tuple[DriverCommand, ...],
        chunk_size: int,
        frame_interval_s: float,
        extrapolation_offset_s: float,
    ) -> TrajectoryChunk:
        """Advance a frame-aligned trajectory chunk."""
        ...

    def close(self) -> None:
        """Release simulation resources."""
        ...


@dataclass(frozen=True, slots=True)
class GameUpdate:
    """Game-owned results aligned with one simulated trajectory."""

    frames: tuple[object, ...]
    dynamic_actors: tuple[DynamicActorTrajectory, ...] = ()


class GameRules(Protocol):
    """Application rules injected into the reusable game engine."""

    @property
    def is_running(self) -> bool:
        """Whether another world-model block should be generated."""
        ...

    def snapshot(self, vehicle_state: VehicleState) -> object:
        """Return immutable game state at a simulation boundary."""
        ...

    def advance_frames(
        self,
        trajectory: TrajectoryChunk,
        frame_interval_s: float,
    ) -> GameUpdate:
        """Advance rules once per simulated frame."""
        ...

    def submit_text(self, value: str, vehicle_state: VehicleState) -> object:
        """Consume application text such as a leaderboard name."""
        ...


class ConditionRenderer(Protocol):
    """Render model conditioning on the owning model thread."""

    def load_scene(self, scene: SceneDefinition) -> None:
        """Upload immutable scene data."""
        ...

    def render(self, trajectory: TrajectoryChunk) -> ConditionBatch:
        """Render synchronized semantic camera and optional BEV frames."""
        ...

    def close(self) -> None:
        """Release renderer resources."""
        ...
