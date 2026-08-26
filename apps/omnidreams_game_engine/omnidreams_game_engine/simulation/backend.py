# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from collections.abc import Sequence
from typing import Protocol

from omnidreams_game_engine.types import (
    DriverCommand,
    TrajectoryChunk,
    VehicleState,
)


class SimulationBackend(Protocol):
    @property
    def current_state(self) -> VehicleState: ...

    def set_physx_debug_enabled(self, enabled: bool) -> None:
        """Toggle collider snapshot capture for future pose chunks."""
        ...

    def pose_chunk(
        self,
        commands: Sequence[DriverCommand],
        chunk_size: int,
        frame_interval_s: float,
        extrapolation_offset_s: float,
    ) -> TrajectoryChunk:
        """Advance with one command per frame and return the trajectory.

        Mutates state to ``trajectory.boundary_state_after_chunk``. Sim wall-clock
        time advances by ``chunk_size * frame_interval_s`` per call, regardless of
        how often the loop calls this method.
        """
        ...
