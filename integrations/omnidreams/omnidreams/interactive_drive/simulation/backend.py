# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import Protocol

from omnidreams.interactive_drive.types import (
    DriverCommand,
    TrajectoryChunk,
    VehicleState,
)


class SimulationBackend(Protocol):
    @property
    def current_state(self) -> VehicleState: ...

    def pose_chunk(
        self,
        command: DriverCommand,
        chunk_size: int,
        frame_interval_s: float,
        extrapolation_offset_s: float,
    ) -> TrajectoryChunk:
        """Advance authoritative state by ``chunk_size`` frames and return the trajectory.

        Mutates state to ``trajectory.boundary_state_after_chunk``. Sim wall-clock
        time advances by ``chunk_size * frame_interval_s`` per call, regardless of
        how often the loop calls this method.
        """
        ...
