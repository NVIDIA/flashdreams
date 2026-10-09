# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unrestricted driving rules for Crazy Robotaxi."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from omnidreams_game_engine.contracts import GameUpdate
from omnidreams_game_engine.types import TrajectoryChunk, VehicleState


@dataclass(frozen=True, slots=True)
class FreeRoamSnapshot:
    """Frame-aligned state for driving without objectives or a time limit."""

    session_state: str = "playing"


class FreeRoamGameRules:
    """Keep simulation and optional live-edit abilities running indefinitely."""

    def __init__(
        self,
        frame_advance: Callable[[VehicleState, bool], int] | None = None,
    ) -> None:
        self._frame_advance = frame_advance
        self._snapshot = FreeRoamSnapshot()

    @property
    def is_running(self) -> bool:
        return True

    def snapshot(self, vehicle_state: VehicleState) -> FreeRoamSnapshot:
        return self._snapshot

    def advance_frames(
        self,
        trajectory: TrajectoryChunk,
        frame_interval_s: float,
    ) -> GameUpdate:
        if self._frame_advance is not None:
            for state in trajectory.vehicle_states:
                self._frame_advance(state, True)
        return GameUpdate(frames=(self._snapshot,) * len(trajectory.vehicle_states))

    def submit_text(self, value: str, vehicle_state: VehicleState) -> FreeRoamSnapshot:
        return self._snapshot
