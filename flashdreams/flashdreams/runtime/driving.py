# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transport-neutral driving commands, vehicle state, and trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]
UInt8Array = npt.NDArray[np.uint8]
Int32Array = npt.NDArray[np.int32]


@dataclass(frozen=True)
class DriverCommand:
    """One tick of driver intent, normalized away from any input device."""

    throttle: float = 0.0
    brake: float = 0.0
    steer: float = 0.0
    stop: bool = False
    reverse: bool = False
    steer_is_direct: bool = False
    """Treat :attr:`steer` as the wheel angle itself rather than a rate."""

    manual_control: bool = False
    """Whether the command carries direct absolute device values."""


@dataclass
class VehicleState:
    """Ego pose and motion in world coordinates."""

    x_m: float
    y_m: float
    z_m: float
    yaw_rad: float
    speed_mps: float
    steer_rad: float
    pitch_rad: float = 0.0
    roll_rad: float = 0.0


@dataclass(frozen=True)
class TrajectoryChunk:
    """Poses for one generated chunk and its boundary state."""

    timestamps_us: npt.NDArray[np.int64]
    rig_poses_world: FloatArray
    boundary_state_after_chunk: VehicleState
    """State seeding the following chunk."""


@dataclass
class ControlSnapshot:
    """Currently held keys and selected view."""

    pressed: set[str] = field(default_factory=set)
    view_mode: str = "rgb"


__all__ = [
    "ControlSnapshot",
    "DriverCommand",
    "FloatArray",
    "Int32Array",
    "TrajectoryChunk",
    "UInt8Array",
    "VehicleState",
]
