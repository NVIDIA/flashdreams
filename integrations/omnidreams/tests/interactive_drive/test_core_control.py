# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import pytest
from omnidreams.interactive_drive.config import ChunkConfig, VehicleConfig
from omnidreams.interactive_drive.control import (
    command_from_snapshot,
    integrate_vehicle,
    sample_chunk_trajectory,
)
from omnidreams.interactive_drive.types import (
    ControlSnapshot,
    DriverCommand,
    VehicleState,
)


def test_command_from_snapshot_maps_keyboard_state() -> None:
    snapshot = ControlSnapshot(pressed={"w", "a"})
    command = command_from_snapshot(snapshot)
    assert command.throttle == 1.0
    assert command.brake == 0.0
    assert command.steer == 1.0


def test_sample_chunk_trajectory_advances_pose_and_time() -> None:
    state = VehicleState(
        x_m=0.0, y_m=0.0, z_m=0.0, yaw_rad=0.0, speed_mps=0.0, steer_rad=0.0
    )
    snapshot = ControlSnapshot(pressed={"w"})
    command = command_from_snapshot(snapshot)

    chunk = sample_chunk_trajectory(
        start_state=state,
        start_timestamp_us=1000,
        command=command,
        chunk_size=4,
        chunk_config=ChunkConfig(fps=10, initial_chunk_frames=2, chunk_frames=2),
        vehicle_config=VehicleConfig(),
        ground_snapper=None,
    )

    assert list(chunk.timestamps_us) == [1000, 101000, 201000, 301000]
    assert chunk.rig_poses_world.shape == (4, 4, 4)
    assert chunk.boundary_state_after_chunk.x_m > 0.0
    assert chunk.boundary_state_after_chunk.speed_mps > 0.0


def test_integrate_vehicle_accumulates_steering_gradually() -> None:
    vehicle = VehicleConfig(
        max_steer_rad=0.5, steer_rate_rad_per_s=1.0, steer_return_rate_rad_per_s=0.5
    )
    state = VehicleState(
        x_m=0.0, y_m=0.0, z_m=0.0, yaw_rad=0.0, speed_mps=0.0, steer_rad=0.0
    )

    state = integrate_vehicle(
        state, DriverCommand(steer=1.0), dt_s=0.1, vehicle=vehicle
    )
    assert state.steer_rad == pytest.approx(0.1)

    state = integrate_vehicle(
        state, DriverCommand(steer=1.0), dt_s=0.1, vehicle=vehicle
    )
    assert state.steer_rad == pytest.approx(0.2)


def test_integrate_vehicle_recenters_steering_after_release() -> None:
    vehicle = VehicleConfig(
        max_steer_rad=0.5, steer_rate_rad_per_s=1.0, steer_return_rate_rad_per_s=0.5
    )
    state = VehicleState(
        x_m=0.0, y_m=0.0, z_m=0.0, yaw_rad=0.0, speed_mps=0.0, steer_rad=0.2
    )

    released = integrate_vehicle(
        state, DriverCommand(steer=0.0), dt_s=0.1, vehicle=vehicle
    )
    assert released.steer_rad == pytest.approx(0.15)

    released = integrate_vehicle(
        released, DriverCommand(steer=0.0), dt_s=0.3, vehicle=vehicle
    )
    assert released.steer_rad == pytest.approx(0.0)
