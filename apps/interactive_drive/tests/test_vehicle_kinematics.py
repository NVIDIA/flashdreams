# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Camera continuity and frame-rate-independent vehicle integration."""

import math

import numpy as np
import pytest
from interactive_drive.config import ChunkConfig, VehicleConfig
from interactive_drive.math3d import rig_pose_from_state
from interactive_drive.simulation.ego_vehicle_kinematics import (
    integrate_vehicle,
    sample_chunk_trajectory,
    state_from_initial_pose,
)
from interactive_drive.types import DriverCommand, VehicleState

pytestmark = pytest.mark.ci_cpu


def test_initial_pose_preserves_pitch_roll_and_position():
    pose = rig_pose_from_state(1, 2, 3, 0.4, 0.1, -0.05)
    state = state_from_initial_pose(pose, 0.4, 3)
    restored = rig_pose_from_state(
        state.x_m, state.y_m, state.z_m, state.yaw_rad, state.pitch_rad, state.roll_rad
    )
    np.testing.assert_allclose(restored, pose, atol=1e-7)


@pytest.mark.parametrize("fps", [10, 30])
def test_pose_timestamps_begin_at_the_supplied_state_and_continue_across_chunks(fps):
    state = VehicleState(0, 0, 0, 0, 3, 0)
    chunk = ChunkConfig(fps=fps)
    vehicle = VehicleConfig(drag_mps2=0)
    first = sample_chunk_trajectory(
        state, 1_000_000, DriverCommand(), fps, chunk, vehicle, None
    )
    second = sample_chunk_trajectory(
        first.boundary_state_after_chunk,
        2_000_000,
        DriverCommand(),
        fps,
        chunk,
        vehicle,
        None,
    )
    for trajectory in (first, second):
        seconds = (trajectory.timestamps_us - 1_000_000) / 1e6
        np.testing.assert_allclose(
            trajectory.rig_poses_world[:, 0, 3], 3 * seconds, atol=2e-5
        )
    assert second.boundary_state_after_chunk.x_m == pytest.approx(6, abs=1e-5)


@pytest.mark.parametrize("speed", [1.0, 3.0, 15.0])
def test_vehicle_motion_agrees_at_ten_thirty_and_120_fps(speed):
    command = DriverCommand(throttle=0.35, steer=0.5, steer_is_direct=True)
    positions = []
    for fps in (10, 30, 120):
        state = VehicleState(0, 0, 0, 0, speed, 0)
        for _ in range(2 * fps):
            state = integrate_vehicle(state, command, 1 / fps, VehicleConfig())
        positions.append([state.x_m, state.y_m, state.yaw_rad, state.speed_mps])
    np.testing.assert_allclose(positions[:2], [positions[2]] * 2, atol=1e-5)
    assert all(math.isfinite(value) for value in positions[-1])
