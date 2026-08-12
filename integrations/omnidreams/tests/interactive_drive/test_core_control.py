# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import math

import pytest
from omnidreams.interactive_drive.config import ChunkConfig, VehicleConfig
from omnidreams.interactive_drive.input.keyboard import command_from_snapshot
from omnidreams.interactive_drive.simulation.components import (
    vehicle_dynamics_from_config,
)
from omnidreams.interactive_drive.simulation.ego_vehicle_kinematics import (
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


def test_s_key_requests_reverse_propulsion() -> None:
    command = command_from_snapshot(ControlSnapshot(pressed={"s"}))

    assert command.throttle == 1.0
    assert command.brake == 0.0
    assert command.reverse is True


def test_opposing_keyboard_directions_brake() -> None:
    command = command_from_snapshot(ControlSnapshot(pressed={"w", "s"}))

    assert command.throttle == 0.0
    assert command.brake == 1.0
    assert command.reverse is False


def test_keyboard_reverse_applies_throttle_in_opposite_direction() -> None:
    vehicle = VehicleConfig()
    command = command_from_snapshot(ControlSnapshot(pressed={"s"}))
    state = VehicleState(
        x_m=0.0, y_m=0.0, z_m=0.0, yaw_rad=0.0, speed_mps=1.0, steer_rad=0.0
    )

    first_reverse_step = integrate_vehicle(state, command, dt_s=0.1, vehicle=vehicle)
    assert first_reverse_step.speed_mps < state.speed_mps

    state = first_reverse_step
    for _ in range(19):
        state = integrate_vehicle(state, command, dt_s=0.1, vehicle=vehicle)

    assert state.speed_mps < 0.0
    assert state.x_m < 0.0


def test_manual_release_does_not_creep_forward() -> None:
    vehicle = VehicleConfig()
    stopped = VehicleState(
        x_m=0.0, y_m=0.0, z_m=0.0, yaw_rad=0.0, speed_mps=0.0, steer_rad=0.0
    )

    released = integrate_vehicle(
        stopped,
        DriverCommand(manual_control=True),
        dt_s=0.1,
        vehicle=vehicle,
    )

    assert released.speed_mps == 0.0
    assert released.x_m == 0.0


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


def test_manual_brake_overrides_throttle_to_a_stop() -> None:
    """Gas + brake pressed together must bleed speed toward a stop.

    Regression for the HUD/ego mismatch: the manual-control branch used to
    give throttle priority, so holding both pedals built speed. Brake now
    wins, matching the HUD's speed readout and real-car behaviour.
    """
    vehicle = VehicleConfig()
    state = VehicleState(
        x_m=0.0, y_m=0.0, z_m=0.0, yaw_rad=0.0, speed_mps=10.0, steer_rad=0.0
    )
    both = DriverCommand(throttle=1.0, brake=1.0, manual_control=True)

    decelerating = integrate_vehicle(state, both, dt_s=0.1, vehicle=vehicle)
    assert decelerating.speed_mps < state.speed_mps

    # Held long enough, the vehicle comes to rest rather than creeping.
    for _ in range(200):
        state = integrate_vehicle(state, both, dt_s=0.1, vehicle=vehicle)
    assert state.speed_mps == pytest.approx(0.0, abs=1e-6)


def test_manual_throttle_only_still_accelerates() -> None:
    """Throttle without brake keeps its acceleration behaviour."""
    vehicle = VehicleConfig()
    state = VehicleState(
        x_m=0.0, y_m=0.0, z_m=0.0, yaw_rad=0.0, speed_mps=0.0, steer_rad=0.0
    )
    throttle = DriverCommand(throttle=1.0, brake=0.0, manual_control=True)

    advanced = integrate_vehicle(state, throttle, dt_s=0.1, vehicle=vehicle)
    assert advanced.speed_mps > state.speed_mps


@pytest.mark.parametrize("manual_control", [False, True])
def test_speed_limit_is_only_applied_when_enabled(manual_control: bool) -> None:
    state = VehicleState(
        x_m=0.0, y_m=0.0, z_m=0.0, yaw_rad=0.0, speed_mps=18.0, steer_rad=0.0
    )
    throttle = DriverCommand(throttle=1.0, manual_control=manual_control)

    limited = integrate_vehicle(
        state,
        throttle,
        dt_s=0.1,
        vehicle=VehicleConfig(speed_limit_enabled=True, max_speed_mps=18.0),
    )
    unlimited = integrate_vehicle(
        state,
        throttle,
        dt_s=0.1,
        vehicle=VehicleConfig(speed_limit_enabled=False, max_speed_mps=18.0),
    )

    assert limited.speed_mps == pytest.approx(18.0)
    assert unlimited.speed_mps > 18.0


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


@pytest.mark.parametrize("speed_mps", [0.5, -4.0])
def test_low_speed_and_reverse_turns_keep_the_rear_axle_no_slip(
    speed_mps: float,
) -> None:
    vehicle = VehicleConfig(drag_mps2=0.0)
    design = vehicle_dynamics_from_config(vehicle)
    command = DriverCommand(steer=0.6, steer_is_direct=True)
    state = VehicleState(
        x_m=0.0,
        y_m=0.0,
        z_m=0.0,
        yaw_rad=0.0,
        speed_mps=speed_mps,
        steer_rad=0.0,
        velocity_x_mps=speed_mps,
        velocity_y_mps=0.0,
    )

    for _ in range(200):
        state = integrate_vehicle(state, command, dt_s=0.02, vehicle=vehicle)

    expected_yaw_rate = speed_mps / vehicle.wheel_base_m * math.tan(state.steer_rad)
    left = (-math.sin(state.yaw_rad), math.cos(state.yaw_rad))
    cg_lateral_speed = state.velocity_x_mps * left[0] + state.velocity_y_mps * left[1]
    rear_axle_lateral_speed = (
        cg_lateral_speed - design.rear_axle_to_cg_m * state.yaw_rate_radps
    )

    assert state.yaw_rate_radps == pytest.approx(expected_yaw_rate, rel=1e-6)
    assert rear_axle_lateral_speed == pytest.approx(0.0, abs=1e-6)
