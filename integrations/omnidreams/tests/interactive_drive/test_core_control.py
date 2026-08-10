# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import math

import pytest
from omnidreams.interactive_drive.config import ChunkConfig, VehicleConfig
from omnidreams.interactive_drive.demo import KeyboardDriveState
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


class _RecordingDriveSink:
    def __init__(self) -> None:
        self.commands: list[dict[str, float | bool]] = []

    def set_drive(self, **command: float | bool) -> None:
        self.commands.append(command)


def test_command_from_snapshot_maps_keyboard_state() -> None:
    snapshot = ControlSnapshot(pressed={"w", "a"})
    command = command_from_snapshot(snapshot)
    assert command.throttle == 1.0
    assert command.brake == 0.0
    assert command.steer == 1.0


def test_space_maps_to_handbrake_without_pressing_normal_brake() -> None:
    command = command_from_snapshot(ControlSnapshot(pressed={"space"}))

    assert command.handbrake is True
    assert command.brake == 0.0
    assert command.stop is False


def test_keyboard_drive_state_publishes_space_as_handbrake() -> None:
    sink = _RecordingDriveSink()
    keyboard = KeyboardDriveState(sink)
    keyboard.set_key("space", True)

    state = keyboard.update()

    assert state.brake == 0.0
    assert sink.commands[-1]["brake"] == 0.0
    assert sink.commands[-1]["handbrake"] is True


def test_keyboard_brake_target_enters_reverse_from_rest() -> None:
    sink = _RecordingDriveSink()
    keyboard = KeyboardDriveState(sink)
    keyboard.set_key("s", True)
    keyboard._last_update_s -= 0.1

    state = keyboard.update()

    assert state.target_speed_mps < 0.0
    assert state.reverse is True
    assert sink.commands[-1]["brake"] == 1.0
    assert sink.commands[-1]["handbrake"] is False


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


def test_keyboard_target_remains_stopped_without_input() -> None:
    sink = _RecordingDriveSink()
    keyboard = KeyboardDriveState(sink)
    keyboard._last_update_s -= 0.1

    state = keyboard.update()

    assert state.target_speed_mps == 0.0
    assert sink.commands[-1]["throttle"] == 0.0
    assert sink.commands[-1]["brake"] == 0.0


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


@pytest.mark.parametrize("manual_control", [False, True])
def test_brake_transitions_from_forward_to_reverse_after_stopping(
    manual_control: bool,
) -> None:
    vehicle = VehicleConfig()
    state = VehicleState(
        x_m=0.0, y_m=0.0, z_m=0.0, yaw_rad=0.0, speed_mps=0.1, steer_rad=0.0
    )
    brake = DriverCommand(brake=1.0, manual_control=manual_control)

    stopped = integrate_vehicle(state, brake, dt_s=0.1, vehicle=vehicle)
    reversing = integrate_vehicle(stopped, brake, dt_s=0.1, vehicle=vehicle)

    assert stopped.speed_mps == 0.0
    assert reversing.speed_mps < 0.0


@pytest.mark.parametrize("manual_control", [False, True])
def test_brake_ignores_tiny_forward_physics_drift_when_entering_reverse(
    manual_control: bool,
) -> None:
    vehicle = VehicleConfig()
    state = VehicleState(
        x_m=0.0,
        y_m=0.0,
        z_m=0.0,
        yaw_rad=0.0,
        speed_mps=0.001,
        steer_rad=0.0,
    )

    reversing = integrate_vehicle(
        state,
        DriverCommand(brake=1.0, manual_control=manual_control),
        dt_s=0.1,
        vehicle=vehicle,
    )

    assert reversing.speed_mps < 0.0


@pytest.mark.parametrize("initial_speed_mps", [-5.0, 5.0])
def test_handbrake_stops_without_reversing_direction(
    initial_speed_mps: float,
) -> None:
    vehicle = VehicleConfig()
    state = VehicleState(
        x_m=0.0,
        y_m=0.0,
        z_m=0.0,
        yaw_rad=0.0,
        speed_mps=initial_speed_mps,
        steer_rad=0.0,
    )
    handbrake = DriverCommand(handbrake=True, manual_control=True)

    for _ in range(20):
        state = integrate_vehicle(state, handbrake, dt_s=0.1, vehicle=vehicle)

    assert state.speed_mps == 0.0


def test_releasing_brake_while_reversing_coasts_toward_stop() -> None:
    vehicle = VehicleConfig()
    state = VehicleState(
        x_m=0.0, y_m=0.0, z_m=0.0, yaw_rad=0.0, speed_mps=-2.0, steer_rad=0.0
    )

    released = integrate_vehicle(
        state, DriverCommand(manual_control=True), dt_s=0.1, vehicle=vehicle
    )

    assert state.speed_mps < released.speed_mps < 0.0


def test_manual_control_without_input_stays_stopped() -> None:
    vehicle = VehicleConfig()
    state = VehicleState(
        x_m=0.0, y_m=0.0, z_m=0.0, yaw_rad=0.0, speed_mps=0.0, steer_rad=0.0
    )

    released = integrate_vehicle(
        state, DriverCommand(manual_control=True), dt_s=0.1, vehicle=vehicle
    )

    assert released.speed_mps == 0.0


def test_releasing_controls_while_moving_coasts_toward_stop() -> None:
    vehicle = VehicleConfig()
    state = VehicleState(
        x_m=0.0, y_m=0.0, z_m=0.0, yaw_rad=0.0, speed_mps=5.0, steer_rad=0.0
    )

    released = integrate_vehicle(
        state, DriverCommand(manual_control=True), dt_s=0.1, vehicle=vehicle
    )

    assert 0.0 < released.speed_mps < state.speed_mps


def test_manual_throttle_uses_configured_acceleration() -> None:
    vehicle = VehicleConfig()
    state = VehicleState(
        x_m=0.0, y_m=0.0, z_m=0.0, yaw_rad=0.0, speed_mps=0.0, steer_rad=0.0
    )
    throttle = DriverCommand(throttle=1.0, brake=0.0, manual_control=True)

    advanced = integrate_vehicle(state, throttle, dt_s=0.1, vehicle=vehicle)
    assert advanced.speed_mps == pytest.approx(vehicle.max_accel_mps2 * 0.1)


def test_keyboard_throttle_uses_arcade_acceleration() -> None:
    sink = _RecordingDriveSink()
    keyboard = KeyboardDriveState(sink)
    keyboard.set_key("w", True)
    keyboard._last_update_s -= 0.1

    state = keyboard.update()

    assert state.target_speed_mps == pytest.approx(VehicleConfig().max_accel_mps2 * 0.1)


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
