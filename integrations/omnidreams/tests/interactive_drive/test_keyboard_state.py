# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for :class:`KeyboardState`'s :class:`RuntimeControls` contract.

The display loop depends on the rising-edge consume semantics for reset:
exactly one ``consume_reset_request`` call returns ``True`` per call to
``request_reset``, and rapid presses must coalesce so a single reset isn't
processed twice.
"""

from types import SimpleNamespace

import pytest
from omnidreams.interactive_drive.crazy_robotaxi.game import TaxiGameSnapshot
from omnidreams.interactive_drive.crazy_robotaxi.input import (
    CrazyRobotaxiKeyboardState,
)
from omnidreams.interactive_drive.demo import KeyboardDriveState
from omnidreams.interactive_drive.input.keyboard import KeyboardState
from omnidreams.interactive_drive.streaming_presenter import (
    _BROWSER_KEY_TO_VIEW_MODE,
)
from omnidreams.interactive_drive.types import DriverCommand, VehicleState

pytestmark = pytest.mark.ci_cpu


class _DriveSink:
    def __init__(self) -> None:
        self.command = SimpleNamespace()

    def set_drive(self, **command: object) -> None:
        self.command = SimpleNamespace(**command)

    def release_all(self) -> None:
        pass


def test_consume_reset_request_returns_false_when_no_reset_pending() -> None:
    keyboard = KeyboardState()
    assert keyboard.consume_reset_request() is False


def test_consume_reset_request_returns_true_once_per_request() -> None:
    keyboard = KeyboardState()
    keyboard.request_reset()
    assert keyboard.consume_reset_request() is True
    assert keyboard.consume_reset_request() is False


def test_repeated_request_reset_coalesces_to_one_consume() -> None:
    """Multiple presses of ``r`` between consumes must not double-fire.

    The loop tears down and rebuilds sim/pipeline on every ``True``; if
    rapid presses produced multiple ``True`` returns, the user would see
    the loading frame N times for N presses instead of once.
    """
    keyboard = KeyboardState()
    keyboard.request_reset()
    keyboard.request_reset()
    keyboard.request_reset()
    assert keyboard.consume_reset_request() is True
    assert keyboard.consume_reset_request() is False


def test_view_mode_reflects_set_view_mode() -> None:
    keyboard = KeyboardState()
    assert keyboard.view_mode == "rgb"
    keyboard.set_view_mode("hdmap")
    assert keyboard.view_mode == "hdmap"


def test_browser_key_three_selects_physx_view() -> None:
    assert _BROWSER_KEY_TO_VIEW_MODE["3"] == "physx"


def test_keyboard_state_uses_shared_key_normalization() -> None:
    keyboard = KeyboardState()
    keyboard.set_key("ArrowUp", True)
    keyboard.set_key("ArrowLeft", True)

    command = keyboard.command()

    assert command.throttle == 1.0
    assert command.steer == 1.0


def test_keyboard_drive_command_overrides_connected_wheel_command() -> None:
    keyboard = KeyboardState()
    keyboard.set_drive_command(
        DriverCommand(throttle=0.0, manual_control=True), source="wheel"
    )
    keyboard.set_drive_command(
        DriverCommand(throttle=1.0, manual_control=True), source="keyboard"
    )

    assert keyboard.command().throttle == 1.0

    keyboard.set_drive_command(None, source="keyboard")
    assert keyboard.command().throttle == 0.0


def test_space_overrides_active_drive_command_with_handbrake() -> None:
    keyboard = CrazyRobotaxiKeyboardState()
    keyboard.set_drive_command(
        DriverCommand(
            throttle=1.0,
            steer=0.25,
            steer_is_direct=True,
            manual_control=True,
        )
    )
    keyboard.set_key("space", True)

    command = keyboard.command()

    assert command.handbrake is True
    assert command.throttle == 0.0
    assert command.brake == 0.0
    assert command.steer == 0.25


def test_consume_exit_scene_request_returns_false_when_none_pending() -> None:
    keyboard = KeyboardState()
    assert keyboard.consume_exit_scene_request() is False


def test_consume_exit_scene_request_returns_true_once_per_request() -> None:
    """The presenter drains the wheel-button exit request exactly once.

    Same rising-edge contract as reset: a single exit-to-selection must not
    re-fire across ticks once consumed.
    """
    keyboard = KeyboardState()
    keyboard.request_exit_scene()
    keyboard.request_exit_scene()
    assert keyboard.consume_exit_scene_request() is True
    assert keyboard.consume_exit_scene_request() is False


def test_runtime_state_publishes_vehicle_and_taxi_atomically() -> None:
    keyboard = CrazyRobotaxiKeyboardState()
    vehicle = VehicleState(1.0, 2.0, 0.0, 0.0, 3.0, 0.0)
    taxi = TaxiGameSnapshot(
        phase="seeking_pickup",
        target_xyz_m=(10.0, 0.0, 0.0),
        distance_m=9.0,
        relative_bearing_rad=0.0,
        target_radius_m=5.0,
        remaining_time_s=None,
        score=0,
    )

    keyboard.update_runtime_state(vehicle, taxi)

    assert keyboard.runtime_state == (vehicle, taxi)
    assert keyboard.vehicle_state is vehicle
    assert keyboard.taxi_game_state is taxi

    keyboard.clear_telemetry()
    assert keyboard.runtime_state == (None, None)


def test_keyboard_state_validates_and_consumes_taxi_name_once() -> None:
    keyboard = CrazyRobotaxiKeyboardState()

    assert keyboard.submit_taxi_name(" Player 1 ") is True
    assert keyboard.consume_taxi_name_submission() == "Player 1"
    assert keyboard.consume_taxi_name_submission() is None
    assert keyboard.submit_taxi_name("bad.name") is False


def test_keyboard_state_suppresses_driving_after_taxi_game_over() -> None:
    keyboard = CrazyRobotaxiKeyboardState()
    keyboard.set_key("w", True)
    vehicle = VehicleState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    taxi = TaxiGameSnapshot(
        phase="seeking_pickup",
        target_xyz_m=(10.0, 0.0, 0.0),
        distance_m=10.0,
        relative_bearing_rad=0.0,
        target_radius_m=5.0,
        remaining_time_s=None,
        score=0,
        session_state="leaderboard",
    )
    keyboard.update_runtime_state(vehicle, taxi)

    assert keyboard.command().throttle == 0.0
