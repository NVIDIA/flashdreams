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
from omnidreams.interactive_drive.demo import KeyboardDriveState
from omnidreams.interactive_drive.input.keyboard import KeyboardState
from omnidreams.interactive_drive.streaming_presenter import (
    _BROWSER_KEY_TO_VIEW_MODE,
)
from omnidreams.interactive_drive.types import DriverCommand

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


def test_keyboard_state_maps_arrow_down_to_reverse() -> None:
    keyboard = KeyboardState()
    keyboard.set_key("ArrowDown", True)

    command = keyboard.command()

    assert command.throttle == 1.0
    assert command.brake == 0.0
    assert command.reverse is True


def test_interactive_drive_s_key_publishes_reverse_command() -> None:
    sink = _DriveSink()
    keyboard = KeyboardDriveState(sink)

    assert keyboard.set_key("s", True) is True
    state = keyboard.update()

    assert sink.command.throttle == 1.0
    assert sink.command.brake == 0.0
    assert sink.command.reverse is True
    assert state.reverse is True


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
