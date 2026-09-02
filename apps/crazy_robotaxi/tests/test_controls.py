# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for configurable Crazy Robotaxi controls."""

from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np
import pytest
from crazy_robotaxi.controls import (
    BoundActionState,
    ControlsConfig,
    ControlsDocument,
    ControlsError,
    GamepadButtonStyle,
    InputBinding,
    binding_display,
    capture_binding,
    gamepad_driver_command,
    keyboard_drive_key,
    keyboard_driver_command,
    update_binding,
)
from omnidreams_game_engine.input import DriverInput

from flashdreams.runtime_v2.user_input_event import (
    GamepadUserInputEvent,
    KeyboardInputState,
    KeyboardUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

pytestmark = pytest.mark.ci_cpu


def _key_event(key: str, state: KeyboardInputState) -> KeyboardUserInputEvent:
    return KeyboardUserInputEvent(timestamp=np.uint64(1), key=key, state=state)


def test_keyboard_bindings_drive_and_dispatch_actions() -> None:
    keyboard = ControlsConfig().keyboard
    keyboard = replace(
        keyboard,
        drive_forward=(InputBinding("key", "i"), None),
        restart=(InputBinding("key", "p"), None),
    )
    controls = ControlsConfig(keyboard=keyboard)

    assert keyboard_drive_key(keyboard, "I") == "i"
    assert keyboard_drive_key(keyboard, "w") is None
    assert keyboard_driver_command(keyboard, {"i"}).throttle == 1.0
    assert BoundActionState(controls).apply(
        UserInputEvents([_key_event("P", KeyboardInputState.PRESSED)])
    ) == {"restart"}

    driver_input = DriverInput(
        keyboard_command=partial(keyboard_driver_command, keyboard),
        key_normalizer=partial(keyboard_drive_key, keyboard),
    )
    driver_input.apply(UserInputEvents([_key_event("I", KeyboardInputState.PRESSED)]))
    assert driver_input.command().throttle == 1.0


def test_gamepad_restart_uses_digital_or_analog_button_state() -> None:
    settings = ControlsConfig().gamepad
    digital = GamepadUserInputEvent(
        timestamp=np.uint64(1),
        action="state",
        pressed=(*((False,) * 9), True),
    )
    analog = GamepadUserInputEvent(
        timestamp=np.uint64(2),
        action="state",
        buttons=(*((0.0,) * 9), 1.0),
    )

    assert BoundActionState(ControlsConfig()).apply(UserInputEvents([digital])) == {
        "restart"
    }
    assert BoundActionState(ControlsConfig()).apply(UserInputEvents([analog])) == {
        "restart"
    }
    assert gamepad_driver_command(settings, analog) is not None


@pytest.mark.parametrize(
    ("style", "expected"),
    (("Xbox", "A"), ("PlayStation", "CROSS"), ("Nintendo Switch", "B")),
)
def test_gamepad_button_display_uses_one_configured_style(
    style: GamepadButtonStyle, expected: str
) -> None:
    assert binding_display("gamepad", InputBinding("button", 0), style) == expected


def test_duplicate_capture_swaps_the_previous_binding() -> None:
    original = ControlsConfig().keyboard

    updated = update_binding(original, "toggle_hints", 0, InputBinding("key", "r"))

    assert updated.toggle_hints == (InputBinding("key", "r"), None)
    assert updated.restart == (InputBinding("key", "h"), None)


def test_axis_capture_ignores_baseline_and_derives_steering_inversion() -> None:
    baseline = GamepadUserInputEvent(
        timestamp=np.uint64(1), action="state", axes=(0.0, 0.0)
    )
    neutral = GamepadUserInputEvent(
        timestamp=np.uint64(2), action="state", axes=(-0.2, 0.0)
    )
    moved_left = GamepadUserInputEvent(
        timestamp=np.uint64(3), action="state", axes=(-0.8, 0.0)
    )

    assert capture_binding("gamepad", "steering", neutral, baseline) is None
    assert capture_binding("gamepad", "steering", moved_left, baseline) == InputBinding(
        "axis", 0, direction="bidirectional", invert=True
    )

    held = replace(baseline, axes=(-0.8, 0.0))
    assert capture_binding("gamepad", "steering", baseline, held) is None


def test_controls_document_round_trips_sparse_yaml_and_comments(tmp_path: Path) -> None:
    path = tmp_path / "keyboard.yaml"
    path.write_text(
        "# Keep this note.\nschema_version: 1\nrestart: [p]\n",
        encoding="utf-8",
    )
    document = ControlsDocument.load(path, "keyboard")
    assert document.settings.restart == (InputBinding("key", "p"), None)

    document.save(
        update_binding(document.settings, "toggle_hints", 0, InputBinding("key", "j"))
    )

    saved = path.read_text(encoding="utf-8")
    assert "# Keep this note." in saved
    assert "drive_forward" not in saved
    assert ControlsDocument.load(path, "keyboard").settings.toggle_hints == (
        InputBinding("key", "j"),
        None,
    )


def test_controls_document_loads_round_trip_yaml_button_indices(
    tmp_path: Path,
) -> None:
    path = tmp_path / "gamepad.yaml"
    path.write_text(
        "schema_version: 1\nhandbrake:\n- button: 0\n-\n",
        encoding="utf-8",
    )

    document = ControlsDocument.load(path, "gamepad")

    assert document.settings.handbrake == (InputBinding("button", 0), None)


@pytest.mark.parametrize(
    "contents",
    (
        "schema_version: 1\nunknown: [q]\n",
        "schema_version: 1\nrestart: [q, w, e]\n",
        "schema_version: 1\nrestart: [q]\ntoggle_hints: [q]\n",
    ),
)
def test_controls_document_rejects_invalid_authored_bindings(
    tmp_path: Path, contents: str
) -> None:
    path = tmp_path / "keyboard.yaml"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ControlsError):
        ControlsDocument.load(path, "keyboard")
