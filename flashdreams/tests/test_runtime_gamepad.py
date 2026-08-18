# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from flashdreams.runtime import (
    DRIVER_COMMAND,
    GAMEPAD_STATE_CAPABILITY,
    GAMEPAD_STATE_EVENT,
    GamepadToDriverCommand,
    InputCanonicalizer,
    KeyboardToDriverCommand,
    TimeWindow,
    UserInputCapability,
    UserInputEvent,
    UserInputs,
    UserInputSchema,
    parse_gamepad_state,
)

pytestmark = pytest.mark.ci_cpu

GAMEPAD_SOURCE = UserInputSchema(capabilities=(GAMEPAD_STATE_CAPABILITY,))
COMBINED_SOURCE = UserInputSchema(
    capabilities=(
        GAMEPAD_STATE_CAPABILITY,
        UserInputCapability(
            event_type="key_down",
            payload_fields=frozenset({"key"}),
        ),
        UserInputCapability(
            event_type="key_up",
            payload_fields=frozenset({"key"}),
        ),
    )
)


def _gamepad_event(
    *,
    timestamp_s: float,
    connected: bool = True,
    steer: float = 0.0,
    throttle: float = 0.0,
    brake: float = 0.0,
) -> UserInputEvent:
    return UserInputEvent(
        timestamp_s=timestamp_s,
        event_type=GAMEPAD_STATE_EVENT,
        payload={
            "connected": connected,
            "steer": steer,
            "throttle": throttle,
            "brake": brake,
            "reverse": False,
            "stop": False,
        },
    )


def test_gamepad_state_fails_fast_on_invalid_analog_input() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        parse_gamepad_state(
            {
                "connected": True,
                "steer": float("nan"),
                "throttle": 0.0,
                "brake": 0.0,
                "reverse": False,
                "stop": False,
            }
        )


def test_gamepad_converter_preserves_analog_values_and_timing() -> None:
    converter = GamepadToDriverCommand(deadzone=0.0)
    result = converter.convert(
        UserInputs(
            events=(
                _gamepad_event(
                    timestamp_s=0.25,
                    steer=-0.3,
                    throttle=0.75,
                ),
            )
        ),
        TimeWindow(start_s=0.0, end_s=1.0),
    )

    assert result is not None
    assert {name: result[name] for name in DRIVER_COMMAND.payload_fields} == {
        "throttle": 0.75,
        "brake": 0.0,
        "steer": -0.3,
        "stop": False,
        "reverse": False,
    }
    assert result["segments"] == (
        (
            0.0,
            0.25,
            {
                "throttle": 0.0,
                "brake": 0.0,
                "steer": 0.0,
                "stop": False,
                "reverse": False,
            },
        ),
        (
            0.25,
            1.0,
            {
                "throttle": 0.75,
                "brake": 0.0,
                "steer": -0.3,
                "stop": False,
                "reverse": False,
            },
        ),
    )


def test_keyboard_resumes_when_gamepad_disconnects() -> None:
    canonicalizer = InputCanonicalizer(
        (GamepadToDriverCommand(), KeyboardToDriverCommand())
    )
    canonicalizer.canonicalize(
        UserInputs(events=(_gamepad_event(timestamp_s=0.1, throttle=0.8),)),
        window=TimeWindow(start_s=0.0, end_s=0.5),
        source_schema=COMBINED_SOURCE,
    )

    result = canonicalizer.canonicalize(
        UserInputs(
            events=(
                _gamepad_event(timestamp_s=0.5, connected=False),
                UserInputEvent(
                    timestamp_s=0.5,
                    event_type="key_down",
                    payload={"key": "w"},
                ),
            )
        ),
        window=TimeWindow(start_s=0.5, end_s=1.0),
        source_schema=COMBINED_SOURCE,
    )

    command = result.values[DRIVER_COMMAND.name]
    assert command["throttle"] == 1.0
    assert result.metadata["canonical_sources"] == {DRIVER_COMMAND.name: "keyboard"}
