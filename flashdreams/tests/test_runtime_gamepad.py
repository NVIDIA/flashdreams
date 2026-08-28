# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from flashdreams.runtime import (
    DRIVER_COMMAND,
    DrivingInputConverter,
    GAMEPAD_STATE_CAPABILITY,
    GAMEPAD_STATE_EVENT,
    InputCanonicalizer,
    TimeWindow,
    UserInputCapability,
    UserInputEvent,
    UserInputs,
    UserInputSchema,
    parse_gamepad_state,
)

pytestmark = pytest.mark.ci_cpu

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
            }
        )


def test_gamepad_converter_preserves_analog_values_and_timing() -> None:
    canonicalizer = InputCanonicalizer((DrivingInputConverter(),))
    result = canonicalizer.canonicalize(
        UserInputs(
            events=(
                _gamepad_event(
                    timestamp_s=0.25,
                    steer=-0.3,
                    throttle=0.75,
                ),
            )
        ),
        window=TimeWindow(start_s=0.0, end_s=1.0),
        source_schema=COMBINED_SOURCE,
    )

    command = result.values[DRIVER_COMMAND.name]
    assert {name: command[name] for name in DRIVER_COMMAND.payload_fields} == {
        "throttle": 0.75,
        "brake": 0.0,
        "steer": -0.3,
        "stop": False,
        "reverse": False,
    }
    assert command["segments"] == (
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
    canonicalizer = InputCanonicalizer((DrivingInputConverter(),))
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


def test_keyboard_and_gamepad_arbitrate_at_segment_boundaries() -> None:
    canonicalizer = InputCanonicalizer((DrivingInputConverter(),))

    result = canonicalizer.canonicalize(
        UserInputs(
            events=(
                UserInputEvent(
                    timestamp_s=0.0,
                    event_type="key_down",
                    payload={"key": "w"},
                ),
                _gamepad_event(timestamp_s=0.25, throttle=0.5),
                _gamepad_event(timestamp_s=0.75),
            )
        ),
        window=TimeWindow(start_s=0.0, end_s=1.0),
        source_schema=COMBINED_SOURCE,
    )

    segments = result.values[DRIVER_COMMAND.name]["segments"]
    assert tuple((start, end, level["throttle"]) for start, end, level in segments) == (
        (0.0, 0.25, 1.0),
        (0.25, 0.75, 0.5),
        (0.75, 1.0, 1.0),
    )
