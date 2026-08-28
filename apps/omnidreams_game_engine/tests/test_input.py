# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for V2 event reduction."""

from dataclasses import replace

import numpy as np
import pytest
from omnidreams_game_engine.input import DriverInput

from flashdreams.api_v2.user_input_event import UserInputEvent
from flashdreams.runtime_v2.user_input_event import (
    GamepadUserInputEvent,
    GameWheelUserInputEvent,
    KeyboardInputState,
    KeyboardUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

pytestmark = pytest.mark.ci_cpu


def _events(*events: UserInputEvent) -> UserInputEvents:
    return UserInputEvents(
        [
            replace(event, timestamp=np.uint64(index))
            for index, event in enumerate(events)
        ]
    )


def _timed_events(
    *events: tuple[int, UserInputEvent],
) -> UserInputEvents:
    return UserInputEvents(
        [
            replace(event, timestamp=np.uint64(timestamp_us))
            for timestamp_us, event in events
        ]
    )


def _key(key: str, state: KeyboardInputState) -> KeyboardUserInputEvent:
    return KeyboardUserInputEvent(timestamp=np.uint64(0), key=key, state=state)


def test_held_keys_produce_one_command_per_model_frame() -> None:
    reducer = DriverInput()

    first = reducer.reduce(
        _events(
            _key("w", KeyboardInputState.PRESSED), _key("a", KeyboardInputState.PRESSED)
        ),
        frame_count=3,
    )
    retained = reducer.reduce(
        UserInputEvents([]),
        frame_count=1,
    )

    assert len(first.commands) == 3
    assert all(command == first.commands[0] for command in first.commands)
    assert first.commands[0].steer == 1.0
    assert not first.commands[0].steer_is_direct
    assert not first.commands[0].manual_control
    assert all(command.throttle == 1.0 for command in first.commands)
    assert retained.commands[0] == first.commands[0]
    assert first.transition_timestamps_us == (1, None, None)
    assert first.transition_count == 2
    assert first.coalesced_transition_count == 1
    assert retained.transition_timestamps_us == (None,)


def test_reverse_key_matches_interactive_drive_command() -> None:
    reducer = DriverInput()
    reverse = reducer.reduce(
        _events(_key("ArrowDown", KeyboardInputState.PRESSED)),
        frame_count=1,
    )

    assert reverse.commands[0].brake == 0.0
    assert reverse.commands[0].throttle == 1.0
    assert reverse.commands[0].reverse

    released = reducer.reduce(
        _events(_key("ArrowDown", KeyboardInputState.RELEASED)),
        frame_count=1,
    )

    assert released.commands[0].throttle == 0.0
    assert not released.commands[0].reverse


def test_short_press_and_release_collapse_to_latest_state() -> None:
    reducer = DriverInput()

    result = reducer.reduce(
        _timed_events(
            (1_000_000, _key("w", KeyboardInputState.PRESSED)),
            (1_100_000, _key("w", KeyboardInputState.RELEASED)),
        ),
        frame_count=5,
    )

    assert [command.throttle for command in result.commands] == [0.0] * 5
    assert result.transition_timestamps_us == (
        1_100_000,
        None,
        None,
        None,
        None,
    )
    assert result.transition_count == 2
    assert result.coalesced_transition_count == 1


def test_duplicate_key_events_do_not_create_transitions() -> None:
    reducer = DriverInput()

    result = reducer.reduce(
        _timed_events(
            (10, _key("w", KeyboardInputState.PRESSED)),
            (20, _key("w", KeyboardInputState.PRESSED)),
        ),
        frame_count=2,
    )

    assert result.transition_count == 1
    assert result.transition_timestamps_us == (10, None)
    assert result.ignored_event_count == 1


def test_space_key_requests_the_interactive_drive_stop_command() -> None:
    reducer = DriverInput()

    result = reducer.reduce(
        _timed_events((10, _key("space", KeyboardInputState.PRESSED))),
        frame_count=1,
    )

    assert result.commands[0].stop
    assert not result.commands[0].handbrake


def test_latest_state_conditions_the_entire_next_chunk() -> None:
    reducer = DriverInput()

    first = reducer.reduce(
        _timed_events(
            (10_000, _key("w", KeyboardInputState.PRESSED)),
            (50_000, _key("w", KeyboardInputState.RELEASED)),
            (90_000, _key("w", KeyboardInputState.PRESSED)),
        ),
        frame_count=2,
    )
    retained = reducer.reduce(UserInputEvents([]), frame_count=1)

    assert [command.throttle for command in first.commands] == [1.0, 1.0]
    assert first.transition_timestamps_us == (90_000, None)
    assert first.transition_count == 3
    assert first.coalesced_transition_count == 2
    assert retained.commands[0].throttle == 1.0
    assert retained.transition_timestamps_us == (None,)


def test_gamepad_state_overrides_keyboard_until_disconnect() -> None:
    reducer = DriverInput()
    buttons = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.75)
    gamepad = GamepadUserInputEvent(
        timestamp=np.uint64(20),
        action="state",
        axes=(-0.5,),
        buttons=buttons,
    )

    controlled = reducer.reduce(
        UserInputEvents([_key("w", KeyboardInputState.PRESSED), gamepad]),
        frame_count=2,
    )

    assert controlled.commands[0].throttle == pytest.approx(0.75)
    assert controlled.commands[0].brake == pytest.approx(0.25)
    assert controlled.commands[0].steer == pytest.approx(0.5)
    assert controlled.commands[0].steer_is_direct
    assert controlled.commands[0].manual_control
    assert controlled.commands[1] == controlled.commands[0]
    assert reducer.source() == "wheel/gamepad"

    disconnected = reducer.reduce(
        UserInputEvents(
            [
                GamepadUserInputEvent(
                    timestamp=np.uint64(30),
                    action="disconnected",
                )
            ]
        ),
        frame_count=1,
    )

    assert disconnected.commands[0].throttle == 1.0
    assert not disconnected.commands[0].manual_control
    assert reducer.source() == "keyboard"


def test_wheel_state_uses_direct_pedal_and_steering_values() -> None:
    reducer = DriverInput()

    result = reducer.reduce(
        UserInputEvents(
            [
                GameWheelUserInputEvent(
                    timestamp=np.uint64(40),
                    action="state",
                    steering=-0.4,
                    throttle=0.8,
                    brake=0.1,
                )
            ]
        ),
        frame_count=2,
    )

    assert result.commands[0].steer == pytest.approx(0.4)
    assert result.commands[0].throttle == pytest.approx(0.8)
    assert result.commands[0].brake == pytest.approx(0.1)
    assert result.commands[0].steer_is_direct
    assert result.commands[0].manual_control
    assert result.commands[1] == result.commands[0]
