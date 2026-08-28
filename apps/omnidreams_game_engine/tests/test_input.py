# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for V2 event reduction."""

from dataclasses import replace

import numpy as np
import pytest
from omnidreams_game_engine.config import DriverInputConfig
from omnidreams_game_engine.input import DriverInput

from flashdreams.api_v2.user_input_event import UserInputEvent
from flashdreams.runtime_v2.user_input_event import (
    FocusUserInputEvent,
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
    reducer = DriverInput(
        DriverInputConfig(
            steering_scale=1.0,
            steering_rate_per_s=2.0,
            steering_return_rate_per_s=4.0,
        ),
        samples_per_second=10.0,
    )

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
    assert [command.steer for command in first.commands] == pytest.approx(
        [0.2, 0.4, 0.6]
    )
    assert all(command.throttle == 1.0 for command in first.commands)
    assert retained.commands[0].steer == pytest.approx(0.8)
    assert first.transition_timestamps_us == (1, 1, 1)
    assert first.transition_count == 2
    assert first.dropped_transition_count == 0
    assert retained.transition_timestamps_us == (None,)


def test_focus_loss_releases_drive_state() -> None:
    reducer = DriverInput(samples_per_second=30.0)
    reverse = reducer.reduce(
        _events(_key("ArrowDown", KeyboardInputState.PRESSED)),
        frame_count=1,
    )

    assert reverse.commands[0].brake == 1.0
    assert reverse.commands[0].throttle == 0.0
    assert not reverse.commands[0].reverse

    result = reducer.reduce(
        _events(FocusUserInputEvent(timestamp=np.uint64(0), focused=False)),
        frame_count=1,
    )

    assert result.commands[0].throttle == 0.0
    assert not result.commands[0].reverse


def test_short_press_and_release_are_preserved_in_the_next_chunk() -> None:
    reducer = DriverInput(samples_per_second=20.0)

    result = reducer.reduce(
        _timed_events(
            (1_000_000, _key("w", KeyboardInputState.PRESSED)),
            (1_100_000, _key("w", KeyboardInputState.RELEASED)),
        ),
        frame_count=5,
    )

    assert [command.throttle for command in result.commands] == [
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    assert result.transition_timestamps_us == (
        1_000_000,
        1_000_000,
        1_100_000,
        1_100_000,
        1_100_000,
    )
    assert result.transition_count == 2
    assert result.dropped_transition_count == 0


def test_duplicate_key_events_do_not_create_transitions() -> None:
    reducer = DriverInput(samples_per_second=30.0)

    result = reducer.reduce(
        _timed_events(
            (10, _key("w", KeyboardInputState.PRESSED)),
            (20, _key("w", KeyboardInputState.PRESSED)),
        ),
        frame_count=2,
    )

    assert result.transition_count == 1
    assert result.transition_timestamps_us == (10, 10)
    assert result.ignored_event_count == 0


def test_browser_space_key_engages_the_handbrake() -> None:
    reducer = DriverInput(samples_per_second=30.0)

    result = reducer.reduce(
        _timed_events((10, _key(" ", KeyboardInputState.PRESSED))),
        frame_count=1,
    )

    assert result.commands[0].handbrake
    assert result.commands[0].brake == 1.0


def test_transition_at_window_end_carries_into_the_next_chunk() -> None:
    reducer = DriverInput(samples_per_second=30.0)

    first = reducer.reduce(
        _timed_events(
            (10_000, _key("w", KeyboardInputState.PRESSED)),
            (50_000, _key("w", KeyboardInputState.RELEASED)),
            (90_000, _key("w", KeyboardInputState.PRESSED)),
        ),
        frame_count=2,
    )
    retained = reducer.reduce(UserInputEvents([]), frame_count=1)

    assert [command.throttle for command in first.commands] == [0.0, 0.0]
    assert first.transition_timestamps_us == (50_000, 50_000)
    assert first.transition_count == 2
    assert first.dropped_transition_count == 0
    assert retained.commands[0].throttle == 1.0
    assert retained.transition_timestamps_us == (90_000,)
