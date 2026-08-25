# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for V2 event reduction."""

import numpy as np
import pytest
from omnidreams_game_engine.config import DriverInputConfig
from omnidreams_game_engine.input import DriverInput

from flashdreams.api_v2.user_input_event_data import UserInputEventData
from flashdreams.runtime_v2.user_input_event import (
    FocusUserInputEventData,
    KeyboardInputState,
    KeyboardUserInputEventData,
    UserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

pytestmark = pytest.mark.ci_cpu


def _events(*data: UserInputEventData) -> UserInputEvents:
    return UserInputEvents(
        [
            UserInputEvent(timestamp=np.uint64(index), event_data=value)
            for index, value in enumerate(data)
        ]
    )


def _key(key: str, state: KeyboardInputState) -> KeyboardUserInputEventData:
    return KeyboardUserInputEventData(key=key, state=state)


def test_held_keys_produce_one_command_per_model_frame() -> None:
    reducer = DriverInput(
        DriverInputConfig(
            steering_scale=1.0,
            steering_rate_per_s=2.0,
            steering_return_rate_per_s=4.0,
        )
    )

    first = reducer.reduce(
        _events(
            _key("w", KeyboardInputState.PRESSED), _key("a", KeyboardInputState.PRESSED)
        ),
        frame_count=3,
        frame_interval_s=0.1,
        accepting_text=False,
    )
    retained = reducer.reduce(
        UserInputEvents([]),
        frame_count=1,
        frame_interval_s=0.1,
        accepting_text=False,
    )

    assert len(first.commands) == 3
    assert [command.steer for command in first.commands] == pytest.approx(
        [0.2, 0.4, 0.6]
    )
    assert all(command.throttle == 1.0 for command in first.commands)
    assert retained.commands[0].steer == pytest.approx(0.8)


def test_focus_loss_releases_drive_state() -> None:
    reducer = DriverInput()
    reverse = reducer.reduce(
        _events(_key("ArrowDown", KeyboardInputState.PRESSED)),
        frame_count=1,
        frame_interval_s=1.0 / 30.0,
        accepting_text=False,
    )

    assert reverse.commands[0].brake == 1.0
    assert reverse.commands[0].throttle == 0.0
    assert not reverse.commands[0].reverse

    result = reducer.reduce(
        _events(FocusUserInputEventData(focused=False)),
        frame_count=1,
        frame_interval_s=1.0 / 30.0,
        accepting_text=False,
    )

    assert result.commands[0].throttle == 0.0
    assert not result.commands[0].reverse


def test_terminal_text_is_reduced_from_the_same_v2_events() -> None:
    reducer = DriverInput()
    result = reducer.reduce(
        _events(
            _key("A", KeyboardInputState.PRESSED),
            _key("I", KeyboardInputState.PRESSED),
            _key("enter", KeyboardInputState.PRESSED),
        ),
        frame_count=1,
        frame_interval_s=1.0 / 30.0,
        accepting_text=True,
    )

    assert result.text == "AI"
    assert result.submitted_text == "AI"
