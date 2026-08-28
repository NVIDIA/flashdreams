# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for V2 retained driving input."""

from dataclasses import replace

import numpy as np
import pytest
from omnidreams_game_engine.input import DriverInput
from omnidreams_game_engine.types import DriverCommand

from flashdreams.api_v2.user_input_event import UserInputEvent
from flashdreams.runtime_v2.input_timeline import RealtimeInputTimeline
from flashdreams.runtime_v2.user_input_event import (
    FocusUserInputEvent,
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


def _key(key: str, state: KeyboardInputState) -> KeyboardUserInputEvent:
    return KeyboardUserInputEvent(timestamp=np.uint64(0), key=key, state=state)


def test_held_keyboard_state_survives_empty_model_event_batches() -> None:
    state = DriverInput()
    state.apply(
        _events(
            _key("w", KeyboardInputState.PRESSED),
            _key("a", KeyboardInputState.PRESSED),
        )
    )
    first = state.command()

    state.apply(UserInputEvents([]))

    assert state.command() == first
    assert first.throttle == 1.0
    assert first.steer == 1.0
    assert not first.steer_is_direct
    assert not first.manual_control


def test_arrow_keys_share_interactive_drive_mapping() -> None:
    state = DriverInput()
    state.apply(_events(_key("ArrowDown", KeyboardInputState.PRESSED)))

    reverse = state.command()

    assert reverse.brake == 0.0
    assert reverse.throttle == 1.0
    assert reverse.reverse

    state.apply(_events(_key("ArrowDown", KeyboardInputState.RELEASED)))
    assert state.command().throttle == 0.0
    assert not state.command().reverse


def test_browser_space_key_matches_controller_brake() -> None:
    keyboard = DriverInput()
    keyboard.apply(_events(_key(" ", KeyboardInputState.PRESSED)))

    controller = DriverInput()
    controller.apply(
        UserInputEvents(
            [
                GamepadUserInputEvent(
                    timestamp=np.uint64(20),
                    action="state",
                    axes=(0.0,),
                    buttons=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                )
            ]
        )
    )

    assert keyboard.command().brake == controller.command().brake == 1.0
    assert keyboard.command().manual_control


def test_gamepad_state_overrides_keyboard_until_disconnect() -> None:
    state = DriverInput()
    buttons = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.75)
    gamepad = GamepadUserInputEvent(
        timestamp=np.uint64(20),
        action="state",
        axes=(-0.5,),
        buttons=buttons,
    )
    state.apply(UserInputEvents([_key("w", KeyboardInputState.PRESSED), gamepad]))

    controlled = state.command()

    assert controlled.throttle == pytest.approx(0.75)
    assert controlled.brake == pytest.approx(0.25)
    assert controlled.steer == pytest.approx(0.5)
    assert controlled.steer_is_direct
    assert controlled.manual_control
    assert state.source() == "wheel/gamepad"

    state.apply(
        UserInputEvents(
            [
                GamepadUserInputEvent(
                    timestamp=np.uint64(30),
                    action="disconnected",
                )
            ]
        )
    )

    assert state.command().throttle == 1.0
    assert not state.command().manual_control
    assert state.source() == "keyboard"


def test_gamepad_r_shoulder_selects_reverse_only_while_held() -> None:
    state = DriverInput()
    forward_buttons = (0.0,) * 7 + (0.75,)
    reverse_buttons = (0.0,) * 5 + (1.0, 0.0, 0.75)

    state.apply(
        UserInputEvents(
            [
                GamepadUserInputEvent(
                    timestamp=np.uint64(30),
                    action="state",
                    buttons=reverse_buttons,
                    pressed=(False,) * 5 + (True, False, True),
                )
            ]
        )
    )

    assert state.command().reverse
    assert state.command().throttle == pytest.approx(0.75)

    state.apply(
        UserInputEvents(
            [
                GamepadUserInputEvent(
                    timestamp=np.uint64(40),
                    action="state",
                    buttons=forward_buttons,
                    pressed=(False,) * 8,
                )
            ]
        )
    )

    assert not state.command().reverse
    assert state.command().throttle == pytest.approx(0.75)


def test_wheel_state_uses_direct_pedal_and_steering_values() -> None:
    state = DriverInput()
    state.apply(
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
        )
    )

    command = state.command()

    assert command.steer == pytest.approx(0.4)
    assert command.throttle == pytest.approx(0.8)
    assert command.brake == pytest.approx(0.1)
    assert command.steer_is_direct
    assert command.manual_control


def test_timestamped_tap_is_preserved_across_physics_frames() -> None:
    state = DriverInput()
    timeline = RealtimeInputTimeline(samples_per_second=30.0)
    input_times_s = state.apply(
        UserInputEvents(
            [
                replace(
                    _key("a", KeyboardInputState.PRESSED),
                    timestamp=np.uint64(1_000_000),
                ),
                replace(
                    _key("a", KeyboardInputState.RELEASED),
                    timestamp=np.uint64(1_100_000),
                ),
            ]
        )
    )

    commands, timestamps = state.sample(
        timeline.next_window(5, input_times_s=input_times_s)
    )

    assert [command.steer for command in commands] == [1.0, 1.0, 1.0, 0.0, 0.0]
    assert timestamps == (1_000_000, None, None, 1_100_000, None)


def test_completed_tap_behind_model_clock_gets_one_physics_frame() -> None:
    state = DriverInput()
    timeline = RealtimeInputTimeline(samples_per_second=30.0)
    state.sample(timeline.next_window(8))
    input_times_s = state.apply(
        UserInputEvents(
            [
                replace(
                    _key("d", KeyboardInputState.PRESSED),
                    timestamp=np.uint64(50_000),
                ),
                replace(
                    _key("d", KeyboardInputState.RELEASED),
                    timestamp=np.uint64(150_000),
                ),
            ]
        )
    )

    commands, timestamps = state.sample(
        timeline.next_window(8, input_times_s=input_times_s)
    )
    following, _ = state.sample(timeline.next_window(1))

    assert [command.steer for command in commands] == [-1.0] + [0.0] * 7
    assert timestamps == (50_000, 150_000, None, None, None, None, None, None)
    assert following == (DriverCommand(),)


def test_stale_release_of_sampled_command_is_not_replayed() -> None:
    state = DriverInput()
    timeline = RealtimeInputTimeline(samples_per_second=30.0)
    pressed_at_s = state.apply(
        UserInputEvents(
            [
                replace(
                    _key("d", KeyboardInputState.PRESSED),
                    timestamp=np.uint64(0),
                )
            ]
        )
    )
    held, _ = state.sample(timeline.next_window(8, input_times_s=pressed_at_s))
    released_at_s = state.apply(
        UserInputEvents(
            [
                replace(
                    _key("d", KeyboardInputState.RELEASED),
                    timestamp=np.uint64(150_000),
                )
            ]
        )
    )

    released, timestamps = state.sample(
        timeline.next_window(8, input_times_s=released_at_s)
    )

    assert all(command.steer == -1.0 for command in held)
    assert released == (DriverCommand(),) * 8
    assert timestamps == (150_000, None, None, None, None, None, None, None)


def test_focus_loss_releases_sampled_keyboard_input() -> None:
    state = DriverInput()
    timeline = RealtimeInputTimeline(samples_per_second=30.0)
    pressed_at_s = state.apply(
        UserInputEvents(
            [
                replace(
                    _key("w", KeyboardInputState.PRESSED),
                    timestamp=np.uint64(1_000_000),
                )
            ]
        )
    )
    pressed, _ = state.sample(timeline.next_window(1, input_times_s=pressed_at_s))
    released_at_s = state.apply(
        UserInputEvents(
            [
                FocusUserInputEvent(
                    timestamp=np.uint64(9_000_000),
                    focused=False,
                )
            ]
        )
    )
    released, timestamps = state.sample(
        timeline.next_window(1, input_times_s=released_at_s)
    )

    assert pressed[0].throttle == 1.0
    assert released[0] == DriverCommand()
    assert timestamps == (9_000_000,)
