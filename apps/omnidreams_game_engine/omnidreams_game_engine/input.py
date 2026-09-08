# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retained V2 input state for model-thread driving."""

from __future__ import annotations

from dataclasses import dataclass, field

from flashdreams.runtime.keyboard import normalize_key
from flashdreams.runtime_v2.input_timeline import InputWindow
from flashdreams.runtime_v2.user_input_event import (
    FocusUserInputEvent,
    GamepadUserInputEvent,
    GameWheelUserInputEvent,
    KeyboardInputState,
    KeyboardUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from omnidreams_game_engine.types import DriverCommand


@dataclass(frozen=True, slots=True)
class _DriverTransition:
    """One timestamped change to the effective driving command."""

    timestamp_s: float
    timestamp_us: int
    command: DriverCommand


@dataclass(slots=True)
class DriverInput:
    """Driving input state owned and updated by one runtime loop."""

    pressed_keys: set[str] = field(default_factory=set)
    """Normalized keyboard driving keys currently held down."""

    controller_command: DriverCommand | None = None
    """Latest wheel or gamepad command; ``None`` enables keyboard input."""

    _sampled_command: DriverCommand = field(
        default_factory=DriverCommand,
        init=False,
        repr=False,
    )
    _pending_transitions: list[_DriverTransition] = field(
        default_factory=list,
        init=False,
        repr=False,
    )

    def apply(self, events: UserInputEvents) -> tuple[float, ...]:
        """Retain new input and return command-transition times in seconds."""
        input_times_s: list[float] = []
        for event in events.get_events():
            before = self.command()
            if not self._apply_event(event):
                continue
            command = self.command()
            if command == before:
                continue
            timestamp_us = int(event.get_timestamp())
            timestamp_s = timestamp_us / 1_000_000.0
            self._pending_transitions.append(
                _DriverTransition(timestamp_s, timestamp_us, command)
            )
            input_times_s.append(timestamp_s)
        return tuple(input_times_s)

    def sample(
        self,
        window: InputWindow,
    ) -> tuple[tuple[DriverCommand, ...], tuple[int | None, ...]]:
        """Quantize retained transitions to frame starts in ``window``."""
        transitions = sorted(
            self._pending_transitions,
            key=lambda transition: transition.timestamp_s,
        )
        self._pending_transitions.clear()
        command = self._sampled_command
        transition_index = 0
        while (
            transition_index < len(transitions)
            and transitions[transition_index].timestamp_s <= window.start_s
        ):
            transition_index += 1
        stale_transitions = transitions[:transition_index]
        replayed_transition: _DriverTransition | None = None
        if stale_transitions:
            current_command = stale_transitions[-1].command
            if current_command == command:
                replayed_transition = next(
                    (
                        transition
                        for transition in reversed(stale_transitions[:-1])
                        if transition.command != current_command
                    ),
                    None,
                )
            command = current_command
        commands: list[DriverCommand] = []
        transition_timestamps_us: list[int | None] = []
        # ponytail: A completed tap that arrives behind the model clock gets one
        # physics frame. Interruptible generation is the upgrade for true
        # mid-step timing.
        frame_starts_s = (window.start_s, *window.sample_times_s[:-1])
        for frame_index, frame_start_s in enumerate(frame_starts_s):
            if frame_index == 0 and replayed_transition is not None:
                commands.append(replayed_transition.command)
                transition_timestamps_us.append(replayed_transition.timestamp_us)
                continue
            latest_timestamp_us = (
                stale_transitions[-1].timestamp_us
                if stale_transitions
                and frame_index == (1 if replayed_transition is not None else 0)
                else None
            )
            while (
                transition_index < len(transitions)
                and transitions[transition_index].timestamp_s <= frame_start_s
            ):
                transition = transitions[transition_index]
                command = transition.command
                latest_timestamp_us = transition.timestamp_us
                transition_index += 1
            commands.append(command)
            transition_timestamps_us.append(latest_timestamp_us)

        self._pending_transitions.extend(transitions[transition_index:])
        self._sampled_command = command
        return tuple(commands), tuple(transition_timestamps_us)

    def command(self) -> DriverCommand:
        """Return the command represented by the current retained input state."""
        if self.controller_command is not None:
            return self.controller_command
        return _keyboard_command(self.pressed_keys)

    def source(self) -> str:
        """Return the currently active input source."""
        if self.controller_command is not None:
            return "wheel/gamepad"
        return "keyboard" if self.pressed_keys else "idle"

    def reset(self) -> None:
        """Clear retained, sampled, and pending driving input."""
        self.pressed_keys.clear()
        self.controller_command = None
        self._sampled_command = DriverCommand()
        self._pending_transitions.clear()

    def _apply_event(self, event: object) -> bool:
        if isinstance(event, FocusUserInputEvent):
            if event.focused:
                return False
            self.pressed_keys.clear()
            return True
        if isinstance(event, KeyboardUserInputEvent):
            key = _normalize_drive_key(event.key)
            if key is None:
                return False
            if event.state is KeyboardInputState.PRESSED:
                self.pressed_keys.add(key)
            else:
                self.pressed_keys.discard(key)
            return True
        if isinstance(event, GameWheelUserInputEvent):
            self.controller_command = (
                None
                if event.action == "disconnected"
                else DriverCommand(
                    throttle=event.throttle,
                    brake=event.brake,
                    steer=-event.steering,
                    steer_is_direct=True,
                    manual_control=True,
                )
            )
            return True
        if isinstance(event, GamepadUserInputEvent):
            self.controller_command = _gamepad_command(event)
            return True
        return False


def _keyboard_command(pressed_keys: set[str]) -> DriverCommand:
    """Map retained keyboard state to a simulation command."""
    forward = "w" in pressed_keys
    reverse = "s" in pressed_keys
    brake = "space" in pressed_keys
    steer = 0.0
    if "a" in pressed_keys:
        steer += 1.0
    if "d" in pressed_keys:
        steer -= 1.0
    return DriverCommand(
        throttle=1.0 if forward != reverse and not brake else 0.0,
        brake=1.0 if brake else 0.0,
        steer=steer,
        reverse=reverse and not forward,
        manual_control=brake,
    )


def _normalize_drive_key(key: str) -> str | None:
    key = normalize_key(key)
    return key if key in {"w", "a", "s", "d", "space"} else None


def _gamepad_command(event: GamepadUserInputEvent) -> DriverCommand | None:
    if event.action == "disconnected":
        return None
    if event.action != "state":
        return None
    steer = -(event.axes[0] if event.axes else 0.0)
    throttle = event.buttons[7] if len(event.buttons) > 7 else 0.0
    brake = event.buttons[6] if len(event.buttons) > 6 else 0.0
    reverse = (
        event.pressed[5]
        if len(event.pressed) > 5
        else len(event.buttons) > 5 and event.buttons[5] > 0.0
    )
    return DriverCommand(
        throttle=throttle,
        brake=brake,
        steer=steer,
        reverse=reverse,
        steer_is_direct=True,
        manual_control=True,
    )
