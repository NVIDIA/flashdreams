# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Latest-state V2 input handling for model-thread driving."""

from __future__ import annotations

from dataclasses import dataclass, field

from flashdreams.runtime_v2.user_input_event import (
    GamepadUserInputEvent,
    GameWheelUserInputEvent,
    KeyboardInputState,
    KeyboardUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from omnidreams_game_engine.types import DriverCommand


@dataclass(frozen=True, slots=True)
class DriverInputBatch:
    """Latest driving command repeated across one model step."""

    commands: tuple[DriverCommand, ...]
    """Frame-aligned copies of the current driving command."""

    transition_timestamps_us: tuple[int | None, ...]
    """Newest state transition on the first frame; ``None`` otherwise."""

    transition_count: int
    """Number of active driving-state transitions applied from this event batch."""

    coalesced_transition_count: int
    """Earlier transitions collapsed into the latest state for this model step."""

    ignored_event_count: int
    """Relevant input events that did not change the active driving state."""


@dataclass(slots=True)
class DriverInput:
    """Current keyboard, gamepad, or wheel driving state for model chunks.

    Unread edges are applied in order, and the resulting command conditions the
    whole next model chunk.
    """

    pressed_keys: set[str] = field(default_factory=set)
    """Normalized keyboard driving keys currently held down."""

    controller_command: DriverCommand | None = None
    """Latest wheel or gamepad command; ``None`` enables keyboard input."""

    def reduce(
        self,
        events: UserInputEvents,
        *,
        frame_count: int,
    ) -> DriverInputBatch:
        """Apply ``events`` and repeat the latest command for every model frame."""
        if frame_count <= 0:
            raise ValueError("frame_count must be positive")
        transition_count = 0
        ignored_event_count = 0
        latest_transition_timestamp_us: int | None = None
        for event in events.get_events():
            before = self.command()
            if not self._apply_event(event):
                if isinstance(
                    event,
                    (
                        KeyboardUserInputEvent,
                        GamepadUserInputEvent,
                        GameWheelUserInputEvent,
                    ),
                ):
                    ignored_event_count += 1
                continue
            if self.command() == before:
                ignored_event_count += 1
                continue
            transition_count += 1
            latest_transition_timestamp_us = int(event.get_timestamp())

        command = self.command()
        transition_timestamps_us = (latest_transition_timestamp_us,) + (None,) * (
            frame_count - 1
        )
        return DriverInputBatch(
            commands=(command,) * frame_count,
            transition_timestamps_us=transition_timestamps_us,
            transition_count=transition_count,
            coalesced_transition_count=max(0, transition_count - 1),
            ignored_event_count=ignored_event_count,
        )

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
        """Clear every retained input value."""
        self.pressed_keys.clear()
        self.controller_command = None

    def _apply_event(self, event: object) -> bool:
        """Apply one event using controller-over-keyboard precedence."""
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
    forward = bool({"w", "up"} & pressed_keys)
    reverse = bool({"s", "down"} & pressed_keys)
    opposing_directions = forward and reverse
    throttle = 1.0 if forward != reverse else 0.0
    steer = 0.0
    if {"a", "left"} & pressed_keys:
        steer += 1.0
    if {"d", "right"} & pressed_keys:
        steer -= 1.0
    return DriverCommand(
        throttle=throttle,
        brake=1.0 if opposing_directions else 0.0,
        steer=steer,
        stop="space" in pressed_keys,
        reverse=reverse and not forward,
    )


def _normalize_drive_key(key: str) -> str | None:
    key = key.strip().lower()
    aliases = {
        "arrowup": "w",
        "arrowdown": "s",
        "arrowleft": "a",
        "arrowright": "d",
    }
    key = aliases.get(key, key)
    return key if key in {"w", "a", "s", "d", "space"} else None


def _gamepad_command(event: GamepadUserInputEvent) -> DriverCommand | None:
    if event.action == "disconnected":
        return None
    if event.action != "state":
        return None
    steer = -(event.axes[0] if event.axes else 0.0)
    throttle = event.buttons[7] if len(event.buttons) > 7 else 0.0
    brake = event.buttons[6] if len(event.buttons) > 6 else 0.0
    return DriverCommand(
        throttle=throttle,
        brake=brake,
        steer=steer,
        steer_is_direct=True,
        manual_control=True,
    )
