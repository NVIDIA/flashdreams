# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-neutral keyboard and mouse action snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field

from flashdreams.runtime_v2.user_input_event import (
    FocusUserInputEvent,
    KeyboardInputState,
    KeyboardUserInputEvent,
    MouseUserInputEvent,
    ResetUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents


@dataclass(frozen=True, kw_only=True, slots=True)
class ActionSnapshot:
    """One model-step snapshot of persistent and transient user actions."""

    keys: frozenset[str] = field(default_factory=frozenset)
    """Case-normalized keyboard keys held at the end of the event batch."""

    mouse_buttons: frozenset[int] = field(default_factory=frozenset)
    """Mouse-button indices held at the end of the event batch."""

    mouse_dx: float = 0.0
    """Accumulated horizontal pointer displacement in normalized coordinates."""

    mouse_dy: float = 0.0
    """Accumulated vertical pointer displacement in normalized coordinates."""

    wheel_x: float = 0.0
    """Accumulated horizontal wheel displacement for this model step."""

    wheel_y: float = 0.0
    """Accumulated vertical wheel displacement for this model step."""


class ActionEventAccumulator:
    """Accumulate v2 keyboard and mouse events into per-step action snapshots."""

    def __init__(self) -> None:
        self._held_keys: set[str] = set()
        self._held_mouse_buttons: set[int] = set()
        self._pointer_position: tuple[float, float] | None = None

    def consume(self, events: UserInputEvents) -> ActionSnapshot:
        """Apply one ordered event batch and return its action snapshot."""
        mouse_dx = 0.0
        mouse_dy = 0.0
        wheel_x = 0.0
        wheel_y = 0.0

        for event in events.get_events():
            if isinstance(event, ResetUserInputEvent):
                self.reset()
                mouse_dx = mouse_dy = wheel_x = wheel_y = 0.0
            elif isinstance(event, FocusUserInputEvent):
                if not event.focused:
                    self.reset()
                    mouse_dx = mouse_dy = wheel_x = wheel_y = 0.0
            elif isinstance(event, KeyboardUserInputEvent):
                key = normalize_key(event.key)
                if event.state is KeyboardInputState.PRESSED:
                    self._held_keys.add(key)
                else:
                    self._held_keys.discard(key)
            elif isinstance(event, MouseUserInputEvent):
                if event.action == "move":
                    if self._pointer_position is not None:
                        previous_x, previous_y = self._pointer_position
                        mouse_dx += event.x - previous_x
                        mouse_dy += event.y - previous_y
                    self._pointer_position = (event.x, event.y)
                elif event.action == "button":
                    self._pointer_position = (event.x, event.y)
                    if event.pressed:
                        self._held_mouse_buttons.add(event.button)
                    else:
                        self._held_mouse_buttons.discard(event.button)
                elif event.action == "wheel":
                    self._pointer_position = (event.x, event.y)
                    wheel_x += event.wheel_x
                    wheel_y += event.wheel_y

        return ActionSnapshot(
            keys=frozenset(self._held_keys),
            mouse_buttons=frozenset(self._held_mouse_buttons),
            mouse_dx=mouse_dx,
            mouse_dy=mouse_dy,
            wheel_x=wheel_x,
            wheel_y=wheel_y,
        )

    def reset(self) -> None:
        """Clear held controls and the absolute-pointer origin."""
        self._held_keys.clear()
        self._held_mouse_buttons.clear()
        self._pointer_position = None


def normalize_key(key: str) -> str:
    """Return a case-insensitive key identifier while preserving key names."""
    return (
        key.upper()
        if len(key) == 1
        else key.replace("_", "").replace("-", "").replace(" ", "").upper()
    )


__all__ = [
    "ActionEventAccumulator",
    "ActionSnapshot",
    "normalize_key",
]
