# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate v2 keyboard and mouse events into Waypoint controls."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from waypoint import WaypointControl

from flashdreams.runtime_v2.user_input_event import (
    FocusUserInputEvent,
    KeyboardInputState,
    KeyboardUserInputEvent,
    MouseUserInputEvent,
    ResetUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

_NAMED_KEY_CODES = {
    "ARROWDOWN": 0x28,
    "ARROWLEFT": 0x25,
    "ARROWRIGHT": 0x27,
    "ARROWUP": 0x26,
    "CONTROL": 0x11,
    "CTRL": 0x11,
    "ENTER": 0x0D,
    "SHIFT": 0x10,
    "SPACE": 0x20,
    "SPACEBAR": 0x20,
    "TAB": 0x09,
}

# Browser/SlangPy order is left, middle, right. Waypoint follows the Windows
# virtual-key mouse IDs used by the official Biome client.
_MOUSE_BUTTON_CODES = {0: 0x01, 1: 0x04, 2: 0x02}


@dataclass(slots=True)
class WaypointControlEventAdapter:
    """Coalesce one ordered v2 event batch into a Waypoint action.

    Keyboard and mouse-button edges update held state. Mouse coordinates are
    absolute and normalized by the v2 clients, so moves become pixel deltas
    relative to the preceding pointer position. The first move only establishes
    an origin. Motion and wheel input are consumed by one action, while held
    buttons persist until release, reset, or focus loss.

    Args:
        video_width: Width used to convert normalized horizontal motion to pixels.
        video_height: Height used to convert normalized vertical motion to pixels.
        mouse_sensitivity: Finite non-negative multiplier applied after pixel
            conversion.

    Raises:
        ValueError: A dimension or sensitivity is invalid.
    """

    video_width: int
    video_height: int
    mouse_sensitivity: float = 1.0
    _held_buttons: set[int] = field(default_factory=set, init=False)
    _pointer_position: tuple[float, float] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.video_width <= 0 or self.video_height <= 0:
            raise ValueError("video dimensions must be positive")
        if not math.isfinite(self.mouse_sensitivity) or self.mouse_sensitivity < 0:
            raise ValueError("mouse_sensitivity must be finite and non-negative")

    def consume(self, events: UserInputEvents) -> WaypointControl:
        """Apply ``events`` in timestamp order and return one coalesced action."""
        mouse_dx = 0.0
        mouse_dy = 0.0
        wheel = 0.0

        for event in events.get_events():
            if isinstance(event, ResetUserInputEvent):
                self.reset()
                mouse_dx = mouse_dy = wheel = 0.0
            elif isinstance(event, FocusUserInputEvent):
                if not event.focused:
                    self.reset()
                    mouse_dx = mouse_dy = wheel = 0.0
            elif isinstance(event, KeyboardUserInputEvent):
                button = _key_code(event.key)
                if button is None:
                    continue
                if event.state is KeyboardInputState.PRESSED:
                    self._held_buttons.add(button)
                else:
                    self._held_buttons.discard(button)
            elif isinstance(event, MouseUserInputEvent):
                if event.action == "move":
                    if self._pointer_position is not None:
                        previous_x, previous_y = self._pointer_position
                        mouse_dx += (
                            (event.x - previous_x)
                            * self.video_width
                            * self.mouse_sensitivity
                        )
                        mouse_dy += (
                            (event.y - previous_y)
                            * self.video_height
                            * self.mouse_sensitivity
                        )
                    self._pointer_position = (event.x, event.y)
                elif event.action == "button":
                    self._pointer_position = (event.x, event.y)
                    button = _MOUSE_BUTTON_CODES.get(event.button)
                    if button is None:
                        continue
                    if event.pressed:
                        self._held_buttons.add(button)
                    else:
                        self._held_buttons.discard(button)
                elif event.action == "wheel":
                    self._pointer_position = (event.x, event.y)
                    wheel += event.wheel_y

        scroll_wheel = 1 if wheel > 0 else -1 if wheel < 0 else 0
        return WaypointControl(
            buttons=frozenset(self._held_buttons),
            mouse_dx=mouse_dx,
            mouse_dy=mouse_dy,
            scroll_wheel=scroll_wheel,
        )

    def reset(self) -> None:
        """Clear held controls and the absolute-pointer origin."""
        self._held_buttons.clear()
        self._pointer_position = None


def _key_code(key: str) -> int | None:
    if key == " ":
        return 0x20
    if len(key) == 1:
        upper = key.upper()
        if "A" <= upper <= "Z" or "0" <= upper <= "9":
            return ord(upper)
        return None
    normalized = key.replace("_", "").replace("-", "").replace(" ", "").upper()
    return _NAMED_KEY_CODES.get(normalized)


__all__ = ["WaypointControlEventAdapter"]
