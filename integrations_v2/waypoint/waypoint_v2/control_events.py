# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate v2 keyboard and mouse events into Waypoint controls."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from flashdreams.runtime_v2.user_input_event import (
    FocusUserInputEventData,
    KeyboardInputState,
    KeyboardUserInputEventData,
    MouseUserInputEventData,
    ResetUserInputEventData,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from waypoint import WaypointControl

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
class ControlEventAdapter:
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
            data = event.get_event_data()
            if isinstance(data, ResetUserInputEventData):
                self.reset()
                mouse_dx = mouse_dy = wheel = 0.0
            elif isinstance(data, FocusUserInputEventData):
                if not data.focused:
                    self.reset()
                    mouse_dx = mouse_dy = wheel = 0.0
            elif isinstance(data, KeyboardUserInputEventData):
                button = _key_code(data.key)
                if button is None:
                    continue
                if data.state is KeyboardInputState.PRESSED:
                    self._held_buttons.add(button)
                else:
                    self._held_buttons.discard(button)
            elif isinstance(data, MouseUserInputEventData):
                if data.action == "move":
                    if self._pointer_position is not None:
                        previous_x, previous_y = self._pointer_position
                        mouse_dx += (
                            (data.x - previous_x)
                            * self.video_width
                            * self.mouse_sensitivity
                        )
                        mouse_dy += (
                            (data.y - previous_y)
                            * self.video_height
                            * self.mouse_sensitivity
                        )
                    self._pointer_position = (data.x, data.y)
                elif data.action == "button":
                    self._pointer_position = (data.x, data.y)
                    button = _MOUSE_BUTTON_CODES.get(data.button)
                    if button is None:
                        continue
                    if data.pressed:
                        self._held_buttons.add(button)
                    else:
                        self._held_buttons.discard(button)
                elif data.action == "wheel":
                    self._pointer_position = (data.x, data.y)
                    wheel += data.wheel_y

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


__all__ = ["ControlEventAdapter"]
