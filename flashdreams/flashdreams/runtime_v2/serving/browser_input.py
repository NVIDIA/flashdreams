# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Browser message validation and v2 input event conversion."""

from __future__ import annotations

import json
import math
from typing import Literal, cast

import numpy as np

from flashdreams.runtime_v2.user_input_event import (
    CloseUserInputEvent,
    FocusUserInputEvent,
    GamepadUserInputEvent,
    GameWheelUserInputEvent,
    KeyboardInputState,
    KeyboardUserInputEvent,
    MouseUserInputEvent,
    ResetUserInputEvent,
    TouchUserInputEvent,
    UserInputEvent,
    XRControllerUserInputEvent,
)


def decode_browser_message(raw_message: object) -> dict[str, object]:
    """Decode a browser data-channel message.

    Raises:
        ValueError: The message is not a JSON string containing an object,
            or its nesting exceeds the decoder's recursion limit.
    """
    if not isinstance(raw_message, str):
        raise ValueError("Browser event must be a JSON string.")
    try:
        payload = json.loads(raw_message)
    except json.JSONDecodeError as error:
        raise ValueError("Browser event must contain valid JSON.") from error
    except RecursionError as error:
        raise ValueError("Browser event nesting is too deep.") from error
    if not isinstance(payload, dict):
        raise ValueError("Browser event must be a JSON object.")

    return payload


def parse_browser_event(
    payload: dict[str, object], *, timestamp: np.uint64
) -> UserInputEvent:
    """Convert a decoded browser message into a validated input event.

    Args:
        payload: Object returned by ``decode_browser_message``.
        timestamp: Server-assigned microseconds since the session began.

    Raises:
        ValueError: The event type or its fields are invalid.
    """
    event_type = payload.get("type")
    if event_type == "keyboard":
        key = payload.get("key")
        pressed = payload.get("pressed")
        if not isinstance(key, str) or not key:
            raise ValueError("Keyboard event requires a non-empty key.")
        if not isinstance(pressed, bool):
            raise ValueError("Keyboard event requires a boolean pressed value.")
        event = KeyboardUserInputEvent(
            timestamp=timestamp,
            key=key,
            state=(
                KeyboardInputState.PRESSED if pressed else KeyboardInputState.RELEASED
            ),
        )
    elif event_type == "mouse":
        action = payload.get("action")
        if not isinstance(action, str) or action not in {"move", "button", "wheel"}:
            raise ValueError("Mouse event action must be 'move', 'button', or 'wheel'.")
        x = _finite_number(payload.get("x"), label="Mouse x")
        y = _finite_number(payload.get("y"), label="Mouse y")
        button = payload.get("button", 0)
        pressed = payload.get("pressed", False)
        wheel_x = _finite_number(payload.get("wheel_x", 0.0), label="wheel_x")
        wheel_y = _finite_number(payload.get("wheel_y", 0.0), label="wheel_y")
        if isinstance(button, bool) or not isinstance(button, int) or button < 0:
            raise ValueError("Mouse button must be a non-negative integer.")
        if not isinstance(pressed, bool):
            raise ValueError("Mouse pressed must be a boolean.")
        event = MouseUserInputEvent(
            timestamp=timestamp,
            action=cast(Literal["move", "button", "wheel"], action),
            x=x,
            y=y,
            button=button,
            pressed=pressed,
            wheel_x=wheel_x,
            wheel_y=wheel_y,
        )
    elif event_type == "focus":
        focused = payload.get("focused")
        if not isinstance(focused, bool):
            raise ValueError("Focus event requires a boolean focused value.")
        event = FocusUserInputEvent(
            timestamp=timestamp,
            focused=focused,
        )
    elif event_type == "touch":
        action = payload.get("action")
        if not isinstance(action, str) or action not in {
            "start",
            "move",
            "end",
            "cancel",
        }:
            raise ValueError(
                "Touch event action must be 'start', 'move', 'end', or 'cancel'."
            )
        primary = payload.get("primary", False)
        if not isinstance(primary, bool):
            raise ValueError("Touch primary must be a boolean.")
        event = TouchUserInputEvent(
            timestamp=timestamp,
            action=cast(Literal["start", "move", "end", "cancel"], action),
            touch_id=_nonnegative_int(payload.get("touch_id"), label="touch_id"),
            x=_normalized_coordinate(payload.get("x"), label="Touch x"),
            y=_normalized_coordinate(payload.get("y"), label="Touch y"),
            pressure=_unit_number(payload.get("pressure", 0.0), label="Touch pressure"),
            primary=primary,
        )
    elif event_type == "gamepad":
        buttons = _number_tuple(payload.get("buttons", ()), label="buttons")
        pressed = _bool_tuple(payload.get("pressed", ()), label="pressed")
        if len(buttons) != len(pressed):
            raise ValueError("Gamepad buttons and pressed must have equal length.")
        event = GamepadUserInputEvent(
            timestamp=timestamp,
            action=_controller_action(payload),
            index=_nonnegative_int(payload.get("index", 0), label="index"),
            controller_id=_string(
                payload.get("controller_id", payload.get("id", "")),
                label="controller_id",
            ),
            mapping=_string(payload.get("mapping", ""), label="mapping"),
            axes=_number_tuple(payload.get("axes", ()), label="axes"),
            buttons=buttons,
            pressed=pressed,
        )
    elif event_type == "game_wheel":
        event = GameWheelUserInputEvent(
            timestamp=timestamp,
            action=_controller_action(payload),
            index=_nonnegative_int(payload.get("index", 0), label="index"),
            controller_id=_string(
                payload.get("controller_id", payload.get("id", "")),
                label="controller_id",
            ),
            steering=_bounded_number(
                payload.get("steering", 0.0),
                label="steering",
                low=-1.0,
                high=1.0,
            ),
            throttle=_unit_number(payload.get("throttle", 0.0), label="throttle"),
            brake=_unit_number(payload.get("brake", 0.0), label="brake"),
            clutch=_unit_number(payload.get("clutch", 0.0), label="clutch"),
            buttons=_bool_tuple(payload.get("buttons", ()), label="buttons"),
        )
    elif event_type == "xr_controller":
        handedness = payload.get("handedness", "none")
        if not isinstance(handedness, str) or handedness not in {
            "left",
            "right",
            "none",
        }:
            raise ValueError("XR handedness must be 'left', 'right', or 'none'.")
        buttons = _number_tuple(payload.get("buttons", ()), label="buttons")
        pressed = _bool_tuple(payload.get("pressed", ()), label="pressed")
        if len(buttons) != len(pressed):
            raise ValueError("XR buttons and pressed must have equal length.")
        event = XRControllerUserInputEvent(
            timestamp=timestamp,
            action=_controller_action(payload),
            handedness=cast(Literal["left", "right", "none"], handedness),
            controller_id=_string(
                payload.get("controller_id", payload.get("id", "")),
                label="controller_id",
            ),
            axes=_number_tuple(payload.get("axes", ()), label="axes"),
            buttons=buttons,
            pressed=pressed,
            position=cast(
                tuple[float, float, float] | None,
                _fixed_number_tuple(
                    payload.get("position"), label="position", length=3
                ),
            ),
            orientation=cast(
                tuple[float, float, float, float] | None,
                _fixed_number_tuple(
                    payload.get("orientation"), label="orientation", length=4
                ),
            ),
        )
    elif event_type == "reset":
        event = ResetUserInputEvent(timestamp=timestamp)
    elif event_type == "close":
        event = CloseUserInputEvent(timestamp=timestamp)
    else:
        raise ValueError("Unsupported browser event type.")
    return event


def _finite_number(value: object, *, label: str) -> float:
    """Return a finite browser-input number."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be numeric.")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{label} must be finite.") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite.")
    return result


def _normalized_coordinate(value: object, *, label: str) -> float:
    """Return a normalized browser pointer coordinate."""
    result = _finite_number(value, label=label)
    if result < 0.0 or result > 1.0:
        raise ValueError(f"{label} must be between 0 and 1.")
    return result


def _bounded_number(value: object, *, label: str, low: float, high: float) -> float:
    """Return a finite browser-input number within an inclusive range."""
    result = _finite_number(value, label=label)
    if result < low or result > high:
        raise ValueError(f"{label} must be between {low} and {high}.")
    return result


def _unit_number(value: object, *, label: str) -> float:
    """Return a browser-input number in ``[0, 1]``."""
    return _bounded_number(value, label=label, low=0.0, high=1.0)


def _nonnegative_int(value: object, *, label: str) -> int:
    """Return a non-negative browser-input integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer.")
    return value


def _string(value: object, *, label: str) -> str:
    """Return a browser-input string."""
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    return value


def _number_tuple(value: object, *, label: str) -> tuple[float, ...]:
    """Return a tuple of finite browser-input numbers."""
    if not isinstance(value, list | tuple):
        raise ValueError(f"{label} must be an array.")
    return tuple(
        _finite_number(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    )


def _bool_tuple(value: object, *, label: str) -> tuple[bool, ...]:
    """Return a tuple of browser-input booleans."""
    if not isinstance(value, list | tuple) or not all(
        isinstance(item, bool) for item in value
    ):
        raise ValueError(f"{label} must be a boolean array.")
    return cast(tuple[bool, ...], tuple(value))


def _fixed_number_tuple(
    value: object, *, label: str, length: int
) -> tuple[float, ...] | None:
    """Return an optional fixed-length tuple of finite numbers."""
    if value is None:
        return None
    result = _number_tuple(value, label=label)
    if len(result) != length:
        raise ValueError(f"{label} must contain exactly {length} values.")
    return result


def _controller_action(
    payload: dict[str, object],
) -> Literal["connected", "disconnected", "state"]:
    """Return a validated controller lifecycle action."""
    action = payload.get("action", "state")
    if not isinstance(action, str) or action not in {
        "connected",
        "disconnected",
        "state",
    }:
        raise ValueError(
            "Controller action must be 'connected', 'disconnected', or 'state'."
        )
    return cast(Literal["connected", "disconnected", "state"], action)
