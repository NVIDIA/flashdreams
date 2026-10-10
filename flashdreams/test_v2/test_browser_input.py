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

"""CPU checks for browser message decoding and input validation."""

import json
import sys
from dataclasses import asdict

import numpy as np
import pytest

from flashdreams.runtime_v2.serving.browser_input import (
    decode_browser_message,
    parse_browser_event,
)
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

pytestmark = pytest.mark.ci_cpu


@pytest.mark.parametrize(
    ("payload", "event_class", "fields"),
    [
        (
            {"type": "keyboard", "key": "w", "pressed": True},
            KeyboardUserInputEvent,
            {"key": "w", "state": KeyboardInputState.PRESSED},
        ),
        (
            {"type": "keyboard", "key": "w", "pressed": False},
            KeyboardUserInputEvent,
            {"key": "w", "state": KeyboardInputState.RELEASED},
        ),
        (
            {"type": "mouse", "action": "move", "x": 1.25, "y": -0.5},
            MouseUserInputEvent,
            {
                "action": "move",
                "x": 1.25,
                "y": -0.5,
                "button": 0,
                "pressed": False,
                "wheel_x": 0.0,
                "wheel_y": 0.0,
            },
        ),
        ({"type": "focus", "focused": False}, FocusUserInputEvent, {"focused": False}),
        (
            {"type": "touch", "action": "start", "touch_id": 2, "x": 0, "y": 1},
            TouchUserInputEvent,
            {
                "action": "start",
                "touch_id": 2,
                "x": 0.0,
                "y": 1.0,
                "pressure": 0.0,
                "primary": False,
            },
        ),
        (
            {
                "type": "gamepad",
                "id": "fallback",
                "controller_id": "pad",
                "axes": [-1, 1],
                "buttons": [0, 1],
                "pressed": [False, True],
            },
            GamepadUserInputEvent,
            {
                "action": "state",
                "index": 0,
                "controller_id": "pad",
                "mapping": "",
                "axes": (-1.0, 1.0),
                "buttons": (0.0, 1.0),
                "pressed": (False, True),
            },
        ),
        (
            {"type": "game_wheel", "id": "wheel"},
            GameWheelUserInputEvent,
            {
                "action": "state",
                "index": 0,
                "controller_id": "wheel",
                "steering": 0.0,
                "throttle": 0.0,
                "brake": 0.0,
                "clutch": 0.0,
                "buttons": (),
            },
        ),
        (
            {"type": "xr_controller"},
            XRControllerUserInputEvent,
            {
                "action": "state",
                "handedness": "none",
                "controller_id": "",
                "axes": (),
                "buttons": (),
                "pressed": (),
                "position": None,
                "orientation": None,
            },
        ),
        (
            {
                "type": "xr_controller",
                "action": "connected",
                "handedness": "left",
                "position": [1, 2, 3],
                "orientation": [0, 0, 0, 1],
            },
            XRControllerUserInputEvent,
            {
                "action": "connected",
                "handedness": "left",
                "controller_id": "",
                "axes": (),
                "buttons": (),
                "pressed": (),
                "position": (1.0, 2.0, 3.0),
                "orientation": (0.0, 0.0, 0.0, 1.0),
            },
        ),
        ({"type": "reset"}, ResetUserInputEvent, {}),
        ({"type": "close"}, CloseUserInputEvent, {}),
    ],
)
def test_browser_events(
    payload: dict[str, object],
    event_class: type[UserInputEvent],
    fields: dict[str, object],
) -> None:
    timestamp = np.uint64(1234)
    event = parse_browser_event(
        decode_browser_message(json.dumps(payload)), timestamp=timestamp
    )
    assert type(event) is event_class
    assert asdict(event) == {"timestamp": timestamp, **fields}
    assert event.get_timestamp() is timestamp


@pytest.mark.parametrize("raw", [None, b"{}", {}, "{", "[]", "null", "1", '"text"'])
def test_invalid_message(raw: object) -> None:
    with pytest.raises(ValueError, match="Browser event must"):
        decode_browser_message(raw)


def test_deeply_nested_message_reports_validation_error() -> None:
    # CPython's C JSON decoder can have a higher limit than Python recursion.
    depth = max(10_000, sys.getrecursionlimit() + 1)
    raw = '{"extra":' + "[" * depth + "0" + "]" * depth + "}"
    with pytest.raises(
        ValueError, match=r"^Browser event nesting is too deep\.$"
    ) as caught:
        decode_browser_message(raw)
    assert isinstance(caught.value.__cause__, RecursionError)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"type": "unknown"},
        {"type": []},
        {"type": "keyboard", "key": "", "pressed": True},
        {"type": "keyboard", "key": "w", "pressed": 1},
        {"type": "focus", "focused": 1},
        {"type": "mouse", "action": "move", "x": True, "y": 0},
        {"type": "mouse", "action": "move", "x": 0, "y": 0, "button": -1},
        {"type": "touch", "action": "start", "touch_id": True, "x": 0, "y": 0},
        {"type": "touch", "action": "start", "touch_id": 0, "x": -0.1, "y": 0},
        {"type": "gamepad", "buttons": [1], "pressed": []},
        {"type": "gamepad", "axes": "bad"},
        {"type": "gamepad", "pressed": [1]},
        {"type": "gamepad", "controller_id": 3},
        {"type": "game_wheel", "steering": 1.1},
        {"type": "game_wheel", "throttle": -0.1},
        {"type": "xr_controller", "buttons": [1], "pressed": []},
        {"type": "xr_controller", "position": [0, 1]},
        {"type": "xr_controller", "orientation": [0, 0, 1]},
    ],
)
def test_invalid_event_fields(payload: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        parse_browser_event(payload, timestamp=np.uint64(0))


@pytest.mark.parametrize(
    ("event_type", "field"),
    [
        ("mouse", "action"),
        ("touch", "action"),
        ("gamepad", "action"),
        ("game_wheel", "action"),
        ("xr_controller", "action"),
        ("xr_controller", "handedness"),
    ],
)
@pytest.mark.parametrize("value", [[], {}, None, 1, "invalid"])
def test_invalid_enum_fields(event_type: str, field: str, value: object) -> None:
    with pytest.raises(ValueError, match="action|handedness"):
        parse_browser_event({"type": event_type, field: value}, timestamp=np.uint64(0))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 10**400])
@pytest.mark.parametrize("event_type", ["mouse", "gamepad"])
def test_nonfinite_numbers(event_type: str, value: int | float) -> None:
    payload = (
        {"type": event_type, "action": "move", "x": value, "y": 0}
        if event_type == "mouse"
        else {"type": event_type, "axes": [value]}
    )
    with pytest.raises(ValueError, match="must be finite"):
        parse_browser_event(
            decode_browser_message(json.dumps(payload)), timestamp=np.uint64(0)
        )
