# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for browser message decoding and input validation."""

import json
import sys
from contextlib import contextmanager
from dataclasses import asdict
from unittest.mock import Mock

import numpy as np
import pytest

from flashdreams.runtime_v2.serving.browser_input import (
    decode_browser_message,
    link_gesture,
    parse_browser_event,
    parse_focus_context,
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
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

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


def _gesture(payload: dict[str, object]):
    return link_gesture(payload, parse_browser_event(payload, timestamp=np.uint64(1)))


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            {"type": "mouse", "action": "button", "x": 3, "y": 4, "pressed": True},
            (("mouse", 0), "start", 3.0, 4.0, True),
        ),
        (
            {"type": "mouse", "action": "button", "x": 3, "y": 4, "in_content": False},
            (("mouse", 0), "end", 3.0, 4.0, False),
        ),
        (
            {"type": "mouse", "action": "button", "x": 3, "y": 4, "cancelled": True},
            (("mouse", 0), "cancel", 3.0, 4.0, True),
        ),
        (
            {
                "type": "touch",
                "action": "start",
                "touch_id": 7,
                "x": 0.1,
                "y": 0.2,
                "primary": True,
            },
            (("touch", 7), "start", 0.1, 0.2, True),
        ),
        (
            {"type": "touch", "action": "move", "touch_id": 7, "x": 0.1, "y": 0.2},
            (("touch", 7), "move", 0.1, 0.2, True),
        ),
    ],
)
def test_link_gesture_describes_primary_pointer(
    payload: dict[str, object], expected: tuple[object, ...]
) -> None:
    assert tuple(_gesture(payload) or ()) == expected


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "mouse", "action": "button", "x": 0, "y": 0, "button": 2},
        {"type": "mouse", "action": "move", "x": 0, "y": 0},
        {"type": "touch", "action": "start", "touch_id": 1, "x": 0, "y": 0},
        {"type": "keyboard", "key": "w", "pressed": True},
        {"type": "focus", "focused": False},
    ],
)
def test_link_gesture_ignores_other_events(payload: dict[str, object]) -> None:
    assert _gesture(payload) is None


@pytest.mark.parametrize("value", [None, 1, "false", [], {}])
@pytest.mark.parametrize(
    "payload",
    [
        {"type": "mouse", "action": "button", "x": 0, "y": 0, "in_content": None},
        {
            "type": "mouse",
            "action": "button",
            "x": 0,
            "y": 0,
            "button": 2,
            "cancelled": None,
        },
        {
            "type": "touch",
            "action": "end",
            "touch_id": 1,
            "x": 0,
            "y": 0,
            "in_content": None,
        },
    ],
)
def test_link_gesture_flags_require_booleans(
    payload: dict[str, object], value: object
) -> None:
    flag = next(key for key, item in payload.items() if item is None)
    with pytest.raises(ValueError, match="boolean"):
        _gesture({**payload, flag: value})


def test_focus_context_is_optional_and_parsed() -> None:
    assert parse_focus_context({"type": "keyboard"}) is None
    context = {"generation": 3, "focused": True}
    assert parse_focus_context({"focus_context": context}) == (3, True)


@pytest.mark.parametrize(
    "context",
    [
        None,
        [],
        {},
        {"generation": 1},
        {"generation": 1, "focused": 1},
        {"generation": -1, "focused": True},
        {"generation": True, "focused": True},
        {"generation": "1", "focused": True},
    ],
)
def test_invalid_focus_context(context: object) -> None:
    with pytest.raises(ValueError):
        parse_focus_context({"focus_context": context})


@contextmanager
def _open_window():
    pytest.importorskip("aiortc")
    from flashdreams.runtime_v2.webrtc_client_window import WebRTCClientWindow

    window = WebRTCClientWindow()
    window.open(
        SessionDesc(output_layout=VideoTensorLayout.tchw, video_width=4, video_height=4)
    )
    try:
        yield window
    finally:
        window.close()


def test_server_checks_json_before_ignoring_events_without_a_session() -> None:
    pytest.importorskip("aiortc")
    from flashdreams.runtime_v2.serving import webrtc_server

    server = webrtc_server.WebRTCServer()
    callback = Mock()
    server.register_input_callback(callback)
    for message in (b"{}", "{", "[]"):
        with pytest.raises(ValueError):
            server._buffer_browser_message(message)
    server._buffer_browser_message('{"type": "mouse", "action": []}')
    server._buffer_browser_message('{"type": "close"}')
    callback.assert_not_called()


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            {"type": "file_selector_result", "id": []},
            "File selector result requires a non-empty id.",
        ),
        (
            {"type": "file_selector_result", "id": "unused", "status": []},
            "File selector result status is invalid.",
        ),
        (
            {"type": "file_selector_result", "id": "unused", "status": "ok"},
            "File selector result must not set status ok; send files "
            "with POST /api/files.",
        ),
        (
            {"type": "mouse", "action": []},
            "Mouse event action must be 'move', 'button', or 'wheel'.",
        ),
        (
            {"type": "mouse", "action": "move", "x": 10**400, "y": 0},
            "Mouse x must be finite.",
        ),
        (
            '{"type":"close","extra":' + "[" * 20_000 + "0" + "]" * 20_000 + "}",
            "Browser event nesting is too deep.",
        ),
    ],
)
def test_invalid_browser_events_raise_and_allow_later_input(
    message: object, expected: str
) -> None:
    with _open_window() as window:
        payload = message if isinstance(message, str) else json.dumps(message)
        with pytest.raises(ValueError) as error:
            window.server._buffer_browser_message(payload)
        assert str(error.value) == expected
        assert window.get_user_input_events().get_events() == []

        window.server._buffer_browser_message(
            json.dumps({"type": "keyboard", "key": "w", "pressed": True})
        )
        (event,) = window.get_user_input_events().get_events()
        assert isinstance(event, KeyboardUserInputEvent)
        assert (event.key, event.state) == ("w", KeyboardInputState.PRESSED)
