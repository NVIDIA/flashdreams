# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for focus ordering across WebRTC input channels."""

import json
from contextlib import contextmanager

import pytest

from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.user_input_event import FocusUserInputEvent
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu


@contextmanager
def _window():
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


def test_focus_snapshot_precedes_input_and_stale_packets_are_ignored():
    with _window() as window:

        def send(payload):
            window.server._buffer_browser_message(json.dumps(payload))

        send(
            {
                "type": "focus",
                "focused": False,
                "focus_context": {"generation": 1, "focused": False},
            }
        )
        window.get_user_input_events()
        press = {
            "type": "mouse",
            "action": "button",
            "x": 0.5,
            "y": 0.5,
            "button": 0,
            "pressed": True,
        }
        send({**press, "focus_context": {"generation": 2, "focused": True}})
        send(
            {
                "type": "focus",
                "focused": True,
                "focus_context": {"generation": 2, "focused": True},
            }
        )
        send(
            {
                "type": "focus",
                "focused": False,
                "focus_context": {"generation": 1, "focused": False},
            }
        )
        events = window.get_user_input_events().get_events()
        assert [event.get_type_name() for event in events] == ["focus", "mouse"]
        assert events[0].focused is True
        assert events[1].pressed is True


@pytest.mark.parametrize(
    "context",
    [
        None,
        [],
        {"generation": True, "focused": True},
        {"generation": -1, "focused": True},
        {"generation": 1.5, "focused": True},
        {"generation": 1, "focused": "true"},
    ],
)
def test_invalid_focus_context_does_not_publish_input(context):
    with _window() as window:
        with pytest.raises(ValueError):
            window.server._buffer_browser_message(
                json.dumps(
                    {
                        "type": "keyboard",
                        "key": "w",
                        "pressed": True,
                        "focus_context": context,
                    }
                )
            )
        assert window.get_user_input_events().get_events() == []


def test_focus_context_must_agree_with_its_generation_and_focus_event():
    with _window() as window:

        def send(payload):
            window.server._buffer_browser_message(json.dumps(payload))

        send(
            {
                "type": "focus",
                "focused": True,
                "focus_context": {"generation": 1, "focused": True},
            }
        )
        window.get_user_input_events()
        for payload in [
            {
                "type": "focus",
                "focused": False,
                "focus_context": {"generation": 1, "focused": False},
            },
            {
                "type": "focus",
                "focused": False,
                "focus_context": {"generation": 2, "focused": True},
            },
        ]:
            with pytest.raises(ValueError):
                send(payload)
        assert window.get_user_input_events().get_events() == []


def test_focus_state_survives_a_session_handoff_and_legacy_input_still_works():
    with _window() as window:

        def send(payload):
            window.server._buffer_browser_message(json.dumps(payload))

        send(
            {
                "type": "focus",
                "focused": False,
                "focus_context": {"generation": 1, "focused": False},
            }
        )
        window.open(
            SessionDesc(
                output_layout=VideoTensorLayout.tchw, video_width=4, video_height=4
            )
        )
        [focus] = window.get_user_input_events().get_events()
        assert isinstance(focus, FocusUserInputEvent)
        assert focus.focused is False
        send({"type": "keyboard", "key": "w", "pressed": True})
        [key] = window.get_user_input_events().get_events()
        assert key.key == "w"
