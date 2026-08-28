# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the v2 SlangPy UI renderer."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from numpy import uint64

from flashdreams.runtime_v2.slangpy_ui_renderer import (
    _rgba8_to_compositing_frame,
    _route_input_events,
)
from flashdreams.runtime_v2.user_input_event import (
    KeyboardInputState,
    KeyboardUserInputEvent,
    MouseUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

pytestmark = pytest.mark.ci_cpu


def test_rgba_conversion_normalizes_directly_into_contiguous_chw() -> None:
    frame = torch.tensor(
        [[[0, 127, 255, 0], [255, 0, 127, 255]]],
        dtype=torch.uint8,
    )

    converted = _rgba8_to_compositing_frame(frame)

    expected = torch.tensor(
        [
            [[-1.0, 1.0]],
            [[127.0 * 2.0 / 255.0 - 1.0, -1.0]],
            [[1.0, 127.0 * 2.0 / 255.0 - 1.0]],
            [[0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    assert converted.is_contiguous()
    torch.testing.assert_close(converted, expected)


def test_space_key_routes_a_text_input_codepoint() -> None:
    ui_context = Mock()
    slangpy = SimpleNamespace(
        KeyboardEvent=lambda: SimpleNamespace(),
        KeyboardEventType=SimpleNamespace(
            key_press="key_press",
            key_release="key_release",
            input="input",
        ),
        KeyCode=SimpleNamespace(space="space"),
        KeyModifierFlags=SimpleNamespace(none="none"),
    )
    events = UserInputEvents(
        [
            KeyboardUserInputEvent(
                timestamp=uint64(0),
                key=" ",
                state=KeyboardInputState.PRESSED,
            )
        ]
    )

    _route_input_events(
        events,
        ui_context=ui_context,
        slangpy=slangpy,
        width=1,
        height=1,
    )

    key_event, text_event = [
        call.args[0] for call in ui_context.handle_keyboard_event.call_args_list
    ]
    assert (key_event.type, key_event.key) == ("key_press", "space")
    assert (text_event.type, text_event.codepoint) == ("input", ord(" "))


def test_shifted_character_preserves_key_and_text_input() -> None:
    ui_context = Mock()
    slangpy = SimpleNamespace(
        KeyboardEvent=lambda: SimpleNamespace(),
        KeyboardEventType=SimpleNamespace(
            key_press="key_press",
            key_release="key_release",
            input="input",
        ),
        KeyCode=SimpleNamespace(a="a"),
        KeyModifierFlags=SimpleNamespace(none="none"),
    )
    events = UserInputEvents(
        [
            KeyboardUserInputEvent(
                timestamp=uint64(0),
                key="A",
                state=KeyboardInputState.PRESSED,
            )
        ]
    )

    _route_input_events(
        events,
        ui_context=ui_context,
        slangpy=slangpy,
        width=1,
        height=1,
    )

    key_event, text_event = [
        call.args[0] for call in ui_context.handle_keyboard_event.call_args_list
    ]
    assert (key_event.type, key_event.key) == ("key_press", "a")
    assert (text_event.type, text_event.codepoint) == ("input", ord("A"))


def test_mouse_input_is_routed_through_slangpy_ui_context() -> None:
    ui_context = Mock()
    slangpy = SimpleNamespace(
        KeyModifierFlags=SimpleNamespace(none="none"),
        MouseButton=SimpleNamespace(left="left", middle="middle", right="right"),
        MouseEvent=lambda: SimpleNamespace(),
        MouseEventType=SimpleNamespace(
            button_down="button_down",
            button_up="button_up",
            move="move",
            scroll="scroll",
        ),
    )
    events = UserInputEvents(
        [
            MouseUserInputEvent(
                timestamp=uint64(0),
                action="button",
                x=0.25,
                y=0.75,
                button=0,
                pressed=True,
            )
        ]
    )

    _route_input_events(
        events,
        ui_context=ui_context,
        slangpy=slangpy,
        width=400,
        height=200,
    )

    routed = ui_context.handle_mouse_event.call_args.args[0]
    assert routed.type == "button_down"
    assert routed.button == "left"
    assert routed.pos == (100.0, 150.0)
