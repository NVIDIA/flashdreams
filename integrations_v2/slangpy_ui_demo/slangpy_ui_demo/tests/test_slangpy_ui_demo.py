# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU smoke tests for the v2 SlangPy UI demos."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from slangpy_ui_demo.model_output_app import (
    ModelOutputSlangPyUIThread,
    ModelOutputSession,
    ModelOutputThread,
)
from slangpy_ui_demo.text_input_app import TextInputSlangPyUIThread, TextInputState
from numpy import uint64

from flashdreams.runtime_v2._slangpy_ui_renderer import _route_input_events
from flashdreams.runtime_v2.presentation_manager import PresentationManager
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.user_input_event import (
    KeyboardInputState,
    KeyboardUserInputEventData,
    UserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

pytestmark = pytest.mark.ci_cpu


def test_text_input_updates_ui_owned_state() -> None:
    state = TextInputState()
    thread = TextInputSlangPyUIThread(
        state=state,
        frequency=60,
        output_layout=SessionDesc().output_layout,
        presentation_manager=PresentationManager(),
        renderer=Mock(),
    )
    ui = SimpleNamespace(
        screen=object(),
        Window=Mock(return_value=object()),
        Text=Mock(side_effect=(object(), SimpleNamespace(text=""))),
        InputText=Mock(return_value=SimpleNamespace(value="")),
    )

    thread.draw_ui(ui, 0, UserInputEvents([]))
    callback = ui.InputText.call_args.args[3]
    callback("hello world")

    assert state.text == "hello world"
    assert state.value_widget is not None
    assert state.value_widget.text == "Value: hello world"


def test_slangpy_ui_routes_pressed_and_released_key_edges() -> None:
    ui_context = Mock()
    slangpy = SimpleNamespace(
        KeyboardEvent=lambda: SimpleNamespace(),
        KeyboardEventType=SimpleNamespace(
            key_press="press",
            key_release="release",
            input="input",
        ),
        KeyCode=SimpleNamespace(left="left"),
        KeyModifierFlags=SimpleNamespace(none="none"),
    )
    events = UserInputEvents(
        [
            UserInputEvent(
                timestamp=uint64(index),
                event_data=KeyboardUserInputEventData(key="ArrowLeft", state=state),
            )
            for index, state in enumerate(
                (KeyboardInputState.PRESSED, KeyboardInputState.RELEASED)
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

    routed = [call.args[0] for call in ui_context.handle_keyboard_event.call_args_list]
    assert [(event.type, event.key) for event in routed] == [
        ("press", "left"),
        ("release", "left"),
    ]


def test_model_output_emits_repeating_selectable_fade_channels() -> None:
    session = ModelOutputSession(
        SessionDesc(video_width=4, video_height=3), device="cpu"
    )
    session.init()
    model_thread = session.model_thread
    ui_thread = session.ui_thread
    assert isinstance(model_thread, ModelOutputThread)
    assert isinstance(ui_thread, ModelOutputSlangPyUIThread)

    chunk = model_thread.step(0, UserInputEvents([]))
    repeated = model_thread.step(1, UserInputEvents([]))
    assert isinstance(chunk, list) and isinstance(repeated, list)
    assert len(chunk) == 3
    expected = torch.linspace(255, 0, 60).round().to(torch.uint8)
    for index, (result, again) in enumerate(zip(chunk, repeated, strict=True)):
        pixels = ((result.output[:, :3] + 1.0) * 127.5).round().to(torch.uint8)
        assert result.output.shape == (60, 4, 3, 4)
        assert torch.equal(pixels[:, index, 0, 0], expected)
        assert result.output[0, 3, 0, 0] == (1.0 if index == 0 else 0.5)
        assert torch.equal(result.output, again.output)

    session._presentation_manager.publish(0, chunk)
    assert session._presentation_manager.advance(0)[0]
    frame = ui_thread.presented_model_frame(1)
    assert frame is not None
    assert frame.data_ptr() == chunk[1].output[0].data_ptr()
