# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU smoke tests for the v2 ImGui demos."""

from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
import torch
from imgui_demo.model_output_app import (
    ModelOutputApplication,
    ModelOutputImGUIThread,
    ModelOutputSession,
    ModelOutputThread,
)
from imgui_demo.text_input_app import TextInputImGUIThread, TextInputState
from numpy import uint64

from flashdreams.api_v2.thread import BlitModelOutputToScreenThread
from flashdreams.runtime_v2.imgui_thread import _route_input_events
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
    thread = TextInputImGUIThread(
        state=state,
        frequency=60,
        output_layout=SessionDesc().output_layout,
        presentation_manager=PresentationManager(),
        renderer=Mock(),
    )
    imgui = Mock()
    imgui.input_text.return_value = (True, "hello world")

    thread.draw_ui(imgui, 0, UserInputEvents([]))

    assert state.text == "hello world"
    assert not state.request_focus


def test_imgui_routes_pressed_and_released_key_edges() -> None:
    io = Mock()
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
        io=io,
        imgui=SimpleNamespace(Key=SimpleNamespace(left_arrow="left")),
        width=1,
        height=1,
    )

    assert io.add_key_event.call_args_list == [call("left", True), call("left", False)]


def test_model_output_emits_repeating_selectable_fade_channels() -> None:
    session = ModelOutputSession(
        SessionDesc(video_width=4, video_height=3), device="cpu"
    )
    session.init()
    model_thread = session.model_thread
    ui_thread = session.ui_thread
    assert isinstance(model_thread, ModelOutputThread)
    assert isinstance(ui_thread, ModelOutputImGUIThread)

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


def test_model_output_demo_can_omit_ui_registration() -> None:
    application = ModelOutputApplication(device="cpu")
    application.init(["--no-ui"])
    session = application.create_session(SessionDesc(video_width=4, video_height=3))
    assert isinstance(session, ModelOutputSession)
    session.init()

    ui_thread, model_thread = session._take_threads()
    assert isinstance(ui_thread, BlitModelOutputToScreenThread)
    assert ui_thread._presentation_manager is session._presentation_manager
    chunk = model_thread.step(0, UserInputEvents([]))
    assert isinstance(chunk, list)
    session._presentation_manager.publish(0, chunk)
    assert session._presentation_manager.advance(0)[0]
    presented = ui_thread.step(0, UserInputEvents([]))
    assert presented is not None
    expected = None
    for result in chunk:
        expected = session._presentation_manager.composite(expected, result.output[0])
    assert expected is not None
    assert torch.equal(presented.output[0], expected)
