# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU smoke tests for the v2 SlangPy UI demos."""

import queue
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from numpy import uint64
from slangpy_ui_demo.invoke_async_app import (
    ColorToggleModelLoop,
    ColorToggleModelState,
    ColorToggleSlangPyUILoop,
    ColorToggleUIState,
)
from slangpy_ui_demo.model_output_app import (
    ModelOutputLoop,
    ModelOutputSession,
    ModelOutputSlangPyUILoop,
)
from slangpy_ui_demo.text_input_app import TextInputSlangPyUILoop, TextInputState

from flashdreams.runtime_v2.presentation_manager import PresentationManager
from flashdreams.runtime_v2.session_desc import BackpressureMode, SessionDesc
from flashdreams.runtime_v2.slangpy_ui_renderer import _route_input_events
from flashdreams.runtime_v2.user_input_event import (
    KeyboardInputState,
    KeyboardUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

pytestmark = pytest.mark.ci_cpu


def test_invoke_async_toggles_model_owned_color_on_w_press() -> None:
    desc = SessionDesc(video_width=4, video_height=3)
    model_state = ColorToggleModelState(session_desc=desc, device="cpu")
    shutdown_event = threading.Event()
    failure_queue: queue.Queue[BaseException] = queue.Queue()
    model_loop = ColorToggleModelLoop()
    model_loop.register_session_loop_objects(
        state=model_state,
        frequency=30,
        shutdown_event=shutdown_event,
        failure_queue=failure_queue,
    )
    ui_loop = ColorToggleSlangPyUILoop(
        renderer=Mock(),
    )
    ui_loop.register_session_loop_objects(
        state=ColorToggleUIState(model_loop=model_loop),
        frequency=60,
        shutdown_event=shutdown_event,
        failure_queue=failure_queue,
    )
    ui_loop.register_session_ui_loop_objects(
        output_layout=desc.output_layout,
        presentation_manager=PresentationManager(),
    )
    ui = SimpleNamespace(
        screen=object(),
        Window=Mock(return_value=object()),
        Text=Mock(return_value=object()),
    )
    w_pressed = UserInputEvents(
        [
            KeyboardUserInputEvent(
                timestamp=uint64(0),
                key="W",
                state=KeyboardInputState.PRESSED,
            )
        ]
    )

    red = model_loop.step(0, UserInputEvents([]))[0].read_output()
    ui_loop.step_ui(ui, 0, w_pressed)
    assert not model_state.blue

    step_index = model_loop._begin_run(UserInputEvents([]), generation=0)
    assert step_index == 0
    blue_results = model_loop.step(step_index, UserInputEvents([]))
    blue = blue_results[0].read_output()
    model_loop._finish_run(blue_results)

    ui_loop.step_ui(ui, 1, w_pressed)
    step_index = model_loop._begin_run(UserInputEvents([]), generation=0)
    assert step_index == 1
    red_again = model_loop.step(step_index, UserInputEvents([]))[0].read_output()

    assert not model_state.blue
    assert torch.equal(red[0, :, 0, 0], torch.tensor([1.0, -1.0, -1.0]))
    assert torch.equal(blue[0, :, 0, 0], torch.tensor([-1.0, -1.0, 1.0]))
    assert torch.equal(red_again, red)


def test_text_input_updates_ui_owned_state() -> None:
    state = TextInputState()
    loop = TextInputSlangPyUILoop(
        renderer=Mock(),
    )
    loop.register_session_loop_objects(
        state=state,
        frequency=60,
        shutdown_event=threading.Event(),
        failure_queue=queue.Queue(),
    )
    loop.register_session_ui_loop_objects(
        output_layout=SessionDesc().output_layout,
        presentation_manager=PresentationManager(),
    )
    ui = SimpleNamespace(
        screen=object(),
        Window=Mock(return_value=object()),
        Text=Mock(side_effect=(object(), SimpleNamespace(text=""))),
        InputText=Mock(return_value=SimpleNamespace(value="")),
    )

    loop.step_ui(ui, 0, UserInputEvents([]))
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
            KeyboardUserInputEvent(
                timestamp=uint64(index),
                key="ArrowLeft",
                state=state,
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
    model_loop = session.model_loop
    ui_loop = session.ui_loop
    assert isinstance(model_loop, ModelOutputLoop)
    assert isinstance(ui_loop, ModelOutputSlangPyUILoop)

    chunk = model_loop.step(0, UserInputEvents([]))
    repeated = model_loop.step(1, UserInputEvents([]))
    assert isinstance(chunk, list) and isinstance(repeated, list)
    assert len(chunk) == 3
    expected = torch.linspace(255, 0, 60).round().to(torch.uint8)
    for index, (result, again) in enumerate(zip(chunk, repeated, strict=True)):
        output = result.read_output()
        pixels = ((output[:, :3] + 1.0) * 127.5).round().to(torch.uint8)
        assert output.shape == (60, 4, 3, 4)
        assert torch.equal(pixels[:, index, 0, 0], expected)
        assert output[0, 3, 0, 0] == (1.0 if index == 0 else 0.5)
        assert torch.equal(output, again.read_output())

    session._presentation_manager.configure(
        backpressure_mode=BackpressureMode.BLOCK,
        stop=threading.Event(),
        put_timeout=0.01,
    )
    session._presentation_manager.publish(0, chunk)
    assert session._presentation_manager.advance(0)[0]
    frame = ui_loop.presented_model_frame(1)
    assert frame is not None
    assert frame.data_ptr() == chunk[1].read_output()[0].data_ptr()
