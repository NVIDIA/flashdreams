# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for interactive text-to-video prompt controls."""

import queue
import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from numpy import uint64
from t2v.ui import T2VImGuiUILoop, T2VUIState

from flashdreams.runtime_v2.presentation_manager import PresentationManager
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.user_input_event import (
    CloseUserInputEvent,
    NumeralKeypadUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents
pytestmark = pytest.mark.ci_cpu


def _loop(state: T2VUIState) -> T2VImGuiUILoop:
    session_desc = SessionDesc(metadata={"existing": "value"})
    loop = T2VImGuiUILoop(renderer=Mock())
    loop.register_session_loop_objects(
        state=state,
        frequency=60,
        shutdown_event=threading.Event(),
        failure_queue=queue.Queue(),
    )
    loop.register_session_ui_loop_objects(
        session_desc=session_desc,
        presentation_manager=PresentationManager(),
    )
    return loop


def _imgui(*, prompt: str, submit: bool) -> SimpleNamespace:
    return SimpleNamespace(
        ImVec2=lambda x, y: (x, y),
        Cond_=SimpleNamespace(once="once"),
        set_next_window_pos=Mock(),
        set_next_window_size=Mock(),
        begin=Mock(),
        end=Mock(),
        text=Mock(),
        input_text=Mock(return_value=(True, prompt)),
        button=Mock(return_value=submit),
    )


def test_new_session_button_submits_the_trimmed_prompt() -> None:
    state = T2VUIState(prompt="old prompt")
    loop = _loop(state)
    imgui = _imgui(prompt="  a cat surfing  ", submit=True)

    loop.step_ui(imgui, 0, UserInputEvents([]))
    run = loop._begin_run(UserInputEvents([]), generation=0)

    assert state.prompt == "a cat surfing"
    assert state.message == "Starting new session…"
    assert run.new_session_request == replace(
        loop.session_desc,
        metadata={"existing": "value", "prompt": "a cat surfing"},
    )
    assert (
        loop._begin_run(UserInputEvents([]), generation=0).new_session_request is None
    )
    imgui.text.assert_any_call(state.message)
    imgui.end.assert_called_once_with()


def test_new_session_button_rejects_an_empty_prompt() -> None:
    state = T2VUIState()
    loop = _loop(state)
    imgui = _imgui(prompt="   ", submit=True)

    loop.step_ui(imgui, 0, UserInputEvents([]))
    run = loop._begin_run(UserInputEvents([]), generation=0)

    assert run.new_session_request is None
    assert state.message == "Enter a prompt before starting a session."
    imgui.text.assert_any_call(state.message)


def test_begin_run_returns_close_without_setting_the_shutdown_event() -> None:
    loop = _loop(T2VUIState())

    run = loop._begin_run(
        UserInputEvents([CloseUserInputEvent(timestamp=uint64(0))]),
        generation=0,
    )

    assert run.stop_requested
    assert not loop._shutdown_event.is_set()


def test_begin_run_keeps_input_until_the_caller_runs_the_step() -> None:
    loop = _loop(T2VUIState())
    event = NumeralKeypadUserInputEvent(timestamp=uint64(0), value=7)

    first = loop._begin_run(UserInputEvents([event]), generation=0)
    second = loop._begin_run(UserInputEvents([]), generation=0)

    assert first.step_index == second.step_index == 0
    assert loop.user_events.get_events() == [event]
