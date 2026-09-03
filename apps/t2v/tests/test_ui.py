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

def test_begin_run_returns_close_without_setting_the_shutdown_event() -> None:
    loop = _loop(T2VUIState())

    run = loop._begin_run(
        UserInputEvents([CloseUserInputEvent(timestamp=uint64(0))]),
        generation=0,
    )
    loop._finish_run(None, step_completed=False)

    assert run.stop_requested
    assert not loop._shutdown_event.is_set()


def test_finish_run_advances_and_consumes_input_when_the_step_is_skipped() -> None:
    loop = _loop(T2VUIState())
    event = NumeralKeypadUserInputEvent(timestamp=uint64(0), value=7)

    first = loop._begin_run(UserInputEvents([event]), generation=0)
    loop._finish_run(None, step_completed=False)
    second = loop._begin_run(UserInputEvents([]), generation=0)
    loop._finish_run(None, step_completed=False)

    assert first.step_index == 0
    assert second.step_index == 1
    assert loop.user_events.get_events() == []
