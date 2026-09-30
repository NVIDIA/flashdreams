# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU smoke tests for the v2 ImGui UI demo."""

import queue
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from imgui_ui_demo.file_picker_app import FilePickerImGuiUILoop, FilePickerState
from imgui_ui_demo.text_input_app import TextInputImGuiUILoop, TextInputState
from numpy import uint64

from flashdreams.runtime_v2.presentation_manager import PresentationManager
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.user_input_event import (
    SelectedFile,
    SelectedFilesUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

pytestmark = pytest.mark.ci_cpu


def test_text_input_updates_ui_owned_state() -> None:
    state = TextInputState()
    loop = TextInputImGuiUILoop(renderer=Mock())
    loop.register_session_loop_objects(
        state=state,
        frequency=60,
        shutdown_event=threading.Event(),
        failure_queue=queue.Queue(),
    )
    loop.register_session_ui_loop_objects(
        session_desc=SessionDesc(),
        presentation_manager=PresentationManager(),
    )
    imgui = SimpleNamespace(
        ImVec2=lambda x, y: (x, y),
        Cond_=SimpleNamespace(once="once"),
        set_next_window_pos=Mock(),
        set_next_window_size=Mock(),
        begin=Mock(),
        end=Mock(),
        text=Mock(),
        input_text=Mock(return_value=(True, "hello world")),
    )

    loop.step_ui(imgui, 0, UserInputEvents([]))

    assert state.text == "hello world"
    imgui.text.assert_called_with("Value: hello world")
    imgui.end.assert_called_once_with()


def _file_picker_loop() -> tuple[FilePickerState, FilePickerImGuiUILoop]:
    state = FilePickerState()
    loop = FilePickerImGuiUILoop(renderer=Mock())
    loop.register_session_loop_objects(
        state=state,
        frequency=60,
        shutdown_event=threading.Event(),
        failure_queue=queue.Queue(),
    )
    loop.register_session_ui_loop_objects(
        session_desc=SessionDesc(),
        presentation_manager=PresentationManager(),
    )
    return state, loop


def test_open_file_button_requests_a_client_file() -> None:
    _state, loop = _file_picker_loop()
    imgui = SimpleNamespace(
        ImVec2=lambda x, y: (x, y),
        Cond_=SimpleNamespace(once="once"),
        set_next_window_pos=Mock(),
        set_next_window_size=Mock(),
        begin=Mock(),
        end=Mock(),
        text=Mock(),
        button=Mock(return_value=True),
    )

    loop.step_ui(imgui, 0, UserInputEvents([]))
    requests = loop.flush_ui_loop_requests()

    assert requests is not None
    assert len(requests.file_selections) == 1
    assert requests.file_selections[0].request_id


def test_selected_files_event_updates_file_picker_status() -> None:
    state, loop = _file_picker_loop()
    imgui = SimpleNamespace(
        ImVec2=lambda x, y: (x, y),
        Cond_=SimpleNamespace(once="once"),
        set_next_window_pos=Mock(),
        set_next_window_size=Mock(),
        begin=Mock(),
        end=Mock(),
        text=Mock(),
        button=Mock(return_value=False),
    )

    loop.step_ui(
        imgui,
        0,
        UserInputEvents(
            [
                SelectedFilesUserInputEvent(
                    timestamp=uint64(0),
                    request_id="open-1",
                    files=(SelectedFile(name="seed.png", data=b"xx"),),
                )
            ]
        ),
    )

    assert state.status == "seed.png (2 bytes)"
    imgui.text.assert_called_with("seed.png (2 bytes)")
