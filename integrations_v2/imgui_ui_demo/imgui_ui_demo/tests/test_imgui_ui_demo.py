# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU smoke tests for the v2 ImGui UI demo."""

import queue
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from imgui_ui_demo.file_picker_app import (
    FilePickerImGuiUILoop,
    FilePickerState,
    _FILE_SELECTION_ERROR_COLOR,
    _FILE_SELECTION_STATUS_TEXT,
)
from imgui_ui_demo.text_input_app import TextInputImGuiUILoop, TextInputState
from numpy import uint64

from flashdreams.runtime_v2.presentation_manager import PresentationManager
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.user_input_event import (
    MAX_SELECTED_FILE_BYTES,
    SelectedFile,
    SelectedFilesStatus,
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


def _file_picker_imgui(*, button: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        ImVec2=lambda x, y: (x, y),
        ImVec4=lambda x, y, z, w: (x, y, z, w),
        Col_=SimpleNamespace(text="text"),
        Cond_=SimpleNamespace(once="once"),
        set_next_window_pos=Mock(),
        set_next_window_size=Mock(),
        begin=Mock(),
        end=Mock(),
        text=Mock(),
        text_wrapped=Mock(),
        push_style_color=Mock(),
        pop_style_color=Mock(),
        button=Mock(return_value=button),
    )


def test_open_file_button_requests_a_client_file() -> None:
    _state, loop = _file_picker_loop()
    imgui = _file_picker_imgui(button=True)

    loop.step_ui(imgui, 0, UserInputEvents([]))
    requests = loop.flush_ui_loop_requests()

    assert requests is not None
    assert len(requests.file_selections) == 1
    assert requests.file_selections[0].request_id
    assert requests.file_selections[0].accept == (".bin", ".raw")
    assert requests.file_selections[0].max_bytes == MAX_SELECTED_FILE_BYTES


def test_selected_files_event_updates_file_picker_status() -> None:
    state, loop = _file_picker_loop()
    imgui = _file_picker_imgui()

    loop.step_ui(
        imgui,
        0,
        UserInputEvents(
            [
                SelectedFilesUserInputEvent(
                    timestamp=uint64(0),
                    request_id="open-1",
                    status=SelectedFilesStatus.OK,
                    files=(SelectedFile(name="seed.png", data=b"xx"),),
                )
            ]
        ),
    )

    assert state.status == "seed.png (2 bytes)"
    imgui.text.assert_called_with("seed.png (2 bytes)")
    imgui.text_wrapped.assert_not_called()
    imgui.push_style_color.assert_not_called()


def test_file_picker_shows_error_status_in_red() -> None:
    assert set(_FILE_SELECTION_STATUS_TEXT) == {
        status
        for status in SelectedFilesStatus
        if status is not SelectedFilesStatus.OK
    }
    state, loop = _file_picker_loop()
    imgui = _file_picker_imgui()
    cancelled = SelectedFilesStatus.CANCELLED

    loop.step_ui(
        imgui,
        0,
        UserInputEvents(
            [
                SelectedFilesUserInputEvent(
                    timestamp=uint64(0),
                    request_id="open-1",
                    status=cancelled,
                )
            ]
        ),
    )

    message = _FILE_SELECTION_STATUS_TEXT[cancelled]
    assert state.status == message
    imgui.push_style_color.assert_called_with("text", _FILE_SELECTION_ERROR_COLOR)
    imgui.text_wrapped.assert_called_with(message)
    imgui.pop_style_color.assert_called_once_with()
    imgui.text.assert_not_called()
