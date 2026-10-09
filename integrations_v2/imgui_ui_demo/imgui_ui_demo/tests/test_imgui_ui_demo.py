# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU smoke tests for the v2 ImGui UI demo."""

import queue
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from imgui_ui_demo.file_picker_app import (
    _FILE_SELECTION_ERROR_COLOR,
    _FILE_SELECTION_STATUS_TEXT,
    _OPEN_A_THEN_B_LABEL,
    _OPEN_FILE_A_LABEL,
    _OPEN_FILE_A_REQUEST_ID,
    _OPEN_FILE_B_LABEL,
    _OPEN_FILE_B_REQUEST_ID,
    _OPEN_FILES_LABEL,
    _OPEN_FILES_REQUEST_ID,
    FilePickerImGuiUILoop,
    FilePickerState,
)
from imgui_ui_demo.text_input_app import TextInputImGuiUILoop, TextInputState
from numpy import uint64

from flashdreams.runtime_v2.presentation_manager import PresentationManager
from flashdreams.runtime_v2.selected_file import (
    MAX_SELECTED_FILE_BYTES,
    SelectedFile,
    SelectedFilesStatus,
)
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.user_input_event import SelectedFilesUserInputEvent
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


def _file_picker_imgui(*, pressed: str | tuple[str, ...] = ()) -> SimpleNamespace:
    labels = (pressed,) if isinstance(pressed, str) else pressed
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
        button=Mock(side_effect=lambda label: label in labels),
    )


@pytest.mark.parametrize(
    ("label", "request_id"),
    [
        (_OPEN_FILE_A_LABEL, _OPEN_FILE_A_REQUEST_ID),
        (_OPEN_FILE_B_LABEL, _OPEN_FILE_B_REQUEST_ID),
    ],
)
def test_open_file_button_requests_a_client_file(label: str, request_id: str) -> None:
    _state, loop = _file_picker_loop()
    imgui = _file_picker_imgui(pressed=label)

    loop.step_ui(imgui, 0, UserInputEvents([]))
    loop.step_ui(imgui, 1, UserInputEvents([]))
    requests = loop.flush_ui_loop_requests()

    assert requests is not None
    assert [item.request_id for item in requests.file_selections] == [request_id]
    assert requests.file_selections[0].accept == (".bin", ".raw")
    assert requests.file_selections[0].max_bytes == MAX_SELECTED_FILE_BYTES
    assert requests.file_selections[0].multiple is False


def test_open_files_button_requests_multiple_client_files() -> None:
    _state, loop = _file_picker_loop()
    imgui = _file_picker_imgui(pressed=_OPEN_FILES_LABEL)

    loop.step_ui(imgui, 0, UserInputEvents([]))
    requests = loop.flush_ui_loop_requests()

    assert requests is not None
    assert [item.request_id for item in requests.file_selections] == [
        _OPEN_FILES_REQUEST_ID
    ]
    assert requests.file_selections[0].accept == (".bin", ".raw")
    assert requests.file_selections[0].max_bytes == MAX_SELECTED_FILE_BYTES
    assert requests.file_selections[0].multiple is True


def test_open_a_then_b_queues_both_request_ids() -> None:
    _state, loop = _file_picker_loop()
    imgui = _file_picker_imgui(pressed=_OPEN_A_THEN_B_LABEL)

    loop.step_ui(imgui, 0, UserInputEvents([]))
    requests = loop.flush_ui_loop_requests()

    assert requests is not None
    assert [item.request_id for item in requests.file_selections] == [
        _OPEN_FILE_A_REQUEST_ID,
        _OPEN_FILE_B_REQUEST_ID,
    ]


def test_selected_files_event_updates_file_picker_status() -> None:
    state, loop = _file_picker_loop()
    imgui = _file_picker_imgui()

    loop.user_events = UserInputEvents(
        [
            SelectedFilesUserInputEvent(
                timestamp=uint64(0),
                request_id=_OPEN_FILE_A_REQUEST_ID,
                status=SelectedFilesStatus.OK,
                files=(SelectedFile(name="seed.png", data=b"xx"),),
            )
        ]
    )
    loop.step_ui(imgui, 0, loop.user_events)

    assert state.status == "seed.png (2 bytes)"
    imgui.text.assert_called_with("seed.png (2 bytes)")
    imgui.text_wrapped.assert_not_called()
    imgui.push_style_color.assert_not_called()


def test_selected_files_event_lists_every_chosen_file() -> None:
    state, loop = _file_picker_loop()
    imgui = _file_picker_imgui()

    loop.user_events = UserInputEvents(
        [
            SelectedFilesUserInputEvent(
                timestamp=uint64(0),
                request_id=_OPEN_FILES_REQUEST_ID,
                status=SelectedFilesStatus.OK,
                files=(
                    SelectedFile(name="a.bin", data=b"aa"),
                    SelectedFile(name="b.raw", data=b"bbb"),
                ),
            )
        ]
    )
    loop.step_ui(imgui, 0, loop.user_events)

    assert state.status == "a.bin (2 bytes), b.raw (3 bytes)"
    imgui.text.assert_called_with("a.bin (2 bytes), b.raw (3 bytes)")
    imgui.text_wrapped.assert_not_called()
    imgui.push_style_color.assert_not_called()


def test_file_picker_shows_error_status_in_red() -> None:
    assert set(_FILE_SELECTION_STATUS_TEXT) == {
        status for status in SelectedFilesStatus if status is not SelectedFilesStatus.OK
    }
    state, loop = _file_picker_loop()
    imgui = _file_picker_imgui()
    cancelled = SelectedFilesStatus.CANCELLED

    loop.user_events = UserInputEvents(
        [
            SelectedFilesUserInputEvent(
                timestamp=uint64(0),
                request_id=_OPEN_FILE_A_REQUEST_ID,
                status=cancelled,
            )
        ]
    )
    loop.step_ui(imgui, 0, loop.user_events)

    message = _FILE_SELECTION_STATUS_TEXT[cancelled]
    assert state.status == message
    imgui.push_style_color.assert_called_with("text", _FILE_SELECTION_ERROR_COLOR)
    imgui.text_wrapped.assert_called_with(message)
    imgui.pop_style_color.assert_called_once_with()
    imgui.text.assert_not_called()
