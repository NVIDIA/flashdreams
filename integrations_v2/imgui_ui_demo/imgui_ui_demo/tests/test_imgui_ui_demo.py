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
    _MAX_FILE_BYTES,
    _MAX_TOTAL_BYTES,
    _OPEN_FILE_A_LABEL,
    _OPEN_FILE_A_REQUEST_ID,
    _OPEN_FILE_B_LABEL,
    _OPEN_FILE_B_REQUEST_ID,
    _OPEN_FILES_LABEL,
    _OPEN_FILES_MAX_FILE_COUNT,
    _OPEN_FILES_REQUEST_ID,
    FilePickerImGuiUILoop,
    FilePickerState,
)
from imgui_ui_demo.text_input_app import TextInputImGuiUILoop, TextInputState
from numpy import uint64

from flashdreams.runtime_v2.presentation_manager import PresentationManager
from flashdreams.runtime_v2.selected_file import (
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
    requests = loop.flush_ui_loop_requests()

    assert requests is not None
    [selection] = requests.file_selections
    assert selection.request_id == request_id
    assert selection.accept == (".bin", ".raw")
    assert selection.max_file_bytes == _MAX_FILE_BYTES
    assert selection.max_total_bytes == _MAX_TOTAL_BYTES
    assert selection.max_file_count == 1


def test_open_files_button_requests_multiple_client_files() -> None:
    _state, loop = _file_picker_loop()
    imgui = _file_picker_imgui(pressed=_OPEN_FILES_LABEL)

    loop.step_ui(imgui, 0, UserInputEvents([]))
    requests = loop.flush_ui_loop_requests()

    assert requests is not None
    [selection] = requests.file_selections
    assert selection.request_id == _OPEN_FILES_REQUEST_ID
    assert selection.accept == (".bin", ".raw")
    assert selection.max_file_bytes == _MAX_FILE_BYTES
    assert selection.max_total_bytes == _MAX_TOTAL_BYTES
    assert selection.max_file_count == _OPEN_FILES_MAX_FILE_COUNT


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

    assert state.statuses == {_OPEN_FILE_A_REQUEST_ID: "seed.png (2 bytes)"}
    assert [call.args[0] for call in imgui.text.call_args_list] == [
        f"{_OPEN_FILE_A_LABEL}: seed.png (2 bytes)",
        f"{_OPEN_FILE_B_LABEL}: No file selected.",
        f"{_OPEN_FILES_LABEL}: No file selected.",
    ]
    imgui.text_wrapped.assert_not_called()
    imgui.push_style_color.assert_not_called()


def test_each_button_shows_only_its_own_result() -> None:
    state, loop = _file_picker_loop()
    loop.user_events = UserInputEvents(
        [
            SelectedFilesUserInputEvent(
                timestamp=uint64(0),
                request_id=_OPEN_FILE_B_REQUEST_ID,
                status=SelectedFilesStatus.OK,
                files=(SelectedFile(name="b.bin", data=b"b"),),
            ),
            SelectedFilesUserInputEvent(
                timestamp=uint64(0),
                request_id=_OPEN_FILES_REQUEST_ID,
                status=SelectedFilesStatus.UNAVAILABLE,
            ),
        ]
    )
    loop.step_ui(_file_picker_imgui(), 0, loop.user_events)

    assert state.statuses == {
        _OPEN_FILE_B_REQUEST_ID: "b.bin (1 bytes)",
        _OPEN_FILES_REQUEST_ID: _FILE_SELECTION_STATUS_TEXT[
            SelectedFilesStatus.UNAVAILABLE
        ],
    }


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

    assert state.statuses == {
        _OPEN_FILES_REQUEST_ID: "a.bin (2 bytes), b.raw (3 bytes)"
    }
    imgui.text.assert_called_with(
        f"{_OPEN_FILES_LABEL}: a.bin (2 bytes), b.raw (3 bytes)"
    )
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
    assert state.statuses == {_OPEN_FILE_A_REQUEST_ID: message}
    imgui.push_style_color.assert_called_once_with("text", _FILE_SELECTION_ERROR_COLOR)
    imgui.text_wrapped.assert_called_once_with(f"{_OPEN_FILE_A_LABEL}: {message}")
    imgui.pop_style_color.assert_called_once_with()
    assert imgui.text.call_count == 2
