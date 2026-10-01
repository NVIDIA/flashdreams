# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for client file selection through the v2 window contract."""

import queue
import threading
from pathlib import Path

import pytest
from numpy import uint64

from flashdreams.api_v2.loop import IUILoop
from flashdreams.runtime_v2.mp4_client_window import Mp4ClientWindow
from flashdreams.runtime_v2.presentation_manager import PresentationManager
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    MAX_SELECTED_FILE_BYTES,
    SelectedFile,
    SelectedFilesStatus,
    SelectedFilesUserInputEvent,
    clamp_selected_file_max_bytes,
    normalize_selected_file_accept,
    selected_file_policy_status,
    selected_file_suffix_allowed,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

pytestmark = pytest.mark.ci_cpu


class _UILoop(IUILoop[None]):
    def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]:
        del step_index, events
        return []

    def reset(self) -> None:
        return


def _ui_loop() -> _UILoop:
    loop = _UILoop()
    loop.register_session_loop_objects(
        state=None,
        frequency=0,
        shutdown_event=threading.Event(),
        failure_queue=queue.Queue(),
    )
    loop.register_session_ui_loop_objects(
        session_desc=SessionDesc(),
        presentation_manager=PresentationManager(),
    )
    return loop


def test_selected_files_event_carries_bytes_and_request_id() -> None:
    event = SelectedFilesUserInputEvent(
        timestamp=uint64(1),
        request_id="open-1",
        status=SelectedFilesStatus.OK,
        files=(SelectedFile(name="seed.png", data=b"png"),),
    )

    assert event.get_type_name() == "selected_files"
    assert event.request_id == "open-1"
    assert event.status is SelectedFilesStatus.OK
    assert event.files[0].data == b"png"


def test_ui_loop_queues_file_selection_policy_without_clamping() -> None:
    loop = _ui_loop()
    loop.request_selected_files(
        "open-1",
        "/tmp",
        accept=(".png", ".JPG"),
        max_bytes=MAX_SELECTED_FILE_BYTES * 2,
    )
    loop.request_selected_files("open-1", "/var")

    requests = loop.flush_ui_loop_requests()
    assert requests is not None
    assert len(requests.file_selections) == 1
    selection = requests.file_selections[0]
    assert selection.request_id == "open-1"
    assert selection.initial_path == "/tmp"
    assert selection.accept == (".png", ".JPG")
    assert selection.max_bytes == MAX_SELECTED_FILE_BYTES * 2


def test_selected_file_policy_helpers() -> None:
    assert clamp_selected_file_max_bytes(None) == MAX_SELECTED_FILE_BYTES
    assert clamp_selected_file_max_bytes(16) == 16
    assert clamp_selected_file_max_bytes(MAX_SELECTED_FILE_BYTES * 2) == (
        MAX_SELECTED_FILE_BYTES
    )
    assert normalize_selected_file_accept([".png", ".jpg"]) == (".png", ".jpg")
    assert selected_file_suffix_allowed("seed.PNG", (".png",))
    assert not selected_file_suffix_allowed("seed.txt", (".png",))
    assert selected_file_suffix_allowed("seed.txt", ())
    assert (
        selected_file_policy_status("seed.txt", 99, accept=(".png",), max_bytes=3)
        is SelectedFilesStatus.DISALLOWED_TYPE
    )
    assert (
        selected_file_policy_status("seed.png", 99, accept=(".png",), max_bytes=3)
        is SelectedFilesStatus.TOO_LARGE
    )
    assert (
        selected_file_policy_status("seed.png", 2, accept=(".png",), max_bytes=3)
        is None
    )


def test_ui_loop_rejects_invalid_accept() -> None:
    loop = _ui_loop()
    with pytest.raises(ValueError, match="accept suffixes"):
        loop.request_selected_files("open-1", accept=("png",))


def test_selected_files_event_rejects_inconsistent_status() -> None:
    with pytest.raises(ValueError, match="must include files"):
        SelectedFilesUserInputEvent(
            timestamp=uint64(1),
            request_id="open-1",
            status=SelectedFilesStatus.OK,
        )
    with pytest.raises(ValueError, match="must not include files"):
        SelectedFilesUserInputEvent(
            timestamp=uint64(1),
            request_id="open-1",
            status=SelectedFilesStatus.CANCELLED,
            files=(SelectedFile(name="seed.png", data=b"png"),),
        )


def test_mp4_window_completes_file_selection_as_unavailable(tmp_path: Path) -> None:
    window = Mp4ClientWindow(tmp_path / "clip.mp4")
    window.request_selected_files("open-1", "/tmp")

    events = window.get_user_input_events().get_events()

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, SelectedFilesUserInputEvent)
    assert event.request_id == "open-1"
    assert event.status is SelectedFilesStatus.UNAVAILABLE
    assert event.files == ()
    assert window.get_user_input_events().get_events() == []
