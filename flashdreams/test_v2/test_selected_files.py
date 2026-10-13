# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for client file selection through the v2 window contract."""

import queue
import threading
from pathlib import Path

import pytest
from numpy import uint64

from flashdreams.api_v2.loop import IUILoop
from flashdreams.api_v2.user_input_event import UserInputEvent
from flashdreams.runtime_v2.mp4_client_window import Mp4ClientWindow
from flashdreams.runtime_v2.null_client_window import NullClientWindow
from flashdreams.runtime_v2.presentation_manager import PresentationManager
from flashdreams.runtime_v2.selected_file import (
    MAX_SELECTED_FILE_BATCH_BYTES,
    MAX_SELECTED_FILE_BYTES,
    FileSelectionRequest,
    SelectedFile,
    SelectedFilesStatus,
    clamp_selected_file_max_bytes,
    normalize_selected_file_accept,
    selected_file_policy_status,
    selected_file_suffix_allowed,
    selected_files_policy_status,
)
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.session_runner import _FileSelections
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    CloseUserInputEvent,
    FileSelectorOpenedUserInputEvent,
    SelectedFilesUserInputEvent,
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


def test_ui_loop_queues_file_selection_policy_without_clamping() -> None:
    loop = _ui_loop()
    loop.file_selector(
        "open-1",
        "/tmp",
        open=True,
        accept=(".png", ".JPG"),
        max_file_bytes=MAX_SELECTED_FILE_BYTES * 2,
        multiple=True,
    )

    requests = loop.flush_ui_loop_requests()
    assert requests is not None
    assert len(requests.file_selections) == 1
    selection = requests.file_selections[0]
    assert selection.request_id == "open-1"
    assert selection.initial_path == "/tmp"
    assert selection.accept == (".png", ".JPG")
    assert selection.max_bytes == MAX_SELECTED_FILE_BYTES * 2
    assert selection.multiple is True


def test_ui_loop_rejects_request_id_reused_in_one_tick() -> None:
    loop = _ui_loop()
    loop.file_selector("open-1")
    with pytest.raises(ValueError, match="more than once this tick"):
        loop.file_selector("open-1", open=True)

    loop.flush_ui_loop_requests()
    loop.file_selector("open-1", open=True)
    requests = loop.flush_ui_loop_requests()
    assert requests is not None
    assert [item.request_id for item in requests.file_selections] == ["open-1"]


def test_ui_loop_uses_home_when_initial_path_is_omitted() -> None:
    loop = _ui_loop()
    loop.file_selector("open-1", open=True)

    requests = loop.flush_ui_loop_requests()
    assert requests is not None
    assert requests.file_selections[0].initial_path == str(Path.home())


def test_ui_loop_forwards_distinct_file_selection_ids() -> None:
    loop = _ui_loop()
    loop.file_selector("open-1", "/tmp", open=True)
    loop.file_selector("open-2", "/var", open=True)

    requests = loop.flush_ui_loop_requests()
    assert requests is not None
    assert [item.request_id for item in requests.file_selections] == [
        "open-1",
        "open-2",
    ]


def test_ui_loop_keeps_file_results_until_a_step_runs() -> None:
    loop = _ui_loop()
    picked = SelectedFilesUserInputEvent(
        timestamp=uint64(0),
        request_id="open-1",
        status=SelectedFilesStatus.CANCELLED,
    )

    loop._begin_run(UserInputEvents([picked]), generation=0)
    loop._finish_run(None, step_completed=False)
    loop._begin_run(UserInputEvents([]), generation=0)
    assert loop.user_events.get_events() == [picked]

    loop._finish_run([], step_completed=True)
    loop._begin_run(UserInputEvents([]), generation=0)
    assert loop.user_events.get_events() == []


class _PickerWindow(NullClientWindow):
    def __init__(self) -> None:
        self.started: list[str] = []

    def request_selected_files(self, request: FileSelectionRequest) -> None:
        self.started.append(request.request_id)


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _request(request_id: str) -> FileSelectionRequest:
    return FileSelectionRequest(request_id=request_id, initial_path="/tmp")


def _opened(request_id: str) -> FileSelectorOpenedUserInputEvent:
    return FileSelectorOpenedUserInputEvent(timestamp=uint64(0), request_id=request_id)


def _result(request_id: str) -> SelectedFilesUserInputEvent:
    return SelectedFilesUserInputEvent(
        timestamp=uint64(0),
        request_id=request_id,
        status=SelectedFilesStatus.CANCELLED,
    )


def _statuses(events: list[UserInputEvent]) -> list[tuple[str, SelectedFilesStatus]]:
    return [
        (event.request_id, event.status)
        for event in events
        if isinstance(event, SelectedFilesUserInputEvent)
    ]


def test_session_shows_one_picker_at_a_time() -> None:
    window = _PickerWindow()
    selections = _FileSelections(window, timeout_seconds=1.0, clock=_Clock())
    assert selections.request(_request("open-1")) == []
    assert selections.request(_request("open-1")) == []
    busy = selections.request(_request("open-2"))
    assert _statuses(busy) == [("open-2", SelectedFilesStatus.UNAVAILABLE)]
    assert window.started == ["open-1"]

    assert selections.filter([_result("open-2")]) == []
    kept = selections.filter([_result("open-1")])
    assert _statuses(kept) == [("open-1", SelectedFilesStatus.CANCELLED)]
    assert selections.request(_request("open-2")) == []
    assert window.started == ["open-1", "open-2"]


def test_session_times_out_unopened_picker_and_drops_its_late_result() -> None:
    window = _PickerWindow()
    clock = _Clock()
    selections = _FileSelections(window, timeout_seconds=1.0, clock=clock)
    selections.request(_request("open-1"))

    clock.now = 1.0
    kept = selections.filter([])
    assert _statuses(kept) == [("open-1", SelectedFilesStatus.UNAVAILABLE)]
    assert selections.filter([_result("open-1")]) == []
    assert selections.request(_request("open-2")) == []
    assert window.started == ["open-1", "open-2"]


def test_session_opened_picker_waits_for_the_user_without_deadline() -> None:
    window = _PickerWindow()
    clock = _Clock()
    selections = _FileSelections(window, timeout_seconds=1.0, clock=clock)
    selections.request(_request("open-1"))
    assert selections.filter([_opened("open-1"), _opened("other")]) == []

    clock.now = 100.0
    assert selections.filter([]) == []
    kept = selections.filter([_result("open-1")])
    assert _statuses(kept) == [("open-1", SelectedFilesStatus.CANCELLED)]


def test_session_passes_through_immediate_results_and_other_input() -> None:
    selections = _FileSelections(NullClientWindow(), clock=_Clock())
    kept = selections.request(_request("open-1"))
    assert _statuses(kept) == [("open-1", SelectedFilesStatus.UNAVAILABLE)]
    close = CloseUserInputEvent(timestamp=uint64(0))
    assert selections.filter([close]) == [close]


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
        selected_file_policy_status("seed.txt", None, accept=(".png",), max_bytes=3)
        is SelectedFilesStatus.DISALLOWED_TYPE
    )
    assert (
        selected_file_policy_status("seed.png", None, accept=(".png",), max_bytes=3)
        is None
    )
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
    assert (
        selected_files_policy_status(
            [("a.bin", 2), ("b.txt", 2)],
            accept=(".bin",),
            max_bytes=3,
        )
        is SelectedFilesStatus.DISALLOWED_TYPE
    )
    assert (
        selected_files_policy_status(
            [("a.bin", 2), ("b.bin", 99)],
            accept=(".bin",),
            max_bytes=3,
        )
        is SelectedFilesStatus.TOO_LARGE
    )
    assert (
        selected_files_policy_status(
            [("a.bin", MAX_SELECTED_FILE_BATCH_BYTES), ("b.bin", 1)],
            accept=(".bin",),
            max_bytes=MAX_SELECTED_FILE_BYTES,
        )
        is SelectedFilesStatus.TOO_LARGE
    )
    assert (
        selected_files_policy_status(
            [("a.bin", 2), ("b.bin", 2)],
            accept=(".bin",),
            max_bytes=3,
        )
        is None
    )


def test_ui_loop_does_not_queue_when_open_is_false() -> None:
    loop = _ui_loop()
    assert loop.file_selector("open-1") is None
    assert loop.flush_ui_loop_requests() is None


def test_ui_loop_returns_this_tick_selected_files_event() -> None:
    loop = _ui_loop()
    event = SelectedFilesUserInputEvent(
        timestamp=uint64(1),
        request_id="open-1",
        status=SelectedFilesStatus.OK,
        files=(SelectedFile(name="seed.png", data=b"png"),),
    )
    other = SelectedFilesUserInputEvent(
        timestamp=uint64(2),
        request_id="open-2",
        status=SelectedFilesStatus.CANCELLED,
    )
    loop.user_events = UserInputEvents([event, other])

    assert loop.file_selector("open-1") is event
    assert loop.file_selector("open-2") is other
    assert loop.file_selector("open-3") is None
    assert loop.flush_ui_loop_requests() is None


def test_ui_loop_rejects_empty_request_id() -> None:
    loop = _ui_loop()
    with pytest.raises(ValueError, match="non-empty string"):
        loop.file_selector("")


def test_ui_loop_rejects_invalid_accept() -> None:
    loop = _ui_loop()
    with pytest.raises(ValueError, match="accept suffixes"):
        loop.file_selector("open-1", open=True, accept=("png",))


def test_ui_loop_rejects_invalid_multiple() -> None:
    loop = _ui_loop()
    with pytest.raises(TypeError, match="multiple must be a bool"):
        loop.file_selector("open-1", open=True, multiple=1)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


def test_ui_loop_rejects_invalid_open() -> None:
    loop = _ui_loop()
    with pytest.raises(TypeError, match="open must be a bool"):
        loop.file_selector("open-1", open=1)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


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


@pytest.mark.parametrize("kind", ["mp4", "null"])
def test_window_without_a_selector_answers_unavailable(
    kind: str, tmp_path: Path
) -> None:
    window = (
        Mp4ClientWindow(tmp_path / "clip.mp4") if kind == "mp4" else NullClientWindow()
    )
    pick = window.request_selected_files(
        FileSelectionRequest(request_id="open-1", initial_path="/tmp")
    )

    assert pick is not None
    assert pick.request_id == "open-1"
    assert pick.status is SelectedFilesStatus.UNAVAILABLE
    assert pick.files == ()
