# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for client file selection through the v2 window contract."""

import queue
import threading
from pathlib import Path

import pytest
from numpy import uint64

from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.api_v2.loop import IUILoop
from flashdreams.runtime_v2.mp4_client_window import Mp4ClientWindow
from flashdreams.runtime_v2.null_client_window import NullClientWindow
from flashdreams.runtime_v2.presentation_manager import PresentationManager
from flashdreams.runtime_v2.selected_file import (
    FILE_SELECTION_PRESENTATION_TIMEOUT_SECONDS,
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
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
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
    loop.request_file_selector(
        "open-1",
        "/tmp",
        accept=(".png", ".JPG"),
        max_file_bytes=MAX_SELECTED_FILE_BYTES * 2,
        multiple=True,
    )

    requests = loop.flush_ui_loop_requests()
    assert requests is not None
    selection = requests.file_selection
    assert selection is not None
    assert selection.request_id == "open-1"
    assert selection.initial_path == "/tmp"
    assert selection.accept == (".png", ".JPG")
    assert selection.max_bytes == MAX_SELECTED_FILE_BYTES * 2
    assert selection.multiple is True


def test_ui_loop_uses_home_when_initial_path_is_omitted() -> None:
    loop = _ui_loop()
    loop.request_file_selector("open-1")

    requests = loop.flush_ui_loop_requests()
    assert requests is not None
    assert requests.file_selection is not None
    assert requests.file_selection.initial_path == str(Path.home())


def test_ui_loop_keeps_the_first_file_selection_of_a_tick() -> None:
    loop = _ui_loop()
    loop.request_file_selector("open-1", "/tmp")
    loop.request_file_selector("open-2", "/var")

    requests = loop.flush_ui_loop_requests()
    assert requests is not None
    assert requests.file_selection is not None
    assert requests.file_selection.request_id == "open-1"


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
        super().__init__()
        self.now = 0.0
        self._file_selector_clock = lambda: self.now
        self.started: list[tuple[int, str]] = []
        self.cancelled = 0

    def _request_file_selector(
        self, selector_id: int, request: FileSelectionRequest
    ) -> None:
        self.started.append((selector_id, request.request_id))

    def _cancel_active_file_selectors(self) -> None:
        self.cancelled += 1


def _request(request_id: str) -> FileSelectionRequest:
    return FileSelectionRequest(request_id=request_id, initial_path="/tmp")


def _statuses(window: IClientWindow) -> list[tuple[str, SelectedFilesStatus]]:
    return [
        (event.request_id, event.status)
        for event in window.get_user_input_events().get_events()
        if isinstance(event, SelectedFilesUserInputEvent)
    ]


def test_window_shows_one_picker_at_a_time() -> None:
    window = _PickerWindow()
    window.request_file_selector(_request("open-1"))
    window.request_file_selector(_request("open-1"))
    window.request_file_selector(_request("open-2"))
    assert [request_id for _, request_id in window.started] == ["open-1"]
    assert _statuses(window) == []

    selector_id = window.started[0][0]
    window._complete_file_selector(selector_id, SelectedFilesStatus.CANCELLED)
    window._complete_file_selector(selector_id, SelectedFilesStatus.CANCELLED)
    assert _statuses(window) == [("open-1", SelectedFilesStatus.CANCELLED)]
    window.request_file_selector(_request("open-2"))
    assert [request_id for _, request_id in window.started] == ["open-1", "open-2"]


def test_window_times_out_unshown_picker_and_drops_its_late_result() -> None:
    window = _PickerWindow()
    window.request_file_selector(_request("open-1"))
    window.now = FILE_SELECTION_PRESENTATION_TIMEOUT_SECONDS
    window._expire_file_selectors()
    assert _statuses(window) == [("open-1", SelectedFilesStatus.UNAVAILABLE)]
    assert window.cancelled == 1

    window._complete_file_selector(window.started[0][0], SelectedFilesStatus.CANCELLED)
    assert _statuses(window) == []


def test_null_window_open_drops_a_result_left_by_the_previous_session() -> None:
    window = NullClientWindow()
    window.request_file_selector(_request("open-1"))
    window.open(SessionDesc())
    assert _statuses(window) == []


def test_window_expires_a_stuck_picker_when_the_next_request_arrives() -> None:
    window = _PickerWindow()
    window.request_file_selector(_request("open-1"))
    window.now = FILE_SELECTION_PRESENTATION_TIMEOUT_SECONDS
    window.request_file_selector(_request("open-2"))
    assert _statuses(window) == [("open-1", SelectedFilesStatus.UNAVAILABLE)]
    assert [request_id for _, request_id in window.started] == ["open-1", "open-2"]


def test_window_shown_picker_waits_for_the_user_without_deadline() -> None:
    window = _PickerWindow()
    window.request_file_selector(_request("open-1"))
    window._file_selector_opened(window.started[0][0])
    window.now = 1_000.0
    window._expire_file_selectors()
    assert _statuses(window) == []


def test_flush_forgets_the_open_picker_without_a_result() -> None:
    window = _PickerWindow()
    window.request_file_selector(_request("open-1"))
    window.flush_file_selectors()
    window._complete_file_selector(window.started[0][0], SelectedFilesStatus.CANCELLED)
    assert _statuses(window) == []
    assert window.cancelled == 1

    window.flush_file_selectors()
    assert window.cancelled == 1
    window.request_file_selector(_request("open-1"))
    assert len(window.started) == 2


def test_window_answers_unavailable_when_its_picker_fails_to_start() -> None:
    class _BrokenWindow(NullClientWindow):
        def _request_file_selector(
            self, selector_id: int, request: FileSelectionRequest
        ) -> None:
            raise RuntimeError("no picker")

    window = _BrokenWindow()
    window.request_file_selector(_request("open-1"))
    assert _statuses(window) == [("open-1", SelectedFilesStatus.UNAVAILABLE)]


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


def test_ui_loop_rejects_empty_request_id() -> None:
    loop = _ui_loop()
    with pytest.raises(ValueError, match="non-empty string"):
        loop.request_file_selector("")


def test_ui_loop_rejects_invalid_accept() -> None:
    loop = _ui_loop()
    with pytest.raises(ValueError, match="accept suffixes"):
        loop.request_file_selector("open-1", accept=("png",))


def test_ui_loop_rejects_invalid_multiple() -> None:
    loop = _ui_loop()
    with pytest.raises(TypeError, match="multiple must be a bool"):
        loop.request_file_selector("open-1", multiple=1)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


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
    window.request_file_selector(
        FileSelectionRequest(request_id="open-1", initial_path="/tmp")
    )

    assert _statuses(window) == [("open-1", SelectedFilesStatus.UNAVAILABLE)]
