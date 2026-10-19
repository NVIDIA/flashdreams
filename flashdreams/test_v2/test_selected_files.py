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
    MAX_SELECTED_FILE_BATCH_BYTES,
    MAX_SELECTED_FILE_BYTES,
    MAX_SELECTED_FILE_COUNT,
    FileSelectionRequest,
    SelectedFile,
    SelectedFilesStatus,
    clamp_selected_file_max_bytes,
    normalize_selected_file_accept,
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
    [selection] = requests.file_selections
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
    assert requests.file_selections[0].initial_path == str(Path.home())


def test_ui_loop_forwards_every_file_selection_of_a_tick() -> None:
    loop = _ui_loop()
    loop.request_file_selector("open-1", "/tmp")
    loop.request_file_selector("open-2", "/var")

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
        super().__init__()
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
    assert _statuses(window) == [
        ("open-1", SelectedFilesStatus.BUSY),
        ("open-2", SelectedFilesStatus.BUSY),
    ]

    selector_id = window.started[0][0]
    window._complete_file_selector(selector_id, SelectedFilesStatus.CANCELLED)
    window._complete_file_selector(selector_id, SelectedFilesStatus.CANCELLED)
    assert _statuses(window) == [("open-1", SelectedFilesStatus.CANCELLED)]
    window.request_file_selector(_request("open-2"))
    assert [request_id for _, request_id in window.started] == ["open-1", "open-2"]


def test_window_checks_a_result_against_its_request() -> None:
    def complete(
        request: FileSelectionRequest,
        status: SelectedFilesStatus,
        files: tuple[SelectedFile, ...],
    ) -> list[SelectedFilesUserInputEvent]:
        window = _PickerWindow()
        window.request_file_selector(request)
        window._complete_file_selector(window.started[0][0], status, files)
        events = window.get_user_input_events().get_events()
        selected = [e for e in events if isinstance(e, SelectedFilesUserInputEvent)]
        assert selected == events
        return selected

    png = FileSelectionRequest(
        request_id="open-1", initial_path="/tmp", accept=(".png",), max_bytes=4
    )
    a = SelectedFile(name="a.png", data=b"1")
    b = SelectedFile(name="b.png", data=b"2")
    cases = [
        (png, SelectedFilesStatus.OK, (), SelectedFilesStatus.UNAVAILABLE),
        (png, SelectedFilesStatus.OK, (a, b), SelectedFilesStatus.UNAVAILABLE),
        (
            png,
            SelectedFilesStatus.OK,
            (SelectedFile(name="a.txt", data=b"1"),),
            SelectedFilesStatus.DISALLOWED_TYPE,
        ),
        (
            png,
            SelectedFilesStatus.OK,
            (SelectedFile(name="a.png", data=b"12345"),),
            SelectedFilesStatus.TOO_LARGE,
        ),
        (png, SelectedFilesStatus.CANCELLED, (a,), SelectedFilesStatus.CANCELLED),
        (png, SelectedFilesStatus.BUSY, (a,), SelectedFilesStatus.BUSY),
    ]
    for request, status, files, expected in cases:
        [event] = complete(request, status, files)
        assert (event.status, event.files) == (expected, ())

    many = FileSelectionRequest(request_id="open-1", initial_path="/tmp", multiple=True)
    [event] = complete(many, SelectedFilesStatus.OK, (a, b))
    assert (event.status, event.files) == (SelectedFilesStatus.OK, (a, b))

    pathy = (
        SelectedFile(name="../outside.png", data=b"1"),
        SelectedFile(name="/home/user/in.png", data=b"2"),
    )
    [event] = complete(many, SelectedFilesStatus.OK, pathy)
    assert [chosen.name for chosen in event.files] == ["outside.png", "in.png"]


def test_window_without_put_user_input_event_cannot_be_created() -> None:
    class _Incomplete(IClientWindow):
        def get_user_input_events(self) -> UserInputEvents:
            return UserInputEvents([])

        def open(self, session_desc: SessionDesc) -> None:
            del session_desc

        def write(self, result: StepResult) -> None:
            del result

        def close(self) -> None:
            return

    with pytest.raises(TypeError, match="_put_user_input_event"):
        _Incomplete()  # type: ignore[abstract]


def test_null_window_open_drops_a_result_left_by_the_previous_session() -> None:
    window = NullClientWindow()
    window.request_file_selector(_request("open-1"))
    window.open(SessionDesc())
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
        selected_files_policy_status(
            [("a.bin", 0)] * (MAX_SELECTED_FILE_COUNT + 1), max_bytes=3
        )
        is SelectedFilesStatus.TOO_LARGE
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
