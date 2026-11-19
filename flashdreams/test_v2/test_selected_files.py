# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for client file selection through the v2 window contract."""

import queue
import threading
from pathlib import Path
from typing import Any

import pytest
from numpy import uint64

from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.api_v2.loop import IUILoop
from flashdreams.runtime_v2.mp4_client_window import Mp4ClientWindow
from flashdreams.runtime_v2.null_client_window import NullClientWindow
from flashdreams.runtime_v2.presentation_manager import PresentationManager
from flashdreams.runtime_v2.selected_file import (
    DEFAULT_MAX_FILE_BYTES,
    DEFAULT_MAX_FILE_COUNT,
    DEFAULT_MAX_TOTAL_BYTES,
    FileSelectionRequest,
    SelectedFile,
    SelectedFilesStatus,
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


def test_ui_loop_queues_the_app_declared_limits() -> None:
    loop = _ui_loop()
    loop.request_file_selector(
        "open-1",
        "/tmp",
        accept=(".png", ".JPG"),
        max_file_bytes=DEFAULT_MAX_FILE_BYTES * 4,
        max_total_bytes=DEFAULT_MAX_TOTAL_BYTES * 4,
        max_file_count=DEFAULT_MAX_FILE_COUNT * 4,
    )
    loop.request_file_selector("open-2", "/tmp")

    requests = loop.flush_ui_loop_requests()
    assert requests is not None
    raised, defaults = requests.file_selections
    assert raised == FileSelectionRequest(
        request_id="open-1",
        initial_path="/tmp",
        accept=(".png", ".JPG"),
        max_file_bytes=DEFAULT_MAX_FILE_BYTES * 4,
        max_total_bytes=DEFAULT_MAX_TOTAL_BYTES * 4,
        max_file_count=DEFAULT_MAX_FILE_COUNT * 4,
    )
    assert (
        defaults.max_file_bytes,
        defaults.max_total_bytes,
        defaults.max_file_count,
    ) == (
        DEFAULT_MAX_FILE_BYTES,
        DEFAULT_MAX_TOTAL_BYTES,
        DEFAULT_MAX_FILE_COUNT,
    )


@pytest.mark.parametrize(
    "limit", ["max_file_bytes", "max_total_bytes", "max_file_count"]
)
def test_ui_loop_rejects_a_non_positive_limit(limit: str) -> None:
    loop = _ui_loop()
    zero: dict[str, Any] = {limit: 0}
    fractional: dict[str, Any] = {limit: 1.5}
    with pytest.raises(ValueError, match=f"{limit} must be > 0"):
        loop.request_file_selector("open-1", **zero)
    with pytest.raises(TypeError, match=f"{limit} must be an integer"):
        loop.request_file_selector("open-1", **fractional)


def test_ui_loop_rejects_a_file_limit_over_the_total_limit() -> None:
    loop = _ui_loop()
    with pytest.raises(ValueError, match="max_file_bytes must not exceed"):
        loop.request_file_selector("open-1", max_file_bytes=2, max_total_bytes=1)
    with pytest.raises(ValueError, match="max_file_bytes must not exceed"):
        loop.request_file_selector("open-1", max_file_bytes=DEFAULT_MAX_TOTAL_BYTES + 1)


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
        request_id="open-1", initial_path="/tmp", accept=(".png",), max_file_bytes=4
    )
    a = SelectedFile(name="a.png", data=b"1")
    b = SelectedFile(name="b.png", data=b"2")
    cases = [
        (png, SelectedFilesStatus.OK, (), SelectedFilesStatus.UNAVAILABLE),
        (png, SelectedFilesStatus.OK, (a, b), SelectedFilesStatus.TOO_LARGE),
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

    many = FileSelectionRequest(
        request_id="open-1", initial_path="/tmp", max_file_count=2
    )
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

    with pytest.raises(TypeError, match="put_user_input_event"):
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
    assert normalize_selected_file_accept([".png", ".jpg"]) == (".png", ".jpg")
    assert selected_file_suffix_allowed("seed.PNG", (".png",))
    assert not selected_file_suffix_allowed("seed.txt", (".png",))
    assert selected_file_suffix_allowed("seed.txt", ())
    bins = FileSelectionRequest(
        request_id="open-1",
        initial_path="/tmp",
        accept=(".bin",),
        max_file_bytes=3,
        max_total_bytes=5,
        max_file_count=2,
    )
    cases = [
        ([("a.bin", 0)] * 3, SelectedFilesStatus.TOO_LARGE),
        ([("a.bin", 2), ("b.txt", 2)], SelectedFilesStatus.DISALLOWED_TYPE),
        ([("a.bin", 2), ("b.bin", 4)], SelectedFilesStatus.TOO_LARGE),
        ([("a.bin", 3), ("b.bin", 3)], SelectedFilesStatus.TOO_LARGE),
        ([("a.bin", 2), ("b.bin", 3)], None),
    ]
    for files, expected in cases:
        assert selected_files_policy_status(files, bins) is expected


def test_ui_loop_rejects_empty_request_id() -> None:
    loop = _ui_loop()
    with pytest.raises(ValueError, match="non-empty string"):
        loop.request_file_selector("")


def test_ui_loop_rejects_invalid_accept() -> None:
    loop = _ui_loop()
    with pytest.raises(ValueError, match="accept suffixes"):
        loop.request_file_selector("open-1", accept=("png",))


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
