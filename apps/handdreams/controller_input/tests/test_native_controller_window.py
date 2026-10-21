# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU integration tests for controller input with native presentation."""

from __future__ import annotations

import queue
import sys
import threading
import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from flashdreams_controller_input import (
    controller_source_v2_window as controller_module,
)
from flashdreams_controller_input.controller_source_v2_window import (
    ControllerSourceWindow,
)
from flashdreams_controller_input.controller_types import Observation
from flashdreams_controller_input.events import ControllerFrameEvent
from flashdreams_controller_input.pose import WRIST_PROXY_POSE_CONVENTION

from flashdreams.runtime_v2.native_window_client_window import NativeWindowClientWindow
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    CloseUserInputEvent,
    KeyboardInputState,
    KeyboardUserInputEvent,
)
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu


class _Presenter:
    """In-memory native presenter recording its calling threads."""

    def __init__(self) -> None:
        self.size = (2, 2)
        self.should_close = False
        self.callbacks: dict[str, Callable[..., None]] = {}
        self.keyboard_events: list[Any] = []
        self.frames: list[torch.Tensor] = []
        self.cursor_options: list[tuple[bool, bool]] = []
        self.resize_requests: list[tuple[int, int]] = []
        self.threads: list[int] = []
        self.open_count = 0
        self.close_count = 0

    def __call__(
        self, *, width: int, height: int, title: str, fullscreen: bool = False
    ) -> _Presenter:
        assert title == "Controller test"
        assert fullscreen is False
        self.size = width, height
        self.threads.append(threading.get_ident())
        self.open_count += 1
        return self

    def set_input_callbacks(self, **callbacks: Callable[..., None]) -> None:
        self.callbacks = callbacks

    def configure_cursor(
        self, *, hide_cursor: bool, lock_cursor_to_window: bool
    ) -> None:
        self.threads.append(threading.get_ident())
        self.cursor_options.append((hide_cursor, lock_cursor_to_window))

    def process_events(self) -> None:
        self.threads.append(threading.get_ident())
        pending, self.keyboard_events = self.keyboard_events, []
        for event in pending:
            self.callbacks["on_keyboard_event"](event)

    def present_frame(self, frame: torch.Tensor) -> bool:
        self.threads.append(threading.get_ident())
        self.frames.append(frame)
        return not self.should_close

    def resize(self, width: int, height: int) -> tuple[int, int]:
        self.threads.append(threading.get_ident())
        self.size = width, height
        self.resize_requests.append(self.size)
        return self.size

    def close(self) -> None:
        self.threads.append(threading.get_ident())
        self.close_count += 1


class _Source:
    """Queued controller input with observable worker ownership."""

    name = "test"

    def __init__(self) -> None:
        self.pending: queue.SimpleQueue[Observation] = queue.SimpleQueue()
        self.started = threading.Event()
        self.start_thread: int | None = None
        self.start_count = 0
        self.stop_count = 0

    def start(self) -> None:
        self.start_thread = threading.get_ident()
        self.start_count += 1
        self.started.set()

    def poll(self) -> Observation | None:
        try:
            return self.pending.get_nowait()
        except queue.Empty:
            return None

    def stop(self) -> None:
        self.stop_count += 1


def _window(
    monkeypatch: pytest.MonkeyPatch,
    now_ns: list[int],
    *,
    initial_window_size: tuple[int, int] | None = None,
) -> tuple[ControllerSourceWindow, _Presenter, _Source]:
    monkeypatch.setitem(
        sys.modules, "flashdreams.runtime_v2.webrtc_client_window", None
    )
    monkeypatch.setitem(sys.modules, "slangpy", None)
    monkeypatch.setattr(
        controller_module, "time", SimpleNamespace(perf_counter_ns=lambda: now_ns[0])
    )
    presenter = _Presenter()
    source = _Source()
    native = NativeWindowClientWindow(
        title="Controller test",
        presenter_factory=cast(Any, presenter),
        clock_ns=lambda: now_ns[0],
    )
    window = ControllerSourceWindow(
        output_window=native,
        source=source,
        source_name="test",
        pose_convention=WRIST_PROXY_POSE_CONVENTION,
        source_thread_name="test-native-controller-input",
        source_failure_message="Controller test input failed",
        initial_window_size=initial_window_size,
    )
    return window, presenter, source


def _buffer_observation(
    window: ControllerSourceWindow, source: _Source, *, capture_ns: int, frame_id: int
) -> None:
    source.pending.put(
        Observation(
            frame={"source_frame_index": frame_id},
            capture_ns=capture_ns,
            source=source.name,
        )
    )
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        with window._lock:
            if window._observations:
                return
        time.sleep(0.005)
    raise AssertionError("controller worker did not buffer the observation")


def _result() -> StepResult:
    return StepResult(
        step_index=0,
        output=torch.full((1, 3, 2, 2), 37, dtype=torch.uint8),
        frame_count=1,
        output_layout=VideoTensorLayout.tchw,
    )


def test_native_controller_window_preserves_display_and_input_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now_ns = [1_000_000]
    window, presenter, source = _window(monkeypatch, now_ns)
    try:
        assert source.started.wait(timeout=1.0)
        window.open(SessionDesc(video_width=2, video_height=2))
        assert not hasattr(window.output_window, "server")
        assert window.metrics_snapshot() == {}
        window.request_hide_cursor(True)
        window.request_lock_cursor_to_window(True)
        assert presenter.cursor_options[-2:] == [(True, False), (True, True)]

        window.write(_result())
        assert torch.equal(
            presenter.frames[-1], torch.full((2, 2, 3), 37, dtype=torch.uint8)
        )
        window.request_new_window_size((4, 3))
        window.write(_result())
        assert torch.equal(
            presenter.frames[-1], torch.full((3, 4, 3), 37, dtype=torch.uint8)
        )

        _buffer_observation(window, source, capture_ns=1_050_000, frame_id=7)
        now_ns[0] = 1_100_000
        presenter.keyboard_events.append(
            SimpleNamespace(
                key=SimpleNamespace(name="escape"),
                is_input=lambda: False,
                is_key_press=lambda: True,
                is_key_release=lambda: False,
            )
        )
        controller, keyboard = window.get_user_input_events().get_events()
        assert isinstance(controller, ControllerFrameEvent)
        assert controller.frame_id == 7
        assert int(controller.timestamp) == 50
        assert isinstance(keyboard, KeyboardUserInputEvent)
        assert keyboard.key == "Escape"
        assert keyboard.state is KeyboardInputState.PRESSED
        assert int(keyboard.timestamp) == 100
        assert window.get_user_input_events().get_events() == []

        presenter.should_close = True
        (close_event,) = window.get_user_input_events().get_events()
        assert isinstance(close_event, CloseUserInputEvent)
        assert window.get_user_input_events().get_events() == []
        window.write(_result())
        assert len(presenter.frames) == 2
    finally:
        window.close()
    window.close()
    assert source.start_thread != threading.get_ident()
    assert source.stop_count == 1
    assert not window._source_thread.is_alive()
    assert presenter.close_count == 1
    assert set(presenter.threads) == {threading.main_thread().ident}


@pytest.mark.parametrize("initial_window_size", [None, (1280, 940)])
def test_native_initial_window_size_preserves_video_and_later_manual_resizing(
    monkeypatch: pytest.MonkeyPatch,
    initial_window_size: tuple[int, int] | None,
) -> None:
    window, presenter, source = _window(
        monkeypatch, [1_000_000], initial_window_size=initial_window_size
    )
    native = cast(NativeWindowClientWindow, window.output_window)
    try:
        assert source.started.wait(timeout=1.0)
        session_desc = SessionDesc(video_width=640, video_height=480)
        window.open(session_desc)
        assert presenter.size == (initial_window_size or (640, 480))
        expected_resizes = [] if initial_window_size is None else [initial_window_size]
        assert presenter.resize_requests == expected_resizes
        assert (session_desc.video_width, session_desc.video_height) == (640, 480)
        assert native._session_desc is session_desc

        presenter.size = (1440, 1080)
        window.get_user_input_events()
        for scene in ("scene-two", "scene-three"):
            replacement = SessionDesc(
                video_width=640, video_height=480, metadata={"sample": scene}
            )
            window.open(replacement)
            assert presenter.size == (1440, 1080)
            assert presenter.resize_requests == expected_resizes
            assert native._session_desc is replacement
            assert native._presenter is presenter
        assert presenter.open_count == 1
        assert source.start_count == 1
    finally:
        window.close()
    assert presenter.close_count == 1
    assert source.stop_count == 1


def test_native_controller_session_handoff_reuses_resources_and_rebases_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now_ns = [1_000_000]
    window, presenter, source = _window(monkeypatch, now_ns)
    try:
        assert source.started.wait(timeout=1.0)
        session_desc = SessionDesc(video_width=2, video_height=2)
        window.open(session_desc)
        _buffer_observation(window, source, capture_ns=1_100_000, frame_id=1)
        now_ns[0] = 2_000_000
        window.open(session_desc)
        assert window.get_user_input_events().get_events() == []
        assert source.start_count == 1
        assert source.stop_count == 0
        assert presenter.open_count == 1
        assert presenter.close_count == 0

        _buffer_observation(window, source, capture_ns=1_900_000, frame_id=2)
        assert window.get_user_input_events().get_events() == []
        _buffer_observation(window, source, capture_ns=2_010_000, frame_id=3)
        (controller,) = window.get_user_input_events().get_events()
        assert isinstance(controller, ControllerFrameEvent)
        assert controller.frame_id == 3
        assert int(controller.timestamp) == 10
        window.write(_result())
        assert len(presenter.frames) == 1
    finally:
        window.close()
    assert source.stop_count == 1
    assert presenter.close_count == 1
