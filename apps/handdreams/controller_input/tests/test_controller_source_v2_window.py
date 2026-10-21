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

"""CPU tests for shared controller windows and remote Isaac input."""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import deque
from types import SimpleNamespace
from typing import Any, cast

import pytest
from flashdreams_controller_input import (
    quest_remote_v2_window as window_module,
)
from flashdreams_controller_input.controller_source_v2_window import (
    ControllerSourceWindow,
    controller_frame_event,
)
from flashdreams_controller_input.controller_types import Observation
from flashdreams_controller_input.events import (
    ControllerConnectionStatus,
    ControllerFrameEvent,
    ControllerStatusEvent,
)
from flashdreams_controller_input.pose import WRIST_PROXY_POSE_CONVENTION
from numpy import uint64

from flashdreams.api_v2.client_window import ClientWindowLink
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.user_input_event import (
    FocusUserInputEvent,
    KeyboardInputState,
    KeyboardUserInputEvent,
    ResetUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.webrtc_client_window import WebRTCClientWindow

pytestmark = pytest.mark.ci_cpu


def _wait(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition was not met before the test timeout")


def _key(key: str, state=KeyboardInputState.PRESSED) -> KeyboardUserInputEvent:
    return KeyboardUserInputEvent(timestamp=uint64(0), key=key, state=state)


def _controller_events(events: UserInputEvents) -> list[ControllerFrameEvent]:
    """Assert that every event is a controller frame and return the batch."""
    frames: list[ControllerFrameEvent] = []
    for event in events.get_events():
        assert isinstance(event, ControllerFrameEvent)
        frames.append(event)
    return frames


def _frame(
    capture_ns: int,
    frame_id: int,
    *,
    source_client_capture_ns: int | None = None,
) -> dict[str, Any]:
    frame = {
        "client_capture_ns": capture_ns,
        "source_frame_index": frame_id,
        "left": {
            "position": [-0.2, 1.0, -0.5],
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "trigger": 0.25,
            "grip": 0.75,
            "axes": [0.1, -0.2],
            "tracked": False,
            "battery": 0.8,
            "primary_button": True,
            "secondary_button": False,
        },
        "right": {
            "position": [0.2, 1.0, -0.5],
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "trigger": 0.9,
            "grip": 0.1,
            "axes": [-0.3, 0.4],
            "tracked": True,
            "battery": 0.6,
            "primary_button": False,
            "secondary_button": True,
        },
    }
    if source_client_capture_ns is not None:
        frame["source_client_capture_ns"] = source_client_capture_ns
    return frame


class _FakeWebRTCWindow:
    def __init__(self) -> None:
        self.server = SimpleNamespace(url="http://127.0.0.1:54321")
        self.events: list = []
        self.opened: list[SessionDesc] = []
        self.requested_sizes: list[tuple[int, int]] = []
        self.written: list[object] = []
        self.cursor_hidden: list[bool] = []
        self.cursor_locked: list[bool] = []
        self.links: list[tuple[ClientWindowLink, ...]] = []
        self.close_count = 0

    def open(self, session_desc: SessionDesc) -> None:
        self.opened.append(session_desc)

    def get_user_input_events(self) -> UserInputEvents:
        events, self.events = self.events, []
        return UserInputEvents(events)

    def request_hide_cursor(self, value: bool) -> None:
        self.cursor_hidden.append(value)

    def request_lock_cursor_to_window(self, value: bool) -> None:
        self.cursor_locked.append(value)

    def request_new_window_size(self, size: tuple[int, int]) -> None:
        assert self.opened
        self.requested_sizes.append(size)

    def request_links(self, links: tuple[ClientWindowLink, ...]) -> None:
        self.links.append(links)

    def write(self, result: object) -> None:
        self.written.append(result)

    def metrics_snapshot(self) -> dict[str, int]:
        return {"sent": len(self.written)}

    def close(self) -> None:
        self.close_count += 1


class _FakeQuestSource:
    name = "quest"
    status_message: str

    def __init__(self, *, start_error: BaseException | None = None) -> None:
        self.start_error = start_error
        self.started = threading.Event()
        self._observations: deque[Observation] = deque()
        self._lock = threading.Lock()
        self.poll_count = 0
        self.stop_count = 0
        self.restart_count = 0

    def start(self) -> None:
        self.started.set()
        if self.start_error is not None:
            raise self.start_error

    def poll(self) -> Observation | None:
        with self._lock:
            if not self._observations:
                return None
            self.poll_count += 1
            return self._observations.popleft()

    def push(self, observation: Observation) -> None:
        with self._lock:
            self._observations.append(observation)

    def stop(self) -> None:
        if self.stop_count == 0:
            self.stop_count = 1

    def restart(self) -> None:
        self.restart_count += 1


class _BlockingQuestSource(_FakeQuestSource):
    """Hold startup until window shutdown asks the source to stop."""

    def __init__(self) -> None:
        super().__init__()
        self.start_returned = threading.Event()
        self._release_start = threading.Event()

    def start(self) -> None:
        self.started.set()
        if not self._release_start.wait(timeout=2.0):
            raise TimeoutError("test source startup was not interrupted")
        self.start_returned.set()
        raise RuntimeError("source stopped while starting")

    def stop(self) -> None:
        super().stop()
        self._release_start.set()


class _FailingStopQuestSource(_FakeQuestSource):
    """Fail the first stop attempt so cleanup retry behavior is observable."""

    def __init__(self) -> None:
        super().__init__()
        self.stop_attempts = 0

    def stop(self) -> None:
        self.stop_attempts += 1
        if self.stop_attempts == 1:
            raise RuntimeError("source stop failed")
        super().stop()


@pytest.mark.parametrize("source_client_capture_ns", [123_500_000, None])
@pytest.mark.parametrize("emulation_metadata", [{}, {"is_emulated": True}])
def test_atomic_frame_becomes_one_raw_controller_event(
    source_client_capture_ns: int | None,
    emulation_metadata: dict,
) -> None:
    capture_ns = 2_000_000_000
    event = controller_frame_event(
        Observation(
            frame={
                **_frame(
                    capture_ns,
                    17,
                    source_client_capture_ns=source_client_capture_ns,
                ),
                **emulation_metadata,
            },
            capture_ns=capture_ns,
            source="quest",
        ),
        session_origin_ns=1_000_000_000,
        frame_id=17,
        pose_convention=WRIST_PROXY_POSE_CONVENTION,
        source_name="Isaac Teleop",
        client_capture_keys=("source_client_capture_ns", "client_capture_ns"),
    )

    assert isinstance(event, ControllerFrameEvent)
    assert event.action == "state"
    assert int(event.timestamp) == 1_000_000
    assert event.frame_id == 17
    assert event.client_capture_ns == (
        capture_ns if source_client_capture_ns is None else source_client_capture_ns
    )
    assert event.pose_convention == WRIST_PROXY_POSE_CONVENTION
    assert event.is_emulated is emulation_metadata.get("is_emulated", False)

    left, right = event.left, event.right
    assert left is not None and right is not None
    assert left.position == (-0.2, 1.0, -0.5)
    assert right.position == (0.2, 1.0, -0.5)
    assert left.orientation == right.orientation == (0.0, 0.0, 0.0, 1.0)
    assert left.axes == (0.1, -0.2)
    assert right.axes == (-0.3, 0.4)
    assert (left.trigger, left.grip) == (0.25, 0.75)
    assert (right.trigger, right.grip) == (0.9, 0.1)
    assert left.primary_button and not left.secondary_button
    assert not right.primary_button and right.secondary_button
    assert left.tracked is False
    assert right.tracked is True
    assert left.battery == 0.8
    assert right.battery == 0.6


def test_converter_preserves_raw_values_without_normalizing_or_aliasing() -> None:
    frame = _frame(2_000_000_000, 17)
    frame["left"].update(
        position=[123.25, -20.5, 0.0123],
        orientation=[0.1, 0.2, 0.3, 2.0],
        axes=[1.25, -0.75, 99.0],
        trigger=1.25,
        grip=-0.5,
    )
    event = controller_frame_event(
        Observation(frame=frame, capture_ns=2_000_000_000, source="quest"),
        session_origin_ns=1_000_000_000,
        frame_id=17,
        pose_convention=WRIST_PROXY_POSE_CONVENTION,
        source_name="Isaac Teleop",
    )
    assert event is not None
    left, right = event.left, event.right
    assert left is not None and right is not None
    assert left.position == (123.25, -20.5, 0.0123)
    assert left.orientation == (0.1, 0.2, 0.3, 2.0)
    assert left.axes == (1.25, -0.75)
    assert (left.trigger, left.grip) == (1.25, -0.5)
    snapshot = event.to_controller_frame()

    frame["left"]["position"][0] = 9.0
    frame["left"]["orientation"][3] = 1.0
    frame["left"]["axes"][0] = 0.0
    frame["left"].update(trigger=0.0, grip=0.0, primary_button=False, battery=None)
    frame["right"].clear()
    frame["client_capture_ns"] = 0
    assert event.to_controller_frame() == snapshot


def test_absent_source_frame_produces_explicit_disconnect() -> None:
    event = controller_frame_event(
        Observation(frame=None, capture_ns=2_000_000_000, source="quest"),
        session_origin_ns=1_000_000_000,
        frame_id=17,
        pose_convention=WRIST_PROXY_POSE_CONVENTION,
        source_name="Isaac Teleop",
    )
    assert event.action == "disconnected"
    assert int(event.timestamp) == 1_000_000
    assert event.left is None and event.right is None


@pytest.mark.parametrize("missing", [("left",), ("right",), ("left", "right")])
def test_missing_controllers_are_inert_untracked_snapshots(missing) -> None:
    frame = _frame(2_000_000_000, 17)
    for side in missing:
        frame[side] = None
    event = controller_frame_event(
        Observation(frame=frame, capture_ns=2_000_000_000, source="quest"),
        session_origin_ns=1_000_000_000,
        frame_id=17,
        pose_convention=WRIST_PROXY_POSE_CONVENTION,
        source_name="Isaac Teleop",
    )
    assert event is not None and event.left is not None and event.right is not None
    assert event.frame_id == 17 and event.client_capture_ns == 2_000_000_000
    for side in missing:
        hand = getattr(event, side)
        assert hand.tracked is False
        assert hand.position == (0.0, 0.0, 0.0)
        assert hand.orientation == (0.0, 0.0, 0.0, 1.0)
        assert hand.trigger == hand.grip == 0.0
        assert hand.axes == () and hand.battery is None
        assert not hand.primary_button and not hand.secondary_button
    mutable = event.to_controller_frame()
    mutable[missing[0]]["tracked"] = True
    mutable[missing[0]]["position"][0] = 42.0
    assert not getattr(event, missing[0]).tracked
    assert getattr(event, missing[0]).position == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("axes", "expected"),
    [
        (None, ()),
        ([], ()),
        ([0.25, -0.5], (0.25, -0.5)),
        ([0.25], (0.25,)),
        ("missing-hand", ()),
    ],
)
def test_atomic_controller_frames_preserve_optional_axes(
    axes: object, expected: tuple[float, ...]
) -> None:
    """Preserve source axes and wrist-proxy semantics through atomic events."""
    frame = _frame(2_000_000_000, 17)
    frame["left"]["tracked"] = True
    if axes == "missing-hand":
        frame["left"] = None
    elif axes is None:
        frame["left"].pop("axes")
    else:
        frame["left"]["axes"] = axes
    event = controller_frame_event(
        Observation(frame=frame, capture_ns=2_000_000_000, source="quest"),
        session_origin_ns=1_000_000_000,
        frame_id=17,
        pose_convention=WRIST_PROXY_POSE_CONVENTION,
        source_name="Isaac Teleop",
    )
    assert event is not None
    assert event.pose_convention == WRIST_PROXY_POSE_CONVENTION
    payload = event.to_controller_frame()
    assert payload["pose_convention"] == WRIST_PROXY_POSE_CONVENTION
    for side in ("left", "right"):
        assert payload[side].get("axes", []) == pytest.approx(
            expected if side == "left" else frame[side]["axes"]
        )
        assert payload[side]["tracked"] is (side == "right" or axes != "missing-hand")


@pytest.mark.parametrize("invalid", [[], "missing", 1])
def test_malformed_controller_is_rejected_at_the_atomic_boundary(invalid) -> None:
    frame = _frame(2_000_000_000, 17)
    frame["left"] = invalid
    with pytest.raises(ValueError, match="invalid left controller"):
        controller_frame_event(
            Observation(frame=frame, capture_ns=2_000_000_000, source="quest"),
            session_origin_ns=1_000_000_000,
            frame_id=17,
            pose_convention=WRIST_PROXY_POSE_CONVENTION,
            source_name="Isaac Teleop",
        )


def test_window_keeps_source_frames_and_survives_session_replacement() -> None:
    source = _FakeQuestSource()
    browser = _FakeWebRTCWindow()
    window = ControllerSourceWindow(
        output_window=cast(WebRTCClientWindow, browser),
        source=source,
        source_name="Isaac Teleop",
        pose_convention=WRIST_PROXY_POSE_CONVENTION,
        source_thread_name="test-controller-input",
        source_failure_message="Isaac Teleop input failed",
        client_capture_keys=("source_client_capture_ns", "client_capture_ns"),
    )
    assert source.started.wait(timeout=1.0)
    session_desc = SessionDesc()
    window.open(session_desc)

    first_capture_ns = time.perf_counter_ns() + 1_000_000
    source.push(
        Observation(
            frame=_frame(first_capture_ns, 1),
            capture_ns=first_capture_ns,
            source="quest",
        )
    )
    _wait(lambda: source.poll_count == 1)
    second_capture_ns = first_capture_ns + 1_000_000
    source.push(
        Observation(
            frame=_frame(second_capture_ns, 2),
            capture_ns=second_capture_ns,
            source="quest",
        )
    )
    _wait(lambda: source.poll_count == 2)
    browser.events.append(
        KeyboardUserInputEvent(
            timestamp=uint64(0),
            key="r",
            state=KeyboardInputState.PRESSED,
        )
    )

    events = window.get_user_input_events().get_events()
    controllers = [event for event in events if isinstance(event, ControllerFrameEvent)]
    assert isinstance(events[0], KeyboardUserInputEvent)
    assert len(controllers) == 2
    assert [event.frame_id for event in controllers] == [1, 2]
    assert all(
        event.left is not None and event.right is not None for event in controllers
    )

    stale_capture_ns = second_capture_ns + 1_000_000
    source.push(
        Observation(
            frame=_frame(stale_capture_ns, 3),
            capture_ns=stale_capture_ns,
            source="quest",
        )
    )
    _wait(lambda: source.poll_count == 3)
    window.open(session_desc)
    assert window.get_user_input_events().get_events() == []
    assert len(browser.opened) == 2
    assert source.started.is_set()

    window.request_hide_cursor(True)
    window.request_lock_cursor_to_window(True)
    links = (
        ClientWindowLink("Connect", "https://teleop.example/", (0.2, 0.5, 0.8, 0.6)),
    )
    window.request_links(links)
    window.request_links(())
    assert browser.cursor_hidden == [True]
    assert browser.cursor_locked == [True]
    assert browser.links == [links, ()]
    assert window.metrics_snapshot() == {"sent": 0}

    window.close()
    window.close()
    assert source.stop_count == 1
    assert browser.close_count == 1


@pytest.mark.parametrize("key", ["b", "B"])
def test_keyboard_b_restarts_only_once_per_press_and_preserves_active_input(key):
    source, browser = _FakeQuestSource(), _FakeWebRTCWindow()
    requests = []
    window = ControllerSourceWindow(
        output_window=cast(WebRTCClientWindow, browser),
        source=source,
        source_name="Isaac Teleop",
        pose_convention=WRIST_PROXY_POSE_CONVENTION,
        source_thread_name="test-controller-input",
        source_failure_message="Isaac Teleop input failed",
        restart_headset=lambda: requests.append("restart"),
    )
    try:
        assert source.started.wait(timeout=1.0)
        window.open(SessionDesc())
        now = time.perf_counter_ns()
        source.push(Observation(_frame(now, 1), now, "other active client"))
        _wait(lambda: source.poll_count == 1)
        escape = _key("Escape")
        browser.events.extend([_key(key), _key(key), escape])
        events = window.get_user_input_events().get_events()
        assert requests == ["restart"]
        assert events[0] is escape
        assert len(events) == 2
        assert isinstance(events[1], ControllerFrameEvent)
        assert events[1].action == "state" and events[1].frame_id == 1
        assert events[1].right is not None
        assert events[1].right.secondary_button  # Physical B is not keyboard B.
        assert source.stop_count == 0

        browser.events.append(_key(key))
        assert window.get_user_input_events().get_events() == []
        assert requests == ["restart"]
        # Scene replacement while B is held must not create another press.
        window.open(SessionDesc())
        browser.events.append(_key(key))
        assert window.get_user_input_events().get_events() == []
        assert requests == ["restart"]

        browser.events.extend(
            [_key(key, KeyboardInputState.RELEASED), _key(key.swapcase())]
        )
        assert window.get_user_input_events().get_events() == []
        assert requests == ["restart", "restart"]
        lost_focus = FocusUserInputEvent(timestamp=uint64(0), focused=False)
        browser.events.extend([lost_focus, _key(key)])
        assert window.get_user_input_events().get_events() == [lost_focus]
        assert requests == ["restart"] * 3
    finally:
        window.close()


def test_keyboard_b_remains_ordinary_input_without_headset_restart_callback():
    source, browser = _FakeQuestSource(), _FakeWebRTCWindow()
    window = ControllerSourceWindow(
        output_window=cast(WebRTCClientWindow, browser),
        source=source,
        source_name="Isaac Teleop",
        pose_convention=WRIST_PROXY_POSE_CONVENTION,
        source_thread_name="test-controller-input",
        source_failure_message="Isaac Teleop input failed",
    )
    try:
        window.open(SessionDesc())
        events = [_key("b"), _key("b", KeyboardInputState.RELEASED)]
        browser.events.extend(events)
        assert window.get_user_input_events().get_events() == events
    finally:
        window.close()


def test_source_queue_overflow_signals_discontinuity_before_retained_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flashdreams_controller_input import (
        controller_source_v2_window as controller_window,
    )

    monkeypatch.setattr(controller_window, "_MAX_PENDING_OBSERVATIONS", 3)
    source = _FakeQuestSource()
    window = ControllerSourceWindow(
        output_window=cast(WebRTCClientWindow, _FakeWebRTCWindow()),
        source=source,
        source_name="Isaac Teleop",
        pose_convention=WRIST_PROXY_POSE_CONVENTION,
        source_thread_name="test-controller-input",
        source_failure_message="Isaac Teleop input failed",
        client_capture_keys=("source_client_capture_ns", "client_capture_ns"),
    )
    try:
        assert source.started.wait(timeout=1.0)
        window.open(SessionDesc())
        origin_ns = time.perf_counter_ns() + 1_000_000
        for frame_id in range(5):
            capture_ns = origin_ns + frame_id * 1_000_000
            source.push(
                Observation(
                    frame=_frame(capture_ns, frame_id),
                    capture_ns=capture_ns,
                    source="quest",
                )
            )
        _wait(lambda: source.poll_count == 5)
        assert len(window._observations) <= 3
        events = _controller_events(window.get_user_input_events())
        assert [event.action for event in events] == [
            "disconnected",
            "state",
            "state",
        ]
        assert [event.frame_id for event in events[1:]] == [3, 4]
        assert window.get_user_input_events().get_events() == []
    finally:
        window.close()


@pytest.mark.parametrize("consume_before_silence", [True, False])
@pytest.mark.parametrize("partial", [False, True])
def test_silent_live_source_emits_one_disconnect_and_recovers(
    consume_before_silence: bool,
    partial: bool,
) -> None:
    source = _FakeQuestSource()
    browser = _FakeWebRTCWindow()
    silence_clock_ns = [0]
    window = ControllerSourceWindow(
        output_window=cast(WebRTCClientWindow, browser),
        source=source,
        source_name="Isaac Teleop",
        pose_convention=WRIST_PROXY_POSE_CONVENTION,
        source_thread_name="test-controller-input",
        source_failure_message="Isaac Teleop input failed",
        client_capture_keys=("source_client_capture_ns", "client_capture_ns"),
        _source_silence_timeout_s=1.0,
        _silence_clock_ns=lambda: silence_clock_ns[0],
    )
    assert source.started.wait(timeout=1.0)
    window.open(SessionDesc())

    first_capture_ns = time.perf_counter_ns() + 1_000_000
    first_frame = _frame(first_capture_ns, 10)
    if partial:
        first_frame["right"] = None
    source.push(
        Observation(
            frame=first_frame,
            capture_ns=first_capture_ns,
            source="quest",
        )
    )
    _wait(lambda: source.poll_count == 1)
    if consume_before_silence:
        first = _controller_events(window.get_user_input_events())
        assert [event.action for event in first] == ["state"]
        assert first[0].right is not None
        assert first[0].right.tracked is not partial

    silence_clock_ns[0] = 1_000_000_000
    disconnected = window.get_user_input_events().get_events()
    assert len(disconnected) == 1
    assert isinstance(disconnected[0], ControllerFrameEvent)
    assert disconnected[0].action == "disconnected"
    assert disconnected[0].left is None and disconnected[0].right is None
    assert window.get_user_input_events().get_events() == []

    second_capture_ns = first_capture_ns + 1_000_000
    source.push(
        Observation(
            frame=_frame(second_capture_ns, 11),
            capture_ns=second_capture_ns,
            source="quest",
        )
    )
    _wait(lambda: source.poll_count == 2)
    recovered = _controller_events(window.get_user_input_events())
    assert [event.action for event in recovered] == ["state"]
    assert {event.frame_id for event in recovered} == {11}
    assert recovered[0].right is not None
    assert recovered[0].right.tracked is True

    window.close()


def test_asynchronous_source_failure_is_reported_and_cleanup_still_runs() -> None:
    source = _FakeQuestSource(start_error=TimeoutError("no controller pair"))
    browser = _FakeWebRTCWindow()
    window = ControllerSourceWindow(
        output_window=cast(WebRTCClientWindow, browser),
        source=source,
        source_name="Isaac Teleop",
        pose_convention=WRIST_PROXY_POSE_CONVENTION,
        source_thread_name="test-controller-input",
        source_failure_message="Isaac Teleop input failed",
        client_capture_keys=("source_client_capture_ns", "client_capture_ns"),
    )
    window.open(SessionDesc())
    _wait(lambda: window._source_failure is not None)

    with pytest.raises(RuntimeError, match="Isaac Teleop input failed") as raised:
        window.get_user_input_events()
    assert isinstance(raised.value.__cause__, TimeoutError)

    window.close()
    assert source.stop_count == 1
    assert browser.close_count == 1


def test_close_interrupts_a_source_blocked_during_startup() -> None:
    source = _BlockingQuestSource()
    browser = _FakeWebRTCWindow()
    window = ControllerSourceWindow(
        output_window=cast(WebRTCClientWindow, browser),
        source=source,
        source_name="Isaac Teleop",
        pose_convention=WRIST_PROXY_POSE_CONVENTION,
        source_thread_name="test-controller-input",
        source_failure_message="Isaac Teleop input failed",
        client_capture_keys=("source_client_capture_ns", "client_capture_ns"),
    )
    assert source.started.wait(timeout=1.0)

    window.close()

    assert source.start_returned.is_set()
    assert not window._source_thread.is_alive()
    assert window._source_failure is None
    assert source.stop_count == 1
    assert browser.close_count == 1


def test_status_updates_while_startup_waits_and_replays_once_per_session() -> None:
    source = _BlockingQuestSource()
    browser = _FakeWebRTCWindow()
    current = [ControllerConnectionStatus(headset="Press the power button")]
    window = ControllerSourceWindow(
        output_window=cast(WebRTCClientWindow, browser),
        source=source,
        source_name="Isaac Teleop",
        pose_convention=WRIST_PROXY_POSE_CONVENTION,
        source_thread_name="test-controller-input",
        source_failure_message="Isaac Teleop input failed",
        status_provider=lambda: current[0],
    )
    try:
        assert source.started.wait(timeout=1.0)
        window.open(SessionDesc())
        (event,) = window.get_user_input_events().get_events()
        assert isinstance(event, ControllerStatusEvent)
        assert event.status is current[0]
        assert event.timestamp >= 0
        assert window.get_user_input_events().get_events() == []
        current[0] = ControllerConnectionStatus(
            headset="Press the power button", web_client="Ready to connect"
        )
        (event,) = window.get_user_input_events().get_events()
        assert isinstance(event, ControllerStatusEvent)
        assert event.status is current[0]
        assert window.get_user_input_events().get_events() == []
        window.open(SessionDesc())
        (event,) = window.get_user_input_events().get_events()
        assert isinstance(event, ControllerStatusEvent)
        assert event.status is current[0]
        assert window.get_user_input_events().get_events() == []
        assert source.poll_count == 0
        assert not source.start_returned.is_set()
    finally:
        window.close()
    assert source.stop_count == 1
    assert not window._source_thread.is_alive()


def test_close_retries_failed_source_cleanup_without_reclosing_webrtc() -> None:
    source = _FailingStopQuestSource()
    browser = _FakeWebRTCWindow()
    window = ControllerSourceWindow(
        output_window=cast(WebRTCClientWindow, browser),
        source=source,
        source_name="Isaac Teleop",
        pose_convention=WRIST_PROXY_POSE_CONVENTION,
        source_thread_name="test-controller-input",
        source_failure_message="Isaac Teleop input failed",
        client_capture_keys=("source_client_capture_ns", "client_capture_ns"),
    )
    assert source.started.wait(timeout=1.0)

    with pytest.raises(RuntimeError, match="source stop failed"):
        window.close()
    assert not window._source_thread.is_alive()
    assert browser.close_count == 1

    window.close()
    window.close()
    assert source.stop_attempts == 2
    assert source.stop_count == 1
    assert browser.close_count == 1


def test_explicit_disconnect_preserves_boundary_and_does_not_repeat_on_silence() -> (
    None
):
    source = _FakeQuestSource()
    clock = [0]
    window = ControllerSourceWindow(
        output_window=cast(WebRTCClientWindow, _FakeWebRTCWindow()),
        source=source,
        source_name="Isaac Teleop",
        pose_convention=WRIST_PROXY_POSE_CONVENTION,
        source_thread_name="test-controller-input",
        source_failure_message="Isaac Teleop input failed",
        client_capture_keys=("source_client_capture_ns", "client_capture_ns"),
        _silence_clock_ns=lambda: clock[0],
    )
    try:
        assert source.started.wait(timeout=1.0)
        window.open(SessionDesc())
        now = time.perf_counter_ns()
        for frame in (_frame(now, 10), None, None, _frame(now, 11), None):
            source.push(Observation(frame=frame, capture_ns=now, source="quest"))
        _wait(lambda: source.poll_count == 5)
        events = _controller_events(window.get_user_input_events())
        assert [event.action for event in events] == [
            "state",
            "disconnected",
            "state",
            "disconnected",
        ]
        assert [event.frame_id for event in events if event.action == "state"] == [
            10,
            11,
        ]
        assert len({int(event.timestamp) for event in events}) == 1
        clock[0] = 2_000_000_000
        assert window.get_user_input_events().get_events() == []
        source.push(Observation(frame=_frame(now, 12), capture_ns=now, source="quest"))
        _wait(lambda: source.poll_count == 6)
        assert [
            event.frame_id
            for event in _controller_events(window.get_user_input_events())
        ] == [12]
        window.open(SessionDesc())
        assert window.get_user_input_events().get_events() == []
        assert source.stop_count == 0
    finally:
        window.close()


@pytest.mark.parametrize(
    "mode",
    [
        window_module.QUEST_REMOTE_WEBRTC_MODE,
        window_module.QUEST_REMOTE_NATIVE_WINDOW_MODE,
    ],
    ids=["webrtc", "native-window"],
)
def test_shared_mode_keeps_remote_frames_atomic_and_both_sources_alive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    mode,
) -> None:
    from flashdreams_controller_input import quest_usb_local, quest_webxr
    from flashdreams_controller_input.isaac_teleop_v2_window import (
        ISAAC_TELEOP_WEBRTC_MODE,
    )

    parser = argparse.ArgumentParser()
    window_module.QUEST_REMOTE_WEBRTC_MODE.add_arguments(parser)
    window_module.QUEST_REMOTE_NATIVE_WINDOW_MODE.add_arguments(parser)
    ISAAC_TELEOP_WEBRTC_MODE.add_arguments(parser)
    # In the unified CLI every mode contributes arguments. Non-remote modes
    # must remain parseable without TLS files or a token.
    parsed = parser.parse_args(["--isaac-teleop-accept-eula"])
    parsed.host, parsed.port = "127.0.0.1", 8081
    monkeypatch.delenv("FLASHDREAMS_CONTROLLER_TOKEN", raising=False)
    mode.check_arguments(parsed)
    parsed.controller_cert = tmp_path / "server.crt"
    parsed.controller_key = tmp_path / "server.key"
    parsed.controller_cert.touch()
    parsed.controller_key.touch()
    monkeypatch.delenv("FLASHDREAMS_CONTROLLER_TOKEN", raising=False)
    with pytest.raises(ValueError, match="FLASHDREAMS_CONTROLLER_TOKEN"):
        mode.check_arguments(parsed)
    monkeypatch.setenv("FLASHDREAMS_CONTROLLER_TOKEN", "test-token-" * 4)
    parsed.controller_port = 0
    with pytest.raises(ValueError, match="controller-port"):
        mode.check_arguments(parsed)
    parsed.controller_port = 8445
    mode.check_arguments(parsed)

    source = _FakeQuestSource()
    received = {}

    def remote_source(**kwargs):
        received.update(kwargs)
        return source

    monkeypatch.setitem(
        sys.modules,
        "flashdreams_controller_input.remote_quest",
        SimpleNamespace(RemoteQuestSource=remote_source),
    )

    class LocalSource(_FakeQuestSource):
        ready_for_clients = True

        def start(self, *, wait_for_frame=True):
            assert wait_for_frame is False
            super().start()

    local = LocalSource()
    monkeypatch.setattr(window_module, "IsaacTeleopSource", lambda **_: local)
    headset = _FakeQuestSource()
    headset.status_message = "Press the power button"
    monkeypatch.setattr(quest_webxr, "QuestWebXRProducer", lambda *_, **__: headset)
    monkeypatch.setattr(quest_usb_local, "cached_web_client_dir", lambda: tmp_path)
    browser = _FakeWebRTCWindow()

    def presentation_mode(name, **_):
        assert name == mode.output_mode
        return SimpleNamespace(create=lambda _: browser)

    monkeypatch.setattr(window_module, "client_window_mode", presentation_mode)
    window = mode.create(parsed)
    try:
        assert source.started.wait(timeout=1.0)
        assert local.started.wait(timeout=1.0)
        assert received == dict(
            host="127.0.0.1",
            port=8445,
            certfile=parsed.controller_cert,
            keyfile=parsed.controller_key,
            token="test-token-" * 4,
            wrist_offset_m=parsed.isaac_teleop_wrist_offset_m,
        )
        window.open(SessionDesc())
        expected_sizes = [(1280, 940)] if mode.output_mode == "native-window" else []
        assert browser.requested_sizes == expected_sizes
        assert browser.opened[-1] == SessionDesc()
        now = time.perf_counter_ns()
        frame = _frame(now, 22, source_client_capture_ns=123)
        frame["left"]["tracked"] = True
        source.push(
            Observation(
                frame=frame,
                capture_ns=now,
                source="remote Quest",
            )
        )
        _wait(lambda: source.poll_count == 1)
        events = window.get_user_input_events().get_events()
        (event,) = [
            event for event in events if isinstance(event, ControllerFrameEvent)
        ]
        (status,) = [
            event for event in events if isinstance(event, ControllerStatusEvent)
        ]
        assert status.status.headset == "Press the power button"
        assert status.status.active_input == "Remote headset"
        assert event.client_capture_ns == 123
        assert event.pose_convention == WRIST_PROXY_POSE_CONVENTION
        assert event.frame_id == 22
        browser.events.append(_key("b"))
        assert window.get_user_input_events().get_events() == []
        assert headset.restart_count == 1
        assert source.stop_count == local.stop_count == headset.stop_count == 0
        window.open(SessionDesc())
        assert browser.requested_sizes == expected_sizes
        assert source.stop_count == 0 and local.stop_count == 0
        (status,) = window.get_user_input_events().get_events()
        assert isinstance(status, ControllerStatusEvent)
        assert status.status.active_input == "Remote headset"
    finally:
        window.close()
    assert source.stop_count == 1 and local.stop_count == 1 and browser.close_count == 1
    assert headset.stop_count == 1


def test_new_tracking_origin_resets_using_standard_runtime_events() -> None:
    from flashdreams.runtime_v2.event_buffer import EventBuffer

    source = _FakeQuestSource()
    window = ControllerSourceWindow(
        output_window=cast(WebRTCClientWindow, _FakeWebRTCWindow()),
        source=source,
        source_name="Isaac Teleop",
        pose_convention=WRIST_PROXY_POSE_CONVENTION,
        source_thread_name="test-controller-input",
        source_failure_message="Isaac Teleop input failed",
        client_capture_keys=("source_client_capture_ns", "client_capture_ns"),
    )
    try:
        assert source.started.wait(timeout=1.0)
        window.open(SessionDesc())
        now = time.perf_counter_ns()
        for generation, frame_id in [(0, 10), (1, 11), (0, 12), (1, 13)]:
            source.push(
                Observation(
                    frame=_frame(now, frame_id),
                    capture_ns=now,
                    source="quest",
                    generation=generation,
                )
            )
        _wait(lambda: source.poll_count == 4)
        events = window.get_user_input_events()
        received = events.get_events()
        assert [type(event) for event in received] == [
            ControllerFrameEvent,
            ControllerFrameEvent,
            ResetUserInputEvent,
            ControllerFrameEvent,
            ControllerFrameEvent,
        ]
        assert isinstance(received[1], ControllerFrameEvent)
        assert received[1].action == "disconnected"
        assert [
            event.frame_id
            for event in received
            if isinstance(event, ControllerFrameEvent) and event.action == "state"
        ] == [10, 11, 13]
        buffer = EventBuffer()
        buffer.register(0)
        buffer.append(events)
        _, generation = buffer.read(0)
        assert generation == 1
        assert window.get_user_input_events().get_events() == []
        # Scene replacement already initializes a new session; do not reset it
        # merely because its first controller sample has a nonzero generation.
        window.open(SessionDesc())
        now = time.perf_counter_ns()
        source.push(
            Observation(
                frame=_frame(now, 14), capture_ns=now, source="quest", generation=1
            )
        )
        _wait(lambda: source.poll_count == 5)
        received_frames = _controller_events(window.get_user_input_events())
        assert len(received_frames) == 1 and received_frames[0].frame_id == 14
    finally:
        window.close()
