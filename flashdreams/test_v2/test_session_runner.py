# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU test for the v2 session loop, independent of any application."""

import logging
import threading

import pytest
import torch
from numpy import uint64

from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.api_v2.session import ISession
from flashdreams.api_v2.user_input_event_data import UserInputEventData
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.session_runner import WhenFull, run_session
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    CloseUserInputEventData,
    KeyboardUserInputEventData,
    ResetUserInputEventData,
    UserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu

_IO_THREAD_NAME = "flashdreams-io"
"""Name the runner gives its I/O thread."""

_RUNNER_LOGGER = "flashdreams.runtime_v2.session_runner"
"""Logger the runner reports discarded results on."""


class CallLog:
    """Record calls made from either thread, with the thread that made them."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._calls: list[tuple[str, str]] = []

    def record(self, call: str) -> None:
        """Append one call and the name of the calling thread."""
        with self._lock:
            self._calls.append((call, threading.current_thread().name))

    @property
    def calls(self) -> list[str]:
        """Return the calls in the order they were made."""
        with self._lock:
            return [call for call, _ in self._calls]

    def threads_for(self, call: str) -> set[str]:
        """Return the names of the threads that made ``call``."""
        with self._lock:
            return {thread for made, thread in self._calls if made == call}


class FakeSession(ISession):
    """Emit one blank frame per step and record what the runner asks for."""

    def __init__(
        self,
        session_desc: SessionDesc,
        log: CallLog,
        *,
        fail_at: int | None = None,
        release_writes: threading.Event | None = None,
        release_writes_at: int | None = None,
    ) -> None:
        """
        Args:
            session_desc: Description this session reports as resolved.
            log: Shared log both fakes record into.
            fail_at: Step index to raise at, for exercising cleanup on failure.
            release_writes: Event to set once ``release_writes_at`` has been
                generated, for holding the window back until then.
            release_writes_at: Step index that sets ``release_writes``.
        """
        self._session_desc = session_desc
        self._log = log
        self._fail_at = fail_at
        self._release_writes = release_writes
        self._release_writes_at = release_writes_at
        self.observed_events: list[UserInputEvents] = []

    def init(self) -> None:
        self._log.record("session.init")

    @property
    def session_desc(self) -> SessionDesc:
        return self._session_desc

    def step(self, step_index: int, events: UserInputEvents) -> StepResult:
        self._log.record(f"session.step({step_index})")
        self.observed_events.append(events)
        if step_index == self._fail_at:
            raise RuntimeError("step failed")
        if self._release_writes is not None and step_index == self._release_writes_at:
            self._release_writes.set()
        return StepResult(
            step_index=step_index,
            output=torch.zeros((1, 3, 1, 1, 1), dtype=torch.float32),
            frame_count=1,
            output_layout=self._session_desc.output_layout,
        )

    def step_ui(self, events: UserInputEvents) -> None:
        self._log.record("session.step_ui")

    def reset(self) -> None:
        self._log.record("session.reset")

    def close(self) -> None:
        self._log.record("session.close")


class RecordingClientWindow(IClientWindow):
    """Report scripted input and record every call the runner makes."""

    def __init__(
        self,
        log: CallLog,
        scripted_events: list[UserInputEvents] | None = None,
        *,
        fail_to_open: bool = False,
        hold_writes: threading.Event | None = None,
    ) -> None:
        """
        Args:
            log: Shared log both fakes record into.
            scripted_events: Events to report, one entry per poll. Polls past the
                end of the script report nothing.
            fail_to_open: Whether :meth:`open` raises.
            hold_writes: Event that has to be set before a write completes, for
                holding this window behind generation on purpose.
        """
        self._log = log
        self._scripted = list(scripted_events or [])
        self._fail_to_open = fail_to_open
        self._hold_writes = hold_writes
        self._lock = threading.Lock()
        self.session_desc: SessionDesc | None = None
        self.results: list[StepResult] = []

    def get_user_input_events(self) -> UserInputEvents:
        self._log.record("window.get_user_input_events")
        with self._lock:
            if self._scripted:
                return self._scripted.pop(0)
        return UserInputEvents([])

    def open(self, session_desc: SessionDesc) -> None:
        self._log.record("window.open")
        if self._fail_to_open:
            raise RuntimeError("open failed")
        self.session_desc = session_desc

    def write(self, result: StepResult) -> None:
        if self._hold_writes is not None:
            self._hold_writes.wait()
        self._log.record(f"window.write({result.step_index})")
        self.results.append(result)

    def close(self) -> None:
        self._log.record("window.close")


def _session_desc() -> SessionDesc:
    return SessionDesc(
        output_layout=VideoTensorLayout.bcthw,
        frames_per_second_for_ui=100,
        frames_per_second_for_step=1,
        video_width=1,
        video_height=1,
    )


def _key_event() -> UserInputEvents:
    return UserInputEvents(
        [
            UserInputEvent(
                timestamp=uint64(0),
                event_data=KeyboardUserInputEventData(key="a", pressed=True),
            )
        ]
    )


def _lifecycle_event(event_data: UserInputEventData) -> UserInputEvents:
    return UserInputEvents([UserInputEvent(timestamp=uint64(0), event_data=event_data)])


def test_run_session_presents_every_step_in_order() -> None:
    log = CallLog()
    session = FakeSession(_session_desc(), log)
    window = RecordingClientWindow(log)

    run_session(session, window, steps=3)

    assert [result.step_index for result in window.results] == [0, 1, 2]
    steps = [call for call in log.calls if call.startswith("session.step(")]
    assert steps == ["session.step(0)", "session.step(1)", "session.step(2)"]


def test_run_session_opens_before_writing_and_closes_after() -> None:
    log = CallLog()
    session = FakeSession(_session_desc(), log)
    window = RecordingClientWindow(log)

    run_session(session, window, steps=2)

    calls = log.calls
    # Interleaving between the threads varies, but these orderings cannot.
    assert calls[0] == "session.init"
    assert calls.index("window.open") < calls.index("window.write(0)")
    assert calls[-2:] == ["window.close", "session.close"]


def test_run_session_touches_the_window_only_from_the_io_thread() -> None:
    log = CallLog()
    session = FakeSession(_session_desc(), log)
    window = RecordingClientWindow(log)

    run_session(session, window, steps=2)

    # A native window has to be pumped by the thread that opened it.
    for call in ("window.open", "window.get_user_input_events", "window.close"):
        assert log.threads_for(call) == {_IO_THREAD_NAME}
    assert log.threads_for("window.write(0)") == {_IO_THREAD_NAME}


def test_run_session_calls_step_ui_on_the_io_thread() -> None:
    log = CallLog()
    session = FakeSession(_session_desc(), log)
    window = RecordingClientWindow(log)

    run_session(session, window, steps=2)

    assert log.threads_for("session.step_ui") == {_IO_THREAD_NAME}
    assert log.threads_for("session.step(0)") == {threading.current_thread().name}


def test_run_session_opens_window_with_the_resolved_session_desc() -> None:
    log = CallLog()
    resolved = _session_desc()
    session = FakeSession(resolved, log)
    window = RecordingClientWindow(log)

    run_session(session, window, steps=1)

    assert window.session_desc is resolved


def test_run_session_gives_the_first_step_input_already_collected() -> None:
    log = CallLog()
    session = FakeSession(_session_desc(), log)
    window = RecordingClientWindow(log, [_key_event()])

    run_session(session, window, steps=2)

    # The I/O thread collects once before generation starts, so input the window
    # already holds is not missed by step 0.
    assert len(session.observed_events[0].get_events()) == 1


def test_run_session_stops_when_the_window_reports_a_close() -> None:
    log = CallLog()
    session = FakeSession(_session_desc(), log)
    window = RecordingClientWindow(log, [_lifecycle_event(CloseUserInputEventData())])

    # No step count at all: the close is the only thing that ends this run.
    run_session(session, window, steps=None)

    assert "session.step(0)" not in log.calls
    assert log.calls[-2:] == ["window.close", "session.close"]


def test_run_session_resets_the_session_and_the_step_index() -> None:
    log = CallLog()
    session = FakeSession(_session_desc(), log)
    window = RecordingClientWindow(log, [_lifecycle_event(ResetUserInputEventData())])

    run_session(session, window, steps=2)

    # step_ui is the I/O thread's and can land anywhere among these.
    calls = [call for call in log.calls if call.startswith("session.reset")] + [
        call for call in log.calls if call.startswith("session.step(")
    ]
    assert calls == ["session.reset", "session.step(0)", "session.step(1)"]
    assert log.calls.index("session.reset") < log.calls.index("session.step(0)")
    # A reset restarts the index without granting extra steps.
    assert [result.step_index for result in window.results] == [0, 1]


def test_run_session_closes_a_session_that_failed_to_init() -> None:
    log = CallLog()

    class FailingSession(FakeSession):
        def init(self) -> None:
            super().init()
            raise RuntimeError("init failed")

    session = FailingSession(_session_desc(), log)
    window = RecordingClientWindow(log)

    with pytest.raises(RuntimeError, match="init failed"):
        run_session(session, window, steps=1)

    # A session that got halfway through starting still has to be released, and
    # the window is never opened for a session that cannot run.
    assert log.calls == ["session.init", "session.close"]


def test_run_session_gives_the_step_after_a_reset_the_whole_batch() -> None:
    log = CallLog()
    session = FakeSession(_session_desc(), log)
    held_key = _key_event().get_events()[0]
    window = RecordingClientWindow(
        log,
        [
            UserInputEvents(
                [
                    held_key,
                    UserInputEvent(
                        timestamp=uint64(1), event_data=ResetUserInputEventData()
                    ),
                ]
            )
        ],
    )

    run_session(session, window, steps=1)

    # Events are edges, so a key held down when the client restarts is still held
    # after: the batch is not split at the reset, and the edge that said so is
    # what carries the state.
    assert held_key in session.observed_events[0].get_events()


def test_run_session_presents_every_result_when_blocking() -> None:
    log = CallLog()
    session = FakeSession(_session_desc(), log)
    window = RecordingClientWindow(log)

    # Room for one result and three more coming, so generation has to wait for
    # the window rather than run ahead of it.
    run_session(session, window, steps=4, max_pending=1, when_full=WhenFull.BLOCK)

    assert [result.step_index for result in window.results] == [0, 1, 2, 3]


def test_run_session_drops_the_oldest_waiting_result() -> None:
    log = CallLog()
    # Hold the window until every step is generated, so which results are dropped
    # does not depend on how the two threads happen to be scheduled.
    generated = threading.Event()
    session = FakeSession(
        _session_desc(), log, release_writes=generated, release_writes_at=3
    )
    window = RecordingClientWindow(log, hold_writes=generated)

    run_session(session, window, steps=4, max_pending=1, when_full=WhenFull.DROP_OLDEST)

    presented = [result.step_index for result in window.results]
    # Room for one result and four generated behind a window that cannot write
    # until the end, so what is stale is lost and the newest always arrives.
    assert presented == sorted(presented)
    assert len(presented) < 4
    assert presented[-1] == 3


def test_run_session_discards_results_generated_before_a_reset(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log = CallLog()
    session = FakeSession(_session_desc(), log)
    window = RecordingClientWindow(
        log,
        [
            UserInputEvents([]),
            _lifecycle_event(ResetUserInputEventData()),
            _lifecycle_event(CloseUserInputEventData()),
        ],
    )

    with caplog.at_level(logging.INFO, logger=_RUNNER_LOGGER):
        run_session(session, window, steps=None, max_pending=2)

    # The client asked to start over, so what the abandoned generation produced is
    # thrown away rather than presented after the restart. The runner logs this only
    # when it discarded at least one result, and the count it reports depends on how
    # far generation got before the reset landed.
    assert any("before a reset" in record.getMessage() for record in caplog.records)


def test_run_session_rejects_a_pending_bound_of_zero() -> None:
    log = CallLog()
    session = FakeSession(_session_desc(), log)
    window = RecordingClientWindow(log)

    with pytest.raises(ValueError, match="max_pending"):
        run_session(session, window, steps=1, max_pending=0)

    assert log.calls == []


def test_run_session_with_no_steps_still_opens_and_closes() -> None:
    log = CallLog()
    session = FakeSession(_session_desc(), log)
    window = RecordingClientWindow(log)

    run_session(session, window, steps=0)

    assert "window.open" in log.calls
    assert log.calls[-2:] == ["window.close", "session.close"]
    assert window.results == []


def test_run_session_closes_both_when_a_step_raises() -> None:
    log = CallLog()
    session = FakeSession(_session_desc(), log, fail_at=1)
    window = RecordingClientWindow(log)

    with pytest.raises(RuntimeError, match="step failed"):
        run_session(session, window, steps=4)

    # A failed step must not leak the window or the session, and must not be
    # presented as a result.
    assert log.calls[-2:] == ["window.close", "session.close"]
    assert [result.step_index for result in window.results] == [0]


def test_run_session_reports_a_window_that_fails_to_open() -> None:
    log = CallLog()
    session = FakeSession(_session_desc(), log)
    window = RecordingClientWindow(log, fail_to_open=True)

    with pytest.raises(RuntimeError, match="open failed"):
        run_session(session, window, steps=2)

    # Generation never starts, but the session is still closed.
    assert "session.step(0)" not in log.calls
    assert log.calls[-1] == "session.close"
