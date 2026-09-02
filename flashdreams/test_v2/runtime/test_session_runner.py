# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU test for the v2 session loop, independent of any application."""

import logging
import queue
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
import torch
from numpy import uint64

from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.api_v2.loop import IModelLoop, IUILoop, invoke_async
from flashdreams.api_v2.session import ISession
from flashdreams.api_v2.user_input_event import UserInputEvent
from flashdreams.runtime_v2.blit_model_output_to_screen_loop import (
    BlitModelOutputToScreenLoop,
)
from flashdreams.runtime_v2.event_buffer import EventBuffer
from flashdreams.runtime_v2.presentation_manager import (
    _PRESENTATION_DRAIN_MARGIN,
    PresentationManager,
    _PresentationClock,
)
from flashdreams.runtime_v2.session_desc import (
    BackpressureMode,
    PresentationMode,
    SessionDesc,
)
from flashdreams.runtime_v2.session_runner import run_session
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    CloseUserInputEvent,
    KeyboardInputState,
    KeyboardUserInputEvent,
    ResetUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu

_STEP_THREAD_NAME = "flashdreams-model-generation-thread"
"""Name the runner gives its model-generation thread."""

_RUNNER_LOGGER = "flashdreams.runtime_v2.session_runner"
"""Logger the runner reports discarded results on."""


def test_session_modes_are_independent() -> None:
    assert list(BackpressureMode) == [
        BackpressureMode.BLOCK,
        BackpressureMode.DROP_OLDEST,
    ]
    assert list(PresentationMode) == [
        PresentationMode.ON_DEMAND,
        PresentationMode.CONTINUOUS,
    ]
    assert [mode.value for mode in PresentationMode] == ["on_demand", "continuous"]
    for legacy_value in ("only_present_new", "only_present_newest"):
        with pytest.raises(ValueError):
            PresentationMode(legacy_value)
    assert SessionDesc().backpressure_mode is BackpressureMode.BLOCK
    assert SessionDesc().presentation_mode is PresentationMode.CONTINUOUS


def test_presentation_clock_paces_frames_and_reanchors_after_a_stall() -> None:
    clock = _PresentationClock(frames_per_second=4)

    assert clock.is_due(now=1.0, generation=0)
    clock.mark_advanced(now=1.0)
    assert not clock.is_due(now=1.24, generation=0)
    assert clock.is_due(now=1.25, generation=0)

    clock.mark_advanced(now=2.0)
    assert not clock.is_due(now=2.24, generation=0)
    assert clock.is_due(now=2.25, generation=0)
    assert clock.is_due(now=2.0, generation=1)


def _observe_model_step(
    clock: _PresentationClock,
    completed_at: float,
    frame_count: int,
    elapsed_s: float,
    generation: int = 0,
) -> None:
    clock.observe_model_output(
        now=completed_at,
        generation=generation,
        frame_count=frame_count,
        step_elapsed_s=elapsed_s,
    )


def _presentation_fps(model_fps: float) -> float:
    return model_fps / _PRESENTATION_DRAIN_MARGIN


def test_presentation_clock_uses_recent_model_fps() -> None:
    clock = _PresentationClock(frames_per_second=16)

    _observe_model_step(clock, 1.0, 12, 0.9)
    assert clock.frames_per_second == 16

    _observe_model_step(clock, 1.9, 12, 0.9)
    assert clock.frames_per_second == pytest.approx(_presentation_fps(12 / 0.9))

    assert clock.is_due(now=2.0, generation=0)
    clock.mark_advanced(now=2.0)
    assert not clock.is_due(now=2.067, generation=0)
    assert clock.is_due(now=2.068, generation=0)


def test_presentation_clock_clamps_model_fps_to_ui_fps() -> None:
    clock = _PresentationClock(
        frames_per_second=30,
        maximum_frames_per_second=60,
    )

    _observe_model_step(clock, 1.0, 120, 1.0)
    _observe_model_step(clock, 2.0, 120, 1.0)

    assert clock.frames_per_second == 60


def test_presentation_clock_allows_backlog_before_paced_deadline() -> None:
    clock = _PresentationClock(
        frames_per_second=30,
        maximum_frames_per_second=60,
    )

    assert clock.is_due(now=1.0, generation=0)
    clock.mark_advanced(now=1.0)
    assert not clock.is_due(now=1.016, generation=0)

    assert clock.is_due(now=1.016, generation=0, backlog=True)
    clock.mark_advanced(now=1.016, backlog=True)

    assert not clock.is_due(now=1.032, generation=0)
    assert clock.is_due(now=1.033, generation=0)


def test_presentation_clock_limits_estimate_to_recent_two_seconds() -> None:
    clock = _PresentationClock(frames_per_second=30)

    _observe_model_step(clock, 0.0, 10, 1.0)
    _observe_model_step(clock, 1.0, 10, 1.0)
    assert clock.frames_per_second == pytest.approx(_presentation_fps(10.0))

    _observe_model_step(clock, 2.0, 20, 1.0)
    assert clock.frames_per_second == pytest.approx(_presentation_fps(15.0))

    _observe_model_step(clock, 3.0, 20, 1.0)
    assert clock.frames_per_second == pytest.approx(_presentation_fps(20.0))


def test_presentation_clock_ignores_gaps_between_model_steps() -> None:
    clock = _PresentationClock(frames_per_second=16)

    _observe_model_step(clock, 1.0, 12, 0.9)
    _observe_model_step(clock, 1.9, 12, 0.9)
    assert clock.frames_per_second == pytest.approx(_presentation_fps(12 / 0.9))

    _observe_model_step(clock, 6.8, 12, 0.9)
    assert clock.frames_per_second == pytest.approx(_presentation_fps(12 / 0.9))


def test_presentation_clock_resets_estimate_for_a_new_generation() -> None:
    clock = _PresentationClock(frames_per_second=16)
    _observe_model_step(clock, 1.0, 12, 1.0)
    _observe_model_step(clock, 2.0, 12, 1.0)
    assert clock.frames_per_second == pytest.approx(_presentation_fps(12.0))

    assert clock.is_due(now=2.1, generation=1)
    assert clock.frames_per_second == 16

    _observe_model_step(clock, 2.2, 120, 0.01, generation=0)
    assert clock.frames_per_second == 16


def test_model_loop_excludes_publish_stalls_from_step_timing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 0.0

    class TimedModelLoop(IModelLoop[None]):
        def step(
            self,
            step_index: int,
            events: UserInputEvents,
        ) -> list[StepResult]:
            nonlocal now
            del events
            now += 0.9
            return [
                StepResult(
                    step_index=step_index,
                    output=torch.zeros((1, 3, 1, 1, 1)),
                    frame_count=1,
                    output_layout=VideoTensorLayout.bcthw,
                )
            ]

    monkeypatch.setattr("flashdreams.api_v2.loop.time.monotonic", lambda: now)
    failure_queue: queue.Queue[BaseException] = queue.Queue()
    model_loop = TimedModelLoop()
    model_loop.register_session_loop_objects(
        state=None,
        frequency=0,
        shutdown_event=threading.Event(),
        failure_queue=failure_queue,
    )
    event_buffer = EventBuffer()
    event_buffer.register(0)
    step_timings: list[float] = []

    def publish(
        generation: int,
        results: list[StepResult],
        step_elapsed_s: float,
    ) -> None:
        nonlocal now
        del generation, results
        step_timings.append(step_elapsed_s)
        now += 4.0

    model_loop._run_model_loop(
        event_buffer=event_buffer,
        reader_id=0,
        publish=publish,
        max_steps=2,
    )

    assert failure_queue.empty()
    assert step_timings == pytest.approx([0.9, 0.9])


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


class FakeModelLoop(IModelLoop["FakeSession"]):
    """Delegate standard model-loop hooks to the test session."""

    def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]:
        return [self.state.step(step_index, events)]

    def is_finished(self) -> bool:
        return self.state.is_finished()

    def reset(self) -> None:
        self.state.reset()


class FakeUILoop(IUILoop["FakeSession"]):
    """Delegate direct UI rendering to the test session."""

    def step(self, step_index: int, events: UserInputEvents) -> StepResult | None:
        return self.state.run_ui(step_index, events)

    def reset(self) -> None:
        return


class FakeSession(ISession):
    """Emit one blank frame per step and record what the runner asks for."""

    def __init__(
        self,
        session_desc: SessionDesc,
        log: CallLog,
        *,
        fail_at: int | None = None,
        fail_to_close: bool = False,
        release_writes: threading.Event | None = None,
        release_writes_at: int | None = None,
    ) -> None:
        """
        Args:
            session_desc: Description this session reports as resolved.
            log: Shared log both fakes record into.
            fail_at: Step index to raise at, for exercising cleanup on failure.
            fail_to_close: Whether :meth:`close` raises, as a session that
                cannot release what it holds would.
            release_writes: Event to set once ``release_writes_at`` has been
                generated, for holding the window back until then.
            release_writes_at: Step index that sets ``release_writes``.
        """
        self._session_desc = session_desc
        self._log = log
        self._fail_at = fail_at
        self._fail_to_close = fail_to_close
        self._release_writes = release_writes
        self._release_writes_at = release_writes_at
        self.observed_events: list[UserInputEvents] = []

    def init(self) -> None:
        self._log.record("session.init")
        self.register_ui_loop(FakeUILoop, state=self)
        self.register_model_loop(FakeModelLoop, state=self)

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
            output=torch.full((1, 3, 1, 1, 1), step_index, dtype=torch.float32),
            frame_count=1,
            output_layout=self._session_desc.output_layout,
        )

    def run_ui(self, step_index: int, events: UserInputEvents) -> StepResult | None:
        del events
        self._log.record("ui_loop.step")
        frame = self._presentation_manager.presented_frame(0)
        if frame is None:
            return None
        return StepResult(
            step_index=step_index,
            output=frame.unsqueeze(0).unsqueeze(2),
            frame_count=1,
            output_layout=self.session_desc.output_layout,
            metrics={"ui_ms": 0.25},
        )

    def is_finished(self) -> bool:
        return False

    def reset(self) -> None:
        self._log.record("session.reset")

    def close(self) -> None:
        self._log.record("session.close")
        if self._fail_to_close:
            raise RuntimeError("session close failed")


def test_registration_attaches_loop_lifecycle_events() -> None:
    session = FakeSession(_session_desc(), CallLog())

    session.init()

    assert session.model_loop._shutdown_event is session._shutdown_event
    assert session.ui_loop._shutdown_event is session._shutdown_event
    assert session.model_loop._failure_queue is session._failure_queue
    assert session.ui_loop._failure_queue is session._failure_queue


def test_session_shutdown_closes_every_registered_loop() -> None:
    closed: list[str] = []

    class FailingModelLoop(FakeModelLoop):
        def close(self) -> None:
            closed.append("model")
            raise RuntimeError("model close failed")

    class ClosingUILoop(FakeUILoop):
        def close(self) -> None:
            closed.append("ui")

    class ShutdownSession(FakeSession):
        def init(self) -> None:
            self.register_model_loop(FailingModelLoop, state=self)
            self.register_ui_loop(ClosingUILoop, state=self)

    session = ShutdownSession(_session_desc(), CallLog())
    session.init()

    failures = session._shutdown_registered_loops()

    assert closed == ["model", "ui"]
    assert len(failures) == 1
    assert str(failures[0]) == "model close failed"
    assert session._shutdown_event.is_set()


class FiniteSession(FakeSession):
    """A session with a fixed length."""

    def __init__(
        self,
        session_desc: SessionDesc,
        log: CallLog,
        *,
        length: int,
        generated: int = 0,
    ) -> None:
        """
        Args:
            session_desc: Description this session reports as resolved.
            log: Shared log both fakes record into.
            length: Steps to generate before reporting that it has finished.
                Counted from the last reset, as a session starting over would.
            generated: Steps to start out having generated, for a session that
                has finished before the run begins.
        """
        super().__init__(session_desc, log)
        self._length = length
        self._generated = generated

    def is_finished(self) -> bool:
        return self._generated >= self._length

    def step(self, step_index: int, events: UserInputEvents) -> StepResult:
        self._generated += 1
        return super().step(step_index, events)

    def reset(self) -> None:
        self._generated = 0
        super().reset()


class RecordingClientWindow(IClientWindow):
    """Report scripted input and record every call the runner makes."""

    def __init__(
        self,
        log: CallLog,
        scripted_events: list[UserInputEvents] | None = None,
        *,
        fail_to_open: bool = False,
        fail_to_close: bool = False,
        hold_writes: threading.Event | None = None,
    ) -> None:
        """
        Args:
            log: Shared log both fakes record into.
            scripted_events: Events to report, one entry per poll. Polls past the
                end of the script report nothing.
            fail_to_open: Whether :meth:`open` raises.
            fail_to_close: Whether :meth:`close` raises, as a sink that cannot
                finish the writes it was holding does.
            hold_writes: Event that has to be set before a write completes, for
                holding this window behind generation on purpose.
        """
        self._log = log
        self._scripted = list(scripted_events or [])
        self._fail_to_open = fail_to_open
        self._fail_to_close = fail_to_close
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
        if self._fail_to_close:
            raise RuntimeError("close failed")


def _session_desc(
    *,
    backpressure_mode: BackpressureMode = BackpressureMode.BLOCK,
    presentation_mode: PresentationMode = PresentationMode.ON_DEMAND,
    ui_fps: int = 100,
    model_fps: int = 1,
) -> SessionDesc:
    return SessionDesc(
        output_layout=VideoTensorLayout.bcthw,
        backpressure_mode=backpressure_mode,
        presentation_mode=presentation_mode,
        frames_per_second_for_ui=ui_fps,
        frames_per_second_for_step=model_fps,
        video_width=1,
        video_height=1,
    )


def _key_event() -> UserInputEvents:
    return UserInputEvents(
        [
            KeyboardUserInputEvent(
                timestamp=uint64(0),
                key="a",
                state=KeyboardInputState.PRESSED,
            )
        ]
    )


def _lifecycle_event(event_type: type[UserInputEvent]) -> UserInputEvents:
    return UserInputEvents([event_type(timestamp=uint64(0))])


def test_run_session_presents_every_step_in_order() -> None:
    log = CallLog()
    session = FakeSession(_session_desc(), log)
    window = RecordingClientWindow(log)

    run_session(session, window, steps=3)

    assert [result.step_index for result in window.results] == [0, 1, 2]
    assert window.results[-1] is session.ui_loop.latest_result
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


def test_run_session_touches_the_window_only_from_the_ui_thread() -> None:
    log = CallLog()
    session = FakeSession(_session_desc(), log)
    window = RecordingClientWindow(log)

    run_session(session, window, steps=2)

    # All window calls stay on the thread that called run_session.
    ui_thread_name = threading.current_thread().name
    for call in ("window.open", "window.get_user_input_events", "window.close"):
        assert log.threads_for(call) == {ui_thread_name}
    assert log.threads_for("window.write(0)") == {ui_thread_name}


def test_run_session_calls_ui_run_on_the_ui_thread() -> None:
    log = CallLog()
    session = FakeSession(_session_desc(), log)
    window = RecordingClientWindow(log)

    run_session(session, window, steps=2)

    assert log.threads_for("ui_loop.step") == {threading.current_thread().name}
    assert log.threads_for("session.step(0)") == {_STEP_THREAD_NAME}


def test_continuous_ui_processes_input_while_model_generation_waits() -> None:
    """Keep the io-thread responsive during a slow model-generation step."""
    log = CallLog()
    input_processed = threading.Event()

    class SlowModelSession(FakeSession):
        def step(self, step_index: int, events: UserInputEvents) -> StepResult:
            assert input_processed.wait(timeout=1.0)
            return super().step(step_index, events)

        def run_ui(self, step_index: int, events: UserInputEvents) -> StepResult | None:
            if events.get_events():
                input_processed.set()
            return super().run_ui(step_index, events)

    session = SlowModelSession(
        _session_desc(
            presentation_mode=PresentationMode.CONTINUOUS,
            ui_fps=100,
            model_fps=1,
        ),
        log,
    )
    window = RecordingClientWindow(
        log,
        [UserInputEvents([]), _key_event()],
    )

    run_session(session, window, steps=1)

    assert input_processed.is_set()
    assert "ui_loop.step" in log.calls


def test_each_message_queue_runs_on_its_owning_thread() -> None:
    log = CallLog()

    class MessageSession(FakeSession):
        def init(self) -> None:
            super().init()

            def model_message(state: FakeSession) -> None:
                state._log.record("model_loop.message")
                invoke_async(
                    self.model_loop,
                    lambda owner: owner._log.record("model_loop.self_message"),
                )

            invoke_async(
                self.ui_loop, lambda state: state._log.record("ui_loop.message")
            )
            invoke_async(self.model_loop, model_message)

    run_session(
        MessageSession(_session_desc(), log), RecordingClientWindow(log), steps=2
    )

    assert log.threads_for("ui_loop.message") == {threading.current_thread().name}
    assert log.threads_for("model_loop.message") == {_STEP_THREAD_NAME}
    assert log.threads_for("model_loop.self_message") == {_STEP_THREAD_NAME}
    assert log.calls.index("model_loop.self_message") > log.calls.index(
        "session.step(0)"
    )


def test_default_ui_composites_channels_and_holds_the_latest_frame() -> None:
    manager = PresentationManager()
    colors = (
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([0.0, 1.0, 0.0, 0.5]),
        torch.tensor([1.0, 0.0, 0.0, 0.5]),
    )
    manager.publish(
        0,
        [
            StepResult(
                step_index=0,
                output=color.reshape(1, -1, 1, 1),
                frame_count=1,
                output_layout=VideoTensorLayout.tchw,
            )
            for color in colors
        ],
    )
    ui = BlitModelOutputToScreenLoop()
    ui.register_session_ui_loop_objects(
        output_layout=VideoTensorLayout.tchw,
        presentation_manager=manager,
    )

    assert manager.advance(0, now=1.0)[0]
    first = ui.step(0, UserInputEvents([]))
    assert first is not None
    assert first.read_output()[0, :, 0, 0].tolist() == [0.5, 0.25, 0.0]
    assert not manager.advance(0)[0]
    held = ui.step(1, UserInputEvents([]))
    assert held is not None
    assert torch.equal(held.read_output(), first.read_output())
    assert not manager.advance(1)[0]
    assert manager.presented_frame_count == 0
    assert ui.step(2, UserInputEvents([])) is None


def test_presentation_manager_defaults_to_one_pending_chunk() -> None:
    manager = PresentationManager(device=torch.device("cpu"))

    assert manager.buffer_capacity == 1
    assert manager.buffered_chunk_capacity == 1
    assert manager.buffered_chunk_count == 0
    assert not manager.is_backlogged


def test_presentation_manager_reports_backlog_by_chunk_queue_capacity() -> None:
    manager = PresentationManager(device=torch.device("cpu"))

    def result(step_index: int) -> StepResult:
        return StepResult(
            step_index=step_index,
            output=torch.arange(3, dtype=torch.float32)
            .reshape(3, 1, 1, 1)
            .expand(-1, 3, -1, -1),
            frame_count=3,
            output_layout=VideoTensorLayout.tchw,
        )

    manager.publish(0, [result(0)])

    assert manager.buffer_capacity == 1
    assert manager.buffered_chunk_count == 1
    assert manager.buffered_chunk_capacity == 1
    assert manager.is_backlogged

    assert manager.advance(0, now=1.0)[0]

    assert manager.buffered_chunk_count == 0
    assert manager.has_pending_frames()
    assert not manager.is_backlogged


def test_presentation_manager_drains_active_chunk_frame_by_frame() -> None:
    manager = PresentationManager(device=torch.device("cpu"))

    def result(step_index: int) -> StepResult:
        return StepResult(
            step_index=step_index,
            output=torch.arange(3, dtype=torch.float32)
            .reshape(3, 1, 1, 1)
            .expand(-1, 3, -1, -1),
            frame_count=3,
            output_layout=VideoTensorLayout.tchw,
        )

    manager.publish(0, [result(0)])

    assert manager.buffered_chunk_count == 1
    assert manager.buffered_chunk_capacity == 1
    assert manager.is_backlogged

    assert manager.advance(0, now=1.1)[0]
    frame = manager.presented_frame(0)
    assert frame is not None
    assert frame.shape == (3, 1, 1)
    assert frame[0, 0, 0] == 0
    assert manager.buffered_chunk_count == 0
    assert manager.has_pending_frames()
    assert not manager.is_backlogged

    assert manager.advance(0, now=1.2)[0]
    frame = manager.presented_frame(0)
    assert frame is not None
    assert frame[0, 0, 0] == 1
    assert manager.buffered_chunk_count == 0

    assert manager.advance(0, now=1.3)[0]
    frame = manager.presented_frame(0)
    assert frame is not None
    assert frame[0, 0, 0] == 2
    assert manager.buffered_chunk_count == 0
    assert not manager.is_backlogged


def test_presentation_manager_advances_when_due_or_backlogged() -> None:
    manager = PresentationManager(device=torch.device("cpu"))
    manager.configure(
        backpressure_mode=BackpressureMode.BLOCK,
        stop=threading.Event(),
        put_timeout=0.01,
        frames_per_second=10,
        maximum_frames_per_second=60,
    )

    def result(step_index: int, frames: int) -> StepResult:
        return StepResult(
            step_index=step_index,
            output=torch.arange(frames, dtype=torch.float32)
            .reshape(frames, 1, 1, 1)
            .expand(-1, 3, -1, -1),
            frame_count=frames,
            output_layout=VideoTensorLayout.tchw,
        )

    manager.publish(0, [result(0, 2)])
    assert manager.advance(0, now=1.0)[0]
    assert not manager.advance(0, now=1.01)[0]

    manager.publish(0, [result(1, 1)])
    assert manager.is_backlogged
    assert manager.advance(0, now=1.01)[0]


def test_presentation_manager_publish_updates_presentation_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = PresentationManager(device=torch.device("cpu"))
    manager.configure(
        backpressure_mode=BackpressureMode.BLOCK,
        stop=threading.Event(),
        put_timeout=0.01,
        frames_per_second=16,
        maximum_frames_per_second=60,
    )
    completions = iter((1.0, 1.9))
    monkeypatch.setattr(
        "flashdreams.runtime_v2.presentation_manager.time.monotonic",
        lambda: next(completions),
    )

    def result(step_index: int) -> StepResult:
        return StepResult(
            step_index=step_index,
            output=torch.zeros((12, 3, 1, 1)),
            frame_count=12,
            output_layout=VideoTensorLayout.tchw,
        )

    manager.publish(0, [result(0)], step_elapsed_s=0.9)
    assert manager._presentation_clock.frames_per_second == 16
    manager.clear()

    manager.publish(0, [result(1)], step_elapsed_s=0.9)
    assert manager._presentation_clock.frames_per_second == pytest.approx(
        _presentation_fps(12 / 0.9)
    )


def test_composite_rejects_frames_with_different_dimensions() -> None:
    manager = PresentationManager(device=torch.device("cpu"))
    bottom = torch.full((3, 2, 3), -1.0)
    overlay = torch.ones((4, 4, 5))

    with pytest.raises(ValueError, match="same dimensions"):
        manager.composite(bottom, overlay)


def test_composite_clamps_alpha_before_interpolation() -> None:
    manager = PresentationManager()
    bottom = torch.full((3, 1, 2), -1.0)
    overlay = torch.tensor([[[1.0, 1.0]], [[0.5, 0.5]], [[0.0, 0.0]], [[-0.5, 1.5]]])

    composited = manager.composite(bottom, overlay)

    assert torch.equal(composited[:, :, 0], bottom[:, :, 0])
    assert torch.equal(composited[:, :, 1], overlay[:3, :, 1])


def test_default_ui_presents_each_frame_from_a_model_chunk() -> None:
    log = CallLog()

    class MultiFrameSession(FakeSession):
        def step(self, step_index: int, events: UserInputEvents) -> StepResult:
            del events
            self._log.record(f"session.step({step_index})")
            return StepResult(
                step_index=step_index,
                output=torch.arange(36, dtype=torch.float32).reshape(1, 3, 12, 1, 1),
                frame_count=12,
                output_layout=self.session_desc.output_layout,
                metrics={"total_ms": 1.5},
            )

    class RecordingMetricsSink:
        def __init__(self) -> None:
            self.results: list[StepResult] = []

        def open(self, session_desc: SessionDesc) -> None:
            del session_desc

        def write(self, result: StepResult) -> None:
            self.results.append(result)

        def close(self) -> None:
            return

    window = RecordingClientWindow(log)
    metrics = RecordingMetricsSink()
    run_session(
        MultiFrameSession(_session_desc(), log),
        window,
        metrics_output_sink=metrics,
        steps=1,
    )

    assert [result.frame_count for result in window.results] == [1] * 12
    assert [
        result.read_output()[0, 0, 0, 0, 0].item() for result in window.results
    ] == list(range(12))
    assert [result.metrics for result in window.results] == [{"ui_ms": 0.25}] * 12
    assert len(metrics.results) == 1
    assert metrics.results[0].metrics == {"total_ms": 1.5}


def test_default_ui_does_not_redraw_an_unchanged_model_frame() -> None:
    log = CallLog()

    class DefaultUISession(FakeSession):
        def init(self) -> None:
            self._log.record("session.init")
            self.register_model_loop(FakeModelLoop, state=self)

    session = DefaultUISession(
        _session_desc(
            presentation_mode=PresentationMode.ON_DEMAND,
            ui_fps=100,
            model_fps=30,
        ),
        log,
    )
    window = RecordingClientWindow(log)

    run_session(session, window, steps=3)

    assert [result.step_index for result in window.results] == [0, 1, 2]


def test_drop_oldest_finishes_active_chunk_before_newest_waiting_chunk() -> None:
    manager = PresentationManager()
    manager.configure(
        backpressure_mode=BackpressureMode.DROP_OLDEST,
        stop=threading.Event(),
        put_timeout=0.01,
    )

    def result(step_index: int, frames: int) -> StepResult:
        return StepResult(
            step_index=step_index,
            output=(
                torch.arange(frames, dtype=torch.float32).reshape(frames, 1, 1, 1)
                + step_index * 10
            ).expand(-1, 3, -1, -1),
            frame_count=frames,
            output_layout=VideoTensorLayout.tchw,
        )

    manager.publish(0, [result(0, 3)])
    assert manager.advance(0)[0]
    first = manager.presented_frame(0)
    assert first is not None
    assert first[0, 0, 0] == 0
    assert manager.presented_frame_count == 1
    manager.publish(0, [result(1, 1)])
    manager.publish(0, [result(2, 1)])

    assert manager.advance(0, now=1.1)[0]
    second = manager.presented_frame(0)
    assert second is not None
    assert second[0, 0, 0] == 1
    assert manager.presented_frame_count == 2

    assert manager.advance(0, now=1.2)[0]
    third = manager.presented_frame(0)
    assert third is not None
    assert third[0, 0, 0] == 2
    assert manager.presented_frame_count == 3

    assert manager.advance(0, now=1.3)[0]
    newest = manager.presented_frame(0)
    assert newest is not None
    assert newest[0, 0, 0] == 20
    assert manager.presented_frame_count == 4
    assert manager.dropped_for_space == 1


def test_run_session_opens_window_with_the_resolved_session_desc() -> None:
    log = CallLog()
    resolved = _session_desc()
    session = FakeSession(resolved, log)
    window = RecordingClientWindow(log)

    run_session(session, window, steps=1)

    assert window.session_desc is resolved


def test_window_write_stays_in_the_presentation_context() -> None:
    log = CallLog()
    session = FakeSession(_session_desc(), log)

    class ContextRecordingManager(PresentationManager):
        active_depth = 0

        @contextmanager
        def presentation_context(self) -> Iterator[None]:
            self.active_depth += 1
            try:
                yield
            finally:
                self.active_depth -= 1

    manager = ContextRecordingManager()
    session.__dict__["_presentation_manager"] = manager

    class ContextCheckingWindow(RecordingClientWindow):
        def write(self, result: StepResult) -> None:
            assert manager.active_depth > 0
            super().write(result)

    run_session(session, ContextCheckingWindow(log), steps=1)


def test_run_session_gives_the_first_step_input_already_collected() -> None:
    log = CallLog()
    session = FakeSession(_session_desc(), log)
    window = RecordingClientWindow(log, [_key_event()])

    run_session(session, window, steps=2)

    # The UI thread collects once before generation starts, so input the window
    # already holds is not missed by step 0.
    assert len(session.observed_events[0].get_events()) == 1


def test_run_session_stops_when_the_window_reports_a_close() -> None:
    log = CallLog()
    session = FakeSession(_session_desc(), log)
    window = RecordingClientWindow(log, [_lifecycle_event(CloseUserInputEvent)])

    # No step count at all: the close is the only thing that ends this run.
    run_session(session, window, steps=None)

    assert "session.step(0)" not in log.calls
    assert log.calls[-2:] == ["window.close", "session.close"]


def test_run_session_resets_the_session_and_the_step_index() -> None:
    log = CallLog()
    session = FakeSession(_session_desc(), log)
    window = RecordingClientWindow(log, [_lifecycle_event(ResetUserInputEvent)])

    run_session(session, window, steps=2)

    # Ignore UI calls when checking the model-generation-loop order.
    calls = [call for call in log.calls if call.startswith("session.reset")] + [
        call for call in log.calls if call.startswith("session.step(")
    ]
    assert calls == ["session.reset", "session.step(0)", "session.step(1)"]
    assert log.calls.index("session.reset") < log.calls.index("session.step(0)")
    # A reset restarts both loops without granting extra model steps.
    assert [result.step_index for result in window.results] == [0, 1]


def test_run_session_stops_when_the_session_says_it_has_finished() -> None:
    """A model that knows its own length ends its own run, uncounted."""
    log = CallLog()
    session = FiniteSession(_session_desc(), log, length=2)
    window = RecordingClientWindow(log)

    run_session(session, window, steps=None)

    assert [result.step_index for result in window.results] == [0, 1]


def test_run_session_ends_at_whichever_comes_first() -> None:
    """A caller can ask for fewer steps than the session would generate."""
    log = CallLog()
    session = FiniteSession(_session_desc(), log, length=5)
    window = RecordingClientWindow(log)

    run_session(session, window, steps=2)

    assert [result.step_index for result in window.results] == [0, 1]


def test_run_session_lets_a_reset_restart_a_finished_session() -> None:
    """A session that starts over is asked about the run it is starting."""
    log = CallLog()
    session = FiniteSession(_session_desc(), log, length=1, generated=1)
    window = RecordingClientWindow(log, [_lifecycle_event(ResetUserInputEvent)])

    run_session(session, window, steps=3)

    # Finished before the run began, so without the reset nothing would be
    # generated. It is applied before the session runs its length again.
    assert [result.step_index for result in window.results] == [0]
    assert "session.reset" in log.calls


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
                    ResetUserInputEvent(timestamp=uint64(1)),
                ]
            )
        ],
    )

    run_session(session, window, steps=1)

    # Events are edges, so a key held down when the client restarts is still held
    # after: the batch is not split at the reset, and the edge that said so is
    # what carries the state.
    assert held_key in session.observed_events[0].get_events()


def test_run_session_keeps_polling_while_the_final_result_is_pending() -> None:
    log = CallLog()
    session = FakeSession(_session_desc(), log)
    window = RecordingClientWindow(
        log,
        [UserInputEvents([]), _lifecycle_event(ResetUserInputEvent)],
    )

    run_session(session, window, steps=1)

    # The reset arrives before the queued frame is shown, so that frame is dropped.
    assert window.results == []


def test_run_session_drops_a_result_the_reset_interrupted() -> None:
    log = CallLog()
    reset_reported = threading.Event()

    class SlowFirstStep(FakeSession):
        """Stay inside the first step until the window has reported the reset."""

        def step(self, step_index: int, events: UserInputEvents) -> StepResult:
            if step_index == 0 and not reset_reported.is_set():
                reset_reported.wait()
            return super().step(step_index, events)

    class ResettingWindow(RecordingClientWindow):
        """Announce the reset, which is the only input this window reports."""

        def get_user_input_events(self) -> UserInputEvents:
            events = super().get_user_input_events()
            if events.get_events():
                reset_reported.set()
            return events

    session = SlowFirstStep(_session_desc(), log)
    window = ResettingWindow(
        log,
        [UserInputEvents([]), _lifecycle_event(ResetUserInputEvent)],
    )

    run_session(session, window, steps=2)

    # The first step was still running when the client asked to start over, so
    # what it produced belongs to a generation nobody is watching any more. Only
    # the step from after the reset reaches the window.
    assert log.calls.count("session.step(0)") == 2
    assert [result.step_index for result in window.results] == [0]


def test_equality_eval_preserves_every_frame_when_model_is_faster() -> None:
    log = CallLog()
    session = FakeSession(_session_desc(ui_fps=30, model_fps=10_000), log)
    window = RecordingClientWindow(log)

    run_session(session, window, steps=4)

    assert [result.step_index for result in window.results] == [0, 1, 2, 3]
    assert [
        result.read_output()[0, 0, 0, 0, 0].item() for result in window.results
    ] == [0, 1, 2, 3]


def test_ui_stall_does_not_burst_multiple_frames_per_input_tick() -> None:
    log = CallLog()

    class FourFrameSession(FakeSession):
        def step(self, step_index: int, events: UserInputEvents) -> StepResult:
            del events
            self._log.record(f"session.step({step_index})")
            return StepResult(
                step_index=step_index,
                output=torch.arange(4, dtype=torch.float32).reshape(1, 1, 4, 1, 1),
                frame_count=4,
                output_layout=self.session_desc.output_layout,
            )

    class FirstWriteStalls(RecordingClientWindow):
        def __init__(self, call_log: CallLog) -> None:
            super().__init__(call_log)
            self._first_write = True

        def write(self, result: StepResult) -> None:
            if self._first_write:
                self._first_write = False
                time.sleep(0.05)
            super().write(result)

    window = FirstWriteStalls(log)
    run_session(
        FourFrameSession(_session_desc(ui_fps=100, model_fps=100), log),
        window,
        steps=1,
    )

    assert [
        result.read_output()[0, 0, 0, 0, 0].item() for result in window.results
    ] == [0, 1, 2, 3]
    writes_per_input_tick: list[int] = []
    writes = 0
    for call in log.calls:
        if call == "window.get_user_input_events":
            writes_per_input_tick.append(writes)
            writes = 0
        elif call.startswith("window.write("):
            writes += 1
    writes_per_input_tick.append(writes)
    assert max(writes_per_input_tick) == 1


def test_on_demand_runs_ui_once_per_new_frame() -> None:
    log = CallLog()
    session = FakeSession(_session_desc(ui_fps=1_000, model_fps=20), log)
    window = RecordingClientWindow(log)
    run_session(session, window, steps=3)

    assert log.calls.count("ui_loop.step") == 3
    assert [
        result.read_output()[0, 0, 0, 0, 0].item() for result in window.results
    ] == [0, 1, 2]


def test_continuous_runs_ui_eagerly_when_ui_is_faster() -> None:
    log = CallLog()
    session = FakeSession(
        _session_desc(
            presentation_mode=PresentationMode.CONTINUOUS,
            ui_fps=1_000,
            model_fps=20,
        ),
        log,
    )
    window = RecordingClientWindow(log)

    run_session(session, window, steps=3)

    presented = [
        result.read_output()[0, 0, 0, 0, 0].item() for result in window.results
    ]
    assert presented == sorted(presented)
    assert len(presented) > 3
    assert presented[-1] == 2


def test_run_session_drops_the_oldest_waiting_result() -> None:
    log = CallLog()

    # Hold the window until every step is generated, so which results are dropped
    # does not depend on how the two threads happen to be scheduled.
    generated = threading.Event()
    drop_oldest_desc = _session_desc(
        backpressure_mode=BackpressureMode.DROP_OLDEST,
    )
    session = FakeSession(
        drop_oldest_desc, log, release_writes=generated, release_writes_at=3
    )
    window = RecordingClientWindow(log, hold_writes=generated)

    run_session(session, window, steps=4)

    presented = [result.step_index for result in window.results]
    # The single-slot pending chunk queue fills while the window is held, so
    # stale chunks are lost before presentation catches up.
    assert presented == sorted(presented)
    assert len(presented) < 4
    assert window.results[-1].read_output()[0, 0, 0, 0, 0].item() == 3


def test_run_session_discards_results_generated_before_a_reset(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log = CallLog()
    session = FakeSession(_session_desc(), log)
    window = RecordingClientWindow(
        log,
        [
            UserInputEvents([]),
            _lifecycle_event(ResetUserInputEvent),
            _lifecycle_event(CloseUserInputEvent),
        ],
    )

    with caplog.at_level(logging.INFO, logger=_RUNNER_LOGGER):
        run_session(session, window, steps=None)

    # The client asked to start over, so what the abandoned generation produced is
    # thrown away rather than presented after the restart. The runner logs this only
    # when it discarded at least one result, and the count it reports depends on how
    # far generation got before the reset landed.
    assert any("before a reset" in record.getMessage() for record in caplog.records)


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


def test_run_session_reports_a_window_that_fails_to_close() -> None:
    log = CallLog()
    session = FakeSession(_session_desc(), log)
    window = RecordingClientWindow(log, fail_to_close=True)

    # Closing is when a sink finishes what it was holding, so a run whose output
    # never landed must not look like it succeeded.
    with pytest.raises(RuntimeError, match="close failed"):
        run_session(session, window, steps=2)

    assert [result.step_index for result in window.results] == [0, 1]
    assert log.calls[-2:] == ["window.close", "session.close"]


def test_run_session_reports_a_window_that_fails_to_open() -> None:
    log = CallLog()
    session = FakeSession(_session_desc(), log)
    window = RecordingClientWindow(log, fail_to_open=True)

    with pytest.raises(RuntimeError, match="open failed"):
        run_session(session, window, steps=2)

    # Generation never starts, but an open that raised part way through still
    # holds what it had acquired, so both halves are closed anyway.
    assert "session.step(0)" not in log.calls
    assert log.calls[-2:] == ["window.close", "session.close"]


def test_run_session_reports_what_ended_the_run_rather_than_the_close(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log = CallLog()
    session = FakeSession(_session_desc(), log, fail_at=0)
    window = RecordingClientWindow(log, fail_to_close=True)

    # Both the step and the close fail. The step is the one that explains the
    # run, so that is what a caller is given, and the close is logged rather
    # than lost.
    with caplog.at_level(logging.ERROR, logger=_RUNNER_LOGGER):
        with pytest.raises(RuntimeError, match="step failed"):
            run_session(session, window, steps=2)

    assert "close failed" in caplog.text
    assert log.calls[-2:] == ["window.close", "session.close"]


def test_run_session_reports_a_session_that_fails_to_close() -> None:
    log = CallLog()
    session = FakeSession(_session_desc(), log, fail_to_close=True)
    window = RecordingClientWindow(log)

    # Nothing else went wrong, so the only thing wrong with the run is that the
    # session still holds what it was using.
    with pytest.raises(RuntimeError, match="session close failed"):
        run_session(session, window, steps=2)

    assert [result.step_index for result in window.results] == [0, 1]


def test_run_session_reports_the_step_rather_than_the_session_close(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log = CallLog()
    session = FakeSession(_session_desc(), log, fail_at=0, fail_to_close=True)
    window = RecordingClientWindow(log)

    with caplog.at_level(logging.ERROR, logger=_RUNNER_LOGGER):
        with pytest.raises(RuntimeError, match="step failed"):
            run_session(session, window, steps=2)

    assert "session close failed" in caplog.text


def test_run_session_reports_the_init_rather_than_the_session_close(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log = CallLog()

    class FailingSession(FakeSession):
        def init(self) -> None:
            super().init()
            raise RuntimeError("init failed")

    session = FailingSession(_session_desc(), log, fail_to_close=True)

    with caplog.at_level(logging.ERROR, logger=_RUNNER_LOGGER):
        with pytest.raises(RuntimeError, match="init failed"):
            run_session(session, RecordingClientWindow(log), steps=1)

    assert "session close failed" in caplog.text
