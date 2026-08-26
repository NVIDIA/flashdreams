# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU test for the v2 session loop, independent of any application."""

import logging
import threading

import pytest
import torch
from numpy import uint64

from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.api_v2.loop import IModelLoop, IUILoop, invoke_async
from flashdreams.api_v2.session import ISession
from flashdreams.api_v2.user_input_event_data import UserInputEventData
from flashdreams.runtime_v2.audio_output import AudioOutput
from flashdreams.runtime_v2.blit_model_output_to_screen_loop import (
    BlitModelOutputToScreenLoop,
)
from flashdreams.runtime_v2.presentation_manager import PresentationManager
from flashdreams.runtime_v2.session_desc import (
    BackpressureMode,
    PresentationMode,
    SessionDesc,
)
from flashdreams.runtime_v2.session_runner import run_session
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    CloseUserInputEventData,
    KeyboardInputState,
    KeyboardUserInputEventData,
    ResetUserInputEventData,
    UserInputEvent,
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
        PresentationMode.ONLY_PRESENT_NEW,
        PresentationMode.ONLY_PRESENT_NEWEST,
    ]
    assert SessionDesc().backpressure_mode is BackpressureMode.BLOCK
    assert SessionDesc().presentation_mode is PresentationMode.ONLY_PRESENT_NEWEST


@pytest.mark.parametrize(
    "field",
    ["frames_per_second_for_ui", "frames_per_second_for_step"],
)
@pytest.mark.parametrize("value", [True, 24.0, 0, -1])
def test_session_frame_rates_are_strict_positive_integers(
    field: str, value: object
) -> None:
    """Reject values that fail later loop timing and encoder contracts."""
    with pytest.raises(ValueError, match="positive integer"):
        SessionDesc(**{field: value})  # ty: ignore[invalid-argument-type]


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


class TransactionalRecordingClientWindow(RecordingClientWindow):
    """Record an output transaction that can be committed or aborted."""

    def __init__(
        self,
        log: CallLog,
        scripted_events: list[UserInputEvents] | None = None,
        *,
        fail_to_open: bool = False,
        fail_to_write: bool = False,
        fail_to_close: bool = False,
        fail_to_abort: bool = False,
    ) -> None:
        super().__init__(
            log,
            scripted_events,
            fail_to_open=fail_to_open,
            fail_to_close=fail_to_close,
        )
        self._fail_to_write = fail_to_write
        self._fail_to_abort = fail_to_abort

    def write(self, result: StepResult) -> None:
        super().write(result)
        if self._fail_to_write:
            raise RuntimeError("write failed")

    def abort(self) -> None:
        self._log.record("window.abort")
        if self._fail_to_abort:
            raise RuntimeError("abort failed")


def _session_desc(
    *,
    backpressure_mode: BackpressureMode = BackpressureMode.BLOCK,
    presentation_mode: PresentationMode = PresentationMode.ONLY_PRESENT_NEW,
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
            UserInputEvent(
                timestamp=uint64(0),
                event_data=KeyboardUserInputEventData(
                    key="a", state=KeyboardInputState.PRESSED
                ),
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


def test_run_session_touches_the_window_only_from_the_io_thread() -> None:
    log = CallLog()
    session = FakeSession(_session_desc(), log)
    window = RecordingClientWindow(log)

    run_session(session, window, steps=2)

    # All window calls stay on the thread that called run_session.
    io_thread_name = threading.current_thread().name
    for call in ("window.open", "window.get_user_input_events", "window.close"):
        assert log.threads_for(call) == {io_thread_name}
    assert log.threads_for("window.write(0)") == {io_thread_name}


def test_run_session_calls_ui_run_on_the_io_thread() -> None:
    log = CallLog()
    session = FakeSession(_session_desc(), log)
    window = RecordingClientWindow(log)

    run_session(session, window, steps=2)

    assert log.threads_for("ui_loop.step") == {threading.current_thread().name}
    assert log.threads_for("session.step(0)") == {_STEP_THREAD_NAME}


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

    assert manager.advance(0)[0]
    first = ui.step(0, UserInputEvents([]))
    assert first is not None
    assert first.output[0, :, 0, 0].tolist() == [0.5, 0.25, 0.0]
    assert not manager.advance(0)[0]
    held = ui.step(1, UserInputEvents([]))
    assert held is not None
    assert torch.equal(held.output, first.output)
    assert not manager.advance(1)[0]
    assert ui.step(2, UserInputEvents([])) is None


def test_default_ui_forwards_one_audio_payload_once_across_channels() -> None:
    manager = PresentationManager()
    audio = AudioOutput(
        samples=torch.zeros(2, 16),
        sample_rate=8_000,
        sample_offset=32,
    )

    def result(*, audio_output: AudioOutput | None = None) -> StepResult:
        return StepResult(
            step_index=0,
            output=torch.zeros(2, 3, 1, 1),
            frame_count=2,
            output_layout=VideoTensorLayout.tchw,
            audio=audio_output,
        )

    manager.publish(0, [result(), result(audio_output=audio)])
    ui = BlitModelOutputToScreenLoop()
    ui.register_session_ui_loop_objects(
        output_layout=VideoTensorLayout.tchw,
        presentation_manager=manager,
    )

    assert manager.advance(0)[0]
    first = ui.step(0, UserInputEvents([]))
    assert first is not None and first.audio is audio
    held = ui.step(1, UserInputEvents([]))
    assert held is not None and held.audio is None
    assert manager.advance(0)[0]
    second = ui.step(2, UserInputEvents([]))
    assert second is not None and second.audio is None


def test_presentation_rejects_multiple_audio_payloads() -> None:
    manager = PresentationManager()
    audio = AudioOutput(samples=torch.zeros(1, 1), sample_rate=8_000)
    result = StepResult(
        step_index=0,
        output=torch.zeros(1, 3, 1, 1),
        frame_count=1,
        output_layout=VideoTensorLayout.tchw,
        audio=audio,
    )

    with pytest.raises(ValueError, match="only one audio"):
        manager.publish(0, [result, result])


def test_presentation_rejects_an_untyped_audio_payload() -> None:
    manager = PresentationManager()
    result = StepResult(
        step_index=0,
        output=torch.zeros(1, 3, 1, 1),
        frame_count=1,
        output_layout=VideoTensorLayout.tchw,
        audio=object(),  # ty: ignore[invalid-argument-type]
    )

    with pytest.raises(TypeError, match="AudioOutput"):
        manager.publish(0, [result])


def test_default_ui_presents_each_frame_from_a_model_chunk() -> None:
    log = CallLog()

    class MultiFrameSession(FakeSession):
        def step(self, step_index: int, events: UserInputEvents) -> StepResult:
            del events
            self._log.record(f"session.step({step_index})")
            return StepResult(
                step_index=step_index,
                output=torch.arange(6, dtype=torch.float32).reshape(1, 3, 2, 1, 1),
                frame_count=2,
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

    assert [result.frame_count for result in window.results] == [1, 1]
    assert [result.output[0, 0, 0, 0, 0].item() for result in window.results] == [0, 1]
    assert [result.metrics for result in window.results] == [
        {"ui_ms": 0.25},
        {"ui_ms": 0.25},
    ]
    assert len(metrics.results) == 1
    assert metrics.results[0].metrics == {"total_ms": 1.5}


def test_drop_oldest_preempts_the_rest_of_a_stale_chunk() -> None:
    manager = PresentationManager()
    manager.configure(
        max_pending=1,
        backpressure_mode=BackpressureMode.DROP_OLDEST,
        stop=threading.Event(),
        put_timeout=0.01,
    )

    def result(step_index: int, frames: int) -> StepResult:
        return StepResult(
            step_index=step_index,
            output=torch.full((frames, 3, 1, 1), float(step_index)),
            frame_count=frames,
            output_layout=VideoTensorLayout.tchw,
        )

    manager.publish(0, [result(0, 2)])
    assert manager.advance(0)[0]
    manager.publish(0, [result(1, 1)])
    assert manager.advance(0)[0]
    newest = manager.presented_frame(0)
    assert newest is not None
    assert newest[0, 0, 0] == 1


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

    # Ignore UI calls when checking the model thread's order.
    calls = [call for call in log.calls if call.startswith("session.reset")] + [
        call for call in log.calls if call.startswith("session.step(")
    ]
    assert calls == ["session.reset", "session.step(0)", "session.step(1)"]
    assert log.calls.index("session.reset") < log.calls.index("session.step(0)")
    # A reset restarts the index without granting extra steps.
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
    window = RecordingClientWindow(log, [_lifecycle_event(ResetUserInputEventData())])

    run_session(session, window, steps=3)

    # Finished before the run began, so without the reset nothing would be
    # generated. It is applied first, and the session runs its length again.
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


def test_run_session_keeps_polling_while_the_final_result_is_pending() -> None:
    log = CallLog()
    session = FakeSession(_session_desc(), log)
    window = RecordingClientWindow(
        log,
        [UserInputEvents([]), _lifecycle_event(ResetUserInputEventData())],
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
        [UserInputEvents([]), _lifecycle_event(ResetUserInputEventData())],
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

    run_session(session, window, steps=4, max_pending=1)

    assert [result.step_index for result in window.results] == [0, 1, 2, 3]
    assert [result.output[0, 0, 0, 0, 0].item() for result in window.results] == [
        0,
        1,
        2,
        3,
    ]


def test_ONLY_PRESENT_NEW_runs_ui_once_per_new_frame() -> None:
    log = CallLog()
    session = FakeSession(_session_desc(ui_fps=1_000, model_fps=20), log)
    window = RecordingClientWindow(log)
    run_session(session, window, steps=3)

    assert log.calls.count("ui_loop.step") == 3
    assert [result.output[0, 0, 0, 0, 0].item() for result in window.results] == [
        0,
        1,
        2,
    ]


def test_only_present_newest_runs_ui_eagerly_when_ui_is_faster() -> None:
    log = CallLog()
    session = FakeSession(
        _session_desc(
            presentation_mode=PresentationMode.ONLY_PRESENT_NEWEST,
            ui_fps=1_000,
            model_fps=20,
        ),
        log,
    )
    window = RecordingClientWindow(log)

    run_session(session, window, steps=3)

    presented = [result.output[0, 0, 0, 0, 0].item() for result in window.results]
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

    run_session(session, window, steps=4, max_pending=1)

    presented = [result.step_index for result in window.results]
    # Room for one result and four generated behind a window that cannot write
    # until the end, so what is stale is lost and the newest always arrives.
    assert presented == sorted(presented)
    assert len(presented) < 4
    assert window.results[-1].output[0, 0, 0, 0, 0].item() == 3


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


def test_success_closes_an_output_transaction_after_the_session() -> None:
    log = CallLog()
    window = TransactionalRecordingClientWindow(log)

    run_session(FakeSession(_session_desc(), log), window, steps=1)

    assert log.calls[-2:] == ["session.close", "window.close"]
    assert "window.abort" not in log.calls


def test_client_cancellation_aborts_the_output_transaction() -> None:
    log = CallLog()
    window = TransactionalRecordingClientWindow(
        log,
        [_lifecycle_event(CloseUserInputEventData())],
    )

    run_session(FakeSession(_session_desc(), log), window, steps=None)

    assert log.calls[-2:] == ["session.close", "window.abort"]
    assert "window.close" not in log.calls


def test_model_thread_start_failure_aborts_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = CallLog()
    window = TransactionalRecordingClientWindow(log)

    def fail_to_start(thread: threading.Thread) -> None:
        del thread
        raise RuntimeError("model start failed")

    monkeypatch.setattr(threading.Thread, "start", fail_to_start)

    with pytest.raises(RuntimeError, match="model start failed"):
        run_session(FakeSession(_session_desc(), log), window, steps=1)

    assert log.calls[-2:] == ["session.close", "window.abort"]
    assert "window.close" not in log.calls


def test_model_step_failure_aborts_output() -> None:
    log = CallLog()
    window = TransactionalRecordingClientWindow(log)

    with pytest.raises(RuntimeError, match="step failed"):
        run_session(FakeSession(_session_desc(), log, fail_at=0), window, steps=1)

    assert log.calls[-2:] == ["session.close", "window.abort"]
    assert "window.close" not in log.calls


def test_presentation_failure_aborts_output() -> None:
    log = CallLog()

    class InvalidPresentationSession(FakeSession):
        def init(self) -> None:
            self._log.record("session.init")
            self.register_model_loop(FakeModelLoop, state=self)

        def step(self, step_index: int, events: UserInputEvents) -> StepResult:
            del events
            self._log.record(f"session.step({step_index})")
            return StepResult(
                step_index=step_index,
                output=torch.zeros(1, 2, 1, 1, 1),
                frame_count=1,
                output_layout=self.session_desc.output_layout,
            )

    window = TransactionalRecordingClientWindow(log)
    with pytest.raises(ValueError, match="one, three, or four"):
        run_session(InvalidPresentationSession(_session_desc(), log), window, steps=1)

    assert log.calls[-2:] == ["session.close", "window.abort"]
    assert "window.close" not in log.calls


def test_sink_open_failure_aborts_output() -> None:
    log = CallLog()
    window = TransactionalRecordingClientWindow(log, fail_to_open=True)

    with pytest.raises(RuntimeError, match="open failed"):
        run_session(FakeSession(_session_desc(), log), window, steps=1)

    assert log.calls[-2:] == ["session.close", "window.abort"]
    assert "window.close" not in log.calls


def test_sink_write_failure_aborts_output() -> None:
    log = CallLog()
    window = TransactionalRecordingClientWindow(log, fail_to_write=True)

    with pytest.raises(RuntimeError, match="write failed"):
        run_session(FakeSession(_session_desc(), log), window, steps=1)

    assert log.calls[-2:] == ["session.close", "window.abort"]
    assert "window.close" not in log.calls


def test_sink_close_failure_falls_back_to_abort() -> None:
    log = CallLog()
    window = TransactionalRecordingClientWindow(log, fail_to_close=True)

    with pytest.raises(RuntimeError, match="close failed"):
        run_session(FakeSession(_session_desc(), log), window, steps=1)

    assert log.calls[-3:] == ["session.close", "window.close", "window.abort"]


def test_session_close_failure_aborts_output() -> None:
    log = CallLog()
    window = TransactionalRecordingClientWindow(log)

    with pytest.raises(RuntimeError, match="session close failed"):
        run_session(
            FakeSession(_session_desc(), log, fail_to_close=True),
            window,
            steps=1,
        )

    assert log.calls[-2:] == ["session.close", "window.abort"]
    assert "window.close" not in log.calls


def test_first_model_failure_survives_all_cleanup_failures(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log = CallLog()

    class FailingModelLoop(FakeModelLoop):
        def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]:
            del step_index, events
            raise RuntimeError("model step failed first")

        def close(self) -> None:
            raise RuntimeError("model close failed second")

    class FailingCleanupSession(FakeSession):
        def init(self) -> None:
            self._log.record("session.init")
            self.register_model_loop(FailingModelLoop, state=self)

    session = FailingCleanupSession(
        _session_desc(),
        log,
        fail_to_close=True,
    )
    window = TransactionalRecordingClientWindow(log, fail_to_abort=True)

    with caplog.at_level(logging.ERROR, logger=_RUNNER_LOGGER):
        with pytest.raises(RuntimeError, match="model step failed first"):
            run_session(session, window, steps=1)

    assert "model close failed second" in caplog.text
    assert "session close failed" in caplog.text
    assert "abort failed" in caplog.text
    assert log.calls[-2:] == ["session.close", "window.abort"]
    assert log.calls.count("window.abort") == 1


def test_queued_loop_failure_outranks_a_main_thread_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Preserve the failure priority defined by the V2 API contract."""
    log = CallLog()

    class QueuedFailureSession(FakeSession):
        def init(self) -> None:
            super().init()
            self._failure_queue.put(RuntimeError("loop failed first"))

    session = QueuedFailureSession(_session_desc(), log)
    window = RecordingClientWindow(log, fail_to_open=True)

    with caplog.at_level(logging.ERROR, logger=_RUNNER_LOGGER):
        with pytest.raises(RuntimeError, match="loop failed first"):
            run_session(session, window, steps=1)

    assert "open failed" in caplog.text


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
