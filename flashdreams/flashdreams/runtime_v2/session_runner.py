# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a session with a client window."""

import logging
import threading
import time
from collections import deque
from dataclasses import replace

from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.api_v2.loop import IModelLoop, IUILoop
from flashdreams.api_v2.output_sink import OutputSink
from flashdreams.api_v2.session import ISession
from flashdreams.api_v2.user_input_event import UserInputEvent
from flashdreams.runtime_v2.event_buffer import EventBuffer
from flashdreams.runtime_v2.session_desc import PresentationMode
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import CloseUserInputEvent
from flashdreams.runtime_v2.user_input_events import UserInputEvents

_LOGGER = logging.getLogger(__name__)
_MODEL_THREAD_NAME = "flashdreams-model-generation-thread"
_UI_READER_ID = 0
_MODEL_READER_ID = 1
_MODEL_FPS_WINDOW_SECONDS = 2.0
"""Wall-time window used to estimate generated-frame throughput."""

_RUNTIME_METRIC_PREFIX = "runtime_"
"""Metric namespace reserved for measurements added by this runtime."""


class _PresentationClock:
    """Schedule model-frame advances at recent model throughput."""

    def __init__(
        self,
        frames_per_second: int,
        maximum_frames_per_second: int | None = None,
    ) -> None:
        maximum_frames_per_second = maximum_frames_per_second or frames_per_second
        self._minimum_frame_interval = 1.0 / maximum_frames_per_second
        self._fallback_frame_interval = max(
            1.0 / frames_per_second,
            self._minimum_frame_interval,
        )
        self._frame_interval = self._fallback_frame_interval
        self._next_frame_at: float | None = None
        self._generation: int | None = None
        self._observations: deque[tuple[float, int]] = deque()
        self._lock = threading.Lock()

    @property
    def frames_per_second(self) -> float:
        """Return the current model-frame presentation rate."""
        with self._lock:
            return 1.0 / self._frame_interval

    def observe_model_output(
        self,
        *,
        now: float,
        generation: int,
        frame_count: int,
    ) -> None:
        """Add one completed model chunk to the rolling FPS estimate.

        Args:
            now: Monotonic completion time for the chunk.
            generation: Session generation that produced the chunk.
            frame_count: Number of generated frames in the chunk.

        Raises:
            ValueError: ``frame_count`` is not positive or ``now`` precedes the
                latest observation.
        """
        if frame_count <= 0:
            raise ValueError(f"frame_count must be > 0, got {frame_count}.")
        with self._lock:
            if self._generation is None or generation > self._generation:
                self._reset_generation(generation)
            elif generation < self._generation:
                return

            if self._observations and now < self._observations[-1][0]:
                raise ValueError("now must not precede the latest observation.")
            observed_frames = (
                frame_count
                if not self._observations
                else self._observations[-1][1] + frame_count
            )
            if self._observations and now == self._observations[-1][0]:
                self._observations[-1] = (now, observed_frames)
            else:
                self._observations.append((now, observed_frames))
            self._update_frame_interval(now)

    def is_due(self, now: float, generation: int) -> bool:
        """Return whether the next model frame may be selected."""
        with self._lock:
            if generation != self._generation:
                self._reset_generation(generation)
            return self._next_frame_at is None or now >= self._next_frame_at

    def mark_advanced(self, now: float) -> None:
        """Record one selected frame without catching up after a long stall."""
        with self._lock:
            next_frame_at = self._next_frame_at
            if next_frame_at is None or now - next_frame_at >= self._frame_interval:
                self._next_frame_at = now + self._frame_interval
            else:
                self._next_frame_at = next_frame_at + self._frame_interval

    def _reset_generation(self, generation: int) -> None:
        self._generation = generation
        self._frame_interval = self._fallback_frame_interval
        self._next_frame_at = None
        self._observations.clear()

    def _update_frame_interval(self, now: float) -> None:
        cutoff = now - _MODEL_FPS_WINDOW_SECONDS
        while len(self._observations) >= 3 and self._observations[1][0] <= cutoff:
            self._observations.popleft()
        if len(self._observations) < 2:
            return

        first_at, first_frames = self._observations[0]
        last_at, last_frames = self._observations[-1]
        window_started_at = max(first_at, cutoff)
        frames_at_window_start = float(first_frames)
        if window_started_at > first_at:
            second_at, second_frames = self._observations[1]
            fraction = (window_started_at - first_at) / (second_at - first_at)
            frames_at_window_start += fraction * (second_frames - first_frames)

        elapsed = last_at - window_started_at
        generated_frames = last_frames - frames_at_window_start
        if elapsed > 0.0 and generated_frames > 0.0:
            self._frame_interval = max(
                elapsed / generated_frames,
                self._minimum_frame_interval,
            )


def _contains(events: UserInputEvents, event_type: type[UserInputEvent]) -> bool:
    """Return whether ``events`` contains an instance of ``event_type``."""
    return any(isinstance(event, event_type) for event in events.get_events())


def _log_secondary_failure(message: str, error: BaseException) -> None:
    """Log a cleanup failure that cannot replace an earlier exception."""
    _LOGGER.error(message, exc_info=error)


def run_session(
    session: ISession,
    window: IClientWindow,
    *,
    metrics_output_sink: OutputSink | None = None,
    steps: int | None = None,
    max_pending: int = 2,
) -> None:
    """Run a session's UI and model loops.

    The calling UI thread handles the window and UI. A model thread runs the
    model loop. Returns when the client closes the window, when the
    model loop has finished and no generated frames are still waiting, or when
    either loop fails.

    Both loops are shut down, every sink opened is closed, and the session is
    closed, before this returns or raises.

    Args:
        session: Session to run.
        window: Source of input and destination for UI output.
        metrics_output_sink: Sink for model measurements, if requested. Receives
            copies of the model loop's results augmented with
            presentation-queue diagnostics; it never receives UI-loop results.
        steps: Maximum model steps; ``None`` runs until stopped.
        max_pending: Maximum model steps waiting to be shown.

    Raises:
        ValueError: ``steps`` is negative, or ``max_pending`` is not positive.
        BaseException: A loop's failure if one was queued, otherwise this
            function's own, otherwise the first cleanup failure. The rest are
            logged.
    """
    if steps is not None and steps < 0:
        raise ValueError(f"steps must be >= 0 or None, got {steps}.")
    if max_pending <= 0:
        raise ValueError(f"max_pending must be > 0, got {max_pending}.")

    session_desc = session.session_desc
    tick_seconds = 1.0 / session_desc.frames_per_second_for_ui
    presentation_clock = _PresentationClock(
        session_desc.frames_per_second_for_step,
        maximum_frames_per_second=session_desc.frames_per_second_for_ui,
    )
    event_buffer = EventBuffer()
    stop = session._shutdown_event
    presentation_manager = session._presentation_manager
    presentation_manager.configure(
        max_pending=max_pending,
        backpressure_mode=session_desc.backpressure_mode,
        stop=stop,
        put_timeout=tick_seconds,
    )
    model_thread_handle: threading.Thread | None = None
    ui_loop: IUILoop[object] | None = None
    model_loop: IModelLoop[object] | None = None
    high_level_failures: BaseException | None = None
    cleanup_failures: list[BaseException] = []
    attempted_output_sinks: list[OutputSink] = []

    def collect_input() -> None:
        events = window.get_user_input_events()
        event_buffer.append(events)
        if _contains(events, CloseUserInputEvent):
            stop.set()

    def run_ui_once() -> StepResult | None:
        """Run one UI step with events not yet seen by that loop."""
        if ui_loop is None:
            return None
        events, generation = event_buffer.read(_UI_READER_ID)
        step_index = ui_loop._begin_run(events, generation)
        if step_index is None or stop.is_set():
            return None
        result = ui_loop.step(step_index, events)
        if result is not None and not isinstance(result, StepResult):
            raise TypeError("A UI loop must return StepResult or None.")
        ui_loop._finish_run(result)
        return result

    def publish_model_results(
        generation: int,
        results: list[StepResult],
    ) -> None:
        if metrics_output_sink is not None:
            for result in results:
                reserved_names = sorted(
                    name
                    for name in result.metrics
                    if name.startswith(_RUNTIME_METRIC_PREFIX)
                )
                if reserved_names:
                    names = ", ".join(reserved_names)
                    raise ValueError(
                        f"Model metrics use reserved runtime metric names: {names}."
                    )
        if results:
            presentation_clock.observe_model_output(
                now=time.monotonic(),
                generation=generation,
                frame_count=results[0].frame_count,
            )
        if metrics_output_sink is None:
            presentation_manager.publish(generation, results)
            return

        queue_depth_before = presentation_manager.buffered_chunk_count
        publish_started_at = time.perf_counter()
        presentation_manager.publish(generation, results)
        publish_wait_s = time.perf_counter() - publish_started_at
        queue_depth_after = presentation_manager.buffered_chunk_count
        runtime_metrics: dict[str, float | int] = {
            "runtime_presentation_publish_wait_s": publish_wait_s,
            "runtime_presentation_queue_depth_before_count": queue_depth_before,
            "runtime_presentation_queue_depth_after_count": queue_depth_after,
            "runtime_presentation_queue_capacity_count": presentation_manager.buffer_capacity,
        }
        for result in results:
            metrics_output_sink.write(
                replace(
                    result,
                    metrics={**result.metrics, **runtime_metrics},
                )
            )

    def run_and_write_ui_once() -> None:
        """Run one UI step and write its result, if any."""
        result = run_ui_once()
        if result is None:
            return
        window.write(result)

    def tick_ui() -> None:
        assert ui_loop is not None
        generation = event_buffer.generation
        model_advanced = False
        now = time.monotonic()
        if presentation_clock.is_due(now, generation):
            model_advanced, _ = presentation_manager.advance(generation)
        if model_advanced:
            presentation_clock.mark_advanced(now)
            run_and_write_ui_once()
            return
        if session_desc.presentation_mode is PresentationMode.ON_DEMAND:
            return
        run_and_write_ui_once()

    try:
        session.init()
        registered_ui, registered_model = session._take_loops()
        ui_loop = registered_ui
        model_loop = registered_model
        event_buffer.register(_UI_READER_ID)
        event_buffer.register(_MODEL_READER_ID)

        attempted_output_sinks.append(window)
        window.open(session_desc)
        if metrics_output_sink is not None:
            attempted_output_sinks.append(metrics_output_sink)
            metrics_output_sink.open(session_desc)
        collect_input()
        tick_ui()

        if not stop.is_set():
            model_thread_handle = threading.Thread(
                target=model_loop._run_model_loop,
                kwargs={
                    "event_buffer": event_buffer,
                    "reader_id": _MODEL_READER_ID,
                    "publish": publish_model_results,
                    "max_steps": steps,
                },
                name=_MODEL_THREAD_NAME,
            )
            model_thread_handle.start()
            next_tick_at = time.monotonic() + tick_seconds

            # Keep servicing input and presenting queued frames until shutdown,
            # or until the model finishes and no generated frames remain.
            # A finished UI loop produces no further window output.
            while not stop.is_set():
                if (
                    not model_thread_handle.is_alive()
                    and not presentation_manager.has_pending_frames()
                ):
                    break
                wait_seconds = max(0.0, next_tick_at - time.monotonic())
                if stop.wait(wait_seconds):
                    break
                collect_input()
                if stop.is_set():
                    break
                tick_ui()
                event_buffer.collect_garbage()
                next_tick_at += tick_seconds
                completed_at = time.monotonic()
                if next_tick_at <= completed_at:
                    next_tick_at = completed_at + tick_seconds
    except BaseException as error:
        high_level_failures = error
    finally:
        stop.set()
        if model_thread_handle is not None:
            try:
                model_thread_handle.join()
            except BaseException as error:
                cleanup_failures.append(error)

        cleanup_failures.extend(session._shutdown_registered_loops())
        try:
            presentation_manager.clear()
        except BaseException as error:
            cleanup_failures.append(error)
        try:
            event_buffer.unregister(_UI_READER_ID)
            event_buffer.unregister(_MODEL_READER_ID)
            event_buffer.clear()
        except BaseException as error:
            cleanup_failures.append(error)

        for output_sink in attempted_output_sinks:
            try:
                output_sink.close()
            except BaseException as error:
                cleanup_failures.append(error)
        try:
            session.close()
        except BaseException as error:
            cleanup_failures.append(error)

    loop_failures = (
        None if session._failure_queue.empty() else session._failure_queue.get()
    )
    primary_failure = loop_failures or high_level_failures
    if primary_failure is None and cleanup_failures:
        primary_failure = cleanup_failures.pop(0)
    for error in cleanup_failures:
        _log_secondary_failure(
            "Cleanup failed after the session had already failed.", error
        )

    if presentation_manager.dropped_for_space:
        _LOGGER.warning(
            "Dropped %d model steps the window could not keep up with.",
            presentation_manager.dropped_for_space,
        )
    if presentation_manager.discarded_at_reset:
        _LOGGER.info(
            "Discarded %d model steps generated before a reset.",
            presentation_manager.discarded_at_reset,
        )
    if primary_failure is not None:
        raise primary_failure


__all__ = ["run_session"]
