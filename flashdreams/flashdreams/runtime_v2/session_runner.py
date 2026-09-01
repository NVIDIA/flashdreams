# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a session with a client window."""

import logging
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path

from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.api_v2.loop import IModelLoop, IUILoop, ModelInferenceState
from flashdreams.api_v2.output_sink import OutputSink
from flashdreams.api_v2.session import ISession
from flashdreams.runtime_v2.event_buffer import EventBuffer
from flashdreams.runtime_v2.session_desc import PresentationMode, SessionDesc
from flashdreams.runtime_v2.step_result import StepResult

_LOGGER = logging.getLogger(__name__)
_MODEL_THREAD_NAME = "flashdreams-model-generation-thread"
_UI_READER_ID = 0
_MODEL_READER_ID = 1

_TRACE_METADATA_KEY = "trace_chunk_lifecycle"
_TRACE_PATH_METADATA_KEY = "trace_chunk_lifecycle_path"
_TRACE_LOGGER = logging.getLogger("flashdreams.runtime_v2.chunk_trace")
_TRACE_PREFIX = "[runtime-v2-chunk-trace]"


@dataclass(frozen=True, slots=True)
class _ChunkTraceLog:
    """Logger state restored after one traced session."""

    handler: logging.FileHandler
    previous_level: int
    previous_propagate: bool


def _log_secondary_failure(message: str, error: BaseException) -> None:
    """Log a cleanup failure that cannot replace an earlier exception."""
    _LOGGER.error(message, exc_info=error)


def run_session(
    session: ISession,
    window: IClientWindow,
    *,
    metrics_output_sink: OutputSink | None = None,
    steps: int | None = None,
) -> SessionDesc | None:
    """Run a session's UI and model loops.

    The calling UI thread handles the window and UI. A model thread runs the
    model loop. Returns when the client closes the window, requests a new
    session, when the UI finishes, or when either loop fails. While the model
    is not running, an unfinished UI ticks regardless of presentation mode so
    it can request another session.

    Both loops, the metrics sink, and the session are closed before this returns
    or raises. The client window stays open only when a clean replacement was
    requested; otherwise it is closed.

    Args:
        session: Session to run.
        window: Source of input and destination for UI output.
        metrics_output_sink: Sink for model measurements, if requested. Receives
            the model loop's results rather than the UI loop's.
        steps: Maximum model steps before ending the session; ``None`` leaves
            session completion to the UI or client window.

    Returns:
        The resolved description for a requested replacement session, or
        ``None`` when the application run should end.

    Raises:
        ValueError: ``steps`` is negative.
        BaseException: A loop's failure if one was queued, otherwise this
            function's own, otherwise the first cleanup failure. The rest are
            logged.
    """
    if steps is not None and steps < 0:
        raise ValueError(f"steps must be >= 0 or None, got {steps}.")

    event_buffer = EventBuffer()
    model_thread_handle: threading.Thread | None = None
    ui_loop: IUILoop[object] | None = None
    model_loop: IModelLoop[object] | None = None
    high_level_failures: BaseException | None = None
    cleanup_failures: list[BaseException] = []
    next_session_desc: SessionDesc | None = None
    stop: threading.Event | None = None
    presentation_manager = None
    trace_log: _ChunkTraceLog | None = None
    try:
        session_desc = session.session_desc
        tick_seconds = 1.0 / session_desc.frames_per_second_for_ui
        stop = session._shutdown_event
        presentation_manager = session._presentation_manager
        trace_chunk_lifecycle = session_desc.metadata.get(_TRACE_METADATA_KEY) is True
        presentation_manager.configure(
            backpressure_mode=session_desc.backpressure_mode,
            stop=stop,
            put_timeout=tick_seconds,
            trace_chunk_lifecycle=trace_chunk_lifecycle,
            frames_per_second=session_desc.frames_per_second_for_step,
            maximum_frames_per_second=session_desc.frames_per_second_for_ui,
        )

        def collect_input() -> None:
            event_buffer.append(window.get_user_input_events())

        def run_ui_once(*, step_requested: bool = True) -> None:
            """Process UI lifecycle control and run a requested UI step."""
            nonlocal next_session_desc
            if ui_loop is None:
                return
            events, generation = event_buffer.read(_UI_READER_ID)

            loopResult = ui_loop._begin_run(events, generation)
            if loopResult.stop_requested:
                stop.set()
                return
            if loopResult.new_session_request is not None:
                next_session_desc = replace(
                    session_desc,
                    metadata={
                        **session_desc.metadata,
                        **loopResult.new_session_request,
                    },
                )
                stop.set()
                return
            if loopResult.step_index is None or not step_requested:
                return
            result = ui_loop.step(loopResult.step_index, ui_loop.user_events)
            if result is not None and not isinstance(result, StepResult):
                raise TypeError("A UI loop must return StepResult or None.")
            ui_loop._finish_run(result)
            if result is not None:
                window.write(result)

        def publish_model_results(
            generation: int,
            results: list[StepResult],
            step_elapsed_s: float,
        ) -> None:
            presentation_manager.publish(
                generation,
                results,
                step_elapsed_s=step_elapsed_s,
            )
            if metrics_output_sink is not None:
                for result in results:
                    metrics_output_sink.write(result)

        def run_model() -> None:
            assert ui_loop is not None
            assert model_loop is not None
            ui_loop._set_model_inference_state(ModelInferenceState.RUNNING)
            try:
                model_loop._run_model_loop(
                    event_buffer=event_buffer,
                    reader_id=_MODEL_READER_ID,
                    publish=publish_model_results,
                    max_steps=steps,
                )
            finally:
                ui_loop._set_model_inference_state(ModelInferenceState.FINISHED)

        def tick_ui() -> None:
            # ensure that the HIGH PRIORITY presentation context is default for UI loop
            with presentation_manager.presentation_context():
                assert ui_loop is not None
                generation = event_buffer.generation
                model_advanced, _ = presentation_manager.advance(generation)
                inference_state = ui_loop.model_inference_state
                if model_advanced:
                    step_requested = True
                elif inference_state is ModelInferenceState.RUNNING:
                    step_requested = (
                        session_desc.presentation_mode is PresentationMode.CONTINUOUS
                    )
                elif inference_state is ModelInferenceState.NOT_STARTED:
                    step_requested = True
                else:
                    step_requested = (
                        not presentation_manager.has_pending_frames()
                        and session._failure_queue.empty()
                    )
                run_ui_once(step_requested=step_requested)

        trace_log = (
            _open_chunk_trace(session_desc.metadata.get(_TRACE_PATH_METADATA_KEY))
            if trace_chunk_lifecycle
            else None
        )
        if trace_chunk_lifecycle:
            _TRACE_LOGGER.info(
                "%s phase=session_config time_ns=%d backpressure=%s "
                "presentation=%s chunk_buffer_capacity=%d step_fps=%d ui_fps=%d "
                "width=%d height=%d trace_path=%s",
                _TRACE_PREFIX,
                time.monotonic_ns(),
                session_desc.backpressure_mode.value,
                session_desc.presentation_mode.value,
                presentation_manager.buffered_chunk_capacity,
                session_desc.frames_per_second_for_step,
                session_desc.frames_per_second_for_ui,
                session_desc.video_width,
                session_desc.video_height,
                trace_log.handler.baseFilename if trace_log is not None else "none",
            )
        session.init()
        registered_ui, registered_model = session._take_loops()
        ui_loop = registered_ui
        model_loop = registered_model
        event_buffer.register(_UI_READER_ID)
        event_buffer.register(_MODEL_READER_ID)

        window.open(session_desc)
        if metrics_output_sink is not None:
            metrics_output_sink.open(session_desc)
        collect_input()
        tick_ui()

        if not stop.is_set():
            model_thread_handle = threading.Thread(
                target=run_model,
                name=_MODEL_THREAD_NAME,
            )
            model_thread_handle.start()
            next_tick_at = time.monotonic() + tick_seconds

            while not stop.is_set():
                if (
                    not model_thread_handle.is_alive()
                    and not presentation_manager.has_pending_frames()
                    and (
                        not session._failure_queue.empty()
                        or steps is not None
                        or ui_loop.is_finished()
                    )
                ):
                    # Input may have arrived with the final presented frame.
                    # Establish the terminal boundary before deciding that a
                    # naturally completed session has no replacement.
                    collect_input()
                    run_ui_once(step_requested=False)
                    break
                wait_seconds = max(0.0, next_tick_at - time.monotonic())
                if stop.wait(wait_seconds):
                    break
                collect_input()
                tick_ui()
                event_buffer.collect_garbage()
                next_tick_at += tick_seconds
                completed_at = time.monotonic()
                if next_tick_at <= completed_at:
                    next_tick_at = completed_at + tick_seconds
    except BaseException as error:
        high_level_failures = error
    finally:
        if stop is not None:
            stop.set()
        if model_thread_handle is not None:
            try:
                model_thread_handle.join()
            except BaseException as error:
                cleanup_failures.append(error)

        if presentation_manager is not None:
            try:
                presentation_manager.close()
            except BaseException as error:
                cleanup_failures.append(error)
        cleanup_failures.extend(session._shutdown_registered_loops())
        try:
            event_buffer.unregister(_UI_READER_ID)
            event_buffer.unregister(_MODEL_READER_ID)
            event_buffer.clear()
        except BaseException as error:
            cleanup_failures.append(error)

        if metrics_output_sink is not None:
            try:
                metrics_output_sink.close()
            except BaseException as error:
                cleanup_failures.append(error)
        if next_session_desc is None:
            try:
                window.close()
            except BaseException as error:
                cleanup_failures.append(error)
        try:
            session.close()
        except BaseException as error:
            cleanup_failures.append(error)
        if trace_log is not None:
            try:
                _close_chunk_trace(trace_log)
            except BaseException as error:
                cleanup_failures.append(error)

    loop_failures = (
        None if session._failure_queue.empty() else session._failure_queue.get()
    )
    primary_failure = loop_failures or high_level_failures
    if primary_failure is None and cleanup_failures:
        primary_failure = cleanup_failures.pop(0)

    # A replacement may take ownership of the window only after every resource
    # owned by the old session has been released successfully.
    if next_session_desc is not None and primary_failure is not None:
        try:
            window.close()
        except BaseException as error:
            cleanup_failures.append(error)
    for error in cleanup_failures:
        _log_secondary_failure(
            "Cleanup failed after the session had already failed.", error
        )

    if presentation_manager is not None and presentation_manager.dropped_for_space:
        _LOGGER.warning(
            "Dropped %d model chunks the window could not keep up with.",
            presentation_manager.dropped_for_space,
        )
    if presentation_manager is not None and presentation_manager.discarded_at_reset:
        _LOGGER.info(
            "Discarded %d model chunks generated before a reset.",
            presentation_manager.discarded_at_reset,
        )
    if primary_failure is not None:
        raise primary_failure
    return next_session_desc


def _open_chunk_trace(path_value: object) -> _ChunkTraceLog:
    """Open a line-buffered lifecycle trace for one session."""
    if not isinstance(path_value, str | Path):
        raise TypeError(
            f"{_TRACE_PATH_METADATA_KEY} must be a filesystem path when tracing"
        )
    path = Path(path_value).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    # ponytail: this process-global logger assumes one active traced V2 session;
    # pass a per-session sink through the loop contracts if concurrent sessions land.
    previous_level = _TRACE_LOGGER.level
    previous_propagate = _TRACE_LOGGER.propagate
    _TRACE_LOGGER.addHandler(handler)
    _TRACE_LOGGER.setLevel(logging.INFO)
    _TRACE_LOGGER.propagate = False
    return _ChunkTraceLog(handler, previous_level, previous_propagate)


def _close_chunk_trace(trace_log: _ChunkTraceLog) -> None:
    """Flush and close a session trace, restoring the shared logger."""
    _TRACE_LOGGER.removeHandler(trace_log.handler)
    try:
        trace_log.handler.close()
    finally:
        _TRACE_LOGGER.setLevel(trace_log.previous_level)
        _TRACE_LOGGER.propagate = trace_log.previous_propagate


__all__ = ["run_session"]
