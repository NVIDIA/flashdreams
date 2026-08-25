# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a session with a client window."""

import logging
import threading

from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.api_v2.loop import IModelLoop, IUILoop
from flashdreams.api_v2.output_sink import OutputSink
from flashdreams.api_v2.session import ISession
from flashdreams.api_v2.user_input_event_data import UserInputEventData
from flashdreams.runtime_v2.event_buffer import EventBuffer
from flashdreams.runtime_v2.session_desc import PresentationMode
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import CloseUserInputEventData
from flashdreams.runtime_v2.user_input_events import UserInputEvents

_LOGGER = logging.getLogger(__name__)
_MODEL_THREAD_NAME = "flashdreams-model-generation-thread"
_UI_READER_ID = 0
_MODEL_READER_ID = 1


def _contains(events: UserInputEvents, event_type: type[UserInputEventData]) -> bool:
    """Return whether any event in ``events`` carries ``event_type`` data."""
    return any(
        isinstance(event.get_event_data(), event_type) for event in events.get_events()
    )


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

    The calling thread handles the window and UI. The model runs on a separate
    Python thread. Returns when the client closes the window, when the model
    loop has finished and no generated frames are still waiting, or when either
    loop fails.

    Both loops are shut down, every sink opened is closed, and the session is
    closed, before this returns or raises.

    Args:
        session: Session to run.
        window: Source of input and destination for UI output.
        metrics_output_sink: Sink for model measurements, if requested. Receives
            the model loop's results rather than the UI loop's.
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

    def collect_input() -> UserInputEvents:
        events = window.get_user_input_events()
        event_buffer.append(events)
        if _contains(events, CloseUserInputEventData):
            stop.set()
        return events

    def run_ui_once() -> StepResult | None:
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
        presentation_manager.publish(generation, results)
        if metrics_output_sink is not None:
            for result in results:
                metrics_output_sink.write(result)

    def tick_ui() -> None:
        model_advanced, _ = presentation_manager.advance(event_buffer.generation)
        # Safe presentation does not redraw a frame the UI already consumed.
        if (
            session_desc.presentation_mode is PresentationMode.ONLY_PRESENT_NEW
            and not model_advanced
        ):
            return
        result = run_ui_once()
        if result is not None:
            window.write(result)

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

            # Keep servicing input and presenting queued frames until shutdown,
            # or until the model finishes and no generated frames remain.
            # A finished UI loop produces no further window output.
            while not stop.is_set():
                if (
                    not model_thread_handle.is_alive()
                    and not presentation_manager.has_pending_frames()
                ):
                    break
                if stop.wait(tick_seconds):
                    break
                collect_input()
                if stop.is_set():
                    break
                tick_ui()
                event_buffer.collect_garbage()
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
        presentation_manager.clear()
        event_buffer.unregister(_UI_READER_ID)
        event_buffer.unregister(_MODEL_READER_ID)
        event_buffer.clear()

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
