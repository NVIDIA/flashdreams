# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
import queue
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Protocol

from loguru import logger
from omnidreams.interactive_drive.input.backend import InputBackend
from omnidreams.interactive_drive.runtime.runtime_controls import RuntimeControls
from omnidreams.interactive_drive.simulation.backend import SimulationBackend
from omnidreams.interactive_drive.types import DriverCommand, PresentedFrame
from omnidreams.interactive_drive.video_model.chunk_pipeline import (
    ChunkPipeline,
    ChunkRequest,
    QueuedFrame,
)

from flashdreams.infra.acceleration.prewarm import is_warmup_index
from flashdreams.serving.realtime.timing import (
    ChunkHistory,
    ChunkPrediction,
    ChunkTimes,
    InputToPresentProfileWindow,
    TraceComponentValue,
    TraceContext,
    event_dependencies,
    trace_time_ns,
)

_PROFILE_INPUT_TO_PRESENT_ENV = "INTERACTIVE_DRIVE_PROFILE_INPUT_TO_PRESENT"
_PROFILE_INPUT_TO_PRESENT_INTERVAL_S_ENV = (
    "INTERACTIVE_DRIVE_PROFILE_INPUT_TO_PRESENT_INTERVAL_S"
)

_PROFILE_E2E_WINDOW = InputToPresentProfileWindow()


def _profile_input_to_present_enabled() -> bool:
    raw = os.environ.get(_PROFILE_INPUT_TO_PRESENT_ENV, "")
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _profile_input_to_present_interval_s() -> float:
    raw = os.environ.get(_PROFILE_INPUT_TO_PRESENT_INTERVAL_S_ENV, "2").strip()
    try:
        value = float(raw)
    except ValueError:
        return 2.0
    return max(0.25, value)


def reset_input_to_present_profile_window() -> None:
    """Clear accumulated e2e samples when the main loop starts."""
    _PROFILE_E2E_WINDOW.reset(interval_s=_profile_input_to_present_interval_s())


def _chunk_frame_interval_s(chunk_times: ChunkTimes) -> float:
    frames = chunk_times.frames
    if len(frames) >= 2:
        return max(
            0.0, frames[1].intended_present_time - frames[0].intended_present_time
        )
    return 0.0


def _record_input_to_present_for_profile(
    *,
    present_time: float,
    input_sample_time: float,
    frame_index: int,
    frame_interval_s: float,
) -> None:
    _PROFILE_E2E_WINDOW.interval_s = _profile_input_to_present_interval_s()
    summary = _PROFILE_E2E_WINDOW.record(
        present_time=present_time,
        input_sample_time=input_sample_time,
        frame_index=frame_index,
        frame_interval_s=frame_interval_s,
    )
    if summary is not None:
        logger.info(summary.log_message())


class PresenterBackend(Protocol):
    @property
    def should_close(self) -> bool: ...

    def process_events(self) -> None: ...

    def present_frame(self, frame: PresentedFrame, view_mode: str) -> None: ...

    def close(self) -> None: ...


class MainLoopState:
    """Mutable per-iteration counters and timestamps for :func:`run_main_loop`.

    Bundled so helper functions can advance loop state in place instead of
    threading tuples or closures through each call.
    """

    next_present_time: float
    next_chunk_index: int
    frame_count: int
    chunks_outstanding: int
    last_consumed_chunk_index: int | None
    # Out-of-bounds overlay text, refreshed each tick from
    # ``simulation.last_proximity``. ``None`` when solidly in-bounds.
    oob_message: str | None
    # Consecutive chunks at/above ``LoopConfig.oob_respawn_proximity``; the
    # auto-respawn fires once it reaches ``oob_respawn_debounce_chunks``.
    oob_respawn_streak: int

    def __init__(self) -> None:
        self.next_present_time = time.perf_counter()
        self.next_chunk_index = 0
        self.frame_count = 0
        self.chunks_outstanding = 0
        self.last_consumed_chunk_index = None
        self.oob_message = None
        self.oob_respawn_streak = 0


@dataclass(frozen=True)
class LoopConfig:
    initial_chunk_size: int
    chunk_size: int
    frame_interval_s: float
    poll_timeout_s: float = 0.001
    history_capacity: int = 16
    # OOB thresholds applied to ``simulation.last_proximity`` (see
    # :meth:`MapBounds.proximity` for the 0.0 / (0,1] / 2.0 semantics).
    # Defaults match the alpasim driver: warn at the "approaching" 0.6,
    # respawn at the 2.0 off-map sentinel. Both no-op when the scene has no
    # geometry (proximity reads 0.0).
    oob_warn_proximity: float = 0.6
    oob_respawn_proximity: float = 2.0
    # Consecutive chunks at/above ``oob_respawn_proximity`` before the
    # auto-respawn fires. Default 1 matches alpasim (immediate respawn);
    # raise it to debounce measurement noise.
    oob_respawn_debounce_chunks: int = 1
    # When set, the loop exits cleanly once chunk index N has been consumed
    # off the present queue. Chunk 0 is warmup and excluded from the trace,
    # so consuming chunks 0..N yields N traced chunks (1..N).
    stop_after_consumed_chunks: int | None = None


# OOB overlay strings, module-level so the HUD can match on them for styling.
OOB_WARN_MESSAGE = "Approaching map edge, turn back to avoid respawn"
OOB_RESPAWN_MESSAGE = "Respawning..."


def should_request_chunk(state: MainLoopState) -> bool:
    return state.chunks_outstanding < 1


def make_chunk_request(
    state: MainLoopState,
    simulation: SimulationBackend,
    command: DriverCommand,
    input_sample_time: float,
    chunk_history: ChunkHistory,
    config: LoopConfig,
    input_sample_event: int | None = None,
    trace_context: TraceContext | None = None,
) -> ChunkRequest:
    request_time = time.perf_counter()
    request_event = _trace_main_instant(
        trace_context,
        "request",
        time_value=request_time,
        depends_on=event_dependencies(input_sample_event),
        chunk_index=state.next_chunk_index,
    )
    chunk_index = state.next_chunk_index
    chunk_size = config.initial_chunk_size if chunk_index == 0 else config.chunk_size
    trajectory = simulation.pose_chunk(
        command=command,
        chunk_size=chunk_size,
        frame_interval_s=config.frame_interval_s,
        extrapolation_offset_s=0.0,
    )
    request_poses_ready_time = time.perf_counter()
    simulation_event = _trace_main_range(
        trace_context,
        "simulation_step",
        begin_time=request_time,
        end_time=request_poses_ready_time,
        depends_on=event_dependencies(request_event),
        chunk_index=chunk_index,
        chunk_size=chunk_size,
    )
    prediction = ChunkPrediction.create(
        request_time=request_time, frame_interval_s=config.frame_interval_s
    )
    intended_present_times = [
        request_time + config.frame_interval_s * frame for frame in range(chunk_size)
    ]
    chunk_times = ChunkTimes.create(
        chunk_index=chunk_index,
        input_sample_time=input_sample_time,
        request_time=request_time,
        request_poses_ready_time=request_poses_ready_time,
        intended_present_times=intended_present_times,
        prediction=prediction,
    )
    chunk_history.append(chunk_times)
    state.next_chunk_index += 1
    state.chunks_outstanding += 1
    return ChunkRequest(
        trajectory=trajectory,
        chunk_times=chunk_times,
        trace_dependency_event=simulation_event,
    )


def present_queued_frame(
    queued_frame: QueuedFrame,
    presenter: PresenterBackend,
    view_mode: str,
    oob_message: str | None = None,
    trace_context: TraceContext | None = None,
    trace_dependencies: list[int] | None = None,
) -> float:
    """Hand a freshly-dequeued frame to the presenter.

    ``oob_message`` is merged into the frame's ``status_message`` only for
    the duration of this present call (via :func:`dataclasses.replace`),
    so the original ``QueuedFrame`` keeps whatever message the backend
    attached -- e.g. the world-model's "Optimizing world model..."
    transition text on the first chunk's last frame stays intact across
    re-presents that intersperse the warmup window with an OOB warning.
    """
    frame_times = queued_frame.chunk_times.frames[queued_frame.frame_index]
    frame_times.sample_display_pose_time = time.perf_counter()
    display_frame = _frame_with_overlay(queued_frame.frame, oob_message)
    present_call_begin_time = time.perf_counter()
    presenter.present_frame(display_frame, view_mode=view_mode)
    present_time = time.perf_counter()
    frame_times.present_time = present_time
    if trace_context is not None:
        if frame_times.image_ready_time is None:
            raise RuntimeError("queued frame is missing image_ready_time")
        chunk_times = queued_frame.chunk_times
        _trace_main_range(
            trace_context,
            "present_frame",
            begin_time=present_call_begin_time,
            end_time=present_time,
            depends_on=[] if trace_dependencies is None else trace_dependencies,
            chunk_index=chunk_times.chunk_index,
            frame_index=queued_frame.frame_index,
            per_frame_error_ms=(present_time - frame_times.intended_present_time)
            * 1000.0,
            input_sample_time_ns=trace_time_ns(chunk_times.input_sample_time),
            image_ready_time_ns=trace_time_ns(frame_times.image_ready_time),
        )
    if _profile_input_to_present_enabled():
        _record_input_to_present_for_profile(
            present_time=present_time,
            input_sample_time=queued_frame.chunk_times.input_sample_time,
            frame_index=queued_frame.frame_index,
            frame_interval_s=_chunk_frame_interval_s(queued_frame.chunk_times),
        )
    return present_time


def _frame_with_overlay(
    frame: PresentedFrame, oob_message: str | None
) -> PresentedFrame:
    """Return ``frame`` with ``oob_message`` merged into ``status_message``.

    The OOB message wins over an existing ``status_message`` because it's
    a more time-sensitive affordance (the user is about to be teleported);
    returns the frame unchanged when there's no OOB message to merge in.
    """
    if oob_message is None:
        return frame
    return replace(frame, status_message=oob_message)


def update_oob_state(
    state: MainLoopState, simulation: SimulationBackend, config: LoopConfig
) -> bool:
    """Refresh ``state.oob_message`` from the simulation's OOB proximity.

    Reads ``simulation.last_proximity`` defensively (defaults to ``0.0`` for
    backends that don't track OOB). Returns ``True`` only on the chunk that
    fires the auto-respawn, which requires proximity to stay at/above
    :attr:`LoopConfig.oob_respawn_proximity` for
    :attr:`LoopConfig.oob_respawn_debounce_chunks` consecutive chunks
    (debouncing single-chunk spikes from a corner ray missing the mesh).
    """
    proximity = float(getattr(simulation, "last_proximity", 0.0))
    previous_message = state.oob_message

    if proximity >= config.oob_respawn_proximity:
        state.oob_respawn_streak += 1
        state.oob_message = OOB_RESPAWN_MESSAGE
        if state.oob_respawn_streak >= max(1, config.oob_respawn_debounce_chunks):
            _log_oob_transition(
                previous_message,
                OOB_RESPAWN_MESSAGE,
                proximity,
                streak=state.oob_respawn_streak,
                action="firing respawn",
            )
            return True
        if previous_message != OOB_RESPAWN_MESSAGE:
            _log_oob_transition(
                previous_message,
                OOB_RESPAWN_MESSAGE,
                proximity,
                streak=state.oob_respawn_streak,
                action="respawn pending",
            )
        return False

    # Below respawn threshold; reset the debounce streak so a brief dip
    # back into the warning band can't accumulate toward a respawn.
    state.oob_respawn_streak = 0

    if proximity >= config.oob_warn_proximity:
        state.oob_message = OOB_WARN_MESSAGE
        if previous_message != OOB_WARN_MESSAGE:
            _log_oob_transition(
                previous_message,
                OOB_WARN_MESSAGE,
                proximity,
                streak=0,
                action="warning",
            )
        return False

    state.oob_message = None
    if previous_message is not None:
        _log_oob_transition(
            previous_message,
            None,
            proximity,
            streak=0,
            action="cleared",
        )
    return False


def _log_oob_transition(
    previous: str | None,
    current: str | None,
    proximity: float,
    *,
    streak: int,
    action: str,
) -> None:
    """Log OOB state transitions to stderr, once per state edge."""
    prev_label = "in-bounds" if previous is None else _truncate(previous, 32)
    curr_label = "in-bounds" if current is None else _truncate(current, 32)
    logger.info(
        f"[loop] oob {prev_label!r} -> {curr_label!r}"
        f" proximity={proximity:.3f} streak={streak} action={action}",
    )


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "\u2026"


def push_telemetry(
    runtime_controls: RuntimeControls, simulation: SimulationBackend
) -> None:
    """Forward ``simulation.current_state`` to ``runtime_controls``.

    No-ops for controls that don't expose ``update_telemetry`` (test fakes,
    custom controllers); only :class:`KeyboardState` consumes it today.
    """
    update = getattr(runtime_controls, "update_telemetry", None)
    if update is None:
        return
    update(simulation.current_state)


def _prepare_queued_frame(
    queued_frame: QueuedFrame,
    presenter: PresenterBackend,
    view_mode: str,
) -> None:
    prepare_frame = getattr(presenter, "prepare_frame", None)
    if callable(prepare_frame):
        prepare_frame(queued_frame.frame, view_mode=view_mode)


def _drain_pipeline_frames(
    *,
    pipeline: ChunkPipeline,
    ready_frames: "deque[QueuedFrame]",
    presenter: PresenterBackend,
    view_mode: str,
) -> None:
    current_generation = pipeline.current_generation
    while True:
        try:
            queued_frame = pipeline.frame_queue.get_nowait()
        except queue.Empty:
            return
        if queued_frame.generation != current_generation:
            # Stale frame from a superseded rollout/scene (generation bumped);
            # drop it so old content isn't flashed over the new load.
            continue
        _prepare_queued_frame(queued_frame, presenter, view_mode)
        ready_frames.append(queued_frame)


def run_main_loop(
    presenter: PresenterBackend,
    runtime_controls: RuntimeControls,
    initial_presented_frame: PresentedFrame,
    input_backend: InputBackend,
    simulation: SimulationBackend,
    pipeline: ChunkPipeline,
    config: LoopConfig,
    loading_status: Callable[[], str | None] | None = None,
    trace_context: TraceContext | None = None,
) -> bool:
    """Drive the request -> render -> present pipeline.

    Authoritative state advances inside ``simulation.pose_chunk`` per chunk
    request, so sim cadence is gated by display-driven requests, not the poll
    rate. ``initial_presented_frame`` seeds the re-present path used while the
    pipeline warms up; ``loading_status`` (if given) supplies the loading-phase
    overlay text until the first real frame, with the OOB warning taking
    precedence.

    Returns ``True`` when the user requested a reset or the OOB auto-respawn
    fired (caller rebuilds the simulation and re-runs), ``False`` when the
    presenter requested close.
    """
    state = MainLoopState()
    last_presented_frame: PresentedFrame = initial_presented_frame
    ready_frames: deque[QueuedFrame] = deque()
    chunk_history = ChunkHistory(config.history_capacity)
    last_input_sample_event: int | None = None
    last_present_wait_event: int | None = None
    if _profile_input_to_present_enabled():
        reset_input_to_present_profile_window()

    while not presenter.should_close:
        presenter.process_events()
        if presenter.should_close:
            break
        if runtime_controls.consume_reset_request():
            return True
        active_trace = (
            trace_context if state.last_consumed_chunk_index is not None else None
        )
        input_sample_begin = time.perf_counter()
        sampled = input_backend.sample()
        input_sample_end = time.perf_counter()
        last_input_sample_event = _trace_main_range(
            active_trace,
            "input_sample",
            begin_time=input_sample_begin,
            end_time=input_sample_end,
            depends_on=[],
        )

        # Keep one chunk in flight.
        if should_request_chunk(state):
            chunk_request = make_chunk_request(
                state=state,
                simulation=simulation,
                command=sampled.command,
                input_sample_time=sampled.sample_time,
                chunk_history=chunk_history,
                config=config,
                input_sample_event=last_input_sample_event,
                trace_context=active_trace,
            )
            pipeline.request_pose_chunk(chunk_request)
            # The pose chunk just advanced authoritative state, so refresh the
            # OOB overlay from the new boundary frame and auto-respawn (same
            # ``return True`` as a manual reset) when far enough off-map.
            if update_oob_state(state, simulation, config):
                return True
            # Republish telemetry per chunk so read-side observers (e.g. the
            # presenter's ``/state`` endpoint) see the latest state.
            push_telemetry(runtime_controls, simulation)

        view_mode = runtime_controls.view_mode
        _drain_pipeline_frames(
            pipeline=pipeline,
            ready_frames=ready_frames,
            presenter=presenter,
            view_mode=view_mode,
        )

        now = time.perf_counter()
        if now < state.next_present_time:
            wait_begin = now
            time.sleep(
                min(config.poll_timeout_s, max(0.0, state.next_present_time - now))
            )
            wait_end = time.perf_counter()
            last_present_wait_event = _trace_main_range(
                active_trace,
                "present_wait",
                begin_time=wait_begin,
                end_time=wait_end,
                depends_on=[],
            )
            continue

        if ready_frames:
            queued_frame = ready_frames.popleft()
            chunk_transitioned = (
                queued_frame.chunk_times.chunk_index != state.last_consumed_chunk_index
            )
            if queued_frame.chunk_times.chunk_index != state.last_consumed_chunk_index:
                state.last_consumed_chunk_index = queued_frame.chunk_times.chunk_index
                state.chunks_outstanding = max(0, state.chunks_outstanding - 1)
            present_trace = (
                None
                if is_warmup_index(queued_frame.chunk_times.chunk_index)
                else trace_context
            )
            present_queued_frame(
                queued_frame,
                presenter,
                view_mode=view_mode,
                oob_message=state.oob_message,
                trace_context=present_trace,
                trace_dependencies=event_dependencies(
                    queued_frame.worker_ready_event_id,
                    last_present_wait_event,
                ),
            )
            last_present_wait_event = None
            last_presented_frame = queued_frame.frame
            state.frame_count += 1
            if (
                chunk_transitioned
                and config.stop_after_consumed_chunks is not None
                and state.last_consumed_chunk_index is not None
                and state.last_consumed_chunk_index >= config.stop_after_consumed_chunks
            ):
                return False
        else:
            # A re-present consumes the preceding sleep just like a real
            # present, so drop it instead of carrying it forward as a
            # dependency of some later, unrelated present.
            last_present_wait_event = None
            # Re-present the last frame with the current overlay: OOB warning
            # wins, else the loading-phase status until the first real frame.
            # The merged frame is local so ``last_presented_frame`` is unchanged.
            overlay = state.oob_message
            if (
                overlay is None
                and loading_status is not None
                and state.frame_count == 0
            ):
                overlay = loading_status()
            presenter.present_frame(
                _frame_with_overlay(last_presented_frame, overlay),
                view_mode=view_mode,
            )

        state.next_present_time += config.frame_interval_s
    return False


def _trace_main_instant(
    trace_context: TraceContext | None,
    name: str,
    *,
    time_value: float,
    depends_on: list[int],
    chunk_index: int,
) -> int | None:
    if trace_context is None:
        return None
    return trace_context.add_instant(
        name,
        thread=trace_context.main_thread,
        time_ns=trace_time_ns(time_value),
        depends_on=depends_on,
        chunk_index=chunk_index,
    )


def _trace_main_range(
    trace_context: TraceContext | None,
    name: str,
    *,
    begin_time: float,
    end_time: float,
    depends_on: list[int],
    chunk_index: int | None = None,
    chunk_size: int | None = None,
    frame_index: int | None = None,
    per_frame_error_ms: float | None = None,
    input_sample_time_ns: int | None = None,
    image_ready_time_ns: int | None = None,
) -> int | None:
    if trace_context is None:
        return None
    components: dict[str, TraceComponentValue] = {}
    if chunk_index is not None:
        components["chunk_index"] = chunk_index
    if chunk_size is not None:
        components["chunk_size"] = chunk_size
    if frame_index is not None:
        components["frame_index"] = frame_index
    if per_frame_error_ms is not None:
        components["per_frame_error_ms"] = per_frame_error_ms
    if input_sample_time_ns is not None:
        components["input_sample_time_ns"] = input_sample_time_ns
    if image_ready_time_ns is not None:
        components["image_ready_time_ns"] = image_ready_time_ns
    return trace_context.add_range(
        name,
        thread=trace_context.main_thread,
        begin_ns=trace_time_ns(begin_time),
        end_ns=trace_time_ns(end_time),
        depends_on=depends_on,
        **components,
    )
