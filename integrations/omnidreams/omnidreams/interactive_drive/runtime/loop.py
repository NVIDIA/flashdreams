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
from omnidreams.interactive_drive.runtime.timing import (
    ChunkHistory,
    ChunkTimes,
)
from omnidreams.interactive_drive.simulation.backend import SimulationBackend
from omnidreams.interactive_drive.types import DriverCommand, PresentedFrame
from omnidreams.interactive_drive.video_model.chunk_pipeline import (
    ChunkPipeline,
    ChunkRequest,
    QueuedFrame,
)

_PROFILE_INPUT_TO_PRESENT_ENV = "INTERACTIVE_DRIVE_PROFILE_INPUT_TO_PRESENT"
_PROFILE_INPUT_TO_PRESENT_INTERVAL_S_ENV = (
    "INTERACTIVE_DRIVE_PROFILE_INPUT_TO_PRESENT_INTERVAL_S"
)

_PROFILE_E2E_SUM_RAW_MS: float = 0.0
_PROFILE_E2E_SUM_ADJ_MS: float = 0.0
_PROFILE_E2E_COUNT: int = 0
_PROFILE_E2E_WINDOW_START: float | None = None


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
    global _PROFILE_E2E_SUM_RAW_MS
    global _PROFILE_E2E_SUM_ADJ_MS
    global _PROFILE_E2E_COUNT
    global _PROFILE_E2E_WINDOW_START

    _PROFILE_E2E_SUM_RAW_MS = 0.0
    _PROFILE_E2E_SUM_ADJ_MS = 0.0
    _PROFILE_E2E_COUNT = 0
    _PROFILE_E2E_WINDOW_START = None


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
    global _PROFILE_E2E_SUM_RAW_MS
    global _PROFILE_E2E_SUM_ADJ_MS
    global _PROFILE_E2E_COUNT
    global _PROFILE_E2E_WINDOW_START

    raw_ms = (present_time - input_sample_time) * 1000.0
    scheduled_ms = frame_index * (frame_interval_s * 1000.0)
    adj_ms = raw_ms - scheduled_ms
    _PROFILE_E2E_SUM_RAW_MS += raw_ms
    _PROFILE_E2E_SUM_ADJ_MS += adj_ms
    _PROFILE_E2E_COUNT += 1
    if _PROFILE_E2E_WINDOW_START is None:
        _PROFILE_E2E_WINDOW_START = present_time

    interval_s = _profile_input_to_present_interval_s()
    if present_time - _PROFILE_E2E_WINDOW_START < interval_s:
        return

    count = _PROFILE_E2E_COUNT
    if count <= 0:
        return
    window_s = present_time - _PROFILE_E2E_WINDOW_START
    wall_present_fps = float(count) / window_s if window_s > 1e-9 else 0.0
    avg_raw_ms = _PROFILE_E2E_SUM_RAW_MS / float(count)
    avg_adj_ms = _PROFILE_E2E_SUM_ADJ_MS / float(count)
    logger.info(
        "[profile] e2e "
        f"wall_present_fps={wall_present_fps:.1f} "
        f"avg_adj_control_to_present_ms={avg_adj_ms:.2f} "
        f"avg_raw_control_to_present_ms={avg_raw_ms:.2f} "
        f"samples={count}",
    )
    _PROFILE_E2E_SUM_RAW_MS = 0.0
    _PROFILE_E2E_SUM_ADJ_MS = 0.0
    _PROFILE_E2E_COUNT = 0
    _PROFILE_E2E_WINDOW_START = present_time


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
) -> ChunkRequest:
    request_time = time.perf_counter()
    chunk_index = state.next_chunk_index
    chunk_size = config.initial_chunk_size if chunk_index == 0 else config.chunk_size
    trajectory = simulation.pose_chunk(
        command=command,
        chunk_size=chunk_size,
        frame_interval_s=config.frame_interval_s,
        extrapolation_offset_s=0.0,
    )
    request_poses_ready_time = time.perf_counter()
    intended_present_times = [
        request_time + config.frame_interval_s * frame for frame in range(chunk_size)
    ]
    chunk_times = ChunkTimes.create(
        chunk_index=chunk_index,
        input_sample_time=input_sample_time,
        request_time=request_time,
        request_poses_ready_time=request_poses_ready_time,
        intended_present_times=intended_present_times,
    )
    chunk_history.append(chunk_times)
    state.next_chunk_index += 1
    state.chunks_outstanding += 1
    return ChunkRequest(trajectory=trajectory, chunk_times=chunk_times)


def present_queued_frame(
    queued_frame: QueuedFrame,
    presenter: PresenterBackend,
    view_mode: str,
    oob_message: str | None = None,
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
    presenter.present_frame(display_frame, view_mode=view_mode)
    present_time = time.perf_counter()
    frame_times.present_time = present_time
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
    if _profile_input_to_present_enabled():
        reset_input_to_present_profile_window()

    while not presenter.should_close:
        presenter.process_events()
        if presenter.should_close:
            break
        if runtime_controls.consume_reset_request():
            return True
        sampled = input_backend.sample()

        # Keep one chunk in flight.
        if should_request_chunk(state):
            chunk_request = make_chunk_request(
                state=state,
                simulation=simulation,
                command=sampled.command,
                input_sample_time=sampled.sample_time,
                chunk_history=chunk_history,
                config=config,
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
            time.sleep(
                min(config.poll_timeout_s, max(0.0, state.next_present_time - now))
            )
            continue

        if ready_frames:
            queued_frame = ready_frames.popleft()
            if queued_frame.chunk_times.chunk_index != state.last_consumed_chunk_index:
                state.last_consumed_chunk_index = queued_frame.chunk_times.chunk_index
                state.chunks_outstanding = max(0, state.chunks_outstanding - 1)
            present_queued_frame(
                queued_frame,
                presenter,
                view_mode=view_mode,
                oob_message=state.oob_message,
            )
            last_presented_frame = queued_frame.frame
            state.frame_count += 1
        else:
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
