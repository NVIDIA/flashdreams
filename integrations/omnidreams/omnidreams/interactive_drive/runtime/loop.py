# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import queue
import time
from dataclasses import dataclass, replace
from typing import Protocol

from omnidreams.interactive_drive.input.backend import InputBackend
from omnidreams.interactive_drive.runtime.runtime_controls import RuntimeControls
from omnidreams.interactive_drive.runtime.timing import (
    ChunkHistory,
    ChunkPrediction,
    ChunkTimes,
)
from omnidreams.interactive_drive.simulation.backend import SimulationBackend
from omnidreams.interactive_drive.types import DriverCommand, PresentedFrame
from omnidreams.interactive_drive.video_model.chunk_pipeline import (
    ChunkPipeline,
    ChunkRequest,
    QueuedFrame,
)


class PresenterBackend(Protocol):
    @property
    def should_close(self) -> bool: ...

    def process_events(self) -> None: ...

    def present_frame(self, frame: PresentedFrame, view_mode: str) -> None: ...

    # ``close`` is part of every concrete presenter
    # (:class:`SlangPyPresenter`, :class:`SlangPyHudPresenter`) and is
    # invoked from :meth:`InteractiveDriveApp.run`'s teardown path.
    def close(self) -> None: ...


class MainLoopState:
    """Mutable per-iteration counters and timestamps for :func:`run_main_loop`.

    Bundled into a single object so helper functions can advance the loop's
    state directly without returning tuples or capturing mutable closures.
    Kept as a plain class rather than ``@dataclass`` because the workspace
    standard prefers frozen dataclasses for value objects, and this is
    explicitly mutable per-iteration scratch.
    """

    next_present_time: float
    next_chunk_index: int
    frame_count: int
    chunks_outstanding: int
    last_consumed_chunk_index: int | None
    # Out-of-bounds overlay text, refreshed each tick from the simulation's
    # ``last_proximity`` reading. ``None`` means the ego is solidly inside
    # the navigable area; non-``None`` is the warning / respawn message
    # that the loop merges into the displayed frame's ``status_message``.
    oob_message: str | None
    # Number of consecutive chunks whose boundary-state proximity has
    # stayed at or above :attr:`LoopConfig.oob_respawn_proximity`. The
    # auto-respawn only fires once the streak reaches
    # :attr:`LoopConfig.oob_respawn_debounce_chunks`; resets to zero the
    # moment a chunk reads below the respawn threshold.
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
    # Out-of-bounds detection. ``simulation.last_proximity`` mirrors
    # alpasim's ``oob_proximity`` semantics:
    #   0.0  = solidly inside the mesh AABB (expanded by a 50 m margin),
    #          more than 100 m from any edge.
    #   (0,1] = within the 100 m warning zone, ramping linearly with
    #          ``1.0 - dist_to_edge / 100``.
    #   2.0  = "off map" sentinel; the ego has crossed AABB + margin.
    #
    # Defaults match the alpasim driver: warn at >= 0.6 (standard
    # "approaching" threshold from
    # ``alpasim_driver.models.manual_model``'s render path), respawn at
    # >= 2.0 (the binary "you're past the boundary" trigger). The
    # warning ramps over a wide band; the respawn is a hard step that
    # only fires when ``dist_to_edge < 0`` -- so brushing curbs,
    # driving on sidewalks, or other intra-AABB excursions never
    # trigger a teleport. Both checks no-op when the scene shipped no
    # ground mesh (``simulation.last_proximity`` reads ``0.0``).
    oob_warn_proximity: float = 0.6
    oob_respawn_proximity: float = 2.0
    # Number of consecutive chunks the boundary-state proximity must
    # remain at or above ``oob_respawn_proximity`` before the loop fires
    # the auto-respawn. Default ``1`` matches alpasim's behaviour
    # (immediate respawn on the off-map step). Raise this for an
    # added "are you sure you want to teleport" buffer; the alpasim
    # signal is binary so a small debounce mostly catches measurement
    # noise that doesn't really exist for the AABB check.
    oob_respawn_debounce_chunks: int = 1


# Warning text shown when the ego enters the OOB warning band. Kept as a
# module-level constant so the slangpy HUD's status-overlay code can
# special-case it for styling later if needed without re-deriving the
# string.
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
        prediction=prediction,
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

    Reads ``simulation.last_proximity`` defensively so test fakes and
    other ``SimulationBackend`` implementations that don't track OOB
    state default to ``0.0`` (always in-bounds). The respawn threshold
    is debounced: the boundary-state proximity must stay above
    :attr:`LoopConfig.oob_respawn_proximity` for
    :attr:`LoopConfig.oob_respawn_debounce_chunks` consecutive chunks
    before the loop returns ``True``. That smooths out single-chunk
    spikes when a corner ray briefly misses the mesh during a sharp
    turn -- the alpasim driver's auto-respawn was similarly sticky on
    the runtime side. Returns ``True`` only on the chunk that actually
    fires the respawn; the caller's ``app.run`` outer loop then
    rebuilds the simulation from the scene's initial pose and the new
    sim's first chunk re-enters with proximity ``0.0``, auto-clearing
    the streak.
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
    """Log OOB state transitions to stderr.

    Fires once per state edge (in-bounds -> warn -> respawn-pending ->
    fire, plus recovery), so the volume is bounded even on long
    sessions: a typical drive sees only a handful of these lines.
    Including the proximity reading and streak count makes it easy to
    confirm whether the defaults are firing at the right time, and to
    tune :attr:`LoopConfig.oob_warn_proximity` /
    :attr:`LoopConfig.oob_respawn_proximity` /
    :attr:`LoopConfig.oob_respawn_debounce_chunks` if not.
    """
    prev_label = "in-bounds" if previous is None else _truncate(previous, 32)
    curr_label = "in-bounds" if current is None else _truncate(current, 32)
    print(
        f"[loop] oob {prev_label!r} -> {curr_label!r}"
        f" proximity={proximity:.3f} streak={streak} action={action}",
        flush=True,
    )


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "\u2026"


def push_telemetry(
    runtime_controls: RuntimeControls, simulation: SimulationBackend
) -> None:
    """Forward ``simulation.current_state`` to ``runtime_controls`` if it accepts it.

    Defensively no-ops when ``runtime_controls`` is a minimal Protocol
    implementation that doesn't expose ``update_telemetry`` (test fakes,
    custom controllers); only the production :class:`KeyboardState`
    subscribes to this channel today, so an unknown ``runtime_controls``
    just doesn't get a per-chunk telemetry update.
    """
    update = getattr(runtime_controls, "update_telemetry", None)
    if update is None:
        return
    update(simulation.current_state)


def run_main_loop(
    presenter: PresenterBackend,
    runtime_controls: RuntimeControls,
    initial_presented_frame: PresentedFrame,
    input_backend: InputBackend,
    simulation: SimulationBackend,
    pipeline: ChunkPipeline,
    config: LoopConfig,
) -> bool:
    """Drive the request -> render -> present pipeline.

    Authoritative simulation state advances inside ``simulation.pose_chunk``
    when a chunk is requested (``chunk_size * frame_interval_s`` per chunk),
    so sim wall-clock cadence is gated by display-driven chunk requests, not
    by how often this loop's poll fires.

    ``initial_presented_frame`` seeds ``last_presented_frame`` so the loop
    has a single uniform ``present_frame`` path: while the pipeline is
    warming up or hasn't produced a chunk yet, the loop keeps re-presenting
    whatever it last presented, which is the loading-overlay frame the
    caller pre-rendered.

    Returns ``True`` if the loop exited because the user requested a reset
    (caller should call ``pipeline.reset`` and re-run the loop with a fresh
    simulation), ``False`` if it exited because the presenter requested
    close. The OOB auto-respawn path also returns ``True``: the simulation's
    boundary state crossed :attr:`LoopConfig.oob_respawn_proximity`, the
    user-visible warning escalated to ``OOB_RESPAWN_MESSAGE``, and the
    caller is expected to rebuild the simulation from the scene's
    initial pose just as it does for a manual ``R`` press.
    """
    state = MainLoopState()
    last_presented_frame: PresentedFrame = initial_presented_frame
    chunk_history = ChunkHistory.create(config.history_capacity)

    while not presenter.should_close:
        presenter.process_events()
        if runtime_controls.consume_reset_request():
            return True
        sampled = input_backend.sample()

        # Keep one chunk in flight for Stage 1; later stages can use richer scheduling.
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
            # ``simulation.pose_chunk`` (called inside make_chunk_request) just
            # advanced the authoritative state by ``chunk_size`` frames, so its
            # ``last_proximity`` reading is now the OOB status of the boundary
            # frame. Refresh the overlay text here -- and bail with the same
            # ``return True`` the manual reset path uses when the ego is far
            # enough off the map that auto-respawning is the right move.
            if update_oob_state(state, simulation, config):
                return True
            # Republish telemetry on the same per-chunk cadence so
            # downstream observers (e.g. the MJPEG presenter's ``/state``
            # endpoint, which the browser polls for the speed readout)
            # see a snapshot drawn from the same ``current_state`` the
            # OOB check just consulted.
            push_telemetry(runtime_controls, simulation)

        now = time.perf_counter()
        if now < state.next_present_time:
            time.sleep(
                min(config.poll_timeout_s, max(0.0, state.next_present_time - now))
            )
            continue

        view_mode = runtime_controls.view_mode
        try:
            queued_frame = pipeline.frame_queue.get_nowait()
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
        except queue.Empty:
            # Re-present the last frame with whatever OOB overlay is current
            # for this tick; the merged frame is local to this call so the
            # cached ``last_presented_frame`` stays unmodified for the next
            # iteration.
            presenter.present_frame(
                _frame_with_overlay(last_presented_frame, state.oob_message),
                view_mode=view_mode,
            )

        state.next_present_time += config.frame_interval_s
    return False
