# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import queue
import time
from dataclasses import dataclass
from typing import Protocol

from interactive_drive.input.backend import InputBackend
from interactive_drive.runtime.runtime_controls import RuntimeControls
from interactive_drive.runtime.timing import (
    ChunkHistory,
    ChunkPrediction,
    ChunkTimes,
)
from interactive_drive.simulation.backend import SimulationBackend
from interactive_drive.types import DriverCommand, PresentedFrame
from interactive_drive.video_model.chunk_pipeline import ChunkPipeline, ChunkRequest, QueuedFrame


class PresenterBackend(Protocol):
    @property
    def should_close(self) -> bool: ...

    def process_events(self) -> None: ...

    def present_frame(self, frame: PresentedFrame, view_mode: str) -> None: ...

    # ``close`` is part of every concrete presenter
    # (:class:`SlangPyPresenter`, :class:`MJPEGStreamingPresenter`,
    # :class:`SlangPyHudPresenter`) and is invoked from
    # :meth:`InteractiveDriveApp.run`'s teardown path.
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

    def __init__(self) -> None:
        self.next_present_time = time.perf_counter()
        self.next_chunk_index = 0
        self.frame_count = 0
        self.chunks_outstanding = 0
        self.last_consumed_chunk_index = None


@dataclass(frozen=True)
class LoopConfig:
    initial_chunk_size: int
    chunk_size: int
    frame_interval_s: float
    poll_timeout_s: float = 0.001
    history_capacity: int = 16


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
) -> float:
    frame_times = queued_frame.chunk_times.frames[queued_frame.frame_index]
    frame_times.sample_display_pose_time = time.perf_counter()
    presenter.present_frame(queued_frame.frame, view_mode=view_mode)
    present_time = time.perf_counter()
    frame_times.present_time = present_time
    return present_time


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
    close.
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

        now = time.perf_counter()
        if now < state.next_present_time:
            time.sleep(min(config.poll_timeout_s, max(0.0, state.next_present_time - now)))
            continue

        view_mode = runtime_controls.view_mode
        try:
            queued_frame = pipeline.frame_queue.get_nowait()
            if queued_frame.chunk_times.chunk_index != state.last_consumed_chunk_index:
                state.last_consumed_chunk_index = queued_frame.chunk_times.chunk_index
                state.chunks_outstanding = max(0, state.chunks_outstanding - 1)
            present_queued_frame(queued_frame, presenter, view_mode=view_mode)
            last_presented_frame = queued_frame.frame
            state.frame_count += 1
        except queue.Empty:
            presenter.present_frame(last_presented_frame, view_mode=view_mode)

        state.next_present_time += config.frame_interval_s
    return False
