# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
import time
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import omnidreams.interactive_drive.runtime.loop as loop_module
import pytest
from omnidreams.interactive_drive._pipeline_fakes import (
    FakeVideoModelBackend,
    make_trajectory,
    minimal_scene,
)
from omnidreams.interactive_drive.input.backend import SampledInput
from omnidreams.interactive_drive.runtime.loop import (
    LoopConfig,
    present_queued_frame,
    run_main_loop,
)
from omnidreams.interactive_drive.runtime.timing import ChunkPrediction, ChunkTimes
from omnidreams.interactive_drive.types import (
    DriverCommand,
    PresentedFrame,
    TrajectoryChunk,
    VehicleState,
)
from omnidreams.interactive_drive.video_model.chunk_pipeline import (
    ChunkPipeline,
    QueuedFrame,
)


def _on_ci() -> bool:
    """True when running under CI (GitHub Actions et al. set ``CI=true``)."""
    return os.environ.get("CI", "").lower() == "true"


def _backend_frame_wait_budget() -> int:
    """Present-tick ceiling for tests that wait on the render worker thread.

    These tests assert that the background pipeline worker delivered a
    rendered frame. Paired with ``_CountingPresenter(close_on_frame=...)`` the
    loop closes the instant that frame arrives, so this value only bounds the
    *stall* case (a worker that never delivers). Locally a small ceiling keeps
    a genuine stall failing fast; CI runners are heavily loaded and can starve
    the worker thread for many present ticks, so allow a generous ceiling
    there to absorb scheduling jitter rather than flaking on a fixed budget.
    """
    return 500 if _on_ci() else 16


def _chunk_times() -> ChunkTimes:
    now = time.perf_counter()
    return ChunkTimes.create(
        chunk_index=0,
        input_sample_time=now,
        request_time=now,
        request_poses_ready_time=now + 0.001,
        prediction=ChunkPrediction.create(request_time=now, frame_interval_s=0.1),
        intended_present_times=[now + 0.1],
    )


def _make_frame() -> PresentedFrame:
    return PresentedFrame(
        timestamp_us=0,
        rgb_host_uint8=np.zeros((4, 4, 3), dtype=np.uint8),
        depth_host_f32=None,
    )


def _loop_config(*, frame_interval_s: float) -> LoopConfig:
    return LoopConfig(
        initial_chunk_size=1,
        chunk_size=1,
        frame_interval_s=frame_interval_s,
        poll_timeout_s=0.0,
    )


@dataclass(frozen=True)
class _PresentRecord:
    frame: PresentedFrame
    view_mode: str


class _CountingPresenter:
    """Records every presented frame; flips ``should_close`` after a budget."""

    def __init__(
        self,
        present_budget: int,
        *,
        start_closed: bool = False,
        close_on_frame: PresentedFrame | None = None,
    ) -> None:
        self._budget = present_budget
        self._closed = start_closed
        # When set, close as soon as a frame that is *not* this one is
        # presented -- i.e. the render worker delivered a backend frame. Lets
        # timing tests wait on the worker instead of racing a fixed present
        # budget, which flaked under CI load. ``present_budget`` then just
        # bounds the stall case.
        self._close_on_frame = close_on_frame
        self.records: list[_PresentRecord] = []
        self.process_events_calls = 0

    @property
    def should_close(self) -> bool:
        return self._closed

    def process_events(self) -> None:
        self.process_events_calls += 1

    def present_frame(self, frame: PresentedFrame, view_mode: str) -> None:
        self.records.append(_PresentRecord(frame=frame, view_mode=view_mode))
        backend_frame_arrived = (
            self._close_on_frame is not None and frame is not self._close_on_frame
        )
        if backend_frame_arrived or len(self.records) >= self._budget:
            self._closed = True

    def close(self) -> None:
        # ``PresenterBackend`` declares ``close`` because every concrete
        # presenter the engine ships needs it for teardown. The test
        # fixture has nothing to release, so this is a no-op; it just
        # exists to satisfy the Protocol.
        return


class _PreparingPresenter(_CountingPresenter):
    def __init__(
        self, present_budget: int, *, close_on_frame: PresentedFrame | None = None
    ) -> None:
        super().__init__(present_budget=present_budget, close_on_frame=close_on_frame)
        self.prepared_frame_ids: set[int] = set()
        self.unprepared_backend_frame_ids: set[int] = set()

    def prepare_frame(self, frame: PresentedFrame, view_mode: str) -> None:
        del view_mode
        self.prepared_frame_ids.add(id(frame))

    def present_frame(self, frame: PresentedFrame, view_mode: str) -> None:
        if id(frame) not in self.prepared_frame_ids:
            self.unprepared_backend_frame_ids.add(id(frame))
        super().present_frame(frame, view_mode)


class _FakeRuntimeControls:
    def __init__(self, *, reset_after_present: int | None = None) -> None:
        self._reset_after_present = reset_after_present
        self._presenter: _CountingPresenter | None = None
        self.view_mode = "rgb"

    def bind_presenter(self, presenter: _CountingPresenter) -> None:
        self._presenter = presenter

    def consume_reset_request(self) -> bool:
        if self._reset_after_present is None or self._presenter is None:
            return False
        if len(self._presenter.records) >= self._reset_after_present:
            self._reset_after_present = None
            return True
        return False


class _FakeInputBackend:
    def sample(self) -> SampledInput:
        return SampledInput(command=DriverCommand(), sample_time=time.perf_counter())


class _FakeSimulation:
    """Returns a canned trajectory."""

    def __init__(self) -> None:
        self._state = VehicleState(
            x_m=0.0, y_m=0.0, z_m=0.0, yaw_rad=0.0, speed_mps=0.0, steer_rad=0.0
        )

    @property
    def current_state(self) -> VehicleState:
        return self._state

    def pose_chunk(
        self,
        command: DriverCommand,
        chunk_size: int,
        frame_interval_s: float,
        extrapolation_offset_s: float,
    ) -> TrajectoryChunk:
        del command, frame_interval_s, extrapolation_offset_s
        return make_trajectory(chunk_size)


def _drive_loop(
    *,
    presenter: _CountingPresenter,
    controls: _FakeRuntimeControls,
    backend: FakeVideoModelBackend,
    simulation: _FakeSimulation,
    initial: PresentedFrame,
    frame_interval_s: float,
) -> bool:
    pipeline = ChunkPipeline(backend)
    pipeline.request_scene(minimal_scene())
    try:
        return run_main_loop(
            presenter=presenter,
            runtime_controls=controls,
            initial_presented_frame=initial,
            input_backend=_FakeInputBackend(),
            simulation=simulation,
            pipeline=pipeline,
            config=_loop_config(frame_interval_s=frame_interval_s),
        )
    finally:
        pipeline.shutdown()


def test_present_timestamp_recorded_after_present_call_returns() -> None:
    chunk_times = _chunk_times()
    queued = QueuedFrame(frame=_make_frame(), chunk_times=chunk_times, frame_index=0)
    presenter = _CountingPresenter(present_budget=1)
    start = time.perf_counter()
    present_time = present_queued_frame(queued, presenter, view_mode="rgb")
    end = time.perf_counter()
    frame_times = chunk_times.frames[0]
    assert frame_times.sample_display_pose_time is not None
    assert frame_times.present_time is not None
    assert frame_times.sample_display_pose_time <= frame_times.present_time
    assert start <= present_time <= end


def test_input_to_present_profile_records_queued_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, float | int]] = []
    monkeypatch.setenv("INTERACTIVE_DRIVE_PROFILE_INPUT_TO_PRESENT", "1")
    monkeypatch.setattr(
        loop_module,
        "_record_input_to_present_for_profile",
        lambda **kwargs: calls.append(kwargs),
    )

    chunk_times = _chunk_times()
    queued = QueuedFrame(frame=_make_frame(), chunk_times=chunk_times, frame_index=0)
    presenter = _CountingPresenter(present_budget=1)

    present_queued_frame(queued, presenter, view_mode="rgb")

    assert len(calls) == 1
    assert calls[0]["input_sample_time"] == chunk_times.input_sample_time
    assert calls[0]["frame_index"] == 0


def test_input_to_present_profile_prints_window_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    del capsys
    messages: list[str] = []
    monkeypatch.setenv("INTERACTIVE_DRIVE_PROFILE_INPUT_TO_PRESENT_INTERVAL_S", "0.25")
    monkeypatch.setattr(
        loop_module,
        "logger",
        SimpleNamespace(
            info=lambda message, *args, **kwargs: messages.append(
                str(message).format(*args, **kwargs)
            )
        ),
    )
    loop_module.reset_input_to_present_profile_window()

    loop_module._record_input_to_present_for_profile(
        present_time=1.0,
        input_sample_time=0.9,
        frame_index=0,
        frame_interval_s=0.1,
    )
    loop_module._record_input_to_present_for_profile(
        present_time=1.3,
        input_sample_time=0.9,
        frame_index=1,
        frame_interval_s=0.1,
    )

    output = "\n".join(messages)
    assert "[profile] e2e" in output
    assert "wall_present_fps=" in output
    assert "avg_adj_control_to_present_ms=" in output
    assert "avg_raw_control_to_present_ms=" in output
    assert "samples=2" in output
    loop_module.reset_input_to_present_profile_window()


def test_run_main_loop_returns_false_when_presenter_starts_closed() -> None:
    presenter = _CountingPresenter(present_budget=0, start_closed=True)
    controls = _FakeRuntimeControls()

    result = _drive_loop(
        presenter=presenter,
        controls=controls,
        backend=FakeVideoModelBackend(frames_per_render=1),
        simulation=_FakeSimulation(),
        initial=_make_frame(),
        frame_interval_s=0.0,
    )

    assert result is False


def test_run_main_loop_returns_true_when_reset_requested() -> None:
    presenter = _CountingPresenter(present_budget=10)
    controls = _FakeRuntimeControls(reset_after_present=3)
    controls.bind_presenter(presenter)

    result = _drive_loop(
        presenter=presenter,
        controls=controls,
        backend=FakeVideoModelBackend(frames_per_render=1),
        simulation=_FakeSimulation(),
        initial=_make_frame(),
        frame_interval_s=0.001,
    )

    assert result is True
    assert len(presenter.records) == 3


def test_loop_re_presents_initial_frame_while_pipeline_queue_is_empty() -> None:
    """The loading-screen fix path: while the pipeline produces no frames,
    every present tick re-shows whatever was last presented, which the caller
    seeds with the loading overlay frame.
    """
    initial = _make_frame()
    presenter = _CountingPresenter(present_budget=4)
    controls = _FakeRuntimeControls()

    result = _drive_loop(
        presenter=presenter,
        controls=controls,
        backend=FakeVideoModelBackend(frames_per_render=0),
        simulation=_FakeSimulation(),
        initial=initial,
        frame_interval_s=0.001,
    )

    assert result is False
    assert len(presenter.records) == 4
    for record in presenter.records:
        assert record.frame is initial


def test_input_to_present_profile_ignores_represented_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, float | int]] = []
    monkeypatch.setenv("INTERACTIVE_DRIVE_PROFILE_INPUT_TO_PRESENT", "1")
    monkeypatch.setattr(
        loop_module,
        "_record_input_to_present_for_profile",
        lambda **kwargs: calls.append(kwargs),
    )
    initial = _make_frame()
    presenter = _CountingPresenter(present_budget=4)
    controls = _FakeRuntimeControls()

    result = _drive_loop(
        presenter=presenter,
        controls=controls,
        backend=FakeVideoModelBackend(frames_per_render=0),
        simulation=_FakeSimulation(),
        initial=initial,
        frame_interval_s=0.001,
    )

    assert result is False
    assert calls == []


def test_loop_presents_backend_frames_when_available() -> None:
    """Once the pipeline has rendered frames, the loop presents them."""
    initial = _make_frame()
    # Wait for the worker to deliver a backend frame rather than racing a
    # fixed present budget (which flaked under CI load); the budget is just a
    # stall ceiling, larger on CI.
    presenter = _CountingPresenter(
        present_budget=_backend_frame_wait_budget(), close_on_frame=initial
    )
    controls = _FakeRuntimeControls()

    result = _drive_loop(
        presenter=presenter,
        controls=controls,
        backend=FakeVideoModelBackend(frames_per_render=1, rgb_value=7),
        simulation=_FakeSimulation(),
        initial=initial,
        frame_interval_s=0.001,
    )

    assert result is False
    assert any(record.frame is not initial for record in presenter.records)


def test_loop_prepares_backend_frames_before_presenting_them() -> None:
    initial = _make_frame()
    presenter = _PreparingPresenter(
        present_budget=_backend_frame_wait_budget(), close_on_frame=initial
    )
    controls = _FakeRuntimeControls()

    _drive_loop(
        presenter=presenter,
        controls=controls,
        backend=FakeVideoModelBackend(frames_per_render=1, rgb_value=9),
        simulation=_FakeSimulation(),
        initial=initial,
        frame_interval_s=0.001,
    )

    assert presenter.prepared_frame_ids
    assert presenter.unprepared_backend_frame_ids == {id(initial)}


def test_loop_stamps_full_timing_chain_on_same_chunktimes_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end timestamp object identity through ``run_main_loop``.

    Captures every ``ChunkTimes`` the loop creates by wrapping
    :py:meth:`ChunkTimes.create`, then asserts that at least one captured
    instance was stamped at every pipeline stage in chronological order.
    The timestamps are written in different threads (worker for render,
    main for present), so this test fails if the loop or pipeline ever
    swaps the object reference along the way.
    """
    captured: list[ChunkTimes] = []
    original_create = ChunkTimes.create

    def capturing_create(
        chunk_index: int,
        input_sample_time: float,
        request_time: float,
        request_poses_ready_time: float,
        prediction: ChunkPrediction,
        intended_present_times: list[float],
    ) -> ChunkTimes:
        instance = original_create(
            chunk_index=chunk_index,
            input_sample_time=input_sample_time,
            request_time=request_time,
            request_poses_ready_time=request_poses_ready_time,
            prediction=prediction,
            intended_present_times=intended_present_times,
        )
        captured.append(instance)
        return instance

    monkeypatch.setattr(ChunkTimes, "create", capturing_create)

    initial = _make_frame()
    presenter = _CountingPresenter(
        present_budget=_backend_frame_wait_budget(), close_on_frame=initial
    )
    controls = _FakeRuntimeControls()
    _drive_loop(
        presenter=presenter,
        controls=controls,
        backend=FakeVideoModelBackend(frames_per_render=1, rgb_value=9),
        simulation=_FakeSimulation(),
        initial=initial,
        frame_interval_s=0.001,
    )

    fully_stamped = [
        chunk
        for chunk in captured
        if chunk.frames and chunk.frames[0].present_time is not None
    ]
    assert fully_stamped, (
        f"No fully-stamped ChunkTimes among {len(captured)} captured. "
        "If captured > 0 but none reached present_time, the worker or "
        "presenter swapped the object reference."
    )

    chunk = fully_stamped[0]
    frame_times = chunk.frames[0]
    chunk_render_start = chunk.chunk_render_start_time
    chunk_ready = chunk.chunk_ready_time
    image_ready = frame_times.image_ready_time
    sample_display = frame_times.sample_display_pose_time
    present = frame_times.present_time
    assert chunk_render_start is not None
    assert chunk_ready is not None
    assert image_ready is not None
    assert sample_display is not None
    assert present is not None

    assert chunk.request_time <= chunk.request_poses_ready_time
    assert chunk.request_poses_ready_time <= chunk_render_start
    assert chunk_render_start <= chunk_ready
    assert chunk_ready <= image_ready
    assert image_ready <= sample_display
    assert sample_display <= present
