# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import threading
import time

import numpy as np
from omnidreams.interactive_drive._pipeline_fakes import (
    FakeVideoModelBackend,
    make_trajectory,
    minimal_scene,
)
from omnidreams.interactive_drive.runtime.timing import ChunkPrediction, ChunkTimes
from omnidreams.interactive_drive.types import FrameChunk, PresentedFrame
from omnidreams.interactive_drive.video_model.chunk_pipeline import (
    ChunkPipeline,
    ChunkRequest,
)


class _GatedBackend:
    """Backend whose render blocks on a gate so a reset can race it."""

    def __init__(self) -> None:
        self.warmup_model_calls = 0
        self.reset_calls = 0
        self.render_started = threading.Event()
        self.release = threading.Event()

    def warmup_model(self) -> None:
        self.warmup_model_calls += 1

    def load_scene(self, scene: object) -> None:
        del scene

    def reset(self) -> None:
        self.reset_calls += 1

    def render_chunk(self, trajectory: object) -> FrameChunk:
        self.render_started.set()
        self.release.wait(timeout=5.0)
        frame = PresentedFrame(
            timestamp_us=0,
            rgb_host_uint8=np.zeros((4, 4, 3), dtype=np.uint8),
            depth_host_f32=None,
        )
        return FrameChunk(
            frames=(frame,),
            boundary_state_after_chunk=trajectory.boundary_state_after_chunk,
            source_name="gated",
        )


def _chunk_times(chunk_size: int) -> ChunkTimes:
    now = time.perf_counter()
    return ChunkTimes.create(
        chunk_index=0,
        input_sample_time=now,
        request_time=now,
        request_poses_ready_time=now + 0.001,
        prediction=ChunkPrediction.create(request_time=now, frame_interval_s=0.1),
        intended_present_times=[
            now + 0.1 + idx * (1.0 / 30.0) for idx in range(chunk_size)
        ],
    )


def test_chunk_pipeline_stamps_timing_and_orders_frames() -> None:
    backend = FakeVideoModelBackend(frames_per_render=3)
    pipeline = ChunkPipeline(backend)
    pipeline.request_scene(minimal_scene())
    chunk_times = _chunk_times(chunk_size=3)
    pipeline.request_pose_chunk(
        ChunkRequest(trajectory=make_trajectory(3), chunk_times=chunk_times)
    )

    first = pipeline.frame_queue.get(timeout=1.0)
    second = pipeline.frame_queue.get(timeout=1.0)
    third = pipeline.frame_queue.get(timeout=1.0)
    pipeline.shutdown()

    assert [first.frame_index, second.frame_index, third.frame_index] == [0, 1, 2]
    assert first.chunk_times is chunk_times
    assert chunk_times.chunk_render_start_time is not None
    assert chunk_times.chunk_ready_time is not None
    assert chunk_times.frames[0].image_ready_time is not None
    assert backend.warmup_model_calls == 1
    assert backend.load_scene_calls == 1


def test_chunk_pipeline_reset_invokes_backend_reset() -> None:
    backend = FakeVideoModelBackend(frames_per_render=1)
    pipeline = ChunkPipeline(backend)
    pipeline.request_scene(minimal_scene())
    chunk_times = _chunk_times(chunk_size=1)
    pipeline.request_pose_chunk(
        ChunkRequest(trajectory=make_trajectory(1), chunk_times=chunk_times)
    )
    pipeline.frame_queue.get(timeout=1.0)
    pipeline.reset()
    pipeline.shutdown()

    assert backend.warmup_model_calls == 1
    assert backend.reset_calls == 1


def test_chunk_pipeline_reuses_model_across_scene_changes() -> None:
    backend = FakeVideoModelBackend(frames_per_render=1)
    pipeline = ChunkPipeline(backend)
    assert pipeline.model_ready.wait(timeout=1.0)
    for _ in range(3):
        pipeline.request_scene(minimal_scene())
        chunk_times = _chunk_times(chunk_size=1)
        pipeline.request_pose_chunk(
            ChunkRequest(trajectory=make_trajectory(1), chunk_times=chunk_times)
        )
        pipeline.frame_queue.get(timeout=1.0)
    pipeline.shutdown()

    # The model is warmed exactly once even though the scene changed twice.
    assert backend.warmup_model_calls == 1
    assert backend.load_scene_calls == 3


def test_chunk_pipeline_drops_superseded_render_after_reset() -> None:
    backend = _GatedBackend()
    pipeline = ChunkPipeline(backend)
    pipeline.request_scene(minimal_scene())
    pipeline.request_pose_chunk(
        ChunkRequest(trajectory=make_trajectory(1), chunk_times=_chunk_times(1))
    )
    # Wait until the (gated) render is in flight, then reset to supersede it.
    assert backend.render_started.wait(timeout=1.0)
    pipeline.reset()
    backend.release.set()  # let the superseded render finish
    pipeline.shutdown()

    assert backend.reset_calls == 1
    # The in-flight render belonged to the pre-reset generation, so its
    # frame is dropped rather than queued.
    assert pipeline.frame_queue.qsize() == 0
