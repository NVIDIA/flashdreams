# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import time

from interactive_drive._pipeline_fakes import (
    FakeVideoModelBackend,
    make_trajectory,
    minimal_scene,
)

from interactive_drive.runtime.timing import ChunkPrediction, ChunkTimes
from interactive_drive.video_model.chunk_pipeline import ChunkPipeline, ChunkRequest


def _chunk_times(chunk_size: int) -> ChunkTimes:
    now = time.perf_counter()
    return ChunkTimes.create(
        chunk_index=0,
        input_sample_time=now,
        request_time=now,
        request_poses_ready_time=now + 0.001,
        prediction=ChunkPrediction.create(request_time=now, frame_interval_s=0.1),
        intended_present_times=[now + 0.1 + idx * (1.0 / 30.0) for idx in range(chunk_size)],
    )


def test_chunk_pipeline_stamps_timing_and_orders_frames() -> None:
    backend = FakeVideoModelBackend(frames_per_render=3)
    pipeline = ChunkPipeline(backend, minimal_scene())
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
    assert backend.warmup_calls == 1


def test_chunk_pipeline_reset_invokes_backend_reset() -> None:
    backend = FakeVideoModelBackend(frames_per_render=1)
    pipeline = ChunkPipeline(backend, minimal_scene())
    chunk_times = _chunk_times(chunk_size=1)
    pipeline.request_pose_chunk(
        ChunkRequest(trajectory=make_trajectory(1), chunk_times=chunk_times)
    )
    pipeline.frame_queue.get(timeout=1.0)
    pipeline.reset()
    pipeline.shutdown()

    assert backend.warmup_calls == 1
    assert backend.reset_calls == 1
