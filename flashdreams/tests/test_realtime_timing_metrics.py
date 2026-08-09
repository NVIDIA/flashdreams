# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from flashdreams.runtime import InMemoryMetricsRecorder
from flashdreams.serving.realtime.timing import (
    ChunkTimes,
    VideoModelTimings,
    record_chunk_timing_metrics,
    record_video_model_timing_metrics,
)

pytestmark = pytest.mark.ci_cpu


def test_chunk_timing_records_feed_session_metrics() -> None:
    chunk = ChunkTimes.create(
        chunk_index=2,
        input_sample_time=0.0,
        request_time=0.010,
        request_poses_ready_time=0.030,
        intended_present_times=[0.100],
    )
    chunk.chunk_render_start_time = 0.050
    chunk.chunk_ready_time = 0.090
    chunk.frames[0].image_ready_time = 0.110
    chunk.frames[0].present_time = 0.140
    metrics = InMemoryMetricsRecorder()

    record_chunk_timing_metrics(metrics, chunk)
    snapshot = metrics.close()

    assert snapshot.timings["realtime.chunk.input_to_request"] == (
        pytest.approx(0.010),
    )
    assert snapshot.timings["realtime.chunk.request_to_poses_ready"] == (
        pytest.approx(0.020),
    )
    assert snapshot.timings["realtime.chunk.queue_wait"] == (pytest.approx(0.020),)
    assert snapshot.timings["realtime.chunk.chunk_render"] == (pytest.approx(0.040),)
    assert metrics.samples[0].step_index == 2


def test_video_model_timing_records_feed_session_metrics() -> None:
    timings = VideoModelTimings(
        condition_start_time=1.0,
        condition_ready_time=1.010,
        model_start_time=1.020,
        model_ready_time=1.070,
        cache_update_start_time=1.075,
        cache_update_ready_time=1.080,
        decode_start_time=1.085,
        decode_ready_time=1.095,
        merge_start_time=1.100,
        merge_ready_time=1.115,
    )
    metrics = InMemoryMetricsRecorder()

    record_video_model_timing_metrics(metrics, timings, chunk_index=3)
    snapshot = metrics.close()

    assert snapshot.timings["realtime.model.condition"] == (pytest.approx(0.010),)
    assert snapshot.timings["realtime.model.model"] == (pytest.approx(0.050),)
    assert snapshot.timings["realtime.model.cache_update"] == (pytest.approx(0.005),)
    assert snapshot.timings["realtime.model.decode"] == (pytest.approx(0.010),)
    assert snapshot.timings["realtime.model.merge"] == (pytest.approx(0.015),)
    assert snapshot.timings["realtime.model.total"] == (pytest.approx(0.115),)
    assert metrics.samples[0].step_index == 3
