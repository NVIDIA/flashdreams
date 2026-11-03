# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for ticks, frames and chunks (the protocol's worked example)."""

from __future__ import annotations

import pytest
from omnidreams_game_engine.api.v0.testing import FakePipeline
from omnidreams_game_engine.api.v0.timeline import ChunkTimeline

pytestmark = pytest.mark.ci_cpu


def test_worked_example_omnidreams_5_8_at_60_hz() -> None:
    timeline = ChunkTimeline(plan=(5, 8), sim_hz=60, model_fps=30)
    t0 = 1000
    assert timeline.frame_ticks == 2
    assert list(timeline.chunk_frames(0)) == [0, 1, 2, 3, 4]
    assert [timeline.tick_of(g, t0) for g in timeline.chunk_frames(0)] == [
        1000,
        1002,
        1004,
        1006,
        1008,
    ]
    assert timeline.seal_tick(0, t0) == t0 + 8
    assert list(timeline.chunk_frames(1)) == list(range(5, 13))
    assert timeline.tick_of(5, t0) == t0 + 10
    assert timeline.seal_tick(1, t0) == t0 + 24
    assert timeline.seal_tick(2, t0) == t0 + 40
    assert timeline.seal_tick(3, t0) == t0 + 56
    for k in range(1, 50):
        first = 5 + 8 * (k - 1)
        assert timeline.first_frame(k) == first
        assert timeline.tick_of(first, t0) == t0 + 10 + 16 * (k - 1)
        assert timeline.seal_tick(k, t0) == t0 + 24 + 16 * (k - 1)
        for g in timeline.chunk_frames(k):
            assert timeline.chunk_of(g) == k


def test_timestamps_follow_ticks_from_the_seed() -> None:
    timeline = ChunkTimeline()
    assert timeline.frame_interval_us == 33333
    assert timeline.timestamp_us(500, t0=500, base_us=7) == 7
    assert timeline.timestamp_us(502, t0=500, base_us=7) == 7 + 33333
    assert timeline.timestamp_us(560, t0=500, base_us=0) == 30 * 33333
    assert timeline.timestamp_us(501, t0=500, base_us=0) == 16666


def test_grants_cover_whole_chunks_after_the_last_grant() -> None:
    timeline = ChunkTimeline()
    assert list(timeline.grant_frames(2, after_g=0)) == list(range(1, 21))
    assert list(timeline.grant_frames(3, after_g=20)) == list(range(21, 29))
    assert list(timeline.grant_frames(2, after_g=20)) == []


def test_plan_is_taken_from_the_pipeline() -> None:
    assert ChunkTimeline.from_frame_count(
        FakePipeline((5, 8)).get_num_output_frames
    ).plan == (
        5,
        8,
    )
    assert ChunkTimeline.from_frame_count(lambda k: 2).plan == (2,)
    irregular = ChunkTimeline.from_frame_count(
        lambda k: (9, 4, 6, 8)[min(k, 3)], sim_hz=48, model_fps=24
    )
    assert irregular.plan == (9, 4, 6, 8)
    assert irregular.frame_ticks == 2
    assert irregular.first_frame(5) == 9 + 4 + 6 + 8 + 8
    assert irregular.chunk_of(18) == 2 and irregular.chunk_of(19) == 3
    assert irregular.chunk_of(26) == 3 and irregular.chunk_of(27) == 4


def test_invalid_timelines_are_refused() -> None:
    with pytest.raises(ValueError):
        ChunkTimeline(plan=())
    with pytest.raises(ValueError):
        ChunkTimeline(plan=(5, 0))
    with pytest.raises(ValueError):
        ChunkTimeline(sim_hz=50, model_fps=30)
    with pytest.raises(ValueError):
        ChunkTimeline().chunk_of(-1)
    with pytest.raises(ValueError, match="do not settle"):
        ChunkTimeline.from_frame_count(lambda k: k + 1)
