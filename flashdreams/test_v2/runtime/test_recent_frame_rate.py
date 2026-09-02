# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU tests for recent frame-throughput measurement."""

import pytest

from flashdreams.runtime_v2.recent_frame_rate import RecentFrameRateTracker

pytestmark = pytest.mark.ci_cpu


def test_recent_frame_rate_weights_chunks_and_excludes_the_cutoff() -> None:
    tracker = RecentFrameRateTracker(window_seconds=2.0)

    assert tracker.observe(
        completed_at=0.5,
        frame_count=5,
        elapsed_s=0.5,
    ) == pytest.approx(10.0)
    assert tracker.observe(
        completed_at=1.5,
        frame_count=20,
        elapsed_s=1.0,
    ) == pytest.approx(25.0 / 1.5)
    assert tracker.observe(
        completed_at=2.5,
        frame_count=30,
        elapsed_s=1.0,
    ) == pytest.approx(25.0)
    assert tracker.observe(
        completed_at=3.0,
        frame_count=15,
        elapsed_s=0.5,
    ) == pytest.approx(26.0)


def test_recent_frame_rate_snapshot_expires_without_mutating_the_tracker() -> None:
    tracker = RecentFrameRateTracker(window_seconds=2.0)
    tracker.observe(completed_at=1.0, frame_count=13, elapsed_s=1.0)
    snapshot = tracker.snapshot()

    tracker.observe(completed_at=2.0, frame_count=20, elapsed_s=1.0)

    assert snapshot.frames_per_second(now=1.0) == pytest.approx(13.0)
    assert snapshot.window_seconds == pytest.approx(2.0)
    assert snapshot.frames_per_second(now=3.0) == 0.0
    assert tracker.snapshot().frames_per_second() == pytest.approx(16.5)


def test_recent_frame_rate_reset_discards_previous_rollout() -> None:
    tracker = RecentFrameRateTracker(window_seconds=2.0)
    tracker.observe(completed_at=1.0, frame_count=13, elapsed_s=1.0)

    tracker.reset()

    assert tracker.snapshot().frames_per_second() == 0.0
    assert tracker.observe(
        completed_at=10.0,
        frame_count=4,
        elapsed_s=1.0,
    ) == pytest.approx(4.0)


def test_recent_frame_rate_accepts_steps_below_clock_resolution() -> None:
    tracker = RecentFrameRateTracker(window_seconds=2.0)

    assert tracker.observe(completed_at=1.0, frame_count=1, elapsed_s=0.0) == float(
        "inf"
    )
    assert tracker.observe(
        completed_at=2.0,
        frame_count=4,
        elapsed_s=0.5,
    ) == pytest.approx(10.0)


@pytest.mark.parametrize("window_seconds", [0.0, -1.0, float("inf")])
def test_recent_frame_rate_rejects_invalid_window(window_seconds: float) -> None:
    with pytest.raises(ValueError, match="window_seconds"):
        RecentFrameRateTracker(window_seconds=window_seconds)


def test_recent_frame_rate_rejects_non_monotonic_completions() -> None:
    tracker = RecentFrameRateTracker(window_seconds=2.0)
    tracker.observe(completed_at=2.0, frame_count=1, elapsed_s=1.0)

    with pytest.raises(ValueError, match="must not precede"):
        tracker.observe(completed_at=1.0, frame_count=1, elapsed_s=1.0)


def test_recent_frame_rate_validates_explicit_sampling_time() -> None:
    empty_snapshot = RecentFrameRateTracker(window_seconds=2.0).snapshot()
    with pytest.raises(ValueError, match="now must be finite"):
        empty_snapshot.frames_per_second(now=float("nan"))

    tracker = RecentFrameRateTracker(window_seconds=2.0)
    tracker.observe(completed_at=2.0, frame_count=1, elapsed_s=1.0)
    with pytest.raises(ValueError, match="must not precede"):
        tracker.snapshot().frames_per_second(now=1.0)
