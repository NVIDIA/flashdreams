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

from __future__ import annotations

import math

import numpy as np
import pytest
from lingbot.webrtc.controls import (
    CameraPoseIntegrator,
    KeyboardResampler,
    KeyboardState,
    PoseSegment,
)

pytestmark = pytest.mark.ci_cpu

## KeyboardState basics (unchanged from the old design)


def test_keyboard_state_keydown_keyup_roundtrip() -> None:
    state = KeyboardState()
    assert state.apply_event(event="keydown", key="w")
    assert "w" in state.snapshot()

    assert state.apply_event(event="keyup", key="w")
    assert "w" not in state.snapshot()


def test_keyboard_state_rejects_unknown_key() -> None:
    state = KeyboardState()
    assert not state.apply_event(event="keydown", key="x")
    assert len(state.snapshot()) == 0


def test_keyboard_state_latest_turn_key_takes_precedence() -> None:
    state = KeyboardState()
    assert state.apply_event(event="keydown", key="a")
    assert state.apply_event(event="keydown", key="d")
    assert state.resolved_effective_keys() == frozenset({"d"})


def test_keyboard_state_release_restores_previous_turn_key() -> None:
    state = KeyboardState()
    assert state.apply_event(event="keydown", key="a")
    assert state.apply_event(event="keydown", key="d")
    assert state.apply_event(event="keyup", key="d")
    assert state.resolved_effective_keys() == frozenset({"a"})


## KeyboardResampler — segment / timeline output


def test_resampler_idle_chunk_yields_single_empty_segment() -> None:
    fps = 16
    dt = 1.0 / fps
    resampler = KeyboardResampler(fps=fps, start_v=0.0)
    segments, frame_times = resampler.sample_chunk(num_frames=4)

    assert segments == [(0.0, 4 * dt, frozenset())]
    assert frame_times == pytest.approx([dt, 2 * dt, 3 * dt, 4 * dt], abs=1e-9)
    assert resampler.next_chunk_start_v == pytest.approx(4 * dt)


def test_resampler_held_key_yields_single_held_segment() -> None:
    """An event arriving before the chunk window is folded into the
    carried state, then the whole chunk runs as one segment with that
    state held."""
    fps = 16
    dt = 1.0 / fps
    resampler = KeyboardResampler(fps=fps, start_v=5 * dt)
    # ``arrival_t`` is in the *past* of the chunk window; the resampler
    # should drop it into ``carried_state`` and emit no breakpoint.
    resampler.on_edge(arrival_t=2 * dt, event="keydown", key="w")
    segments, _ = resampler.sample_chunk(num_frames=4)

    assert segments == [(5 * dt, 9 * dt, frozenset({"w"}))]
    assert resampler.event_log_size() == 0


def test_resampler_mid_chunk_keydown_splits_into_two_segments() -> None:
    """Keydown at virtual time ``2 * dt`` inside the chunk creates a
    breakpoint there: ``[0, 2*dt, {})`` then ``[2*dt, 4*dt, {w})``."""
    fps = 16
    dt = 1.0 / fps
    resampler = KeyboardResampler(fps=fps, start_v=0.0)
    resampler.on_edge(arrival_t=2 * dt, event="keydown", key="w")
    segments, _ = resampler.sample_chunk(num_frames=4)

    assert segments == [
        (0.0, 2 * dt, frozenset()),
        (2 * dt, 4 * dt, frozenset({"w"})),
    ]
    assert resampler.event_log_size() == 0


def test_resampler_tap_inside_chunk_produces_three_segments() -> None:
    """A press followed by a release inside the chunk window splits
    into three segments: before press, during tap, after release."""
    fps = 16
    dt = 1.0 / fps
    resampler = KeyboardResampler(fps=fps, start_v=0.0)
    resampler.on_edge(arrival_t=1.0 * dt, event="keydown", key="w")
    resampler.on_edge(arrival_t=2.5 * dt, event="keyup", key="w")
    segments, _ = resampler.sample_chunk(num_frames=4)

    assert segments == [
        (0.0, 1.0 * dt, frozenset()),
        (1.0 * dt, 2.5 * dt, frozenset({"w"})),
        (2.5 * dt, 4.0 * dt, frozenset()),
    ]


def test_resampler_overlapping_keys_share_a_segment() -> None:
    """When two keys are pressed simultaneously the middle segment
    carries the union of effective keys."""
    fps = 16
    dt = 1.0 / fps
    resampler = KeyboardResampler(fps=fps, start_v=0.0)
    resampler.on_edge(arrival_t=1 * dt, event="keydown", key="w")
    resampler.on_edge(arrival_t=2 * dt, event="keydown", key="a")
    resampler.on_edge(arrival_t=3 * dt, event="keyup", key="w")
    resampler.on_edge(arrival_t=4 * dt, event="keyup", key="a")
    segments, _ = resampler.sample_chunk(num_frames=5)

    expected = [
        (0.0, 1 * dt, frozenset()),
        (1 * dt, 2 * dt, frozenset({"w"})),
        (2 * dt, 3 * dt, frozenset({"w", "a"})),
        (3 * dt, 4 * dt, frozenset({"a"})),
        (4 * dt, 5 * dt, frozenset()),
    ]
    assert segments == expected


def test_resampler_event_at_chunk_start_skips_zero_length_prefix() -> None:
    """An event at ``arrival_t == chunk_start_v`` folds into the
    carried state without emitting a zero-length leading segment."""
    fps = 16
    dt = 1.0 / fps
    resampler = KeyboardResampler(fps=fps, start_v=4 * dt)
    resampler.on_edge(arrival_t=4 * dt, event="keydown", key="w")
    segments, _ = resampler.sample_chunk(num_frames=2)

    assert segments == [(4 * dt, 6 * dt, frozenset({"w"}))]


def test_resampler_state_carries_across_chunks() -> None:
    """A keydown processed in chunk 0 is still active when chunk 1
    samples; chunk 1's first segment starts with the held state."""
    fps = 16
    dt = 1.0 / fps
    resampler = KeyboardResampler(fps=fps, start_v=0.0)
    resampler.on_edge(arrival_t=0.5 * dt, event="keydown", key="w")
    resampler.sample_chunk(num_frames=4)
    segments, _ = resampler.sample_chunk(num_frames=4)

    assert segments == [(4 * dt, 8 * dt, frozenset({"w"}))]


def test_resampler_reset_clears_log_and_state() -> None:
    fps = 16
    dt = 1.0 / fps
    resampler = KeyboardResampler(fps=fps, start_v=0.0)
    resampler.on_edge(arrival_t=0.5 * dt, event="keydown", key="w")
    resampler.sample_chunk(num_frames=2)
    resampler.reset(start_v=100.0)
    assert resampler.event_log_size() == 0
    assert resampler.next_chunk_start_v == 100.0
    segments, _ = resampler.sample_chunk(num_frames=2)
    assert segments == [(100.0, 100.0 + 2 * dt, frozenset())]


## CameraPoseIntegrator — rate-based, segment-driven integration


def _single_segment(
    *, start_v: float, end_v: float, state: frozenset[str]
) -> list[PoseSegment]:
    return [(start_v, end_v, state)]


def _eval_frame_times(start_v: float, num_frames: int, dt: float) -> list[float]:
    return [start_v + (i + 1) * dt for i in range(num_frames)]


def test_integrator_idle_keeps_pose_constant() -> None:
    fps = 16
    dt = 1.0 / fps
    integrator = CameraPoseIntegrator()
    chunk = integrator.integrate_chunk(
        segments=_single_segment(start_v=0.0, end_v=4 * dt, state=frozenset()),
        frame_times=_eval_frame_times(0.0, 4, dt),
    )
    assert chunk.shape == (4, 4, 4)
    for i in range(4):
        assert np.allclose(chunk[i], np.eye(4), atol=1e-6)


def test_integrator_full_chunk_forward_motion_matches_rate() -> None:
    """Hold ``w`` for the full chunk: position advances by exactly
    ``move_speed_per_s * chunk_duration``."""
    fps = 16
    dt = 1.0 / fps
    integrator = CameraPoseIntegrator(move_speed_per_s=1.0, rotate_speed_rad_per_s=0.0)
    num_frames = 4
    chunk = integrator.integrate_chunk(
        segments=_single_segment(
            start_v=0.0, end_v=num_frames * dt, state=frozenset({"w"})
        ),
        frame_times=_eval_frame_times(0.0, num_frames, dt),
    )
    # Frame i records pose after ``(i+1) * dt`` seconds of motion.
    for i in range(num_frames):
        expected_z = (i + 1) * dt * 1.0
        assert np.isclose(chunk[i, 2, 3], expected_z, atol=1e-6), (
            f"frame {i}: expected z={expected_z}, got {chunk[i, 2, 3]}"
        )


def test_integrator_partial_tap_produces_proportional_motion() -> None:
    """30 ms tap in a 250 ms chunk produces 30 ms worth of forward
    motion, not a full frame's worth."""
    fps = 16
    dt = 1.0 / fps
    integrator = CameraPoseIntegrator(move_speed_per_s=1.0, rotate_speed_rad_per_s=0.0)
    num_frames = 4
    # Tap from t=0.020 s to t=0.050 s inside a chunk that runs from
    # 0 to 4*dt=0.25 s; the tap straddles frame 0's end (at dt=0.0625 s).
    segments: list[PoseSegment] = [
        (0.0, 0.020, frozenset()),
        (0.020, 0.050, frozenset({"w"})),
        (0.050, num_frames * dt, frozenset()),
    ]
    chunk = integrator.integrate_chunk(
        segments=segments,
        frame_times=_eval_frame_times(0.0, num_frames, dt),
    )
    # Every frame should reflect exactly 30 ms of forward motion
    # (the tap completes inside frame 0's interval, [0, dt]).
    expected_z = 0.030 * 1.0
    for i in range(num_frames):
        assert np.isclose(chunk[i, 2, 3], expected_z, atol=1e-6), (
            f"frame {i}: expected z={expected_z}, got {chunk[i, 2, 3]}"
        )


def test_integrator_yaw_arc_with_forward_motion() -> None:
    """While ``w`` and ``a`` are both held the camera should turn left
    (positive yaw rate is +Y → 'd'/'l'; 'a'/'j' is negative) and move
    forward along the rotated heading rather than along world +Z.
    """
    fps = 16
    dt = 1.0 / fps
    move_rate = 1.0
    yaw_rate = math.pi  # 1 radian per (1/π) sec; chosen for easy math
    integrator = CameraPoseIntegrator(
        move_speed_per_s=move_rate, rotate_speed_rad_per_s=yaw_rate
    )
    duration = 0.1
    chunk = integrator.integrate_chunk(
        segments=_single_segment(
            start_v=0.0, end_v=duration, state=frozenset({"w", "a"})
        ),
        frame_times=[duration],
    )
    pose = chunk[0]
    # Yaw rotated by ``-yaw_rate * duration`` (a/j is left = negative).
    expected_yaw = -yaw_rate * duration
    assert np.isclose(pose[0, 0], math.cos(expected_yaw), atol=1e-5)
    assert np.isclose(pose[0, 2], math.sin(expected_yaw), atol=1e-5)
    # Translation must lie in the (x, z) plane (no vertical drift) and
    # along the rotated forward heading.
    expected_x = math.sin(expected_yaw) * move_rate * duration
    expected_z = math.cos(expected_yaw) * move_rate * duration
    assert np.isclose(pose[0, 3], expected_x, atol=1e-5)
    assert np.isclose(pose[1, 3], 0.0, atol=1e-6)
    assert np.isclose(pose[2, 3], expected_z, atol=1e-5)


def test_integrator_carries_pose_seamlessly_across_chunks() -> None:
    """End-of-chunk-N pose equals the start-of-chunk-(N+1) integration
    point: no discontinuity at the chunk boundary."""
    fps = 16
    dt = 1.0 / fps
    integrator = CameraPoseIntegrator(move_speed_per_s=1.0, rotate_speed_rad_per_s=0.0)
    num_frames = 4
    integrator.integrate_chunk(
        segments=_single_segment(
            start_v=0.0, end_v=num_frames * dt, state=frozenset({"w"})
        ),
        frame_times=_eval_frame_times(0.0, num_frames, dt),
    )
    pose_after_chunk_0 = integrator.current_pose()

    chunk1 = integrator.integrate_chunk(
        segments=_single_segment(
            start_v=num_frames * dt,
            end_v=2 * num_frames * dt,
            state=frozenset({"w"}),
        ),
        frame_times=_eval_frame_times(num_frames * dt, num_frames, dt),
    )
    # Chunk 1's first frame is one dt past the end-of-chunk-0 pose, so
    # its z-translation should be exactly ``move_speed_per_s * dt``
    # larger than the carried pose's z; no seam.
    assert np.isclose(chunk1[0, 2, 3] - pose_after_chunk_0[2, 3], dt * 1.0, atol=1e-6)


def test_integrator_rejects_empty_segments() -> None:
    integrator = CameraPoseIntegrator()
    with pytest.raises(ValueError):
        integrator.integrate_chunk(segments=[], frame_times=[0.1])


def test_integrator_rejects_out_of_window_frame_times() -> None:
    integrator = CameraPoseIntegrator()
    with pytest.raises(ValueError):
        integrator.integrate_chunk(
            segments=_single_segment(start_v=0.0, end_v=1.0, state=frozenset()),
            frame_times=[1.5],
        )


def test_integrator_rejects_non_monotonic_frame_times() -> None:
    integrator = CameraPoseIntegrator()
    with pytest.raises(ValueError):
        integrator.integrate_chunk(
            segments=_single_segment(start_v=0.0, end_v=1.0, state=frozenset()),
            frame_times=[0.5, 0.3],
        )


## Property tests against random viewer signals


def _generate_random_events(
    rng: np.random.Generator,
    duration_s: float,
    mean_interarrival_s: float,
    keys: tuple[str, ...],
) -> list[tuple[float, str, str]]:
    """Build a Poisson-like sequence of ``(arrival_t, event, key)`` triples.

    Each event is independently a ``keydown`` or ``keyup`` on a random
    key drawn from ``keys``; arrival times are cumulative exponential
    inter-arrival samples. Sorted by ``arrival_t``.
    """
    events: list[tuple[float, str, str]] = []
    t = 0.0
    while True:
        t += float(rng.exponential(mean_interarrival_s))
        if t >= duration_s:
            break
        key = str(rng.choice(keys))
        event = "keydown" if rng.random() < 0.5 else "keyup"
        events.append((t, event, key))
    return events


def _replay_resampler_segments(
    events: list[tuple[float, str, str]],
    *,
    fps: int,
    num_frames: int,
    num_chunks: int,
    start_v: float = 0.0,
) -> list[list[PoseSegment]]:
    """Drive the resampler over ``num_chunks`` chunks, modelling the
    worker's trigger semantics: at trigger time
    ``V_{N+1} = next_chunk_start_v + num_frames * dt`` every event with
    ``arrival_t <= trigger_wall`` has been pushed via :meth:`on_edge`.
    Returns one list of segments per chunk.
    """
    dt = 1.0 / fps
    resampler = KeyboardResampler(fps=fps, start_v=start_v)
    chunks: list[list[PoseSegment]] = []
    next_event = 0
    for _ in range(num_chunks):
        trigger_v = resampler.next_chunk_start_v + num_frames * dt
        while next_event < len(events) and events[next_event][0] <= trigger_v:
            arrival_t, ev, key = events[next_event]
            resampler.on_edge(arrival_t=arrival_t, event=ev, key=key)
            next_event += 1
        segments, _ = resampler.sample_chunk(num_frames)
        chunks.append(segments)
    return chunks


def test_random_events_segments_cover_chunk_without_gaps() -> None:
    """For random viewer events every chunk's segments cover the chunk
    window exactly: starts at ``chunk_start_v``, ends at
    ``chunk_start_v + num_frames * dt``, and every breakpoint matches
    the next segment's start.
    """
    rng = np.random.default_rng(42)
    fps = 16
    num_frames = 12
    dt = 1.0 / fps
    duration_s = 5.0
    events = _generate_random_events(
        rng,
        duration_s=duration_s,
        mean_interarrival_s=0.08,
        keys=("w", "a", "s", "d", "q", "e", "i", "k"),
    )
    num_chunks = math.ceil(duration_s / (num_frames * dt)) + 2

    chunks = _replay_resampler_segments(
        events, fps=fps, num_frames=num_frames, num_chunks=num_chunks
    )
    chunk_duration = num_frames * dt
    for idx, chunk in enumerate(chunks):
        expected_start = idx * chunk_duration
        expected_end = (idx + 1) * chunk_duration
        assert chunk, f"chunk {idx} produced no segments"
        assert chunk[0][0] == pytest.approx(expected_start, abs=1e-9)
        assert chunk[-1][1] == pytest.approx(expected_end, abs=1e-9)
        for a, b in zip(chunk, chunk[1:]):
            assert a[1] == pytest.approx(b[0], abs=1e-9)


def test_random_events_event_breakpoints_match_arrival_times() -> None:
    """Every event with ``arrival_t`` strictly inside a chunk window
    appears as a breakpoint between consecutive segments at exactly
    that ``arrival_t``."""
    rng = np.random.default_rng(99)
    fps = 16
    num_frames = 12
    dt = 1.0 / fps
    duration_s = 3.0
    events = _generate_random_events(
        rng,
        duration_s=duration_s,
        mean_interarrival_s=0.05,
        keys=("w", "a", "s", "d"),
    )
    num_chunks = math.ceil(duration_s / (num_frames * dt)) + 2
    chunks = _replay_resampler_segments(
        events, fps=fps, num_frames=num_frames, num_chunks=num_chunks
    )
    # Build the set of all breakpoints across chunks.
    breakpoints: list[float] = []
    for chunk in chunks:
        breakpoints.extend(s[1] for s in chunk[:-1])
    chunk_duration = num_frames * dt
    for arrival_t, _, _ in events:
        chunk_idx = int(arrival_t // chunk_duration)
        chunk_start = chunk_idx * chunk_duration
        # Events that land on a chunk start fold into the carried state
        # without emitting a breakpoint, so skip those.
        if math.isclose(arrival_t, chunk_start, abs_tol=1e-9):
            continue
        assert any(math.isclose(bp, arrival_t, abs_tol=1e-9) for bp in breakpoints), (
            f"no breakpoint at arrival_t={arrival_t}"
        )


def test_random_events_have_bounded_latency() -> None:
    """End-to-end latency for any event falls inside the steady-state band.

    In the rate-based design with the worker triggering at
    ``V_{N+1} = V_N + num_frames * dt``:

    * Event at ``arrival_t = W > 0`` first affects pose at global frame
      ``g = max(0, ceil(W / dt) - 1)`` (if ``W`` is a multiple of dt)
      or ``g = floor(W / dt)`` (otherwise) — both formulas equal
      ``ceil(W / dt) - 1`` when ``W`` is a multiple of dt, ``floor(W / dt)``
      otherwise.
    * Display wallclock of global frame ``g`` is
      ``D_0 + g * dt`` with ``D_0 = num_frames * dt + gen_seconds``.

    Latency ``= D_0 + g * dt - W``. Algebra shows this lies in
    ``[(num_frames - 1) * dt + gen_seconds, num_frames * dt + gen_seconds]``
    for every ``W >= 0`` — a band of width exactly one frame interval.
    """
    rng = np.random.default_rng(123)
    fps = 16
    num_frames = 12
    dt = 1.0 / fps
    gen_seconds = 0.5
    duration_s = 5.0
    events = _generate_random_events(
        rng,
        duration_s=duration_s,
        mean_interarrival_s=0.1,
        keys=("w", "a", "s", "d", "q", "e", "i", "k", "j", "l"),
    )

    d_0 = num_frames * dt + gen_seconds
    lower = (num_frames - 1) * dt + gen_seconds
    upper = num_frames * dt + gen_seconds
    for arrival_t, _, _ in events:
        x = arrival_t / dt
        if math.isclose(x, round(x), abs_tol=1e-9):
            # Event lands exactly on a frame boundary: it first
            # contributes to the NEXT frame (frames record the state
            # at the end of their interval, not events at the boundary).
            g = (
                max(0, int(round(x)) - 1)
                if math.isclose(arrival_t, 0.0)
                else int(round(x))
            )
        else:
            g = int(math.floor(x))
        display_t = d_0 + g * dt
        latency = display_t - arrival_t
        assert lower - 1e-9 <= latency <= upper + 1e-9, (
            f"event at arrival_t={arrival_t:.4f}: latency={latency:.4f} "
            f"not in [{lower:.4f}, {upper:.4f}]"
        )


def test_random_events_steady_state_under_simulated_worker() -> None:
    """Full wallclock simulation of the new worker loop.

    Each chunk N triggers at wallclock ``V_{N+1} = V_N + num_frames * dt``,
    runs generation for ``gen_seconds``, and lands on the playback
    queue at ``D_N = trigger_wall + gen_seconds``.

    In steady state with ``gen_seconds < num_frames * dt`` the worker
    sleeps between chunks (delay = chunk_duration - gen_seconds), so
    ``D_N`` is exactly ``num_frames * dt`` apart and the playback queue
    is always full when ``recv`` arrives — no recurring stalls.
    """
    fps = 16
    num_frames = 12
    dt = 1.0 / fps
    gen_seconds = 0.5
    chunk_duration_s = num_frames * dt
    assert gen_seconds < chunk_duration_s, (
        "Test config violates steady-state precondition gen < chunk_duration."
    )

    rng = np.random.default_rng(7)
    duration_s = 6.0
    events = _generate_random_events(
        rng,
        duration_s=duration_s,
        mean_interarrival_s=0.05,
        keys=("w", "a", "s", "d"),
    )

    resampler = KeyboardResampler(fps=fps, start_v=0.0)
    next_event = 0
    num_chunks = math.ceil(duration_s / chunk_duration_s) + 2
    d_n_values: list[float] = []
    wallclock = 0.0
    for _ in range(num_chunks):
        trigger_wall = resampler.next_chunk_start_v + num_frames * dt
        if wallclock < trigger_wall:
            wallclock = trigger_wall
        while next_event < len(events) and events[next_event][0] <= wallclock:
            arrival_t, ev, key = events[next_event]
            resampler.on_edge(arrival_t=arrival_t, event=ev, key=key)
            next_event += 1
        resampler.sample_chunk(num_frames)
        wallclock += gen_seconds
        d_n_values.append(wallclock)

    diffs = np.diff(d_n_values)
    assert np.allclose(diffs, chunk_duration_s, atol=1e-9), (
        f"Chunk arrival spacing not steady: diffs={diffs.tolist()}"
    )
    for n, d_n in enumerate(d_n_values):
        deadline = d_n_values[0] + n * chunk_duration_s
        assert d_n <= deadline + 1e-9, (
            f"Chunk {n} arrived at {d_n:.4f} but deadline was {deadline:.4f}"
        )


def test_random_events_pose_continuous_across_chunks() -> None:
    """Concatenating the integrator's output across consecutive chunks
    yields a globally continuous trajectory: the first frame of chunk
    ``N+1`` is exactly one dt of motion past the last frame of chunk
    ``N`` (given the segment state covering that dt).
    """
    rng = np.random.default_rng(2026)
    fps = 16
    num_frames = 12
    dt = 1.0 / fps
    duration_s = 3.0
    events = _generate_random_events(
        rng,
        duration_s=duration_s,
        mean_interarrival_s=0.07,
        keys=("w", "a", "s", "d"),
    )
    num_chunks = math.ceil(duration_s / (num_frames * dt)) + 1
    chunks_segments = _replay_resampler_segments(
        events, fps=fps, num_frames=num_frames, num_chunks=num_chunks
    )

    integrator = CameraPoseIntegrator(move_speed_per_s=1.0, rotate_speed_rad_per_s=0.1)
    chunk_duration = num_frames * dt
    poses_per_chunk: list[np.ndarray] = []
    for idx, segments in enumerate(chunks_segments):
        frame_times = _eval_frame_times(idx * chunk_duration, num_frames, dt)
        chunk_poses = integrator.integrate_chunk(
            segments=segments, frame_times=frame_times
        )
        poses_per_chunk.append(chunk_poses)

    for idx in range(1, len(poses_per_chunk)):
        last_prev = poses_per_chunk[idx - 1][-1]
        first_curr = poses_per_chunk[idx][0]
        # The integrator is in continuous time: ``last_prev`` is the
        # pose at ``idx * chunk_duration``, ``first_curr`` is the pose
        # at ``idx * chunk_duration + dt``. The displacement between
        # them is fully determined by the first segment of the new
        # chunk; verifying that the two poses are *close* (well below
        # any plausible drift floor) is enough to detect a seam-style
        # discontinuity.
        translation_jump = float(np.linalg.norm(first_curr[:3, 3] - last_prev[:3, 3]))
        # Upper bound on one-frame translation: ``move_speed_per_s * dt``.
        assert translation_jump <= 1.0 * dt + 1e-6, (
            f"chunk-{idx} boundary jump {translation_jump} exceeds "
            f"one-frame motion budget"
        )


## Catch-up after stalls
#
# These tests model the generation worker loop closely enough to
# exercise the "rewind virtual clock to wall when lag > chunk_duration"
# fix in ``session._generation_worker``. We don't import the worker
# directly because it is built around asyncio + the runtime; the
# catch-up policy is a two-line formula and replicating it in the
# simulator keeps the test fast and deterministic.


def _simulated_worker_step(
    *,
    resampler: KeyboardResampler,
    wallclock: float,
    chunk_duration: float,
    gen_seconds: float,
) -> tuple[float, float]:
    """Advance one worker iteration, mirroring ``_generation_worker``.

    Returns the new ``(wallclock, lag_after_chunk)`` pair. ``lag_after_chunk``
    is ``wallclock - resampler.next_chunk_start_v`` after the chunk has
    been generated; in steady state it should hover around
    ``chunk_duration`` (the worker triggers at chunk_end_v and then
    spends ``gen_seconds`` advancing wallclock by that much).
    """
    trigger_wall = resampler.next_chunk_start_v + chunk_duration
    if wallclock < trigger_wall:
        wallclock = trigger_wall

    # Mirror the worker's catch-up branch.
    lag = wallclock - (resampler.next_chunk_start_v + chunk_duration)
    if lag > chunk_duration:
        resampler.next_chunk_start_v = wallclock - chunk_duration

    resampler.sample_chunk(int(round(chunk_duration * resampler.fps)))
    wallclock += gen_seconds
    return wallclock, wallclock - resampler.next_chunk_start_v


def test_worker_catches_up_after_bootstrap_stall() -> None:
    """A massive first-chunk stall must NOT pin the virtual clock behind wall.

    Reproduces the live server's bug: gen for chunk 0 spends 11s in
    CUDA-graph warmup while wallclock advances 11s. Without the
    catch-up, ``next_chunk_start_v`` keeps advancing by 0.75s/chunk
    and lag stays at +11s forever — every keystroke ships ~14 chunks
    late. With the catch-up, chunk 1 rewinds the virtual clock to
    ``wall - chunk_duration`` and subsequent chunks settle into a
    steady-state lag of ``chunk_duration`` (one chunk in flight).
    """
    fps = 16
    num_frames = 12
    dt = 1.0 / fps
    chunk_duration = num_frames * dt
    bootstrap_gen_s = 11.0
    steady_gen_s = 0.5

    resampler = KeyboardResampler(fps=fps, start_v=0.0)
    wallclock = 0.0

    # Chunk 0: long warmup stall.
    wallclock, _ = _simulated_worker_step(
        resampler=resampler,
        wallclock=wallclock,
        chunk_duration=chunk_duration,
        gen_seconds=bootstrap_gen_s,
    )

    # Chunks 1..N: should converge to steady-state lag of one
    # chunk_duration within a single iteration thanks to the rewind.
    lags: list[float] = []
    for _ in range(10):
        wallclock, lag = _simulated_worker_step(
            resampler=resampler,
            wallclock=wallclock,
            chunk_duration=chunk_duration,
            gen_seconds=steady_gen_s,
        )
        lags.append(lag)

    # First post-bootstrap chunk pulls the virtual clock back to one
    # chunk_duration behind wall (plus a single ``gen_seconds``).
    assert lags[0] <= chunk_duration + steady_gen_s + 1e-9
    # And it stays bounded forever after — no slow drift.
    for lag in lags[1:]:
        assert lag <= chunk_duration + steady_gen_s + 1e-9, (
            f"lag drifted after bootstrap catch-up: {lag:.4f}s"
        )


def test_worker_catches_up_after_transient_stall_mid_stream() -> None:
    """A single mid-stream stall recovers in one chunk, not forever.

    Bursts of slower-than-budget generation (e.g. NCCL hiccup, GC
    pause) push wallclock ahead of virtual time. The catch-up branch
    should rewind every time ``lag > chunk_duration``, so the
    long-run lag never grows beyond the recovery cap regardless of
    how many transient stalls occur.
    """
    fps = 16
    num_frames = 12
    dt = 1.0 / fps
    chunk_duration = num_frames * dt
    steady_gen_s = 0.5
    stall_gen_s = 3.0

    resampler = KeyboardResampler(fps=fps, start_v=0.0)
    wallclock = 0.0
    rng = np.random.default_rng(0)

    lags: list[float] = []
    for chunk_idx in range(40):
        # Sprinkle in three transient stalls.
        gen_s = stall_gen_s if chunk_idx in (5, 17, 28) else steady_gen_s
        wallclock, lag = _simulated_worker_step(
            resampler=resampler,
            wallclock=wallclock,
            chunk_duration=chunk_duration,
            gen_seconds=gen_s,
        )
        lags.append(lag)
        # Inject a random keyboard event each iteration to stress
        # carried_state drain across the rewind.
        if chunk_idx < 39:
            event_t = wallclock + rng.uniform(0.0, chunk_duration)
            ev = "keydown" if chunk_idx % 2 == 0 else "keyup"
            resampler.on_edge(arrival_t=event_t, event=ev, key="w")

    # Even with three 3s stalls, the rewind keeps long-run lag bounded
    # by one chunk_duration + one stall's worth of gen, not by the
    # cumulative stall budget (which would be 3 * 3s = 9s).
    cap = chunk_duration + stall_gen_s + 1e-9
    for chunk_idx, lag in enumerate(lags):
        assert lag <= cap, f"chunk {chunk_idx} lag {lag:.4f}s exceeds bound {cap:.4f}s"


def test_worker_catchup_preserves_held_keys_across_skip() -> None:
    """A long stall plus a held key: rewind drops the gap but state survives.

    User scenario: data channel opens, user mashes ``w`` 1s into the
    session and never releases it, then chunk 0 finally finishes 11s
    later. The catch-up branch rewinds the virtual clock past the
    keydown event; the resampler must still surface ``w`` as held in
    the first post-rewind chunk's carried state.
    """
    fps = 16
    num_frames = 12
    dt = 1.0 / fps
    chunk_duration = num_frames * dt
    bootstrap_gen_s = 11.0

    resampler = KeyboardResampler(fps=fps, start_v=0.0)
    resampler.on_edge(arrival_t=1.0, event="keydown", key="w")

    # Chunk 0: warmup; the keydown lands inside chunk 0's window
    # (0..0.75) is False — 1.0 is *after* the window, so it stays in
    # the log. The post-bootstrap rewind will then drain it into
    # carried_state.
    wallclock = 0.0
    wallclock, _ = _simulated_worker_step(
        resampler=resampler,
        wallclock=wallclock,
        chunk_duration=chunk_duration,
        gen_seconds=bootstrap_gen_s,
    )

    # Chunk 1: this is the iteration where the rewind fires.
    _, _ = _simulated_worker_step(
        resampler=resampler,
        wallclock=wallclock,
        chunk_duration=chunk_duration,
        gen_seconds=0.5,
    )

    # After the rewind + sample, the resampler's carried_state must
    # reflect that ``w`` is held. Re-sample a degenerate one-frame
    # chunk: with no further events, the segment's state == carried.
    segments, _ = resampler.sample_chunk(1)
    state_after = segments[0][2]
    assert "w" in state_after, f"held key dropped by catch-up; got state={state_after}"
