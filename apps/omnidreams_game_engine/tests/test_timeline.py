# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generation follows recorded gameplay without mutating its path."""

import threading
from dataclasses import replace

import numpy as np
import pytest

from omnidreams_game_engine.engine import SimulationStep
from omnidreams_game_engine.timeline import WorldSample, WorldTimeline
from omnidreams_game_engine.types import (
    DriverCommand,
    DynamicActorTrajectory,
    TrajectoryChunk,
    VehicleState,
)

pytestmark = pytest.mark.ci_cpu


def sample(tick):
    # An L-shaped path: interpolating start/end would cross the inside corner.
    state = VehicleState(min(tick, 10), max(0, tick - 10), 0, 0, 1, 0)
    pose = np.eye(4, dtype=np.float32)[None]
    pose[0, :3, 3] = state.x_m, state.y_m, 0
    timestamps = np.asarray([round(tick * 1e6 / 120)], dtype=np.int64)
    actor = DynamicActorTrajectory(
        "traffic",
        "car",
        timestamps,
        pose[:, :3, 3].copy(),
        np.asarray([[0, 0, 0, 1]], dtype=np.float32),
        np.ones(3),
    )
    path = TrajectoryChunk(
        timestamps,
        pose,
        (state,),
        state,
        applied_commands=(DriverCommand(throttle=float(tick > 3)),),
        dynamic_actors=(actor,),
        static_collision_detected=tick == 7,
        static_collision_frame_index=0 if tick == 7 else None,
    )
    return WorldSample(SimulationStep(path, (f"tick:{tick}",)), presentation=tick)


def test_recorded_turn_and_intermediate_collision_survive_generation_sampling():
    timeline = WorldTimeline()
    for tick in range(33):
        timeline.append(sample(tick))
    window = timeline.prepare(5, threading.Event())
    assert window is not None
    assert window.indices == (0, 4, 8, 12, 16)
    step = window.recorded_step()
    assert [(s.x_m, s.y_m) for s in step.trajectory.vehicle_states] == [
        (0, 0),
        (4, 0),
        (8, 0),
        (10, 2),
        (10, 6),
    ]
    assert step.trajectory.static_collision_detected
    assert step.trajectory.static_collision_frame_index == 2
    assert len(step.trajectory.dynamic_actors[0].timestamps_us) == 17
    assert step.game_frames == ("tick:0", "tick:4", "tick:8", "tick:12", "tick:16")
    # A completed model chunk is the only operation that commits the cursor.
    assert timeline.generation_tick == -1
    timeline.append(sample(33))
    assert window.recorded_step().game_frames == step.game_frames
    with pytest.raises(RuntimeError, match="Commit"):
        timeline.prepare(4, threading.Event())
    timeline.commit(window)
    assert timeline.generation_tick == 16
    next_window = timeline.prepare(4, threading.Event())
    assert next_window is not None
    assert next_window.start_tick == 17
    assert [next_window.samples[i].presentation for i in next_window.indices] == [
        20,
        24,
        29,
        33,
    ]


def test_sustained_slow_generation_does_not_accumulate_input_latency():
    timeline = WorldTimeline()
    for tick in range(17):
        timeline.append(sample(tick))
    first = timeline.prepare(5, threading.Event())
    assert first is not None and first.end_tick == 16
    timeline.commit(first)
    # Model throughput varies below 24 fps for a full minute. The next chunk
    # must reach the most recent recorded input, not replay an increasing backlog.
    for duration_ticks in (48, 72) * 60:
        start = timeline.gameplay_tick + 1
        latest = timeline.gameplay_tick + duration_ticks
        for tick in range(start, latest + 1):
            entry = sample(tick)
            if tick == latest:
                entry = replace(entry, transition_timestamp_us=tick * 1_000_000 // 120)
            timeline.append(entry)
        window = timeline.prepare(8, threading.Event())
        assert window is not None and window.end_tick == latest
        assert len(window.samples) == duration_ticks
        assert window.transition_timestamps_us()[-1] == latest * 1_000_000 // 120
        assert all(b > a for a, b in zip(window.indices, window.indices[1:]))
        assert window.recorded_step().game_frames[-1] == f"tick:{latest}"
        timeline.commit(window)
        assert timeline.generation_tick == timeline.gameplay_tick
    stop = threading.Event()
    stop.set()
    assert timeline.prepare(8, stop) is None


def test_history_overflow_and_worker_failure_are_explicit():
    timeline = WorldTimeline(capacity=4)
    for tick in range(4):
        timeline.append(sample(tick))
    with pytest.raises(RuntimeError, match="history capacity"):
        timeline.append(sample(4))
    cause = ValueError("physics failed")
    timeline.close(cause)
    with pytest.raises(RuntimeError, match="Gameplay simulation failed") as error:
        timeline.prepare(3, threading.Event())
    assert error.value.__cause__ is cause


def test_close_wakes_generation_waiter():
    timeline = WorldTimeline()
    results = []
    worker = threading.Thread(
        target=lambda: results.append(timeline.prepare(8, threading.Event()))
    )
    worker.start()
    timeline.close()
    worker.join(1)
    assert not worker.is_alive()
    assert results == [None]


def test_terminal_window_freezes_final_state_without_inventing_future_gameplay():
    timeline = WorldTimeline()
    for tick in range(7):
        entry = sample(tick)
        if tick == 3:
            entry = replace(entry, transition_timestamp_us=25_000)
        timeline.append(entry)
    timeline.finish()
    window = timeline.prepare(5, threading.Event())
    assert window is not None
    assert window.end_tick == 6
    assert window.indices == (0, 4, 6, 6, 6)
    assert window.transition_timestamps_us() == (None, 25_000, None, None, None)
    assert window.recorded_step().game_frames[-3:] == ("tick:6",) * 3
    timeline.commit(window)
    assert timeline.generation_tick == timeline.gameplay_tick == 6
    assert timeline.prepare(8, threading.Event()) is None
