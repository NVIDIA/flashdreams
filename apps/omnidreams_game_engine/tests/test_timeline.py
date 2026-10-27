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
        28,
        32,
    ]


def test_catchup_is_bounded_and_never_extrapolates():
    timeline = WorldTimeline()
    for tick in range(60):
        timeline.append(sample(tick))
    first = timeline.prepare(5, threading.Event(), max_rate=1.25)
    assert first is not None and first.end_tick == 16
    timeline.commit(first)
    catchup = timeline.prepare(8, threading.Event(), max_rate=1.25)
    assert catchup is not None and catchup.end_tick == 56
    assert [catchup.start_tick + i for i in catchup.indices] == list(range(21, 57, 5))
    timeline.commit(catchup)
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
