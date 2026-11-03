# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for decoded state and chunk assembly."""

from __future__ import annotations

import json

import numpy as np
import pytest
from omnidreams_game_engine.api.v0.assembly import ChunkAssembler
from omnidreams_game_engine.api.v0.state import ActorState, FluStateCodec, FrameState
from omnidreams_game_engine.api.v0.stub_engine import ScriptedDrive
from omnidreams_game_engine.api.v0.timeline import ChunkTimeline
from omnidreams_game_engine.types import DriverCommand, VehicleState

pytestmark = pytest.mark.ci_cpu

T0 = 100


def _actor(entity_id: str, x: float) -> ActorState:
    return ActorState(
        entity_id=entity_id,
        object_type="Car",
        position_world=np.asarray((x, 0.0, 0.8), dtype=np.float32),
        orientation_xyzw=np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32),
        dimensions_lwh=np.asarray((4.5, 1.9, 1.6), dtype=np.float32),
    )


def _frame(g: int, *, actors: tuple[ActorState, ...] = ()) -> FrameState:
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = float(g)
    return FrameState(
        g=g,
        tick=T0 + 2 * g,
        rig_to_world=pose,
        vehicle=VehicleState(
            x_m=float(g), y_m=0.0, z_m=0.0, yaw_rad=0.0, speed_mps=15.0, steer_rad=0.0
        ),
        command=DriverCommand(throttle=0.1 * (g % 3)),
        actors=actors,
    )


def _assembler(epoch: int = 1) -> ChunkAssembler:
    assembler = ChunkAssembler(ChunkTimeline())
    assembler.begin_epoch(epoch)
    return assembler


def test_chunk_is_ready_only_when_complete_and_in_any_order() -> None:
    assembler = _assembler()
    for g in (4, 2, 0, 3):
        assert assembler.add(1, _frame(g)) == "added"
    assert not assembler.ready(0)
    assert assembler.missing(0) == [1]
    assert assembler.take(0) is None
    assembler.add(1, _frame(1))
    chunk = assembler.take(0)
    assert chunk is not None
    assert [frame.g for frame in chunk.frames] == [0, 1, 2, 3, 4]
    assert chunk.ticks == (100, 102, 104, 106, 108)
    assert chunk.t0_tick == T0
    assert assembler.take(0) is None


def test_duplicates_stale_epochs_and_taken_chunks_are_dropped() -> None:
    assembler = _assembler(epoch=2)
    assert assembler.add(1, _frame(0)) == "stale"
    for g in range(5):
        assembler.add(2, _frame(g))
    assert assembler.add(2, _frame(3)) == "duplicate"
    assert assembler.take(0) is not None
    assert assembler.add(2, _frame(4)) == "stale"
    assert assembler.stats.duplicates == 1
    assert assembler.stats.stale == 2


def test_a_new_epoch_is_a_discontinuity() -> None:
    assembler = _assembler()
    for g in range(3):
        assembler.add(1, _frame(g))
    assembler.begin_epoch(2)
    assert assembler.missing(0) == [0, 1, 2, 3, 4]
    assert assembler.t0_tick is None
    assert assembler.add(1, _frame(3)) == "stale"
    assert not assembler.ready(0)


def test_trajectory_carries_poses_states_commands_and_timestamps() -> None:
    assembler = _assembler()
    for g in range(5):
        assembler.add(1, _frame(g))
    chunk = assembler.take(0)
    assert chunk is not None
    trajectory = chunk.to_trajectory(ChunkTimeline(), base_timestamp_us=5_000_000)
    np.testing.assert_array_equal(
        trajectory.timestamps_us, 5_000_000 + np.arange(5) * 33333
    )
    assert trajectory.rig_poses_world.shape == (5, 4, 4)
    assert trajectory.rig_poses_world[3, 0, 3] == 3.0
    assert [state.x_m for state in trajectory.vehicle_states] == [0, 1, 2, 3, 4]
    assert trajectory.boundary_state_after_chunk.x_m == 4.0
    assert [c.throttle for c in trajectory.applied_commands] == pytest.approx(
        [0.0, 0.1, 0.2, 0.0, 0.1]
    )


def test_actors_appearing_and_leaving_are_split_into_tracks() -> None:
    car = _actor("car-1", 10.0)
    ped = _actor("ped-2", 3.0)
    presence = {
        5: (car,),
        6: (car, ped),
        7: (car, ped),
        8: (ped,),
        9: (car,),
        10: (car,),
        11: (),
        12: (car,),
    }
    assembler = _assembler()
    for g in range(13):
        assembler.add(1, _frame(g, actors=presence.get(g, ())))
    assert assembler.take(0) is not None
    chunk = assembler.take(1)
    assert chunk is not None
    trajectory = chunk.to_trajectory(ChunkTimeline(), base_timestamp_us=0)
    tracks = {actor.entity_id: actor for actor in trajectory.dynamic_actors}
    assert sorted(tracks) == ["car-1", "car-1~1", "car-1~2", "ped-2"]
    timestamps = trajectory.timestamps_us
    np.testing.assert_array_equal(tracks["car-1"].timestamps_us, timestamps[0:3])
    np.testing.assert_array_equal(tracks["car-1~1"].timestamps_us, timestamps[4:6])
    np.testing.assert_array_equal(tracks["car-1~2"].timestamps_us, timestamps[7:8])
    np.testing.assert_array_equal(tracks["ped-2"].timestamps_us, timestamps[1:4])
    assert tracks["ped-2"].translations_world.shape == (3, 3)
    assert all(track.is_simulated for track in tracks.values())


def test_flu_state_codec_round_trips_exactly_through_json() -> None:
    codec = FluStateCodec()
    original = ScriptedDrive().frame_state(4321)
    wire = json.loads(json.dumps(codec.encode(original)))
    decoded = codec.decode(wire, g=17, tick=4321)
    assert decoded.g == 17 and decoded.tick == 4321
    np.testing.assert_array_equal(decoded.rig_to_world, original.rig_to_world)
    assert decoded.rig_to_world.dtype == np.float32
    assert decoded.vehicle == original.vehicle
    assert decoded.command == original.command
    assert [a.entity_id for a in decoded.actors] == [
        a.entity_id for a in original.actors
    ]
    for got, want in zip(decoded.actors, original.actors, strict=True):
        np.testing.assert_array_equal(got.position_world, want.position_world)
        np.testing.assert_array_equal(got.orientation_xyzw, want.orientation_xyzw)
        np.testing.assert_array_equal(got.dimensions_lwh, want.dimensions_lwh)


POSE = [float(v) for v in np.eye(4).reshape(16)]


@pytest.mark.parametrize(
    "state",
    [
        {"vehicle": {}},
        {"rig_to_world": [0.0] * 15, "vehicle": {}},
        {"rig_to_world": POSE, "vehicle": [1, 2]},
        {"rig_to_world": POSE, "vehicle": {}, "command": {"throttle": None}},
        {"rig_to_world": POSE, "vehicle": {}, "command": [1, 2]},
        {"rig_to_world": POSE, "vehicle": {"x_m": [1.0]}},
        {"rig_to_world": POSE, "vehicle": {}, "actors": [{}]},
        {
            "rig_to_world": POSE,
            "vehicle": {},
            "actors": [
                {
                    "entity_id": "a",
                    "position_world": [1, 2],
                    "dimensions_lwh": [1, 1, 1],
                }
            ],
        },
    ],
)
def test_flu_state_codec_raises_value_error_on_unreadable_states(state: dict) -> None:
    with pytest.raises(ValueError, match="Unreadable flu/0 state"):
        FluStateCodec().decode(state, g=0, tick=0)


def test_flu_state_codec_ignores_unknown_keys_and_defaults_from_the_pose() -> None:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = (1.0, 2.0, 3.0)
    state = {
        "rig_to_world": [float(v) for v in pose.reshape(16)],
        "vehicle": {"speed_mps": 4.0, "velocity_x_mps": None, "gear": 3},
    }
    decoded = FluStateCodec().decode(state, g=0, tick=0)
    assert (decoded.vehicle.x_m, decoded.vehicle.y_m, decoded.vehicle.z_m) == (1, 2, 3)
    assert decoded.vehicle.speed_mps == 4.0
    assert decoded.vehicle.velocity_x_mps is None
