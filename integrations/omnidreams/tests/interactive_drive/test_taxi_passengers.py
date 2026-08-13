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

"""CPU tests for Crazy Robotaxi pickup-passenger conditioning tracks."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from omnidreams.interactive_drive.crazy_robotaxi.app import CrazyRobotaxiRuntime
from omnidreams.interactive_drive.crazy_robotaxi.game import (
    TaxiGameController,
    TaxiGameSnapshot,
    TaxiSessionState,
)
from omnidreams.interactive_drive.crazy_robotaxi.input import (
    CrazyRobotaxiKeyboardState,
)
from omnidreams.interactive_drive.crazy_robotaxi.passengers import (
    build_pickup_passenger_trajectories,
)
from omnidreams.interactive_drive.math3d import rig_pose_from_vehicle_state
from omnidreams.interactive_drive.types import (
    DynamicActorTrajectory,
    TrajectoryChunk,
    VehicleState,
)

pytestmark = pytest.mark.ci_cpu


def _snapshot(
    *pickup_targets_xyz_m: tuple[float, float, float],
    session_state: TaxiSessionState = "playing",
    pickup_passengers_xyz_m: tuple[tuple[float, float, float], ...] = (),
) -> TaxiGameSnapshot:
    phase = "seeking_pickup" if pickup_targets_xyz_m else "to_dropoff"
    target_xyz_m = pickup_targets_xyz_m[0] if pickup_targets_xyz_m else (0.0, 0.0, 0.0)
    return TaxiGameSnapshot(
        phase=phase,
        target_xyz_m=target_xyz_m,
        distance_m=0.0,
        relative_bearing_rad=0.0,
        target_radius_m=5.0,
        remaining_time_s=None,
        score=0,
        session_state=session_state,
        pickup_targets_xyz_m=pickup_targets_xyz_m,
        pickup_passengers_xyz_m=pickup_passengers_xyz_m,
    )


def _trajectory(
    timestamps_us: np.ndarray,
    *,
    dynamic_actors: tuple[DynamicActorTrajectory, ...] = (),
) -> TrajectoryChunk:
    states = tuple(
        VehicleState(
            x_m=0.0,
            y_m=0.0,
            z_m=0.0,
            yaw_rad=0.0,
            speed_mps=0.0,
            steer_rad=0.0,
        )
        for _ in timestamps_us
    )
    return TrajectoryChunk(
        timestamps_us=timestamps_us,
        rig_poses_world=np.stack(
            [rig_pose_from_vehicle_state(state) for state in states]
        ),
        vehicle_states=states,
        boundary_state_after_chunk=states[-1],
        dynamic_actors=dynamic_actors,
    )


def _actor(entity_id: str = "existing-traffic") -> DynamicActorTrajectory:
    return DynamicActorTrajectory(
        entity_id=entity_id,
        object_type="Car",
        timestamps_us=np.array([100], dtype=np.int64),
        translations_world=np.zeros((1, 3), dtype=np.float32),
        orientations_xyzw=np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        dimensions_lwh=np.array([4.0, 2.0, 1.5], dtype=np.float32),
    )


def test_builds_one_grounded_pedestrian_for_every_pickup_target() -> None:
    timestamps_us = np.array([100, 200], dtype=np.int64)
    targets = ((1.0, 2.0, 0.25), (5.0, 7.0, -0.5))

    actors = build_pickup_passenger_trajectories(
        (_snapshot(*targets), _snapshot(*targets)), timestamps_us
    )

    assert len(actors) == 2
    actors_by_xy = {tuple(actor.translations_world[0, :2]): actor for actor in actors}
    for target_xyz_m in targets:
        actor = actors_by_xy[target_xyz_m[:2]]
        assert actor.object_type == "Pedestrian"
        assert actor.is_simulated
        np.testing.assert_array_equal(actor.timestamps_us, timestamps_us)
        np.testing.assert_allclose(
            actor.translations_world,
            np.array(
                [
                    [target_xyz_m[0], target_xyz_m[1], target_xyz_m[2] + 0.9],
                    [target_xyz_m[0], target_xyz_m[1], target_xyz_m[2] + 0.9],
                ],
                dtype=np.float32,
            ),
        )
        np.testing.assert_array_equal(
            actor.dimensions_lwh, np.array([0.6, 0.6, 1.8], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            actor.orientations_xyzw,
            np.array(
                [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
        )


def test_runtime_retains_existing_actors_before_adding_passengers() -> None:
    timestamps_us = np.array([100], dtype=np.int64)
    existing_actor = _actor()
    trajectory = _trajectory(timestamps_us, dynamic_actors=(existing_actor,))
    snapshots = (_snapshot((1.0, 2.0, 0.0)),)
    controller = cast(
        TaxiGameController,
        SimpleNamespace(advance_frames=lambda trajectory, frame_interval_s: snapshots),
    )
    keyboard = cast(
        CrazyRobotaxiKeyboardState,
        SimpleNamespace(),
    )
    runtime = CrazyRobotaxiRuntime(controller, keyboard)

    update = runtime.advance_frames(trajectory, frame_interval_s=0.1)

    assert update.frame_application_states == snapshots
    assert update.trajectory.dynamic_actors[0] is existing_actor
    assert update.trajectory.dynamic_actors[1].object_type == "Pedestrian"


def test_passenger_uses_roadside_position_instead_of_marker_center() -> None:
    marker = (1.0, 2.0, 0.0)
    roadside_passenger = (1.0, 4.0, 0.25)

    actors = build_pickup_passenger_trajectories(
        (
            _snapshot(
                marker,
                pickup_passengers_xyz_m=(roadside_passenger,),
            ),
        ),
        np.array([100], dtype=np.int64),
    )

    assert len(actors) == 1
    np.testing.assert_allclose(
        actors[0].translations_world[0],
        [roadside_passenger[0], roadside_passenger[1], 1.15],
    )


def test_pickup_removes_passenger_on_completion_frame_and_adds_next_fare() -> None:
    timestamps_us = np.array([100, 200, 300, 400], dtype=np.int64)
    first_target = (1.0, 2.0, 0.0)
    next_target = (8.0, 9.0, 0.0)

    actors = build_pickup_passenger_trajectories(
        (
            _snapshot(first_target),
            _snapshot(first_target),
            _snapshot(),
            _snapshot(next_target),
        ),
        timestamps_us,
    )

    assert len(actors) == 3
    np.testing.assert_array_equal(actors[0].timestamps_us, [100])
    np.testing.assert_array_equal(actors[1].timestamps_us, [200])
    np.testing.assert_array_equal(actors[2].timestamps_us, [400])
    np.testing.assert_allclose(actors[2].translations_world[0], [8.0, 9.0, 0.9])


def test_repeated_visibility_is_split_into_separate_tracks() -> None:
    timestamps_us = np.array([100, 200, 300], dtype=np.int64)
    target = (1.0, 2.0, 0.0)

    actors = build_pickup_passenger_trajectories(
        (_snapshot(target), _snapshot(), _snapshot(target)), timestamps_us
    )

    assert len(actors) == 2
    assert actors[0].entity_id == actors[1].entity_id
    np.testing.assert_array_equal(actors[0].timestamps_us, [100])
    np.testing.assert_array_equal(actors[1].timestamps_us, [300])


def test_finished_session_hides_pickup_passengers() -> None:
    actors = build_pickup_passenger_trajectories(
        (_snapshot((1.0, 2.0, 0.0), session_state="leaderboard"),),
        np.array([100], dtype=np.int64),
    )

    assert actors == ()
