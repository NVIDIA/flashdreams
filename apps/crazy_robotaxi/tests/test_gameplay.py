# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU regressions for taxi-specific rules, navigation, and dynamics."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from crazy_robotaxi.dynamics import TaxiVehicleConfig, integrate_taxi_vehicle
from crazy_robotaxi.high_scores import HighScoreStore, validate_player_name
from crazy_robotaxi.navigation import (
    LanePosition,
    NavigationLane,
    NavigationWaypoint,
    TaxiNavigationMap,
)
from crazy_robotaxi.passengers import build_pickup_passenger_trajectories
from crazy_robotaxi.rules import TaxiGameConfig, TaxiGameController, TaxiGameSnapshot
from omnidreams_game_engine.math3d import rig_pose_from_vehicle_state
from omnidreams_game_engine.types import DriverCommand, TrajectoryChunk, VehicleState

pytestmark = pytest.mark.ci_cpu


def _state(x_m: float = 0.0, y_m: float = 0.0) -> VehicleState:
    return VehicleState(x_m, y_m, 0.0, 0.0, 0.0, 0.0)


def _trajectory(*positions_xy: tuple[float, float]) -> TrajectoryChunk:
    states = tuple(_state(*position) for position in positions_xy)
    return TrajectoryChunk(
        timestamps_us=np.arange(len(states), dtype=np.int64),
        rig_poses_world=np.stack(
            [rig_pose_from_vehicle_state(state) for state in states]
        ),
        vehicle_states=states,
        boundary_state_after_chunk=states[-1],
    )


def _controller(
    config: TaxiGameConfig | None = None,
    *,
    high_score_store: HighScoreStore | None = None,
) -> TaxiGameController:
    return TaxiGameController(
        scene_id="taxi-test",
        reference_route_world=np.asarray(
            [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]], dtype=np.float32
        ),
        initial_state=_state(),
        config=config or TaxiGameConfig(waypoint_spacing_m=1000.0),
        high_score_store=high_score_store,
    )


def _lane(start: tuple[float, float], end: tuple[float, float]) -> NavigationLane:
    return NavigationLane(np.asarray([[*start, 0.0], [*end, 0.0]], dtype=np.float32))


def test_navigation_uses_directed_road_distance() -> None:
    navigation = TaxiNavigationMap(
        (
            _lane((0.0, 0.0), (10.0, 0.0)),
            _lane((10.0, 0.0), (20.0, 10.0)),
            _lane((20.0, 10.0), (20.0, 20.0)),
        )
    )
    destination = NavigationWaypoint(
        np.asarray([20.0, 20.0, 0.0], dtype=np.float32),
        lane_index=2,
        distance_along_lane_m=10.0,
    )

    route = navigation.route(LanePosition(0, 0.0, 0.0, 0.0), destination)

    assert route is not None
    assert route.lane_indices == (0, 1, 2)
    assert route.distance_m == pytest.approx(20.0 + math.sqrt(200.0))


def test_taxi_brake_from_rest_enters_reverse() -> None:
    result = integrate_taxi_vehicle(
        _state(),
        DriverCommand(brake=1.0, manual_control=True),
        dt_s=0.1,
        vehicle=TaxiVehicleConfig(),
    )

    assert result.speed_mps < 0.0


def test_fare_and_game_over_flow_reaches_v2_name_entry(tmp_path: Path) -> None:
    store = HighScoreStore(tmp_path / "scores.csv")
    controller = _controller(
        TaxiGameConfig(
            waypoint_spacing_m=1000.0,
            global_time_s=1.0,
            dropoff_time_bonus_s=0.0,
            high_scores_path=store.path,
        ),
        high_score_store=store,
    )

    controller.advance(_trajectory((100.0, 0.0), (0.0, 0.0)), 0.0)
    controller.advance(_trajectory((0.0, 0.0)), 1.0)
    game_over = controller.snapshot(_state())

    assert not controller.is_playing
    assert game_over.score == 4100
    assert game_over.session_state == "awaiting_name"

    controller.submit_high_score_name("PLAYER 1")
    leaderboard = controller.snapshot(_state())
    assert leaderboard.session_state == "leaderboard"
    assert [(entry.name, entry.score) for entry in leaderboard.leaderboard] == [
        ("PLAYER 1", 4100)
    ]


def test_high_scores_order_by_score_then_timestamp(tmp_path: Path) -> None:
    store = HighScoreStore(tmp_path / "scores.csv")
    store.record("LATER", 900, achieved_at_utc="2026-08-10T12:00:01+00:00")
    store.record("HIGH", 1200, achieved_at_utc="2026-08-10T12:00:02+00:00")
    store.record("EARLIER", 900, achieved_at_utc="2026-08-10T12:00:00+00:00")

    assert validate_player_name(" A-B_C ") == "A-B_C"
    assert [(entry.name, entry.score) for entry in store.read()] == [
        ("HIGH", 1200),
        ("EARLIER", 900),
        ("LATER", 900),
    ]


def test_passenger_tracks_follow_snapshot_visibility() -> None:
    target = (1.0, 2.0, 0.25)

    def snapshot(*targets: tuple[float, float, float]) -> TaxiGameSnapshot:
        return TaxiGameSnapshot(
            phase="seeking_pickup" if targets else "to_dropoff",
            target_xyz_m=targets[0] if targets else (0.0, 0.0, 0.0),
            distance_m=0.0,
            relative_bearing_rad=0.0,
            target_radius_m=5.0,
            remaining_time_s=None,
            score=0,
            pickup_targets_xyz_m=targets,
        )

    actors = build_pickup_passenger_trajectories(
        (snapshot(target), snapshot(), snapshot(target)),
        np.asarray([100, 200, 300], dtype=np.int64),
    )

    assert len(actors) == 2
    assert actors[0].entity_id == actors[1].entity_id
    np.testing.assert_array_equal(actors[0].timestamps_us, [100])
    np.testing.assert_array_equal(actors[1].timestamps_us, [300])
