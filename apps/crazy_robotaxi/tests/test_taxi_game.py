# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU tests for interactive-drive taxi-game state and projection."""

import math
from pathlib import Path

import numpy as np
import pytest
from crazy_robotaxi.game import (
    TaxiGameConfig,
    TaxiGameController,
    TaxiGameSnapshot,
    project_segment_pose_to_bev,
    project_target_to_bev,
    project_taxi_marker_to_camera,
    project_taxi_markers_to_camera,
    relative_target_bearing_rad,
)
from crazy_robotaxi.high_scores import HighScoreStore
from crazy_robotaxi.navigation import NavigationLane
from omnidreams_game_engine.camera import FThetaCameraModel
from omnidreams_game_engine.config import BevConfig
from omnidreams_game_engine.math3d import rig_pose_from_vehicle_state
from omnidreams_game_engine.simulation.map_bounds import MapBounds
from omnidreams_game_engine.types import (
    CameraCalibration,
    TrajectoryChunk,
    VehicleState,
)


def _state(x_m: float = 0.0, y_m: float = 0.0, yaw_rad: float = 0.0) -> VehicleState:
    return VehicleState(
        x_m=x_m,
        y_m=y_m,
        z_m=0.0,
        yaw_rad=yaw_rad,
        speed_mps=0.0,
        steer_rad=0.0,
    )


def _camera_calibration() -> CameraCalibration:
    return CameraCalibration(
        clipgt_name="camera:test",
        logical_name="camera_test",
        width=100,
        height=80,
        cx=50.0,
        cy=40.0,
        polynomial=np.array([0.0, 0.01], dtype=np.float32),
        is_backward_polynomial=True,
        linear_cde=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        sensor_to_rig_flu=np.eye(4, dtype=np.float32),
    )


def _trajectory(*positions_xy: tuple[float, float]) -> TrajectoryChunk:
    states = tuple(_state(x_m, y_m) for x_m, y_m in positions_xy)
    poses = np.stack([rig_pose_from_vehicle_state(state) for state in states])
    return TrajectoryChunk(
        timestamps_us=np.arange(len(positions_xy), dtype=np.int64),
        rig_poses_world=poses,
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
        reference_route_world=np.array(
            [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]], dtype=np.float32
        ),
        initial_state=_state(),
        config=config or TaxiGameConfig(enabled=True, waypoint_spacing_m=1000.0),
        high_score_store=high_score_store,
    )


def test_seeded_waypoint_layout_is_deterministic() -> None:
    route = np.stack(
        [
            np.linspace(0.0, 200.0, 101),
            np.zeros(101),
            np.zeros(101),
        ],
        axis=1,
    ).astype(np.float32)
    config = TaxiGameConfig(enabled=True, seed=17, waypoint_spacing_m=10.0)

    first = TaxiGameController(
        scene_id="scene",
        reference_route_world=route,
        initial_state=_state(),
        config=config,
    )
    second = TaxiGameController(
        scene_id="scene",
        reference_route_world=route,
        initial_state=_state(),
        config=config,
    )
    different_seed = TaxiGameController(
        scene_id="scene",
        reference_route_world=route,
        initial_state=_state(),
        config=TaxiGameConfig(enabled=True, seed=18, waypoint_spacing_m=10.0),
    )

    assert (
        first.snapshot(_state()).target_xyz_m == second.snapshot(_state()).target_xyz_m
    )
    assert (
        first.snapshot(_state()).target_xyz_m
        != different_seed.snapshot(_state()).target_xyz_m
    )


def test_unseeded_waypoint_layout_requests_fresh_entropy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_seeds: list[int | None] = []
    original_default_rng = np.random.default_rng

    def recording_default_rng(seed: int | None = None) -> np.random.Generator:
        requested_seeds.append(seed)
        return original_default_rng(17)

    monkeypatch.setattr(np.random, "default_rng", recording_default_rng)

    _controller(TaxiGameConfig(enabled=True, waypoint_spacing_m=1000.0))

    assert requested_seeds == [None]


@pytest.mark.parametrize(
    ("initial_yaw_rad", "expected_x_sign"),
    [(0.0, 1.0), (math.pi, -1.0)],
)
def test_initial_pickup_layout_includes_target_in_front_of_ego(
    initial_yaw_rad: float, expected_x_sign: float
) -> None:
    route = np.asarray([[-80.0, 0.0, 0.0], [80.0, 0.0, 0.0]], dtype=np.float32)
    controller = TaxiGameController(
        scene_id="forward-pickup",
        reference_route_world=route,
        initial_state=_state(yaw_rad=initial_yaw_rad),
        config=TaxiGameConfig(enabled=True, seed=17, waypoint_spacing_m=10.0),
        initial_camera=_camera_calibration(),
    )

    pickup = controller.snapshot(_state(yaw_rad=initial_yaw_rad))

    assert any(
        target[0] * expected_x_sign > 0.0 for target in pickup.pickup_targets_xyz_m
    )


def test_initial_pickup_can_be_distant_but_must_project_inside_camera() -> None:
    controller = TaxiGameController(
        scene_id="visible-pickup",
        reference_route_world=np.asarray(
            [[80.0, 0.0, 0.0], [120.0, 0.0, 0.0]], dtype=np.float32
        ),
        navigation_routes_world=(
            np.asarray([[80.0, 0.0, 0.0], [120.0, 0.0, 0.0]], dtype=np.float32),
            np.asarray([[25.0, 80.0, 0.0], [26.0, 80.0, 0.0]], dtype=np.float32),
            np.asarray([[-25.0, 0.0, 0.0], [-26.0, 0.0, 0.0]], dtype=np.float32),
        ),
        initial_state=_state(),
        config=TaxiGameConfig(enabled=True, seed=17, waypoint_spacing_m=1000.0),
        initial_camera=_camera_calibration(),
    )

    pickup = controller.snapshot(_state())

    assert any(
        target[0] >= 80.0 and abs(target[1]) <= 1.0
        for target in pickup.pickup_targets_xyz_m
    )


def test_initial_pickup_prefers_visible_candidate_closest_to_200_meters() -> None:
    controller = TaxiGameController(
        scene_id="bounded-visible-pickup",
        reference_route_world=np.asarray(
            [[150.0, 0.0, 0.0], [250.0, 0.0, 0.0]], dtype=np.float32
        ),
        initial_state=_state(),
        config=TaxiGameConfig(enabled=True, seed=2, waypoint_spacing_m=100.0),
        initial_camera=_camera_calibration(),
    )

    pickup = controller.snapshot(_state())

    assert 20.0 <= pickup.distance_m <= 200.0


def test_initial_pickup_can_exceed_200_when_that_is_the_closest_visible_choice() -> (
    None
):
    controller = TaxiGameController(
        scene_id="nearest-visible-pickup",
        reference_route_world=np.asarray(
            [[210.0, 0.0, 0.0], [450.0, 0.0, 0.0]], dtype=np.float32
        ),
        initial_state=_state(),
        config=TaxiGameConfig(enabled=True, seed=2, waypoint_spacing_m=240.0),
        initial_camera=_camera_calibration(),
    )

    pickup = controller.snapshot(_state())

    assert pickup.distance_m == pytest.approx(math.hypot(210.0, 1.0))


def test_available_pickups_are_sampled_across_the_map() -> None:
    routes = (
        np.asarray([[-100.0, -100.0, 0.0], [100.0, -100.0, 0.0]], dtype=np.float32),
        np.asarray([[100.0, -100.0, 0.0], [100.0, 100.0, 0.0]], dtype=np.float32),
        np.asarray([[100.0, 100.0, 0.0], [-100.0, 100.0, 0.0]], dtype=np.float32),
        np.asarray([[-100.0, 100.0, 0.0], [-100.0, -100.0, 0.0]], dtype=np.float32),
    )
    controller = TaxiGameController(
        scene_id="varied-pickups",
        reference_route_world=routes[0],
        navigation_routes_world=routes,
        initial_state=_state(),
        config=TaxiGameConfig(
            enabled=True,
            seed=17,
            waypoint_spacing_m=20.0,
            pickup_grid_spacing_m=20.0,
        ),
        initial_camera=_camera_calibration(),
    )

    pickup = controller.snapshot(_state())
    selected_xy = {target[:2] for target in pickup.pickup_targets_xyz_m}
    assert len(selected_xy) > 8
    assert any(target[0] < 0.0 for target in selected_xy)
    assert any(target[0] > 0.0 for target in selected_xy)
    assert any(target[1] < 0.0 for target in selected_xy)
    assert any(target[1] > 0.0 for target in selected_xy)


def test_pickup_markers_and_passengers_use_separate_roadside_positions() -> None:
    lane = NavigationLane(
        centerline_world=np.asarray(
            [[0.0, 0.0, 0.0], [200.0, 0.0, 0.0]], dtype=np.float32
        ),
        road_edge_world=np.asarray(
            [[0.0, -2.0, 0.0], [200.0, -2.0, 0.0]], dtype=np.float32
        ),
    )
    controller = TaxiGameController(
        scene_id="roadside-pickups",
        reference_route_world=lane.centerline_world,
        navigation_lanes=(lane,),
        initial_state=_state(),
        config=TaxiGameConfig(
            enabled=True,
            seed=17,
            waypoint_spacing_m=20.0,
            pickup_grid_spacing_m=20.0,
            pickup_min_distance_m=0.0,
        ),
    )

    snapshot = controller.snapshot(_state())

    assert len(snapshot.pickup_targets_xyz_m) == len(snapshot.pickup_passengers_xyz_m)
    assert snapshot.pickup_targets_xyz_m
    assert all(
        target[1] == pytest.approx(-1.0) for target in snapshot.pickup_targets_xyz_m
    )
    assert all(
        passenger[1] == pytest.approx(-2.75)
        for passenger in snapshot.pickup_passengers_xyz_m
    )


def test_pickups_and_dropoffs_exclude_map_boundary_margin() -> None:
    bounds = MapBounds(x_min=0.0, y_min=0.0, x_max=300.0, y_max=100.0)
    controller = TaxiGameController(
        scene_id="bounded-targets",
        reference_route_world=np.asarray(
            [[0.0, 50.0, 0.0], [300.0, 50.0, 0.0]], dtype=np.float32
        ),
        initial_state=_state(150.0, 50.0),
        config=TaxiGameConfig(
            enabled=True,
            seed=17,
            waypoint_spacing_m=10.0,
            pickup_grid_spacing_m=20.0,
            waypoint_edge_margin_m=40.0,
        ),
        map_bounds=bounds,
    )

    seeking = controller.snapshot(_state(150.0, 50.0))
    assert seeking.pickup_targets_xyz_m
    assert all(
        40.0 <= target[0] <= 260.0 and 40.0 <= target[1] <= 60.0
        for target in seeking.pickup_targets_xyz_m
    )

    pickup = seeking.target_xyz_m
    controller.advance(_trajectory(pickup[:2]), 0.0)
    dropoff = controller.snapshot(_state(*pickup[:2]))

    assert dropoff.phase == "to_dropoff"
    assert 40.0 <= dropoff.target_xyz_m[0] <= 260.0
    assert 40.0 <= dropoff.target_xyz_m[1] <= 60.0


def test_default_targets_stay_one_hundred_meters_inside_map_bounds() -> None:
    bounds = MapBounds(x_min=0.0, y_min=0.0, x_max=500.0, y_max=500.0)
    controller = TaxiGameController(
        scene_id="default-bounded-targets",
        reference_route_world=np.asarray(
            [[0.0, 250.0, 0.0], [500.0, 250.0, 0.0]], dtype=np.float32
        ),
        initial_state=_state(250.0, 250.0),
        config=TaxiGameConfig(
            enabled=True,
            seed=17,
            waypoint_spacing_m=10.0,
            pickup_grid_spacing_m=20.0,
        ),
        map_bounds=bounds,
    )

    seeking = controller.snapshot(_state(250.0, 250.0))
    assert seeking.pickup_targets_xyz_m
    assert all(
        100.0 <= target[0] <= 400.0 and 100.0 <= target[1] <= 400.0
        for target in seeking.pickup_targets_xyz_m
    )

    pickup = seeking.target_xyz_m
    controller.advance(_trajectory(pickup[:2]), 0.0)
    dropoff = controller.snapshot(_state(*pickup[:2]))

    assert dropoff.phase == "to_dropoff"
    assert 100.0 <= dropoff.target_xyz_m[0] <= 400.0
    assert 100.0 <= dropoff.target_xyz_m[1] <= 400.0


def test_fare_completion_survives_no_directed_route_to_next_pickup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _controller()
    pickup = controller.snapshot(_state()).target_xyz_m
    controller.advance(_trajectory(pickup[:2]), 0.0)
    dropoff = controller.snapshot(_state(*pickup[:2])).target_xyz_m
    monkeypatch.setattr(
        controller._navigation,
        "route_distances",
        lambda _source, _waypoints: np.full(
            len(controller._waypoints), np.inf, dtype=np.float64
        ),
    )

    controller.advance(_trajectory(dropoff[:2]), 0.0)
    seeking = controller.snapshot(_state(*dropoff[:2]))

    assert seeking.phase == "seeking_pickup"
    assert seeking.event == "fare_complete"
    assert seeking.pickup_targets_xyz_m


def test_every_published_pickup_can_start_a_fare() -> None:
    controller = TaxiGameController(
        scene_id="available-pickups",
        reference_route_world=np.asarray(
            [[0.0, 0.0, 0.0], [200.0, 0.0, 0.0]], dtype=np.float32
        ),
        initial_state=_state(),
        config=TaxiGameConfig(enabled=True, seed=17, waypoint_spacing_m=40.0),
    )
    seeking = controller.snapshot(_state())
    alternate = next(
        target
        for target in seeking.pickup_targets_xyz_m
        if target != seeking.target_xyz_m
    )

    controller.advance(_trajectory(alternate[:2]), 0.0)
    active = controller.snapshot(_state(*alternate[:2]))

    assert active.phase == "to_dropoff"
    assert active.event == "pickup_complete"
    assert active.pickup_targets_xyz_m == ()


def test_pickup_compass_tracks_the_nearest_available_pickup() -> None:
    controller = TaxiGameController(
        scene_id="nearest-pickup-compass",
        reference_route_world=np.asarray(
            [[0.0, 0.0, 0.0], [300.0, 0.0, 0.0]], dtype=np.float32
        ),
        initial_state=_state(),
        config=TaxiGameConfig(
            enabled=True,
            seed=17,
            waypoint_spacing_m=20.0,
            pickup_grid_spacing_m=40.0,
        ),
    )
    initial = controller.snapshot(_state())
    alternate = max(
        initial.pickup_targets_xyz_m,
        key=lambda target: math.hypot(target[0], target[1]),
    )
    nearby_state = _state(alternate[0] + 1.0, alternate[1])

    nearby = controller.snapshot(nearby_state)

    assert nearby.phase == "seeking_pickup"
    assert nearby.target_xyz_m == alternate
    assert nearby.distance_m == pytest.approx(1.0)


def test_taxi_mode_rejects_route_without_travel_distance() -> None:
    route = np.zeros((2, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="no usable travel distance"):
        TaxiGameController(
            scene_id="scene",
            reference_route_world=route,
            initial_state=_state(),
            config=TaxiGameConfig(enabled=True),
        )


def test_dropoffs_stay_within_reachable_road_component() -> None:
    controller = TaxiGameController(
        scene_id="street-network",
        reference_route_world=np.array(
            [[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]], dtype=np.float32
        ),
        navigation_routes_world=(
            np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]], dtype=np.float32),
            np.array([[0.0, 100.0, 0.0], [100.0, 100.0, 0.0]], dtype=np.float32),
        ),
        initial_state=_state(),
        config=TaxiGameConfig(enabled=True, waypoint_spacing_m=1000.0),
    )

    pickup = controller.snapshot(_state())
    controller.advance(
        _trajectory((pickup.target_xyz_m[0], pickup.target_xyz_m[1])), 0.0
    )
    dropoff = controller.snapshot(
        _state(pickup.target_xyz_m[0], pickup.target_xyz_m[1])
    )

    assert dropoff.phase == "to_dropoff"
    assert abs(dropoff.target_xyz_m[1]) <= 1.0


def test_dropoff_is_at_least_two_hundred_route_meters_when_available() -> None:
    controller = TaxiGameController(
        scene_id="long-fare",
        reference_route_world=np.asarray(
            [[0.0, 0.0, 0.0], [500.0, 0.0, 0.0]], dtype=np.float32
        ),
        initial_state=_state(),
        config=TaxiGameConfig(
            enabled=True,
            seed=17,
            waypoint_spacing_m=25.0,
            pickup_grid_spacing_m=50.0,
        ),
    )
    pickup = controller.snapshot(_state()).target_xyz_m

    controller.advance(_trajectory(pickup[:2]), 0.0)
    dropoff = controller.snapshot(_state(*pickup[:2]))

    assert dropoff.phase == "to_dropoff"
    assert dropoff.distance_m >= 200.0


def test_dropoff_falls_back_to_shorter_fare_when_no_long_fare_exists() -> None:
    controller = TaxiGameController(
        scene_id="short-fare-fallback",
        reference_route_world=np.asarray(
            [[0.0, 0.0, 0.0], [150.0, 0.0, 0.0]], dtype=np.float32
        ),
        initial_state=_state(),
        config=TaxiGameConfig(
            enabled=True,
            seed=17,
            waypoint_spacing_m=25.0,
            pickup_grid_spacing_m=50.0,
        ),
    )
    pickup = controller.snapshot(_state()).target_xyz_m

    controller.advance(_trajectory(pickup[:2]), 0.0)
    dropoff = controller.snapshot(_state(*pickup[:2]))

    assert dropoff.phase == "to_dropoff"
    assert 0.0 < dropoff.distance_m < 200.0


def test_fare_uses_routed_distance() -> None:
    lanes = (
        NavigationLane(
            np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32)
        ),
        NavigationLane(
            np.asarray([[10.0, 0.0, 0.0], [20.0, 10.0, 0.0]], dtype=np.float32)
        ),
        NavigationLane(
            np.asarray([[20.0, 10.0, 0.0], [20.0, 20.0, 0.0]], dtype=np.float32)
        ),
    )
    controller = TaxiGameController(
        scene_id="routed-fare",
        reference_route_world=lanes[0].centerline_world,
        navigation_lanes=lanes,
        initial_state=_state(-200.0, 0.0),
        config=TaxiGameConfig(
            enabled=True,
            seed=4,
            waypoint_spacing_m=1000.0,
            fare_min_route_distance_m=15.0,
            fare_max_route_distance_m=100.0,
            target_speed_mps=1.0,
            grace_s=0.0,
            min_time_s=0.0,
            max_time_s=100.0,
            trip_time_multiplier=1.0,
        ),
        initial_camera=_camera_calibration(),
    )
    pickup = controller.snapshot(_state(-200.0, 0.0))
    controller.advance(
        _trajectory((pickup.target_xyz_m[0], pickup.target_xyz_m[1])), 0.0
    )

    active = controller.snapshot(_state(*pickup.target_xyz_m[:2]))

    straight_line_distance = math.hypot(
        active.target_xyz_m[0] - pickup.target_xyz_m[0],
        active.target_xyz_m[1] - pickup.target_xyz_m[1],
    )
    assert active.phase == "to_dropoff"
    assert active.remaining_time_s is not None
    assert active.remaining_time_s > straight_line_distance


def test_pickup_and_dropoff_can_complete_inside_one_chunk() -> None:
    controller = _controller()

    controller.advance(_trajectory((100.0, 0.0), (0.0, 0.0)), 1.0 / 30.0)
    snapshot = controller.snapshot(_state())

    assert snapshot.phase == "seeking_pickup"
    assert snapshot.score == 4100
    assert snapshot.event == "fare_complete"
    assert snapshot.awarded_points == 4100
    assert snapshot.awarded_global_time_s == 30.0


def test_advance_frames_returns_state_for_each_rendered_pose() -> None:
    controller = _controller()

    snapshots = controller.advance_frames(
        _trajectory((100.0, 0.0), (0.0, 0.0)), 1.0 / 30.0
    )

    assert [snapshot.phase for snapshot in snapshots] == [
        "to_dropoff",
        "seeking_pickup",
    ]
    assert snapshots[0].target_radius_m == 6.0
    assert snapshots[0].event == "pickup_complete"
    assert snapshots[0].awarded_global_time_s == 0.0
    assert snapshots[0].global_remaining_time_s == pytest.approx(60.0 - 1.0 / 30.0)
    assert snapshots[1].target_radius_m == 5.0


def test_dropoff_timer_expires_in_simulation_time() -> None:
    controller = _controller()
    controller.advance(_trajectory((100.0, 0.0)), 1.0 / 30.0)
    active = controller.snapshot(_state(100.0, 0.0))
    assert active.phase == "to_dropoff"
    assert active.remaining_time_s == pytest.approx(36.0)

    controller.advance(_trajectory((100.0, 0.0)), 36.0)
    expired = controller.snapshot(_state(100.0, 0.0))

    assert expired.phase == "seeking_pickup"
    assert expired.score == 0
    assert expired.event == "time_expired"
    assert expired.global_remaining_time_s == pytest.approx(24.0 - 1.0 / 30.0)


def test_arrival_wins_same_frame_tie_with_expiry() -> None:
    controller = _controller()
    controller.advance(_trajectory((100.0, 0.0)), 1.0 / 30.0)

    controller.advance(_trajectory((0.0, 0.0)), 100.0)
    snapshot = controller.snapshot(_state())

    assert snapshot.event == "fare_complete"
    assert snapshot.score == 4100


def test_dropoff_with_four_whole_seconds_remaining_awards_900_points() -> None:
    controller = _controller(TaxiGameConfig(enabled=True, waypoint_spacing_m=1000.0))
    controller.advance(_trajectory((100.0, 0.0)), 0.0)
    controller.advance(_trajectory((100.0, 0.0)), 31.5)

    controller.advance(_trajectory((0.0, 0.0)), 0.0)
    snapshot = controller.snapshot(_state())

    assert snapshot.score == 900
    assert snapshot.awarded_points == 900


def test_successful_dropoff_adds_thirty_seconds_to_global_timer() -> None:
    controller = _controller(
        TaxiGameConfig(
            enabled=True,
            waypoint_spacing_m=1000.0,
            global_time_s=1.0,
        )
    )
    controller.advance(_trajectory((100.0, 0.0)), 0.0)

    controller.advance(_trajectory((0.0, 0.0)), 1.0)
    snapshot = controller.snapshot(_state())

    assert snapshot.score == 4100
    assert snapshot.global_remaining_time_s == pytest.approx(30.0)
    assert snapshot.session_state == "playing"


def test_pickup_does_not_add_time_to_global_timer() -> None:
    controller = _controller(
        TaxiGameConfig(
            enabled=True,
            waypoint_spacing_m=1000.0,
            global_time_s=10.0,
        )
    )

    controller.advance(_trajectory((100.0, 0.0)), 0.0)
    snapshot = controller.snapshot(_state(100.0, 0.0))

    assert snapshot.global_remaining_time_s == 10.0
    assert snapshot.event == "pickup_complete"
    assert snapshot.awarded_global_time_s == 0.0


def test_snapshot_exposes_persisted_high_score(tmp_path: Path) -> None:
    store = HighScoreStore(tmp_path / "scores.csv")
    store.record("CHAMP", 4200, achieved_at_utc="2026-01-01T00:00:00+00:00")

    snapshot = _controller(high_score_store=store).snapshot(_state())

    assert snapshot.high_score == 4200
    assert snapshot.leaderboard == ()
    assert snapshot.as_dict()["high_score"] == 4200


def test_snapshot_omits_high_score_when_leaderboard_is_empty(tmp_path: Path) -> None:
    snapshot = _controller(
        high_score_store=HighScoreStore(tmp_path / "scores.csv")
    ).snapshot(_state())

    assert snapshot.high_score is None


def test_global_timer_ends_game_and_accepts_qualifying_name(tmp_path: Path) -> None:
    store = HighScoreStore(tmp_path / "scores.csv")
    controller = _controller(
        TaxiGameConfig(
            enabled=True,
            waypoint_spacing_m=1000.0,
            global_time_s=1.0,
            dropoff_time_bonus_s=0.0,
            high_scores_path=tmp_path / "scores.csv",
        ),
        high_score_store=store,
    )

    controller.advance(_trajectory((100.0, 0.0), (0.0, 0.0)), 0.0)
    controller.advance(_trajectory((0.0, 0.0)), 1.0)
    game_over = controller.snapshot(_state())

    assert controller.is_playing is False
    assert game_over.global_remaining_time_s == 0.0
    assert game_over.session_state == "awaiting_name"
    assert game_over.high_score_rank == 1

    controller.submit_high_score_name("PLAYER 1")
    leaderboard = controller.snapshot(_state())

    assert leaderboard.session_state == "leaderboard"
    assert [(entry.name, entry.score) for entry in leaderboard.leaderboard] == [
        ("PLAYER 1", 4100)
    ]


def test_zero_score_skips_name_entry_and_leaderboard(tmp_path: Path) -> None:
    store = HighScoreStore(tmp_path / "scores.csv")
    controller = _controller(
        TaxiGameConfig(
            enabled=True,
            waypoint_spacing_m=1000.0,
            global_time_s=1.0,
            high_scores_path=tmp_path / "scores.csv",
        ),
        high_score_store=store,
    )

    controller.advance(_trajectory((0.0, 0.0)), 1.0)
    snapshot = controller.snapshot(_state())

    assert snapshot.session_state == "leaderboard"
    assert snapshot.high_score_rank is None
    assert snapshot.leaderboard == ()


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ((10.0, 0.0), 0.0),
        ((0.0, 10.0), math.pi / 2.0),
        ((0.0, -10.0), -math.pi / 2.0),
        ((-10.0, 0.0), -math.pi),
    ],
)
def test_relative_bearing_cardinal_directions(
    target: tuple[float, float], expected: float
) -> None:
    bearing = relative_target_bearing_rad(0.0, 0.0, 0.0, *target)
    assert bearing == pytest.approx(expected)


def test_relative_bearing_wraps_ego_yaw() -> None:
    bearing = relative_target_bearing_rad(0.0, 0.0, math.radians(350.0), 10.0, 0.0)
    assert bearing == pytest.approx(math.radians(10.0))


def test_bev_projection_places_forward_and_left_targets() -> None:
    bev = BevConfig(width=100, height=100, height_m=75.0, fov_deg=60.0, tilt_deg=0.0)
    forward_u, forward_v, forward_visible = project_target_to_bev(
        (10.0, 0.0, 0.0), _state(), bev
    )
    left_u, _, left_visible = project_target_to_bev((0.0, 10.0, 0.0), _state(), bev)

    assert forward_visible is True
    assert forward_u == pytest.approx(0.5)
    assert forward_v < 0.5
    assert left_visible is True
    assert left_u < 0.5


def test_bev_segment_projection_clips_crossing_line_to_viewport() -> None:
    bev = BevConfig(width=100, height=100, height_m=75.0, fov_deg=60.0, tilt_deg=0.0)
    pose = rig_pose_from_vehicle_state(_state())
    segment = np.asarray([[10.0, -100.0, 0.0], [10.0, 100.0, 0.0]], dtype=np.float32)

    projected = project_segment_pose_to_bev(segment, pose, bev)

    assert projected is not None
    assert projected[0][0] == pytest.approx(1.0)
    assert projected[1][0] == pytest.approx(0.0)
    assert all(0.0 <= coordinate <= 1.0 for point in projected for coordinate in point)


def test_camera_marker_is_visible_only_when_world_anchor_is_in_view() -> None:
    calibration = _camera_calibration()
    camera_model = FThetaCameraModel(calibration)
    snapshot = TaxiGameSnapshot(
        phase="seeking_pickup",
        target_xyz_m=(10.0, 0.0, 0.0),
        distance_m=10.0,
        relative_bearing_rad=0.0,
        target_radius_m=2.0,
        remaining_time_s=None,
        score=0,
    )

    visible = project_taxi_marker_to_camera(
        snapshot,
        np.eye(4, dtype=np.float32),
        camera_model,
        image_width=100,
        image_height=80,
    )
    behind = project_taxi_marker_to_camera(
        TaxiGameSnapshot(
            phase="seeking_pickup",
            target_xyz_m=(-10.0, 0.0, 0.0),
            distance_m=10.0,
            relative_bearing_rad=-math.pi,
            target_radius_m=2.0,
            remaining_time_s=None,
            score=0,
        ),
        np.eye(4, dtype=np.float32),
        camera_model,
        image_width=100,
        image_height=80,
    )

    assert visible is not None
    assert visible.anchor_uv == pytest.approx((50.0, 40.0))
    assert visible.ring_edges_uv
    assert behind is None


def test_camera_marker_projection_keeps_only_three_closest_visible_pickups() -> None:
    camera_model = FThetaCameraModel(_camera_calibration())
    snapshot = TaxiGameSnapshot(
        phase="seeking_pickup",
        target_xyz_m=(10.0, 0.0, 0.0),
        distance_m=10.0,
        relative_bearing_rad=0.0,
        target_radius_m=2.0,
        remaining_time_s=None,
        score=0,
        pickup_targets_xyz_m=(
            (40.0, 0.0, 0.0),
            (20.0, 0.0, 0.0),
            (-10.0, 0.0, 0.0),
            (30.0, 0.0, 0.0),
            (10.0, 0.0, 0.0),
        ),
    )

    markers = project_taxi_markers_to_camera(
        snapshot,
        np.eye(4, dtype=np.float32),
        camera_model,
        image_width=100,
        image_height=80,
    )
    advanced_markers = project_taxi_markers_to_camera(
        snapshot,
        rig_pose_from_vehicle_state(_state(15.0, 0.0)),
        camera_model,
        image_width=100,
        image_height=80,
    )

    assert [marker.distance_m for marker in markers] == pytest.approx(
        [10.0, 20.0, 30.0]
    )
    assert [marker.distance_m for marker in advanced_markers] == pytest.approx(
        [5.0, 15.0, 25.0]
    )
