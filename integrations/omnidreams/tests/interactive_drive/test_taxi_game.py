# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU tests for interactive-drive taxi-game state and projection."""

import math
from pathlib import Path

import numpy as np
import pytest
from omnidreams.interactive_drive.camera import FThetaCameraModel
from omnidreams.interactive_drive.config import BevConfig
from omnidreams.interactive_drive.high_scores import HighScoreStore
from omnidreams.interactive_drive.taxi_game import (
    TaxiGameConfig,
    TaxiGameController,
    TaxiGameSnapshot,
    project_target_to_bev,
    project_taxi_marker_to_camera,
    relative_target_bearing_rad,
)
from omnidreams.interactive_drive.types import (
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


def _trajectory(*positions_xy: tuple[float, float]) -> TrajectoryChunk:
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], len(positions_xy), axis=0)
    for pose, (x_m, y_m) in zip(poses, positions_xy, strict=True):
        pose[0, 3] = x_m
        pose[1, 3] = y_m
    boundary_x, boundary_y = positions_xy[-1]
    return TrajectoryChunk(
        timestamps_us=np.arange(len(positions_xy), dtype=np.int64),
        rig_poses_world=poses,
        boundary_state_after_chunk=_state(boundary_x, boundary_y),
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


def test_taxi_mode_rejects_route_without_travel_distance() -> None:
    route = np.zeros((2, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="no usable travel distance"):
        TaxiGameController(
            scene_id="scene",
            reference_route_world=route,
            initial_state=_state(),
            config=TaxiGameConfig(enabled=True),
        )


def test_navigation_routes_move_dropoffs_to_other_streets() -> None:
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

    assert pickup.target_xyz_m[1] == 0.0
    assert dropoff.phase == "to_dropoff"
    assert dropoff.target_xyz_m[1] == 100.0


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
    assert snapshots[0].awarded_global_time_s == 30.0
    assert snapshots[0].global_remaining_time_s == pytest.approx(90.0 - 1.0 / 30.0)
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
    assert expired.global_remaining_time_s == pytest.approx(54.0 - 1.0 / 30.0)


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
    assert snapshot.global_remaining_time_s == pytest.approx(60.0)
    assert snapshot.session_state == "playing"


def test_pickup_adds_configured_time_to_global_timer() -> None:
    controller = _controller(
        TaxiGameConfig(
            enabled=True,
            waypoint_spacing_m=1000.0,
            global_time_s=10.0,
            pickup_time_bonus_s=12.0,
        )
    )

    controller.advance(_trajectory((100.0, 0.0)), 0.0)
    snapshot = controller.snapshot(_state(100.0, 0.0))

    assert snapshot.global_remaining_time_s == 22.0
    assert snapshot.event == "pickup_complete"
    assert snapshot.awarded_global_time_s == 12.0


def test_global_timer_ends_game_and_accepts_qualifying_name(tmp_path: Path) -> None:
    store = HighScoreStore(tmp_path / "scores.csv")
    controller = _controller(
        TaxiGameConfig(
            enabled=True,
            waypoint_spacing_m=1000.0,
            global_time_s=1.0,
            pickup_time_bonus_s=0.0,
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


def test_camera_marker_is_visible_only_when_world_anchor_is_in_view() -> None:
    calibration = CameraCalibration(
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
