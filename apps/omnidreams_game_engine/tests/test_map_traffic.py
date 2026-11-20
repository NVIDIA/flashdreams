# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU regressions for model-thread map traffic control."""

from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pytest
from ludus_renderer import BodyState
from omnidreams_game_engine.config import VehicleConfig
from omnidreams_game_engine.game_map.types import GameMapTrafficVehicle
from omnidreams_game_engine.game_map.vicinity import GameMapVicinity
from omnidreams_game_engine.simulation.map_traffic import (
    MapTrafficController,
    MapTrafficPhase,
)

pytestmark = pytest.mark.ci_cpu


def _traffic_definition(
    centerline: np.ndarray,
    *,
    vehicle_id: str = "traffic",
    start_distance_m: float = 1.0,
) -> GameMapTrafficVehicle:
    return GameMapTrafficVehicle(
        vehicle_id=vehicle_id,
        node_ids=("a", "b"),
        end_behavior="wrap",
        vehicle_type="car",
        dimensions_lwh_m=(4.5, 1.8, 1.5),
        speed_mps=None,
        start_distance_m=start_distance_m,
        centerline_world=centerline,
        speed_limits_mps=np.full(len(centerline), 10.0, dtype=np.float32),
        route_element_ids=("road",) * (len(centerline) - 1),
    )


def _body_at(
    position_xy: tuple[float, float],
    *,
    yaw_rad: float = 0.0,
    linear_velocity_xy: tuple[float, float] = (0.0, 0.0),
) -> BodyState:
    return BodyState(
        position_m=np.asarray([*position_xy, 0.75], dtype=np.float32),
        orientation_xyzw=np.asarray(
            [0.0, 0.0, math.sin(yaw_rad * 0.5), math.cos(yaw_rad * 0.5)],
            dtype=np.float32,
        ),
        linear_velocity_mps=np.asarray([*linear_velocity_xy, 0.0], dtype=np.float32),
        angular_velocity_radps=np.zeros(3, dtype=np.float32),
    )


def _activate(controller: MapTrafficController) -> None:
    controller.set_vicinity(
        GameMapVicinity("road", frozenset({"road"}), frozenset({"road"}))
    )


def test_active_traffic_walks_route_cursor_without_global_search() -> None:
    centerline = np.asarray(
        [[x, 0, 0] for x in range(21)] + [[20, 20, 0], [0, 20, 0], [0, 0, 0]],
        dtype=np.float32,
    )
    controller = MapTrafficController(
        (_traffic_definition(centerline, vehicle_id="cursor"),), VehicleConfig()
    )
    _activate(controller)
    state = controller.state("map-traffic:cursor")
    assert state is not None
    controller.observe_physics(
        state.object_id,
        struck=False,
        body=_body_at((12.4, 0.0), linear_velocity_xy=(10.0, 0.0)),
        dt_s=1.0 / 30.0,
    )
    with patch.object(
        controller,
        "_nearest_route_projection",
        side_effect=AssertionError("normal traversal used a global route search"),
    ):
        targets = controller.prepare_step(_body_at((-100.0, -100.0)), 1.0 / 30.0)

    assert state.route_segment_index == 12
    assert state.timestamp_us == pytest.approx(1_240_000, abs=1.0)
    assert targets[0].timestamp_us == pytest.approx(1_590_000, abs=1.0)


@pytest.mark.parametrize("active", [False, True])
def test_batched_route_samples_match_tracks_at_boundaries(active: bool) -> None:
    routes = (
        np.asarray([[0, 0, 0], [19, 7, 3], [0, 0, 0]], dtype=np.float32),
        np.asarray(
            [[100, 0, 0], [100, 23, 0], [117, 23, 0], [100, 0, 0]], dtype=np.float32
        ),
    )
    controller = MapTrafficController(
        tuple(
            _traffic_definition(route, vehicle_id=str(i))
            for i, route in enumerate(routes)
        ),
        VehicleConfig(),
    )
    if active:
        _activate(controller)
    for boundary in (0, 1, -1):
        for offset in (-1, 0, 1):
            for state in controller._states:
                state.timestamp_us = float(
                    state.scene_object.timestamps_us[boundary] + offset
                )
                state.position_m = np.asarray([77, -33, 2], dtype=np.float32)
            observations = controller._observations()
            for state in controller._states:
                position, _, velocity = state.scene_object.sample(
                    int(state.timestamp_us)
                )
                observation = observations[state.object_id]
                np.testing.assert_array_equal(
                    observation.position_xy,
                    state.position_m[:2] if active else position[:2],
                )
                np.testing.assert_array_equal(observation.velocity_xy, velocity[:2])
                if not active:
                    assert state.route_segment_index == controller._route_segment_index(
                        state.scene_object, state.timestamp_us
                    )


def test_offscreen_following_wrap_and_activation_keep_logical_progress() -> None:
    route = np.asarray([[0, 0, 0], [100, 0, 0], [0, 0, 0]], dtype=np.float32)
    controller = MapTrafficController(
        (
            _traffic_definition(route, vehicle_id="follower", start_distance_m=10),
            _traffic_definition(route, vehicle_id="leader", start_distance_m=20),
        ),
        VehicleConfig(),
    )
    follower, leader = controller._states
    assert controller.prepare_step(_body_at((-1000, -1000)), 1 / 30) == ()
    assert follower.velocity_scale == 0
    assert leader.velocity_scale == 1
    stopped_at = follower.timestamp_us
    controller.prepare_step(_body_at((-1000, -1000)), 1 / 30)
    assert follower.timestamp_us == stopped_at
    _activate(controller)
    expected, _, _ = follower.scene_object.sample(int(stopped_at))
    np.testing.assert_array_equal(follower.position_m, expected)
    controller.set_vicinity(None)
    follower.timestamp_us = follower.duration_us - 10
    follower.velocity_scale = 1
    controller.prepare_step(_body_at((-1000, -1000)), 0.1)
    assert follower.timestamp_us == 99_990
    assert follower.route_segment_index == 0
    assert (
        MapTrafficController((), VehicleConfig()).prepare_step(_body_at((0, 0)), 0.1)
        == ()
    )


def test_collision_recovery_globally_reacquires_route_and_cursor() -> None:
    centerline = np.asarray(
        [[0, 0, 0], [20, 0, 0], [20, 20, 0], [0, 20, 0], [0, 0, 0]],
        dtype=np.float32,
    )
    controller = MapTrafficController(
        (
            _traffic_definition(
                centerline,
                vehicle_id="recovering",
                start_distance_m=2.0,
            ),
        ),
        VehicleConfig(),
    )
    _activate(controller)
    state = controller.state("map-traffic:recovering")
    assert state is not None
    stopped = _body_at((18.0, 8.0), yaw_rad=math.pi)

    controller.observe_physics(state.object_id, struck=True, body=stopped, dt_s=0.25)
    for _ in range(4):
        controller.observe_physics(
            state.object_id, struck=False, body=stopped, dt_s=0.25
        )

    projected_position, _, _ = state.scene_object.sample(int(state.timestamp_us))
    assert state.phase is MapTrafficPhase.RECOVERING
    assert state.route_segment_index == 1
    np.testing.assert_allclose(projected_position[:2], [20.0, 8.0], atol=1.0e-4)
