# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU tests for Crazy Robotaxi directed road routing."""

from __future__ import annotations

import math

import numpy as np
import pytest
from crazy_robotaxi.navigation import (
    LanePosition,
    NavigationFareRegion,
    NavigationLane,
    NavigationWaypoint,
    TaxiNavigationMap,
)


def _lane(
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
) -> NavigationLane:
    return NavigationLane(
        np.asarray(
            [[*start_xy, 0.0], [*end_xy, 0.0]],
            dtype=np.float32,
        )
    )


def _position(lane_index: int, distance_m: float = 0.0) -> LanePosition:
    return LanePosition(lane_index, distance_m, 0.0, 0.0)


def test_shortest_route_uses_directed_connectors_and_road_distance() -> None:
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

    route = navigation.route(_position(0), destination)

    assert route is not None
    assert route.lane_indices == (0, 1, 2)
    assert route.distance_m == pytest.approx(20.0 + math.sqrt(200.0))


def test_route_does_not_traverse_lane_against_its_direction() -> None:
    navigation = TaxiNavigationMap((_lane((0.0, 0.0), (10.0, 0.0)),))
    destination = NavigationWaypoint(
        np.asarray([2.0, 0.0, 0.0], dtype=np.float32),
        lane_index=0,
        distance_along_lane_m=2.0,
    )

    assert navigation.route(_position(0, 8.0), destination) is None


def test_parking_target_is_sampled_in_polygon_and_routes_only_to_entrance() -> None:
    navigation = TaxiNavigationMap(
        (
            NavigationLane(
                np.asarray([[0, 0, 0], [10, 0, 0]], dtype=np.float32),
                lane_id="arrival",
                successor_ids=(),
            ),
            NavigationLane(
                np.asarray([[10, 0, 0], [0, 0, 0]], dtype=np.float32),
                lane_id="departure",
                successor_ids=(),
            ),
        )
    )
    region = NavigationFareRegion(
        region_id="lot",
        kind="area",
        geometry_world=(
            np.asarray(
                [[20, 20, 0], [20, 30, 0], [30, 30, 0], [30, 20, 0]],
                dtype=np.float32,
            ),
        ),
        arrival_lane_ids=("arrival",),
        departure_lane_ids=("departure",),
    )

    waypoint = navigation.sample_fare_regions(
        (region,), spacing_m=20, rng=np.random.default_rng(7)
    )[0]
    route = navigation.route(_position(0, 2), waypoint)

    assert 20 <= waypoint.xyz_m[0] <= 30
    assert 20 <= waypoint.xyz_m[1] <= 30
    assert waypoint.departure_anchors[0].lane_index == 1
    assert route is not None
    assert route.distance_m == pytest.approx(8.0)


def test_concave_parking_targets_stay_inside_polygon() -> None:
    navigation = TaxiNavigationMap(
        (
            NavigationLane(
                np.asarray([[0, 0, 0], [10, 0, 0]], dtype=np.float32),
                lane_id="road",
                successor_ids=(),
            ),
        )
    )
    region = NavigationFareRegion(
        region_id="concave_lot",
        kind="area",
        geometry_world=(
            np.asarray(
                [[0, 0, 0], [0, 10, 0], [4, 10, 0], [4, 4, 0], [10, 4, 0], [10, 0, 0]],
                dtype=np.float32,
            ),
        ),
        arrival_lane_ids=("road",),
        departure_lane_ids=("road",),
    )

    waypoints = navigation.sample_fare_regions(
        (region,), spacing_m=2, rng=np.random.default_rng(19)
    )

    assert waypoints
    assert all(point.xyz_m[0] <= 4 or point.xyz_m[1] <= 4 for point in waypoints)


def test_lane_matching_prefers_vehicle_heading_on_overlapping_lanes() -> None:
    navigation = TaxiNavigationMap(
        (
            _lane((0.0, 0.0), (10.0, 0.0)),
            _lane((10.0, 0.0), (0.0, 0.0)),
        )
    )

    forward = navigation.nearest_lane_positions(5.0, 0.0, 0.0)
    reverse = navigation.nearest_lane_positions(5.0, 0.0, math.pi)

    assert forward[0].lane_index == 0
    assert reverse[0].lane_index == 1


def test_roadside_waypoint_overlaps_edge_with_passenger_beyond_road() -> None:
    navigation = TaxiNavigationMap(
        (
            NavigationLane(
                centerline_world=np.asarray(
                    [[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]], dtype=np.float32
                ),
                road_edge_world=np.asarray(
                    [[0.0, -2.0, 0.0], [20.0, -2.0, 0.0]], dtype=np.float32
                ),
            ),
        )
    )

    waypoint = navigation.sample_waypoints(spacing_m=10.0, offset_m=0.0)[0]

    np.testing.assert_allclose(waypoint.xyz_m, [0.0, -1.0, 0.0])
    assert waypoint.passenger_xyz_m is not None
    np.testing.assert_allclose(waypoint.passenger_xyz_m, [0.0, -2.75, 0.0])
    assert -2.0 < float(waypoint.xyz_m[1]) < 0.0


def test_waypoint_sampling_excludes_road_lanes_without_stopping_edges() -> None:
    navigation = TaxiNavigationMap(
        (
            NavigationLane(
                centerline_world=np.asarray(
                    [[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]], dtype=np.float32
                ),
                allows_taxi_stops=False,
            ),
            NavigationLane(
                centerline_world=np.asarray(
                    [[0.0, 10.0, 0.0], [20.0, 10.0, 0.0]], dtype=np.float32
                ),
                road_edge_world=np.asarray(
                    [[0.0, 12.0, 0.0], [20.0, 12.0, 0.0]], dtype=np.float32
                ),
            ),
        )
    )

    waypoints = navigation.sample_waypoints(spacing_m=10.0, offset_m=0.0)

    assert {waypoint.lane_index for waypoint in waypoints} == {1}


def test_recorded_route_fallback_infers_a_right_hand_road_edge() -> None:
    navigation = TaxiNavigationMap.from_polylines(
        (np.asarray([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]], dtype=np.float32),),
        bidirectional=False,
    )

    waypoint = navigation.sample_waypoints(spacing_m=10.0, offset_m=0.0)[0]

    np.testing.assert_allclose(waypoint.xyz_m, [0.0, -1.0, 0.0])
    assert waypoint.passenger_xyz_m is not None
    np.testing.assert_allclose(waypoint.passenger_xyz_m, [0.0, -2.75, 0.0])
