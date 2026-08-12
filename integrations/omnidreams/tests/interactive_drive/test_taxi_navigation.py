# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU tests for Crazy Robotaxi directed road routing."""

from __future__ import annotations

import math

import numpy as np
import pytest
from omnidreams.interactive_drive.crazy_robotaxi.navigation import (
    LanePosition,
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
