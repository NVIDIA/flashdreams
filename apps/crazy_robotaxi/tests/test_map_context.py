# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU coverage for map-anchored runtime prompt context."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from crazy_robotaxi.live_edit.map_context import MapContextTracker
from omnidreams_game_engine.game_map import load_game_map
from omnidreams_game_engine.game_map.types import GameMapLane, ResolvedGameMap
from omnidreams_game_engine.types import VehicleState

pytestmark = pytest.mark.ci_cpu

_MAPS = Path(__file__).parent / "maps"


def _with_prompt_contexts(game_map: ResolvedGameMap) -> ResolvedGameMap:
    return replace(
        game_map,
        topology=replace(
            game_map.topology,
            nodes=tuple(
                replace(node, prompt_context=f"Landmark at {node.node_id}.")
                for node in game_map.topology.nodes
            ),
            roads=tuple(
                replace(road, prompt_context=f"Setting along {road.road_id}.")
                for road in game_map.topology.roads
            ),
        ),
    )


def _lane(game_map: ResolvedGameMap, lane_id: str) -> GameMapLane:
    return next(lane for lane in game_map.lanes if lane.lane_id == lane_id)


def _sample_polyline(points: np.ndarray, distance_m: float) -> tuple[np.ndarray, float]:
    lengths = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    remaining = min(max(0.0, distance_m), float(lengths.sum()))
    for index, length in enumerate(lengths):
        if remaining <= length or index == len(lengths) - 1:
            alpha = 0.0 if length <= 1.0e-6 else remaining / float(length)
            point = points[index] + alpha * (points[index + 1] - points[index])
            tangent = points[index + 1, :2] - points[index, :2]
            return point, math.atan2(float(tangent[1]), float(tangent[0]))
        remaining -= float(length)
    raise AssertionError("polyline must contain a nonempty segment")


def _state_on_lane(
    lane: GameMapLane,
    distance_m: float,
    *,
    speed_mps: float = 10.0,
    velocity: bool = False,
) -> VehicleState:
    point, travel_yaw = _sample_polyline(lane.centerline_world, distance_m)
    vehicle_yaw = travel_yaw if speed_mps >= 0.0 else travel_yaw - math.pi
    magnitude = abs(speed_mps)
    return VehicleState(
        x_m=float(point[0]),
        y_m=float(point[1]),
        z_m=float(point[2]),
        yaw_rad=vehicle_yaw,
        speed_mps=speed_mps,
        steer_rad=0.0,
        velocity_x_mps=(magnitude * math.cos(travel_yaw) if velocity else None),
        velocity_y_mps=(magnitude * math.sin(travel_yaw) if velocity else None),
    )


def _lane_length(lane: GameMapLane) -> float:
    return float(
        np.linalg.norm(np.diff(lane.centerline_world[:, :2], axis=0), axis=1).sum()
    )


def test_intersection_approach_current_and_outgoing_context_handoff() -> None:
    game_map = _with_prompt_contexts(
        load_game_map(_MAPS / "intersection_geometry.robotaxi.yaml")
    )
    tracker = MapContextTracker(game_map)
    incoming = _lane(game_map, "west_road:lane:2")
    outgoing = _lane(game_map, "east_road:lane:2")
    connector = _lane(game_map, "center:connector:0")

    far = tracker.update(_state_on_lane(incoming, _lane_length(incoming) - 40.0))
    approaching = tracker.update(
        _state_on_lane(incoming, _lane_length(incoming) - 20.0)
    )
    current = tracker.update(_state_on_lane(connector, 0.5 * _lane_length(connector)))
    first_outgoing = tracker.update(_state_on_lane(outgoing, 8.0))
    second_outgoing = tracker.update(_state_on_lane(outgoing, 16.0))

    assert "approaching an intersection" not in far.suffix
    assert "Setting along west_road." in approaching.suffix
    assert "approaching an intersection" in approaching.suffix
    assert "Landmark at center." in approaching.suffix
    assert "traveling through an intersection" in current.suffix
    assert first_outgoing.road_id == "west_road"
    assert "intersection" not in first_outgoing.suffix
    assert second_outgoing.road_id == "east_road"
    assert "Setting along east_road." in second_outgoing.suffix


def test_stopping_shortens_lookahead_and_motion_uses_hysteresis() -> None:
    game_map = load_game_map(_MAPS / "intersection_geometry.robotaxi.yaml")
    tracker = MapContextTracker(game_map)
    incoming = _lane(game_map, "west_road:lane:2")
    near = _lane_length(incoming) - 20.0

    assert tracker.update(_state_on_lane(incoming, near, speed_mps=8.0)).motion == (
        "forward"
    )
    stopped = tracker.update(_state_on_lane(incoming, near, speed_mps=0.0))
    assert stopped.motion == "stationary"
    assert "approaching an intersection" not in stopped.suffix
    assert tracker.update(_state_on_lane(incoming, near, speed_mps=0.5)).motion == (
        "stationary"
    )
    assert tracker.update(_state_on_lane(incoming, near, speed_mps=0.8)).motion == (
        "forward"
    )


def test_reversing_away_recomputes_destination_immediately() -> None:
    game_map = load_game_map(_MAPS / "intersection_geometry.robotaxi.yaml")
    tracker = MapContextTracker(game_map)
    toward_center = _lane(game_map, "west_road:lane:2")
    toward_west_end = _lane(game_map, "west_road:lane:1")

    approaching = tracker.update(
        _state_on_lane(toward_center, _lane_length(toward_center) - 15.0)
    )
    reversing = tracker.update(_state_on_lane(toward_west_end, 15.0, speed_mps=-5.0))

    assert "approaching an intersection" in approaching.suffix
    assert "approaching an intersection" not in reversing.suffix
    assert reversing.motion == "reverse"
    assert (
        "The taxi is reversing; scenery moves forward relative to the camera."
        in reversing.suffix
    )


def test_velocity_selects_actual_travel_direction_over_vehicle_yaw() -> None:
    game_map = load_game_map(_MAPS / "intersection_geometry.robotaxi.yaml")
    tracker = MapContextTracker(game_map)
    lane = _lane(game_map, "west_road:lane:2")
    state = _state_on_lane(lane, _lane_length(lane) - 15.0, velocity=True)
    state.yaw_rad += math.pi

    result = tracker.update(state)

    assert result.road_id == "west_road"
    assert "approaching an intersection" in result.suffix


def test_two_off_map_chunks_clear_scene_context_but_keep_motion() -> None:
    game_map = _with_prompt_contexts(
        load_game_map(_MAPS / "intersection_geometry.robotaxi.yaml")
    )
    tracker = MapContextTracker(game_map)
    lane = _lane(game_map, "west_road:lane:2")
    on_map = tracker.update(_state_on_lane(lane, 30.0))
    off_map = VehicleState(1e6, 1e6, 0.0, 0.0, 4.0, 0.0)

    first = tracker.update(off_map)
    second = tracker.update(off_map)

    assert first == on_map
    assert second.road_id is None
    assert second.node_id is None
    assert second.suffix == "The taxi is driving forward."


def test_culdesac_driveway_and_parking_lot_phrases() -> None:
    game_map = _with_prompt_contexts(
        load_game_map(_MAPS / "parking_driveway.robotaxi.yaml")
    )

    driveway_tracker = MapContextTracker(game_map)
    toward_driveway = _lane(game_map, "west_road:lane:1")
    driveway_lane = _lane(game_map, "lot_driveway:lane:0")
    approaching_driveway = driveway_tracker.update(
        _state_on_lane(toward_driveway, _lane_length(toward_driveway) - 10.0)
    )
    in_driveway = driveway_tracker.update(
        _state_on_lane(driveway_lane, 0.5 * _lane_length(driveway_lane))
    )

    culdesac_tracker = MapContextTracker(game_map)
    toward_culdesac = _lane(game_map, "west_road:lane:0")
    approaching_end = culdesac_tracker.update(
        _state_on_lane(toward_culdesac, _lane_length(toward_culdesac) - 10.0)
    )
    parking = driveway_tracker.update(
        VehicleState(0.0, -40.0, 0.0, -math.pi / 2.0, 3.0, 0.0)
    )

    assert "parking lot entrance is ahead" in approaching_driveway.suffix
    assert "Landmark at lot_driveway." in approaching_driveway.suffix
    assert "passing through a parking lot entrance" in in_driveway.suffix
    assert "road ends ahead in a cul-de-sac" in approaching_end.suffix
    assert "driving through a parking lot" in parking.suffix
    assert "Landmark at parking_lot." in parking.suffix
    assert parking.road_id is None
    assert "Setting along west_road." not in parking.suffix


def test_unique_road_joint_successor_contributes_curve_context() -> None:
    game_map = load_game_map(_MAPS / "traffic_loop.robotaxi.yaml")
    tracker = MapContextTracker(game_map)
    lane = _lane(game_map, "south_east:lane:1")

    result = tracker.update(
        _state_on_lane(lane, _lane_length(lane) - 5.0, speed_mps=10.0)
    )

    assert "road curves left ahead" in result.suffix
    assert "intersection" not in result.suffix
    assert (
        tracker._curve_direction(lane, _lane_length(lane), 15.0, stop_at_branch=True)
        == "left"
    )


@pytest.mark.parametrize(
    ("points", "direction"),
    [
        ([(0, 0), (10, 0), (20, 10), (20, 20)], "left"),
        ([(0, 0), (10, 0), (20, -10), (20, -20)], "right"),
    ],
)
def test_sampled_curved_roads_report_signed_curve_direction(
    points: list[tuple[float, float]], direction: str
) -> None:
    game_map = load_game_map(_MAPS / "traffic_loop.robotaxi.yaml")
    tracker = MapContextTracker(game_map)
    template = game_map.lanes[0]
    centerline = np.asarray([(x, y, 0.0) for x, y in points], dtype=np.float32)
    lane = replace(template, centerline_world=centerline, successor_ids=())

    assert tracker._curve_direction(lane, 0.0, 50.0, stop_at_branch=True) == direction
