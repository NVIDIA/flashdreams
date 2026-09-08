# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Map topology and vehicle-motion clauses for live prompt direction."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from omnidreams_game_engine.game_map.types import (
    GameMapLane,
    GameMapNode,
    ResolvedGameMap,
)
from omnidreams_game_engine.game_map.vicinity import GameMapVicinityResolver
from omnidreams_game_engine.types import VehicleState

from crazy_robotaxi.navigation import NavigationLane, TaxiNavigationMap

_CURVE_THRESHOLD_RAD = math.radians(15.0)

_APPROACH_PHRASES = {
    "intersection": "The vehicle is approaching an intersection.",
    "cul_de_sac": "The road ends ahead in a cul-de-sac.",
    "driveway": "A parking lot entrance is ahead.",
}
_CURRENT_PHRASES = {
    "intersection": "The vehicle is traveling through an intersection.",
    "cul_de_sac": "The vehicle is turning within a cul-de-sac.",
    "driveway": "The vehicle is passing through a parking lot entrance.",
    "parking_lot": "The vehicle is driving through a parking lot.",
}
_MOTION_PHRASES = {
    "forward": "The taxi is driving forward.",
    "stationary": "The taxi is stationary.",
    "reverse": "The taxi is reversing; scenery moves forward relative to the camera.",
}


def _angle_delta(first: float, second: float) -> float:
    return (second - first + math.pi) % (2.0 * math.pi) - math.pi


def _polyline_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1).sum())


def _point_at_distance(points: np.ndarray, distance_m: float) -> tuple[np.ndarray, int]:
    lengths = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    remaining = max(0.0, float(distance_m))
    for index, length in enumerate(lengths):
        if remaining <= length or index == len(lengths) - 1:
            alpha = 0.0 if length <= 1.0e-6 else min(1.0, remaining / float(length))
            return points[index] + alpha * (points[index + 1] - points[index]), index
        remaining -= float(length)
    return points[-1], len(points) - 2


def compose_map_suffix(
    *,
    road_context: str | None,
    node_phrase: str | None,
    node_context: str | None,
    curve: str | None,
    motion: str,
) -> str:
    """Compose one normalized map/motion suffix in stable order."""
    curve_phrase = None if curve is None else f"The road curves {curve} ahead."
    return " ".join(
        part.strip()
        for part in (
            road_context,
            node_phrase,
            node_context,
            curve_phrase,
            _MOTION_PHRASES[motion],
        )
        if part is not None and part.strip()
    )


@dataclass(frozen=True, slots=True)
class MapPromptState:
    """Semantic prompt state selected for one authoritative chunk."""

    suffix: str
    road_id: str | None
    node_id: str | None
    motion: str


class MapContextTracker:
    """Classify map topology with chunk-level stability and hysteresis."""

    def __init__(self, game_map: ResolvedGameMap) -> None:
        self._game_map = game_map
        self._nodes = {node.node_id: node for node in game_map.topology.nodes}
        self._roads = {road.road_id: road for road in game_map.topology.roads}
        self._lanes = {lane.lane_id: lane for lane in game_map.lanes}
        self._navigation = TaxiNavigationMap(
            tuple(
                NavigationLane(
                    centerline_world=lane.centerline_world,
                    lane_id=lane.lane_id,
                    successor_ids=lane.successor_ids,
                    element_id=lane.element_id,
                )
                for lane in game_map.lanes
            )
        )
        self._vicinity = GameMapVicinityResolver(game_map)
        self._access_source = {
            access.access_id: access.source_node_id
            for access in game_map.topology.parking_accesses
        }
        self._motion = "stationary"
        self._travel_yaw = float(game_map.default_spawn.yaw_rad)
        self._retained_road_id: str | None = None
        self._outgoing_candidate: str | None = None
        self._outgoing_chunks = 0
        self._off_map_chunks = 0
        self._last_state = MapPromptState(
            _MOTION_PHRASES["stationary"], None, None, "stationary"
        )

    def reset(self) -> None:
        """Reset rollout-local hysteresis while retaining immutable indexes."""
        self._motion = "stationary"
        self._travel_yaw = float(self._game_map.default_spawn.yaw_rad)
        self._retained_road_id = None
        self._outgoing_candidate = None
        self._outgoing_chunks = 0
        self._off_map_chunks = 0
        self._last_state = MapPromptState(
            _MOTION_PHRASES["stationary"], None, None, "stationary"
        )

    def update(self, state: VehicleState) -> MapPromptState:
        """Classify the final authoritative vehicle state of one chunk."""
        previous_motion = self._motion
        self._motion = self._next_motion(float(state.speed_mps))
        self._travel_yaw = self._resolve_travel_yaw(state)
        vicinity = self._vicinity.resolve(state.x_m, state.y_m, previous=None)
        lane_match = self._navigation.nearest_lane_positions(
            state.x_m, state.y_m, self._travel_yaw, limit=1
        )
        if vicinity is None or not lane_match:
            self._off_map_chunks += 1
            if self._off_map_chunks < 2:
                return self._last_state
            result = MapPromptState(
                _MOTION_PHRASES[self._motion], None, None, self._motion
            )
            self._last_state = result
            return result
        self._off_map_chunks = 0

        match = lane_match[0]
        navigation_lane = self._navigation.lanes[match.lane_index]
        lane = self._lanes[str(navigation_lane.lane_id)]
        location_id = vicinity.location_element_id
        location_node = self._nodes.get(location_id)
        matched_road_id = self._road_id_for_element(lane.element_id)

        if previous_motion != "reverse" and self._motion == "reverse":
            self._retained_road_id = matched_road_id
            self._outgoing_candidate = None
            self._outgoing_chunks = 0
        self._update_retained_road(matched_road_id, location_node is not None)
        road_id = self._retained_road_id or matched_road_id
        if lane.element_id in self._access_source or (
            location_node is not None and location_node.node_type == "parking_lot"
        ):
            road_id = None
        road_context = None if road_id is None else self._roads[road_id].prompt_context

        lookahead_m = min(50.0, max(15.0, abs(float(state.speed_mps)) * 3.0))
        next_node, remaining_m = self._next_node(lane, match.distance_along_lane_m)
        node = location_node
        node_phrase: str | None = None
        if node is not None:
            node_phrase = _CURRENT_PHRASES.get(node.node_type)
        elif next_node is not None and remaining_m <= lookahead_m:
            node = next_node
            node_phrase = _APPROACH_PHRASES.get(node.node_type)

        curve = self._curve_direction(
            lane,
            match.distance_along_lane_m,
            lookahead_m,
            stop_at_branch=True,
        )
        result = MapPromptState(
            suffix=compose_map_suffix(
                road_context=road_context,
                node_phrase=node_phrase,
                node_context=None if node is None else node.prompt_context,
                curve=curve,
                motion=self._motion,
            ),
            road_id=road_id,
            node_id=None if node is None else node.node_id,
            motion=self._motion,
        )
        self._last_state = result
        return result

    def _next_motion(self, speed_mps: float) -> str:
        if self._motion == "reverse":
            if speed_mps <= -0.2:
                return "reverse"
            return "stationary" if abs(speed_mps) < 0.35 else "forward"
        if self._motion == "stationary":
            if abs(speed_mps) <= 0.75:
                return "stationary"
            return "reverse" if speed_mps < 0.0 else "forward"
        if speed_mps < -0.5:
            return "reverse"
        if abs(speed_mps) < 0.35:
            return "stationary"
        return "forward"

    def _resolve_travel_yaw(self, state: VehicleState) -> float:
        if state.velocity_x_mps is not None and state.velocity_y_mps is not None:
            velocity = math.hypot(state.velocity_x_mps, state.velocity_y_mps)
            if velocity > 0.35:
                return math.atan2(state.velocity_y_mps, state.velocity_x_mps)
        if self._motion == "reverse":
            return state.yaw_rad + math.pi
        if self._motion == "forward":
            return state.yaw_rad
        return self._travel_yaw

    def _road_id_for_element(self, element_id: str) -> str | None:
        if element_id in self._roads:
            return element_id
        source = self._access_source.get(element_id)
        if source is None:
            return None
        incident = [
            road.road_id
            for road in self._roads.values()
            if source in {road.from_node_id, road.to_node_id}
        ]
        return incident[0] if len(incident) == 1 else None

    def _update_retained_road(self, road_id: str | None, on_node: bool) -> None:
        if road_id is None or on_node:
            return
        if self._retained_road_id is None:
            self._retained_road_id = road_id
            return
        if road_id == self._retained_road_id:
            self._outgoing_candidate = None
            self._outgoing_chunks = 0
            return
        if road_id != self._outgoing_candidate:
            self._outgoing_candidate = road_id
            self._outgoing_chunks = 1
            return
        self._outgoing_chunks += 1
        if self._outgoing_chunks >= 2:
            self._retained_road_id = road_id
            self._outgoing_candidate = None
            self._outgoing_chunks = 0

    def _next_node(
        self, lane: GameMapLane, distance_m: float
    ) -> tuple[GameMapNode | None, float]:
        road_id = self._road_id_for_element(lane.element_id)
        if road_id is None:
            return None, math.inf
        road = self._roads[road_id]
        end_xy = lane.centerline_world[-1, :2]
        from_node = self._nodes[road.from_node_id]
        to_node = self._nodes[road.to_node_id]
        from_distance = float(
            np.linalg.norm(end_xy - np.asarray([from_node.x_m, from_node.y_m]))
        )
        to_distance = float(
            np.linalg.norm(end_xy - np.asarray([to_node.x_m, to_node.y_m]))
        )
        return (
            from_node if from_distance <= to_distance else to_node,
            max(0.0, _polyline_length(lane.centerline_world) - distance_m),
        )

    def _curve_direction(
        self,
        lane: GameMapLane,
        distance_m: float,
        lookahead_m: float,
        *,
        stop_at_branch: bool,
    ) -> str | None:
        start, segment_index = _point_at_distance(lane.centerline_world, distance_m)
        points = [start, *lane.centerline_world[segment_index + 1 :]]
        remaining = lookahead_m - _polyline_length(np.asarray(points))
        current = lane
        visited = {lane.lane_id}
        while remaining > 0.0 and current.successor_ids:
            candidates = [
                self._lanes[item]
                for item in current.successor_ids
                if item in self._lanes and item not in visited
            ]
            if not candidates or (stop_at_branch and len(candidates) > 1):
                break
            heading_points = (
                points[-2:] if len(points) >= 2 else current.centerline_world[-2:]
            )
            previous_heading = math.atan2(
                heading_points[-1][1] - heading_points[-2][1],
                heading_points[-1][0] - heading_points[-2][0],
            )
            current = min(
                candidates,
                key=lambda item: abs(
                    _angle_delta(
                        previous_heading,
                        math.atan2(
                            item.centerline_world[1, 1] - item.centerline_world[0, 1],
                            item.centerline_world[1, 0] - item.centerline_world[0, 0],
                        ),
                    )
                ),
            )
            visited.add(current.lane_id)
            points.extend(current.centerline_world[1:])
            remaining -= _polyline_length(current.centerline_world)
        path = np.asarray(points)
        if len(path) < 3:
            return None
        trimmed = [path[0]]
        remaining = lookahead_m
        for first, second in zip(path[:-1], path[1:], strict=True):
            length = float(np.linalg.norm(second[:2] - first[:2]))
            if length <= 1.0e-6:
                continue
            if length >= remaining:
                trimmed.append(first + (remaining / length) * (second - first))
                break
            trimmed.append(second)
            remaining -= length
        headings = [
            math.atan2(second[1] - first[1], second[0] - first[0])
            for first, second in zip(trimmed[:-1], trimmed[1:], strict=True)
        ]
        turn = sum(
            _angle_delta(first, second)
            for first, second in zip(headings[:-1], headings[1:], strict=True)
        )
        if abs(turn) < _CURVE_THRESHOLD_RAD:
            return None
        return "left" if turn > 0.0 else "right"


__all__ = ["MapContextTracker", "MapPromptState", "compose_map_suffix"]
