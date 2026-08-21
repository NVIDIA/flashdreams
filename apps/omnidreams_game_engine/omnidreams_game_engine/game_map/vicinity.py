# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Resolve graph-local actor visibility around a map-space vehicle pose."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from omnidreams_game_engine.game_map.types import ResolvedGameMap

_BOUNDARY_EPSILON_M = 1.0e-4


@dataclass(frozen=True)
class GameMapVicinity:
    """Semantic location and actor-visible element sets for one ego pose."""

    location_element_id: str
    traffic_element_ids: frozenset[str]
    pedestrian_element_ids: frozenset[str]


def _point_segment_distance(
    point: np.ndarray, start: np.ndarray, end: np.ndarray
) -> float:
    vector = end - start
    length_sq = float(np.dot(vector, vector))
    alpha = (
        0.0
        if length_sq <= 1.0e-12
        else float(np.dot(point - start, vector)) / length_sq
    )
    closest = start + np.clip(alpha, 0.0, 1.0) * vector
    return float(np.linalg.norm(point - closest))


def _polygon_contains(point: np.ndarray, polygon: np.ndarray) -> bool:
    vertices = np.asarray(polygon[:, :2], dtype=np.float64)
    if len(vertices) > 1 and np.allclose(vertices[0], vertices[-1]):
        vertices = vertices[:-1]
    inside = False
    previous = vertices[-1]
    for current in vertices:
        if _point_segment_distance(point, previous, current) <= _BOUNDARY_EPSILON_M:
            return True
        if (current[1] > point[1]) != (previous[1] > point[1]):
            crossing_x = (previous[0] - current[0]) * (point[1] - current[1]) / (
                previous[1] - current[1]
            ) + current[0]
            if point[0] < crossing_x:
                inside = not inside
        previous = current
    return inside


class GameMapVicinityResolver:
    """Resolve the current road/node neighborhood from compiled map geometry."""

    def __init__(self, game_map: ResolvedGameMap) -> None:
        self._nodes = {node.node_id: node for node in game_map.topology.nodes}
        self._roads = {road.road_id: road for road in game_map.topology.roads}
        self._road_lanes = {
            road_id: tuple(
                lane for lane in game_map.lanes if lane.element_id == road_id
            )
            for road_id in self._roads
        }
        self._incident_roads: dict[str, set[str]] = {
            node_id: set() for node_id in self._nodes
        }
        for road in self._roads.values():
            self._incident_roads[road.from_node_id].add(road.road_id)
            self._incident_roads[road.to_node_id].add(road.road_id)
        self._parking_lots_by_access_node: dict[str, set[str]] = {}
        self._access_source_by_id: dict[str, str] = {}
        for access in game_map.topology.parking_accesses:
            self._parking_lots_by_access_node.setdefault(
                access.source_node_id, set()
            ).add(access.parking_lot_node_id)
            self._access_source_by_id[access.access_id] = access.source_node_id
        elements = {element.element_id: element for element in game_map.elements}
        self._node_polygons = tuple(
            (node_id, elements[node_id].surface_world)
            for node_id in sorted(self._nodes)
            if node_id in elements
        )
        self._road_polygons = tuple(
            (road_id, elements[road_id].surface_world)
            for road_id in sorted(self._roads)
            if road_id in elements
        )
        self._access_polygons = tuple(
            (access_id, elements[access_id].surface_world)
            for access_id in sorted(self._access_source_by_id)
            if access_id in elements
        )

    def _location_element(self, point_xy: np.ndarray) -> str | None:
        for element_id, polygon in self._node_polygons:
            if _polygon_contains(point_xy, polygon):
                return element_id
        for element_id, polygon in self._road_polygons:
            if _polygon_contains(point_xy, polygon):
                return element_id
        for access_id, polygon in self._access_polygons:
            if _polygon_contains(point_xy, polygon):
                return self._access_source_by_id[access_id]
        return None

    def _next_node(self, road_id: str, point_xy: np.ndarray, yaw_rad: float) -> str:
        road = self._roads[road_id]
        forward = np.asarray([math.cos(yaw_rad), math.sin(yaw_rad)], dtype=np.float64)
        best: tuple[float, str] | None = None
        for lane in self._road_lanes[road_id]:
            points = np.asarray(lane.centerline_world[:, :2], dtype=np.float64)
            vectors = np.diff(points, axis=0)
            lengths_sq = np.sum(vectors * vectors, axis=1)
            relative = point_xy[None, :] - points[:-1]
            alpha = np.divide(
                np.sum(relative * vectors, axis=1),
                lengths_sq,
                out=np.zeros_like(lengths_sq),
                where=lengths_sq > 1.0e-12,
            )
            closest = points[:-1] + np.clip(alpha, 0.0, 1.0)[:, None] * vectors
            segment_index = int(np.argmin(np.linalg.norm(closest - point_xy, axis=1)))
            tangent = vectors[segment_index]
            tangent /= max(float(np.linalg.norm(tangent)), 1.0e-12)
            heading_error = math.acos(
                float(np.clip(np.dot(tangent, forward), -1.0, 1.0))
            )
            lateral = float(np.linalg.norm(closest[segment_index] - point_xy))
            end_xy = points[-1]
            from_node = self._nodes[road.from_node_id]
            to_node = self._nodes[road.to_node_id]
            from_xy = np.asarray([from_node.x_m, from_node.y_m])
            to_xy = np.asarray([to_node.x_m, to_node.y_m])
            next_node = (
                road.from_node_id
                if np.linalg.norm(end_xy - from_xy) < np.linalg.norm(end_xy - to_xy)
                else road.to_node_id
            )
            candidate = (lateral + heading_error * 20.0, next_node)
            if best is None or candidate < best:
                best = candidate
        if best is not None:
            return best[1]
        candidates = []
        for node_id in (road.from_node_id, road.to_node_id):
            node = self._nodes[node_id]
            delta = np.asarray([node.x_m, node.y_m]) - point_xy
            delta /= max(float(np.linalg.norm(delta)), 1.0e-12)
            candidates.append((float(np.dot(delta, forward)), node_id))
        return max(candidates)[1]

    def resolve(
        self,
        x_m: float,
        y_m: float,
        yaw_rad: float,
        *,
        previous: GameMapVicinity | None = None,
    ) -> GameMapVicinity | None:
        """Return the graph neighborhood, preserving ``previous`` while off-road."""
        point_xy = np.asarray([x_m, y_m], dtype=np.float64)
        location = self._location_element(point_xy)
        if location is None:
            return previous
        if location in self._roads:
            node_id = self._next_node(location, point_xy, yaw_rad)
            traffic = {location, node_id, *self._incident_roads[node_id]}
        else:
            traffic = {location, *self._incident_roads.get(location, ())}
        pedestrians = set(traffic)
        for node_id in traffic:
            pedestrians.update(self._parking_lots_by_access_node.get(node_id, ()))
        return GameMapVicinity(location, frozenset(traffic), frozenset(pedestrians))


__all__ = ["GameMapVicinity", "GameMapVicinityResolver"]
