# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Resolve graph-local actor visibility around a map-space vehicle pose."""

from __future__ import annotations

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


def _polygon_contains(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Return whether a point is inside or on a pre-normalized polygon."""
    starts = polygon
    ends = np.roll(polygon, -1, axis=0)
    vectors = ends - starts
    lengths_sq = np.einsum("ij,ij->i", vectors, vectors)
    relative = point[None, :] - starts
    alpha = np.divide(
        np.einsum("ij,ij->i", relative, vectors),
        lengths_sq,
        out=np.zeros_like(lengths_sq),
        where=lengths_sq > 1.0e-12,
    )
    closest = starts + np.clip(alpha, 0.0, 1.0)[:, None] * vectors
    offsets = point[None, :] - closest
    if np.any(np.einsum("ij,ij->i", offsets, offsets) <= _BOUNDARY_EPSILON_M**2):
        return True

    crosses_y = (starts[:, 1] > point[1]) != (ends[:, 1] > point[1])
    if not np.any(crosses_y):
        return False
    crossing_starts = starts[crosses_y]
    crossing_vectors = vectors[crosses_y]
    crossing_x = crossing_starts[:, 0] + (
        (point[1] - crossing_starts[:, 1])
        * crossing_vectors[:, 0]
        / crossing_vectors[:, 1]
    )
    return bool(np.count_nonzero(point[0] < crossing_x) % 2)


@dataclass(frozen=True)
class _PolygonLookup:
    """A small vectorized bounding-box index over semantic polygons."""

    element_ids: tuple[str, ...]
    polygons: tuple[np.ndarray, ...]
    minimums_xy: np.ndarray
    maximums_xy: np.ndarray

    @classmethod
    def build(cls, entries: tuple[tuple[str, np.ndarray], ...]) -> _PolygonLookup:
        element_ids: list[str] = []
        polygons: list[np.ndarray] = []
        for element_id, polygon in entries:
            vertices = np.asarray(polygon[:, :2], dtype=np.float64)
            if len(vertices) > 1 and np.allclose(vertices[0], vertices[-1]):
                vertices = vertices[:-1]
            element_ids.append(element_id)
            polygons.append(vertices)
        if not polygons:
            empty = np.empty((0, 2), dtype=np.float64)
            return cls((), (), empty, empty.copy())
        return cls(
            tuple(element_ids),
            tuple(polygons),
            np.asarray([polygon.min(axis=0) for polygon in polygons]),
            np.asarray([polygon.max(axis=0) for polygon in polygons]),
        )

    def containing_element(self, point_xy: np.ndarray) -> str | None:
        """Return the first indexed polygon containing ``point_xy``."""
        within_bounds = np.all(
            (point_xy >= self.minimums_xy - _BOUNDARY_EPSILON_M)
            & (point_xy <= self.maximums_xy + _BOUNDARY_EPSILON_M),
            axis=1,
        )
        for index in np.flatnonzero(within_bounds):
            if _polygon_contains(point_xy, self.polygons[int(index)]):
                return self.element_ids[int(index)]
        return None


class GameMapVicinityResolver:
    """Resolve the current road/node neighborhood from compiled map geometry."""

    def __init__(self, game_map: ResolvedGameMap) -> None:
        self._nodes = {node.node_id: node for node in game_map.topology.nodes}
        self._roads = {road.road_id: road for road in game_map.topology.roads}
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
        self._node_polygons = _PolygonLookup.build(
            tuple(
                (node_id, elements[node_id].surface_world)
                for node_id in sorted(self._nodes)
                if node_id in elements
            )
        )
        self._road_polygons = _PolygonLookup.build(
            tuple(
                (road_id, elements[road_id].surface_world)
                for road_id in sorted(self._roads)
                if road_id in elements
            )
        )
        self._access_polygons = _PolygonLookup.build(
            tuple(
                (
                    self._access_source_by_id[access_id],
                    elements[access_id].surface_world,
                )
                for access_id in sorted(self._access_source_by_id)
                if access_id in elements
            )
        )

    def _location_element(self, point_xy: np.ndarray) -> str | None:
        node_id = self._node_polygons.containing_element(point_xy)
        if node_id is not None:
            return node_id
        road_id = self._road_polygons.containing_element(point_xy)
        if road_id is not None:
            return road_id
        return self._access_polygons.containing_element(point_xy)

    def _expanded_elements(self, location: str) -> set[str]:
        """Return nodes within one public-road hop and all their incident roads."""
        if location in self._roads:
            road = self._roads[location]
            seed_nodes = {road.from_node_id, road.to_node_id}
        else:
            seed_nodes = {location}

        first_roads = {
            road_id
            for node_id in seed_nodes
            for road_id in self._incident_roads.get(node_id, ())
        }
        expanded_nodes = set(seed_nodes)
        for road_id in first_roads:
            road = self._roads[road_id]
            expanded_nodes.update((road.from_node_id, road.to_node_id))
        expanded_roads = {
            road_id
            for node_id in expanded_nodes
            for road_id in self._incident_roads.get(node_id, ())
        }
        return expanded_nodes | expanded_roads

    def resolve(
        self,
        x_m: float,
        y_m: float,
        *,
        previous: GameMapVicinity | None = None,
    ) -> GameMapVicinity | None:
        """Return the graph neighborhood, preserving ``previous`` while off-road."""
        point_xy = np.asarray([x_m, y_m], dtype=np.float64)
        location = self._location_element(point_xy)
        if location is None:
            return previous
        traffic = self._expanded_elements(location)
        pedestrians = set(traffic)
        for node_id in traffic:
            pedestrians.update(self._parking_lots_by_access_node.get(node_id, ()))
        return GameMapVicinity(location, frozenset(traffic), frozenset(pedestrians))


__all__ = ["GameMapVicinity", "GameMapVicinityResolver"]
