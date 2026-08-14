# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Directed road routing for Crazy Robotaxi."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

_MIN_SEGMENT_LENGTH_M = 1.0e-4

_ROAD_MARKER_EDGE_INSET_M = 1.0
"""Maximum distance from a mapped road edge to a Taxi marker center."""

_PASSENGER_EDGE_OFFSET_M = 0.75
"""Distance a waiting passenger stands beyond a mapped road edge."""

_INFERRED_LANE_HALF_WIDTH_M = 2.0
"""Half-width used when a legacy scene provides only a recorded route."""


@dataclass(frozen=True)
class NavigationLane:
    """Directed lane centerline."""

    centerline_world: npt.NDArray[np.float32]
    """Directed lane-center polyline in world coordinates."""

    road_edge_world: npt.NDArray[np.float32] | None = None
    """Curb or outer road-edge polyline suitable for a roadside stop."""

    allows_taxi_stops: bool = True
    """Whether pickup and dropoff candidates may be sampled from this lane."""


@dataclass(frozen=True)
class NavigationWaypoint:
    """Sampled target position tied to a directed lane."""

    xyz_m: npt.NDArray[np.float32]
    """World-space waypoint position."""

    lane_index: int
    """Index of the source lane in the navigation map."""

    distance_along_lane_m: float
    """Arc distance from the source lane's directed start."""

    passenger_xyz_m: npt.NDArray[np.float32] | None = None
    """Waiting-passenger ground point, or ``None`` to use ``xyz_m``."""


@dataclass(frozen=True)
class LanePosition:
    """Closest directed-lane location for a vehicle pose."""

    lane_index: int
    """Index of the matched navigation lane."""

    distance_along_lane_m: float
    """Arc distance from the lane's directed start."""

    lateral_distance_m: float
    """XY distance between the vehicle and the matched centerline."""

    heading_error_rad: float
    """Absolute difference between vehicle and lane headings."""


@dataclass(frozen=True)
class RoutePlan:
    """Shortest legal lane path to one destination waypoint."""

    lane_indices: tuple[int, ...]
    """Directed lanes traversed from the current position to the target."""

    distance_m: float
    """Total routed road distance to the destination."""


class TaxiNavigationMap:
    """Directed lane graph for one Taxi scene."""

    def __init__(
        self,
        lanes: tuple[NavigationLane, ...],
        *,
        endpoint_snap_tolerance_m: float = 1.0,
    ) -> None:
        """Build routing indexes for a scene.

        Args:
            lanes: Directed car-lane centerlines.
            endpoint_snap_tolerance_m: Maximum endpoint gap connected by the graph.

        Raises:
            ValueError: No lane contains usable travel distance or the endpoint
                tolerance is not positive.
        """
        if endpoint_snap_tolerance_m <= 0.0:
            raise ValueError("Taxi endpoint snap tolerance must be positive.")

        normalized_lanes: list[NavigationLane] = []
        cumulative_distances: list[npt.NDArray[np.float32]] = []
        road_edge_cumulative_distances: list[npt.NDArray[np.float32] | None] = []
        for lane in lanes:
            points = _normalize_polyline(lane.centerline_world)
            if points is None:
                continue
            segment_lengths = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
            cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths))).astype(
                np.float32
            )
            road_edge = (
                None
                if lane.road_edge_world is None
                else _normalize_polyline(lane.road_edge_world)
            )
            normalized_lanes.append(
                NavigationLane(points, road_edge, lane.allows_taxi_stops)
            )
            cumulative_distances.append(cumulative)
            road_edge_cumulative_distances.append(
                None if road_edge is None else _cumulative_distances(road_edge)
            )
        if not normalized_lanes:
            raise ValueError("Taxi navigation geometry has no usable travel distance.")

        self._lanes = tuple(normalized_lanes)
        self._cumulative_distances = tuple(cumulative_distances)
        self._road_edge_cumulative_distances = tuple(road_edge_cumulative_distances)
        self._lane_lengths = np.asarray(
            [float(cumulative[-1]) for cumulative in cumulative_distances],
            dtype=np.float64,
        )
        self._adjacency = self._build_adjacency(endpoint_snap_tolerance_m)
        self._build_segment_index()

    @classmethod
    def from_polylines(
        cls,
        routes_world: tuple[npt.NDArray[np.float32], ...],
        *,
        bidirectional: bool,
    ) -> TaxiNavigationMap:
        """Build a navigation map from route polylines.

        Args:
            routes_world: Route polylines in world coordinates.
            bidirectional: Whether to add a reversed lane for every route.

        Returns:
            Navigation map containing the supplied route directions.
        """
        lanes: list[NavigationLane] = []
        for route in routes_world:
            route_array = np.asarray(route, dtype=np.float32)
            lanes.append(
                NavigationLane(
                    route_array,
                    _infer_right_road_edge(route_array),
                )
            )
            if bidirectional:
                reversed_route = route_array[::-1].copy()
                lanes.append(
                    NavigationLane(
                        reversed_route,
                        _infer_right_road_edge(reversed_route),
                    )
                )
        return cls(tuple(lanes))

    @property
    def lanes(self) -> tuple[NavigationLane, ...]:
        """Return the normalized directed lanes."""
        return self._lanes

    def sample_waypoints(
        self, spacing_m: float, offset_m: float
    ) -> tuple[NavigationWaypoint, ...]:
        """Sample spatially distinct target candidates across the lane graph.

        Args:
            spacing_m: Arc distance between samples on each lane.
            offset_m: Shared sampling offset in ``[0, spacing_m)``.

        Returns:
            Deduplicated waypoint candidates with source-lane locations.

        Raises:
            ValueError: ``spacing_m`` is not positive or fewer than two distinct
                waypoints can be produced.
        """
        if spacing_m <= 0.0:
            raise ValueError("Taxi waypoint spacing must be positive.")
        sampled: list[NavigationWaypoint] = []
        occupied_cells: set[tuple[int, int]] = set()
        for lane_index, lane_length in enumerate(self._lane_lengths):
            if not self._lanes[lane_index].allows_taxi_stops:
                continue
            sample_distances = np.arange(
                offset_m, float(lane_length) + 1.0e-6, spacing_m
            )
            if len(sample_distances) < 2:
                sample_distances = np.asarray([0.0, lane_length], dtype=np.float32)
            for distance_m in sample_distances:
                point, passenger_point = self._taxi_stop_points_at(
                    lane_index, float(distance_m)
                )
                cell = (
                    int(round(float(point[0]) * 2.0)),
                    int(round(float(point[1]) * 2.0)),
                )
                if cell in occupied_cells:
                    continue
                occupied_cells.add(cell)
                sampled.append(
                    NavigationWaypoint(
                        point,
                        lane_index,
                        float(distance_m),
                        passenger_point,
                    )
                )
        if len(sampled) < 2:
            raise ValueError("Taxi mode requires at least two distinct road waypoints.")
        return tuple(sampled)

    def point_at(
        self, lane_index: int, distance_along_lane_m: float
    ) -> npt.NDArray[np.float32]:
        """Interpolate a world point along a directed lane."""
        lane = self._lanes[lane_index].centerline_world
        cumulative = self._cumulative_distances[lane_index]
        distance_m = float(np.clip(distance_along_lane_m, 0.0, float(cumulative[-1])))
        return _point_at_distance(lane, cumulative, distance_m)

    def _taxi_stop_points_at(
        self, lane_index: int, distance_along_lane_m: float
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        center = self.point_at(lane_index, distance_along_lane_m)
        lane = self._lanes[lane_index]
        edge_cumulative = self._road_edge_cumulative_distances[lane_index]
        if lane.road_edge_world is None or edge_cumulative is None:
            return center, center.copy()

        lane_fraction = float(
            np.clip(
                distance_along_lane_m / self._lane_lengths[lane_index],
                0.0,
                1.0,
            )
        )
        edge = _point_at_distance(
            lane.road_edge_world,
            edge_cumulative,
            lane_fraction * float(edge_cumulative[-1]),
        )
        inward_xy = center[:2] - edge[:2]
        half_width_m = float(np.linalg.norm(inward_xy))
        if half_width_m <= _MIN_SEGMENT_LENGTH_M:
            return center, center.copy()

        inward_unit_xy = inward_xy / half_width_m
        marker_inset_m = min(_ROAD_MARKER_EDGE_INSET_M, 0.5 * half_width_m)
        marker = edge.copy()
        marker[:2] += marker_inset_m * inward_unit_xy
        passenger = edge.copy()
        passenger[:2] -= _PASSENGER_EDGE_OFFSET_M * inward_unit_xy
        return marker.astype(np.float32), passenger.astype(np.float32)

    def nearest_lane_positions(
        self,
        x_m: float,
        y_m: float,
        yaw_rad: float,
        *,
        limit: int = 8,
    ) -> tuple[LanePosition, ...]:
        """Return nearby lane matches ordered by distance and heading agreement."""
        if limit <= 0:
            return ()
        query = np.asarray([x_m, y_m], dtype=np.float32)
        relative = query[None, :] - self._segment_starts_xy
        parameter = np.clip(
            np.sum(relative * self._segment_vectors_xy, axis=1)
            / self._segment_lengths_sq,
            0.0,
            1.0,
        )
        closest = (
            self._segment_starts_xy + parameter[:, None] * self._segment_vectors_xy
        )
        distances = np.linalg.norm(closest - query[None, :], axis=1)
        heading_errors = np.abs(
            _normalize_angles(self._segment_headings_rad - float(yaw_rad))
        )
        scores = distances + np.where(heading_errors <= math.pi * 0.55, 0.0, 20.0)
        candidate_count = min(len(scores), max(limit * 12, limit))
        candidate_segments = np.argpartition(scores, candidate_count - 1)[
            :candidate_count
        ]
        candidate_segments = candidate_segments[
            np.argsort(scores[candidate_segments], kind="stable")
        ]

        matches: list[LanePosition] = []
        matched_lanes: set[int] = set()
        for segment_index in candidate_segments:
            lane_index = int(self._segment_lane_indices[segment_index])
            if lane_index in matched_lanes:
                continue
            matched_lanes.add(lane_index)
            matches.append(
                LanePosition(
                    lane_index=lane_index,
                    distance_along_lane_m=float(
                        self._segment_start_distances_m[segment_index]
                        + parameter[segment_index]
                        * math.sqrt(float(self._segment_lengths_sq[segment_index]))
                    ),
                    lateral_distance_m=float(distances[segment_index]),
                    heading_error_rad=float(heading_errors[segment_index]),
                )
            )
            if len(matches) >= limit:
                break
        return tuple(matches)

    def route(
        self, start: LanePosition, destination: NavigationWaypoint
    ) -> RoutePlan | None:
        """Return the shortest directed route between two lane positions."""
        distances_to_start, predecessors = self._shortest_tree(start)
        direct_distance = math.inf
        if (
            destination.lane_index == start.lane_index
            and destination.distance_along_lane_m >= start.distance_along_lane_m
        ):
            direct_distance = (
                destination.distance_along_lane_m - start.distance_along_lane_m
            )
        graph_distance = (
            float(distances_to_start[destination.lane_index])
            + destination.distance_along_lane_m
        )
        if math.isfinite(direct_distance) and direct_distance <= graph_distance:
            lane_path = (start.lane_index,)
            distance_m = direct_distance
        elif math.isfinite(graph_distance):
            lane_path = self._reconstruct_path(
                start.lane_index, destination.lane_index, predecessors
            )
            if not lane_path:
                return None
            distance_m = graph_distance
        else:
            return None
        return RoutePlan(
            lane_indices=lane_path,
            distance_m=max(0.0, float(distance_m)),
        )

    def route_distances(
        self,
        start: LanePosition,
        destinations: tuple[NavigationWaypoint, ...],
    ) -> tuple[float, ...]:
        """Return shortest directed distances to candidate waypoints."""
        distances_to_start, _predecessors = self._shortest_tree(start)
        result: list[float] = []
        for destination in destinations:
            direct_distance = math.inf
            if (
                destination.lane_index == start.lane_index
                and destination.distance_along_lane_m >= start.distance_along_lane_m
            ):
                direct_distance = (
                    destination.distance_along_lane_m - start.distance_along_lane_m
                )
            graph_distance = (
                float(distances_to_start[destination.lane_index])
                + destination.distance_along_lane_m
            )
            result.append(min(direct_distance, graph_distance))
        return tuple(result)

    def _build_adjacency(
        self, endpoint_snap_tolerance_m: float
    ) -> tuple[tuple[tuple[int, float], ...], ...]:
        cell_size = endpoint_snap_tolerance_m
        start_buckets: dict[tuple[int, int], list[int]] = {}
        for lane_index, lane in enumerate(self._lanes):
            start = lane.centerline_world[0, :2]
            cell = (
                math.floor(float(start[0]) / cell_size),
                math.floor(float(start[1]) / cell_size),
            )
            start_buckets.setdefault(cell, []).append(lane_index)

        adjacency: list[tuple[tuple[int, float], ...]] = []
        for lane_index, lane in enumerate(self._lanes):
            end = lane.centerline_world[-1, :2]
            end_cell = (
                math.floor(float(end[0]) / cell_size),
                math.floor(float(end[1]) / cell_size),
            )
            connected: list[tuple[int, float]] = []
            for offset_x in (-1, 0, 1):
                for offset_y in (-1, 0, 1):
                    for successor in start_buckets.get(
                        (end_cell[0] + offset_x, end_cell[1] + offset_y), ()
                    ):
                        if successor == lane_index:
                            continue
                        gap = float(
                            np.linalg.norm(
                                end - self._lanes[successor].centerline_world[0, :2]
                            )
                        )
                        if gap <= endpoint_snap_tolerance_m:
                            connected.append((successor, gap))
            adjacency.append(tuple(sorted(set(connected))))
        return tuple(adjacency)

    def _build_segment_index(self) -> None:
        starts: list[npt.NDArray[np.float32]] = []
        vectors: list[npt.NDArray[np.float32]] = []
        lane_indices: list[int] = []
        start_distances: list[float] = []
        for lane_index, lane in enumerate(self._lanes):
            points = lane.centerline_world
            starts.extend(points[:-1, :2])
            vectors.extend(np.diff(points[:, :2], axis=0))
            lane_indices.extend([lane_index] * (len(points) - 1))
            start_distances.extend(self._cumulative_distances[lane_index][:-1])
        self._segment_starts_xy = np.asarray(starts, dtype=np.float32)
        self._segment_vectors_xy = np.asarray(vectors, dtype=np.float32)
        self._segment_lengths_sq = np.sum(
            self._segment_vectors_xy * self._segment_vectors_xy, axis=1
        )
        self._segment_lane_indices = np.asarray(lane_indices, dtype=np.int32)
        self._segment_start_distances_m = np.asarray(start_distances, dtype=np.float32)
        self._segment_headings_rad = np.arctan2(
            self._segment_vectors_xy[:, 1], self._segment_vectors_xy[:, 0]
        )

    def _shortest_tree(
        self, start: LanePosition
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
        lane_count = len(self._lanes)
        distances = np.full(lane_count, math.inf, dtype=np.float64)
        predecessors = np.full(lane_count, -1, dtype=np.int32)
        queue: list[tuple[float, int]] = []
        source_lane = start.lane_index
        remaining_source_distance = max(
            0.0, self._lane_lengths[source_lane] - start.distance_along_lane_m
        )
        for successor, gap in self._adjacency[source_lane]:
            distance = remaining_source_distance + gap
            if distance < distances[successor]:
                distances[successor] = distance
                predecessors[successor] = source_lane
                heapq.heappush(queue, (distance, successor))

        while queue:
            distance, lane_index = heapq.heappop(queue)
            if distance > distances[lane_index] + 1.0e-9:
                continue
            exit_distance = distance + self._lane_lengths[lane_index]
            for successor, gap in self._adjacency[lane_index]:
                candidate = exit_distance + gap
                if candidate + 1.0e-9 >= distances[successor]:
                    continue
                distances[successor] = candidate
                predecessors[successor] = lane_index
                heapq.heappush(queue, (candidate, successor))
        return distances, predecessors

    def _reconstruct_path(
        self,
        source_lane: int,
        destination_lane: int,
        predecessors: npt.NDArray[np.int32],
    ) -> tuple[int, ...]:
        if predecessors[destination_lane] < 0:
            return ()
        reversed_path = [destination_lane]
        current = destination_lane
        for _ in range(len(self._lanes) + 1):
            predecessor = int(predecessors[current])
            if predecessor < 0:
                return ()
            reversed_path.append(predecessor)
            if predecessor == source_lane:
                return tuple(reversed(reversed_path))
            current = predecessor
        return ()


def _point_at_distance(
    points: npt.NDArray[np.float32],
    cumulative: npt.NDArray[np.float32],
    distance_m: float,
) -> npt.NDArray[np.float32]:
    distance_m = float(np.clip(distance_m, 0.0, float(cumulative[-1])))
    right = int(np.searchsorted(cumulative, distance_m, side="right"))
    right = min(max(1, right), len(points) - 1)
    left = right - 1
    span = float(cumulative[right] - cumulative[left])
    alpha = 0.0 if span <= 1.0e-6 else (distance_m - cumulative[left]) / span
    return ((1.0 - alpha) * points[left] + alpha * points[right]).astype(np.float32)


def _cumulative_distances(
    points: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    segment_lengths = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(segment_lengths))).astype(np.float32)


def _infer_right_road_edge(
    centerline_world: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32] | None:
    centerline = _normalize_polyline(centerline_world)
    if centerline is None:
        return None
    tangent_xy = np.empty((len(centerline), 2), dtype=np.float32)
    tangent_xy[0] = centerline[1, :2] - centerline[0, :2]
    tangent_xy[-1] = centerline[-1, :2] - centerline[-2, :2]
    if len(centerline) > 2:
        tangent_xy[1:-1] = centerline[2:, :2] - centerline[:-2, :2]
    tangent_lengths = np.linalg.norm(tangent_xy, axis=1)
    if np.any(tangent_lengths <= _MIN_SEGMENT_LENGTH_M):
        return None
    right_normal_xy = (
        np.stack((tangent_xy[:, 1], -tangent_xy[:, 0]), axis=1)
        / tangent_lengths[:, None]
    )
    road_edge = centerline.copy()
    road_edge[:, :2] += _INFERRED_LANE_HALF_WIDTH_M * right_normal_xy
    return road_edge.astype(np.float32)


def _normalize_polyline(
    points_world: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32] | None:
    points = np.asarray(points_world, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        return None
    if not np.isfinite(points).all():
        return None
    segment_lengths = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    keep = np.concatenate(([True], segment_lengths > _MIN_SEGMENT_LENGTH_M))
    points = points[keep]
    if len(points) < 2:
        return None
    return points


def _normalize_angles(angles_rad: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return (angles_rad + math.pi) % (2.0 * math.pi) - math.pi
