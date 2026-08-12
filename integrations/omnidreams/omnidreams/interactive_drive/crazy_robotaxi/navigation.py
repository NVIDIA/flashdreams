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

"""Directed road routing and turn instructions for Crazy Robotaxi."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

TurnManeuver = Literal["left", "right", "straight", "u_turn"]

_MANEUVER_BY_LANE_LABEL: dict[str, TurnManeuver] = {
    "LEFT_TURN": "left",
    "BRANCH_LEFT": "left",
    "RIGHT_TURN": "right",
    "BRANCH_RIGHT": "right",
    "STRAIGHT_TURN": "straight",
    "BRANCH_STRAIGHT": "straight",
    "U_TURN": "u_turn",
}

_MIN_SEGMENT_LENGTH_M = 1.0e-4
_FLOATING_SIGN_HEIGHT_M = 3.0
_TURN_THRESHOLD_RAD = math.radians(35.0)
_U_TURN_THRESHOLD_RAD = math.radians(145.0)
_INTERSECTION_HEADING_SAMPLE_M = 8.0
_ROUTE_SAMPLE_SPACING_M = 1.0


@dataclass(frozen=True)
class NavigationLane:
    """Directed lane centerline and its mapped maneuver label."""

    centerline_world: npt.NDArray[np.float32]
    """Directed lane-center polyline in world coordinates."""

    maneuver_label: str = "STRAIGHT"
    """ClipGT lane-direction label associated with the centerline."""


@dataclass(frozen=True)
class NavigationWaypoint:
    """Sampled target position tied to a directed lane."""

    xyz_m: npt.NDArray[np.float32]
    """World-space waypoint position."""

    lane_index: int
    """Index of the source lane in the navigation map."""

    distance_along_lane_m: float
    """Arc distance from the source lane's directed start."""


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
class TaxiTurnInstruction:
    """Floating turn arrow anchored to one routed intersection."""

    maneuver: TurnManeuver
    """Direction displayed by the sign."""

    anchor_xyz_m: tuple[float, float, float]
    """Elevated world-space center of the floating arrow."""

    route_distance_m: float
    """Remaining routed distance from the current lane position."""

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable instruction."""
        return {
            "maneuver": self.maneuver,
            "anchor_xyz_m": list(self.anchor_xyz_m),
            "route_distance_m": self.route_distance_m,
        }


@dataclass(frozen=True)
class RoutePlan:
    """Shortest legal lane path to one destination waypoint."""

    lane_indices: tuple[int, ...]
    """Directed lanes traversed from the current position to the target."""

    distance_m: float
    """Total routed road distance to the destination."""

    turn_instructions: tuple[TaxiTurnInstruction, ...]
    """Remaining intersection instructions in travel order."""


class TaxiNavigationMap:
    """Directed lane graph and intersection geometry for one Taxi scene."""

    def __init__(
        self,
        lanes: tuple[NavigationLane, ...],
        intersection_polygons_world: tuple[npt.NDArray[np.float32], ...] = (),
        *,
        endpoint_snap_tolerance_m: float = 1.0,
    ) -> None:
        """Build routing indexes for a scene.

        Args:
            lanes: Directed car-lane centerlines.
            intersection_polygons_world: World-space intersection footprints.
            endpoint_snap_tolerance_m: Maximum endpoint gap connected by the graph.

        Raises:
            ValueError: No lane contains usable travel distance or the endpoint
                tolerance is not positive.
        """
        if endpoint_snap_tolerance_m <= 0.0:
            raise ValueError("Taxi endpoint snap tolerance must be positive.")

        normalized_lanes: list[NavigationLane] = []
        cumulative_distances: list[npt.NDArray[np.float32]] = []
        for lane in lanes:
            points = _normalize_polyline(lane.centerline_world)
            if points is None:
                continue
            segment_lengths = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
            cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths))).astype(
                np.float32
            )
            normalized_lanes.append(
                NavigationLane(points, str(lane.maneuver_label).upper())
            )
            cumulative_distances.append(cumulative)
        if not normalized_lanes:
            raise ValueError("Taxi navigation geometry has no usable travel distance.")

        self._lanes = tuple(normalized_lanes)
        self._cumulative_distances = tuple(cumulative_distances)
        self._lane_lengths = np.asarray(
            [float(cumulative[-1]) for cumulative in cumulative_distances],
            dtype=np.float64,
        )
        self._intersection_polygons = tuple(
            polygon
            for raw_polygon in intersection_polygons_world
            if (polygon := _normalize_polygon(raw_polygon)) is not None
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
        """Build a navigation map from unlabeled route polylines.

        Args:
            routes_world: Route polylines in world coordinates.
            bidirectional: Whether to add a reversed lane for every route.

        Returns:
            Navigation map with straight maneuver labels.
        """
        lanes: list[NavigationLane] = []
        for route in routes_world:
            route_array = np.asarray(route, dtype=np.float32)
            lanes.append(NavigationLane(route_array))
            if bidirectional:
                lanes.append(NavigationLane(route_array[::-1].copy()))
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
            sample_distances = np.arange(
                offset_m, float(lane_length) + 1.0e-6, spacing_m
            )
            if len(sample_distances) < 2:
                sample_distances = np.asarray([0.0, lane_length], dtype=np.float32)
            for distance_m in sample_distances:
                point = self.point_at(lane_index, float(distance_m))
                cell = (
                    int(round(float(point[0]) * 2.0)),
                    int(round(float(point[1]) * 2.0)),
                )
                if cell in occupied_cells:
                    continue
                occupied_cells.add(cell)
                sampled.append(NavigationWaypoint(point, lane_index, float(distance_m)))
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
        right = int(np.searchsorted(cumulative, distance_m, side="right"))
        right = min(max(1, right), len(lane) - 1)
        left = right - 1
        span = float(cumulative[right] - cumulative[left])
        alpha = 0.0 if span <= 1.0e-6 else (distance_m - cumulative[left]) / span
        return ((1.0 - alpha) * lane[left] + alpha * lane[right]).astype(np.float32)

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
            turn_instructions=self._turn_instructions(start, lane_path, destination),
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

    def _turn_instructions(
        self,
        start: LanePosition,
        lane_path: tuple[int, ...],
        destination: NavigationWaypoint,
    ) -> tuple[TaxiTurnInstruction, ...]:
        if self._intersection_polygons:
            return self._polygon_turn_instructions(start, lane_path, destination)

        # Unlabelled fallback routes have no mapped intersections. Preserve lane
        # maneuver labels when a caller supplies them without polygon metadata.
        instructions: list[TaxiTurnInstruction] = []
        occupied_cells: set[tuple[int, int]] = set()
        distance_to_lane_start = -start.distance_along_lane_m
        for path_index, lane_index in enumerate(lane_path):
            lane = self._lanes[lane_index]
            maneuver = _MANEUVER_BY_LANE_LABEL.get(lane.maneuver_label)
            midpoint_distance = self._lane_lengths[lane_index] * 0.5
            route_distance = distance_to_lane_start + midpoint_distance
            if maneuver is not None and route_distance >= 0.0:
                midpoint = self.point_at(lane_index, midpoint_distance)
                spatial_cell = (
                    int(round(float(midpoint[0]) / 15.0)),
                    int(round(float(midpoint[1]) / 15.0)),
                )
                if spatial_cell not in occupied_cells:
                    occupied_cells.add(spatial_cell)
                    instructions.append(
                        TaxiTurnInstruction(
                            maneuver=maneuver,
                            anchor_xyz_m=(
                                float(midpoint[0]),
                                float(midpoint[1]),
                                float(midpoint[2]) + _FLOATING_SIGN_HEIGHT_M,
                            ),
                            route_distance_m=float(route_distance),
                        )
                    )
            if path_index + 1 < len(lane_path):
                next_lane = lane_path[path_index + 1]
                distance_to_lane_start += self._lane_lengths[lane_index]
                distance_to_lane_start += self._edge_gap(lane_index, next_lane)
        return tuple(instructions)

    def _polygon_turn_instructions(
        self,
        start: LanePosition,
        lane_path: tuple[int, ...],
        destination: NavigationWaypoint,
    ) -> tuple[TaxiTurnInstruction, ...]:
        """Derive turns where the routed centerline crosses mapped intersections."""
        if (
            len(lane_path) == 1
            and abs(destination.distance_along_lane_m - start.distance_along_lane_m)
            <= _MIN_SEGMENT_LENGTH_M
        ):
            return ()
        route_points = self._route_polyline(start, lane_path, destination)
        segment_lengths = np.linalg.norm(np.diff(route_points[:, :2], axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths))).astype(
            np.float32
        )
        route_length_m = float(cumulative[-1])
        instructions: list[TaxiTurnInstruction] = []
        for polygon in self._intersection_polygons:
            interval = _polyline_polygon_interval(route_points, cumulative, polygon)
            if interval is None:
                continue
            entry_distance_m, exit_distance_m = interval
            if entry_distance_m <= 1.0 or exit_distance_m >= route_length_m - 1.0:
                continue
            before = _point_at_polyline(
                route_points,
                cumulative,
                max(0.0, entry_distance_m - _INTERSECTION_HEADING_SAMPLE_M),
            )
            entry = _point_at_polyline(route_points, cumulative, entry_distance_m)
            exit = _point_at_polyline(route_points, cumulative, exit_distance_m)
            after = _point_at_polyline(
                route_points,
                cumulative,
                min(
                    route_length_m,
                    exit_distance_m + _INTERSECTION_HEADING_SAMPLE_M,
                ),
            )
            heading_delta = _heading_delta(before, entry, exit, after)
            maneuver = _maneuver_from_heading_delta(heading_delta)
            anchor_distance_m = 0.5 * (entry_distance_m + exit_distance_m)
            anchor = _point_at_polyline(route_points, cumulative, anchor_distance_m)
            instructions.append(
                TaxiTurnInstruction(
                    maneuver=maneuver,
                    anchor_xyz_m=(
                        float(anchor[0]),
                        float(anchor[1]),
                        float(anchor[2]) + _FLOATING_SIGN_HEIGHT_M,
                    ),
                    route_distance_m=anchor_distance_m,
                )
            )
        instructions.sort(key=lambda instruction: instruction.route_distance_m)
        return tuple(instructions)

    def _route_polyline(
        self,
        start: LanePosition,
        lane_path: tuple[int, ...],
        destination: NavigationWaypoint,
    ) -> npt.NDArray[np.float32]:
        """Return the routed centerline clipped to the start and destination."""
        pieces: list[npt.NDArray[np.float32]] = []
        last_path_index = len(lane_path) - 1
        for path_index, lane_index in enumerate(lane_path):
            start_distance_m = start.distance_along_lane_m if path_index == 0 else 0.0
            end_distance_m = (
                destination.distance_along_lane_m
                if path_index == last_path_index
                else float(self._lane_lengths[lane_index])
            )
            pieces.append(
                self._lane_slice(lane_index, start_distance_m, end_distance_m)
            )
        route_points = _normalize_polyline(np.concatenate(pieces))
        assert route_points is not None
        return _densify_polyline(route_points, _ROUTE_SAMPLE_SPACING_M)

    def _lane_slice(
        self,
        lane_index: int,
        start_distance_m: float,
        end_distance_m: float,
    ) -> npt.NDArray[np.float32]:
        """Return one directed lane segment bounded by arc distances."""
        cumulative = self._cumulative_distances[lane_index]
        start_distance_m = float(np.clip(start_distance_m, 0.0, cumulative[-1]))
        end_distance_m = float(np.clip(end_distance_m, 0.0, cumulative[-1]))
        interior = self._lanes[lane_index].centerline_world[
            (cumulative > start_distance_m) & (cumulative < end_distance_m)
        ]
        return np.concatenate(
            (
                self.point_at(lane_index, start_distance_m)[None, :],
                interior,
                self.point_at(lane_index, end_distance_m)[None, :],
            ),
            axis=0,
        )

    def _edge_gap(self, lane_index: int, successor: int) -> float:
        for connected_lane, gap in self._adjacency[lane_index]:
            if connected_lane == successor:
                return gap
        return 0.0


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


def _normalize_polygon(
    points_world: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32] | None:
    points = np.asarray(points_world, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        return None
    if not np.isfinite(points).all():
        return None
    return points


def _normalize_angles(angles_rad: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return (angles_rad + math.pi) % (2.0 * math.pi) - math.pi


def _polyline_polygon_interval(
    points_world: npt.NDArray[np.float32],
    cumulative_distances_m: npt.NDArray[np.float32],
    polygon_world: npt.NDArray[np.float32],
) -> tuple[float, float] | None:
    polygon_xy = polygon_world[:, :2]
    points_xy = points_world[:, :2]
    minimum = np.min(polygon_xy, axis=0)
    maximum = np.max(polygon_xy, axis=0)
    candidates = np.flatnonzero(
        (points_xy[:, 0] >= minimum[0])
        & (points_xy[:, 0] <= maximum[0])
        & (points_xy[:, 1] >= minimum[1])
        & (points_xy[:, 1] <= maximum[1])
    )
    if len(candidates) == 0:
        return None
    inside = candidates[_points_in_polygon(points_xy[candidates], polygon_xy)]
    if len(inside) == 0:
        return None
    first = max(0, int(inside[0]) - 1)
    last = min(len(points_world) - 1, int(inside[-1]) + 1)
    return (
        float(cumulative_distances_m[first]),
        float(cumulative_distances_m[last]),
    )


def _points_in_polygon(
    points_xy: npt.NDArray[np.float32],
    polygon_xy: npt.NDArray[np.float32],
) -> npt.NDArray[np.bool_]:
    x_values = points_xy[:, 0]
    y_values = points_xy[:, 1]
    inside = np.zeros(len(points_xy), dtype=np.bool_)
    previous = len(polygon_xy) - 1
    for current in range(len(polygon_xy)):
        x_current, y_current = polygon_xy[current]
        x_previous, y_previous = polygon_xy[previous]
        crosses_y = (y_current > y_values) != (y_previous > y_values)
        denominator = float(y_previous - y_current)
        if abs(denominator) > 1.0e-8:
            x_crossing = (float(x_previous) - float(x_current)) * (
                y_values - float(y_current)
            ) / denominator + float(x_current)
            inside ^= crosses_y & (x_values < x_crossing)
        previous = current
    return inside


def _point_at_polyline(
    points_world: npt.NDArray[np.float32],
    cumulative_distances_m: npt.NDArray[np.float32],
    distance_m: float,
) -> npt.NDArray[np.float32]:
    distance_m = float(np.clip(distance_m, 0.0, cumulative_distances_m[-1]))
    right = int(np.searchsorted(cumulative_distances_m, distance_m, side="right"))
    right = min(max(1, right), len(points_world) - 1)
    left = right - 1
    span = float(cumulative_distances_m[right] - cumulative_distances_m[left])
    alpha = (
        0.0
        if span <= _MIN_SEGMENT_LENGTH_M
        else (distance_m - float(cumulative_distances_m[left])) / span
    )
    return ((1.0 - alpha) * points_world[left] + alpha * points_world[right]).astype(
        np.float32
    )


def _densify_polyline(
    points_world: npt.NDArray[np.float32], spacing_m: float
) -> npt.NDArray[np.float32]:
    segment_lengths = np.linalg.norm(np.diff(points_world[:, :2], axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths))).astype(np.float32)
    total_distance_m = float(cumulative[-1])
    sample_distances = np.arange(0.0, total_distance_m, spacing_m, dtype=np.float32)
    sample_distances = np.append(sample_distances, np.float32(total_distance_m))
    return np.stack(
        [
            _point_at_polyline(points_world, cumulative, float(distance_m))
            for distance_m in sample_distances
        ]
    )


def _heading_delta(
    before_xyz: npt.NDArray[np.float32],
    entry_xyz: npt.NDArray[np.float32],
    exit_xyz: npt.NDArray[np.float32],
    after_xyz: npt.NDArray[np.float32],
) -> float:
    incoming = entry_xyz[:2] - before_xyz[:2]
    outgoing = after_xyz[:2] - exit_xyz[:2]
    if (
        float(np.linalg.norm(incoming)) <= _MIN_SEGMENT_LENGTH_M
        or float(np.linalg.norm(outgoing)) <= _MIN_SEGMENT_LENGTH_M
    ):
        return 0.0
    incoming_heading = math.atan2(float(incoming[1]), float(incoming[0]))
    outgoing_heading = math.atan2(float(outgoing[1]), float(outgoing[0]))
    return (outgoing_heading - incoming_heading + math.pi) % (2.0 * math.pi) - math.pi


def _maneuver_from_heading_delta(heading_delta_rad: float) -> TurnManeuver:
    magnitude = abs(heading_delta_rad)
    if magnitude >= _U_TURN_THRESHOLD_RAD:
        return "u_turn"
    if magnitude <= _TURN_THRESHOLD_RAD:
        return "straight"
    return "left" if heading_delta_rad > 0.0 else "right"
