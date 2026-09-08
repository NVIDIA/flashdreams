# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compile authored traffic waypoints onto the directed public-road graph."""

from __future__ import annotations

import hashlib
import heapq
import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from omnidreams_game_engine.game_map._schema import GameMapError
from omnidreams_game_engine.game_map.types import (
    GameMapLane,
    GameMapSpawn,
    GameMapTopology,
    GameMapTrafficVehicle,
)

_VEHICLE_DIMENSIONS_LWH_M = {
    "car": (4.5, 1.8, 1.5),
    "truck": (7.0, 2.5, 3.0),
    "bus": (12.0, 2.55, 3.2),
}
_TURN_THRESHOLD_RAD = math.radians(35.0)
_GENERATED_SLOT_SPACING_M = 2.0
_GENERATED_FOOTPRINT_BUFFER_M = 0.5
_GENERATED_SPAWN_CLEARANCE_M = 8.0
_HEADWAY_MIN_CLEARANCE_M = 2.0
_HEADWAY_TIME_S = 1.25
_HEADWAY_LANE_CORRIDOR_M = 2.25
_HEADWAY_MAX_ANGLE_RAD = math.radians(40.0)
_MIN_ROUTE_POINT_SPACING_M = 0.25
_LANE_SEAM_TOLERANCE_M = 0.05
_MAX_ROUTE_YAW_RATE_RADPS = 1.2
_MAX_ROUTE_LATERAL_ACCEL_MPS2 = 2.5
_MAX_ROUTE_ACCEL_MPS2 = 2.5
_MAX_ROUTE_BRAKING_MPS2 = 4.0


@dataclass(frozen=True)
class _RouteTemplate:
    node_ids: tuple[str, ...]
    end_behavior: str
    centerline_world: np.ndarray
    speed_limits_mps: np.ndarray
    route_element_ids: tuple[str, ...]


@dataclass(frozen=True)
class _Placement:
    position_xy: np.ndarray
    forward_xy: np.ndarray
    speed_mps: float
    half_length_m: float
    half_width_m: float


def _polyline_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1).sum())


def _resample(points: np.ndarray, count: int) -> np.ndarray:
    lengths = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total = float(cumulative[-1])
    distances = np.linspace(0.0, total, count)
    result = np.empty((count, 3), dtype=np.float64)
    for index, distance in enumerate(distances):
        segment = min(
            int(np.searchsorted(cumulative, distance, side="right") - 1),
            len(lengths) - 1,
        )
        segment = max(0, segment)
        alpha = (distance - cumulative[segment]) / max(float(lengths[segment]), 1.0e-9)
        result[index] = points[segment] + alpha * (
            points[segment + 1] - points[segment]
        )
    return result


def _append_path(
    points: list[np.ndarray],
    speeds: list[float],
    element_ids: list[str],
    path: np.ndarray,
    speed_mps: float,
    element_id: str,
) -> None:
    for point in path:
        if (
            points
            and float(np.linalg.norm(points[-1][:2] - point[:2]))
            <= _MIN_ROUTE_POINT_SPACING_M
        ):
            speeds[-1] = min(speeds[-1], speed_mps)
            continue
        if points:
            element_ids.append(element_id)
        points.append(np.asarray(point, dtype=np.float64))
        speeds.append(speed_mps)


def _curve_limited_speeds(points: np.ndarray, speeds: np.ndarray) -> np.ndarray:
    """Limit a closed route to speeds its physical follower can turn through."""
    if len(points) < 4:
        return speeds
    closed = float(np.linalg.norm(points[-1, :2] - points[0, :2])) <= 1.0e-4
    if not closed:
        return speeds
    route_points = points[:-1]
    route_speeds = speeds[:-1].astype(np.float64, copy=True)
    count = len(route_points)
    if count < 3:
        return speeds

    segment_vectors = np.roll(route_points[:, :2], -1, axis=0) - route_points[:, :2]
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    headings = np.arctan2(segment_vectors[:, 1], segment_vectors[:, 0])
    heading_changes = np.abs(
        (headings - np.roll(headings, 1) + math.pi) % (2.0 * math.pi) - math.pi
    )
    previous_lengths = np.roll(segment_lengths, 1)
    local_lengths = np.maximum(
        0.5 * (previous_lengths + segment_lengths), _MIN_ROUTE_POINT_SPACING_M
    )
    curvature = heading_changes / local_lengths
    turning = curvature > 1.0e-9
    lateral_caps = np.full(count, math.inf, dtype=np.float64)
    lateral_caps[turning] = np.sqrt(_MAX_ROUTE_LATERAL_ACCEL_MPS2 / curvature[turning])
    yaw_caps = np.full(count, math.inf, dtype=np.float64)
    changing_heading = heading_changes > 1.0e-9
    yaw_caps[changing_heading] = (
        _MAX_ROUTE_YAW_RATE_RADPS
        * previous_lengths[changing_heading]
        / heading_changes[changing_heading]
    )
    route_speeds = np.minimum(route_speeds, np.minimum(lateral_caps, yaw_caps))

    # Propagate each turn's limit backward through its braking distance and
    # forward through acceleration, including across the cyclic route seam.
    for _ in range(count):
        previous_speeds = route_speeds.copy()
        for index in range(count):
            following = (index + 1) % count
            acceleration_cap = math.sqrt(
                route_speeds[index] ** 2
                + 2.0 * _MAX_ROUTE_ACCEL_MPS2 * segment_lengths[index]
            )
            route_speeds[following] = min(route_speeds[following], acceleration_cap)
        for index in range(count - 1, -1, -1):
            following = (index + 1) % count
            braking_cap = math.sqrt(
                route_speeds[following] ** 2
                + 2.0 * _MAX_ROUTE_BRAKING_MPS2 * segment_lengths[index]
            )
            route_speeds[index] = min(route_speeds[index], braking_cap)
        if np.array_equal(route_speeds, previous_speeds):
            break

    route_speeds = np.concatenate((route_speeds, route_speeds[:1]))
    return route_speeds.astype(np.float32)


def _directed_road_lanes(
    topology: GameMapTopology, lanes: tuple[GameMapLane, ...]
) -> dict[tuple[str, str, str], list[GameMapLane]]:
    nodes = {node.node_id: node for node in topology.nodes}
    result: dict[tuple[str, str, str], list[GameMapLane]] = {}
    for road in topology.roads:
        road_lanes = [lane for lane in lanes if lane.element_id == road.road_id]
        for start_id, end_id in (
            (road.from_node_id, road.to_node_id),
            (road.to_node_id, road.from_node_id),
        ):
            start = np.asarray([nodes[start_id].x_m, nodes[start_id].y_m])
            end = np.asarray([nodes[end_id].x_m, nodes[end_id].y_m])
            directed = [
                lane
                for lane in road_lanes
                if float(np.linalg.norm(lane.centerline_world[0, :2] - start))
                < float(np.linalg.norm(lane.centerline_world[-1, :2] - start))
                and float(np.linalg.norm(lane.centerline_world[-1, :2] - end))
                < float(np.linalg.norm(lane.centerline_world[0, :2] - end))
            ]
            if not directed:
                continue
            tangent = (
                directed[0].centerline_world[-1, :2]
                - directed[0].centerline_world[0, :2]
            )
            tangent /= max(float(np.linalg.norm(tangent)), 1.0e-9)
            right = np.asarray([tangent[1], -tangent[0]])
            directed.sort(
                key=lambda lane: -float(
                    np.dot(
                        lane.centerline_world[len(lane.centerline_world) // 2, :2],
                        right,
                    )
                )
            )
            result[(road.road_id, start_id, end_id)] = directed
    return result


def _shortest_roads(
    start_id: str,
    end_id: str,
    topology: GameMapTopology,
    directed: dict[tuple[str, str, str], list[GameMapLane]],
) -> list[tuple[str, str, str]]:
    if start_id == end_id:
        return []
    outgoing: dict[str, list[tuple[str, str, float]]] = {}
    for road in topology.roads:
        for a, b in (
            (road.from_node_id, road.to_node_id),
            (road.to_node_id, road.from_node_id),
        ):
            road_lanes = directed.get((road.road_id, a, b))
            if not road_lanes:
                continue
            weight = min(_polyline_length(lane.centerline_world) for lane in road_lanes)
            outgoing.setdefault(a, []).append((b, road.road_id, weight))
    queue: list[tuple[float, str]] = [(0.0, start_id)]
    distance = {start_id: 0.0}
    previous: dict[str, tuple[str, str]] = {}
    while queue:
        cost, node_id = heapq.heappop(queue)
        if cost != distance.get(node_id):
            continue
        if node_id == end_id:
            break
        for target_id, road_id, weight in sorted(outgoing.get(node_id, ())):
            candidate = cost + weight
            if candidate + 1.0e-9 < distance.get(target_id, math.inf):
                distance[target_id] = candidate
                previous[target_id] = (node_id, road_id)
                heapq.heappush(queue, (candidate, target_id))
    if end_id not in previous:
        raise GameMapError(
            f"Traffic route cannot reach node {end_id!r} from {start_id!r}"
        )
    reversed_path: list[tuple[str, str, str]] = []
    node_id = end_id
    while node_id != start_id:
        source_id, road_id = previous[node_id]
        reversed_path.append((road_id, source_id, node_id))
        node_id = source_id
    return list(reversed(reversed_path))


def _turn_kind(current: list[GameMapLane], following: list[GameMapLane]) -> str:
    incoming = current[0].centerline_world
    outgoing = following[0].centerline_world
    first = incoming[-1, :2] - incoming[-2, :2]
    second = outgoing[1, :2] - outgoing[0, :2]
    first /= max(float(np.linalg.norm(first)), 1.0e-9)
    second /= max(float(np.linalg.norm(second)), 1.0e-9)
    angle = math.atan2(
        float(first[0] * second[1] - first[1] * second[0]), float(np.dot(first, second))
    )
    if abs(angle) <= _TURN_THRESHOLD_RAD:
        return "straight"
    return "left" if angle > 0.0 else "right"


def _connector_path(
    source_id: str,
    target_id: str,
    lane_by_id: dict[str, GameMapLane],
    public_road_ids: set[str],
    parking_access_ids: set[str],
) -> list[GameMapLane]:
    queue: list[tuple[float, float, float, str, tuple[str, ...]]] = [
        (0.0, 0.0, 0.0, source_id, (source_id,))
    ]
    best = {source_id: (0.0, 0.0, 0.0)}
    while queue:
        seam_cost, heading_cost, length_cost, lane_id, path = heapq.heappop(queue)
        if (seam_cost, heading_cost, length_cost) != best.get(lane_id):
            continue
        if lane_id == target_id:
            return [lane_by_id[item] for item in path]
        lane = lane_by_id[lane_id]
        source_tangent = lane.centerline_world[-1, :2] - lane.centerline_world[-2, :2]
        source_tangent /= max(float(np.linalg.norm(source_tangent)), 1.0e-9)
        for successor in lane.successor_ids:
            if successor not in lane_by_id:
                continue
            following = lane_by_id[successor]
            if following.element_id in public_road_ids and successor != target_id:
                continue
            if following.element_id in parking_access_ids:
                continue
            seam_distance = float(
                np.linalg.norm(
                    lane.centerline_world[-1, :2] - following.centerline_world[0, :2]
                )
            )
            target_tangent = (
                following.centerline_world[1, :2] - following.centerline_world[0, :2]
            )
            target_tangent /= max(float(np.linalg.norm(target_tangent)), 1.0e-9)
            heading_delta = math.acos(
                float(np.clip(np.dot(source_tangent, target_tangent), -1.0, 1.0))
            )
            cost = (
                seam_cost + max(0.0, seam_distance - _LANE_SEAM_TOLERANCE_M),
                heading_cost + heading_delta,
                length_cost
                + (
                    0.0
                    if successor == target_id
                    else _polyline_length(following.centerline_world)
                ),
            )
            if cost >= best.get(successor, (math.inf, math.inf, math.inf)):
                continue
            best[successor] = cost
            heapq.heappush(queue, (*cost, successor, (*path, successor)))
    raise GameMapError(
        f"Traffic route has no legal lane connection from {source_id!r} to {target_id!r}"
    )


def _compile_route(
    traversals: list[tuple[str, str, str]],
    topology: GameMapTopology,
    lanes: tuple[GameMapLane, ...],
    speed_cap_mps: float | None,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    if not traversals:
        raise GameMapError("Traffic routes must contain travel between distinct nodes")
    directed = _directed_road_lanes(topology, lanes)
    candidates = [directed[item] for item in traversals]
    node_types = {node.node_id: node.node_type for node in topology.nodes}
    entry: list[int | None] = [None] * len(traversals)
    exit_lane: list[int | None] = [None] * len(traversals)
    for index, current in enumerate(candidates):
        following_index = (index + 1) % len(candidates)
        following = candidates[following_index]
        node_id = traversals[index][2]
        kind = (
            "straight"
            if node_types[node_id] in {"road_joint", "driveway"}
            else _turn_kind(current, following)
        )
        if kind == "right":
            exit_lane[index] = 0
            entry[following_index] = 0
        elif kind == "left":
            exit_lane[index] = len(current) - 1
            entry[following_index] = len(following) - 1
    for _ in range(2):
        for index, current in enumerate(candidates):
            following_index = (index + 1) % len(candidates)
            following = candidates[following_index]
            if exit_lane[index] is None:
                exit_lane[index] = entry[index] if entry[index] is not None else 0
            if entry[following_index] is None:
                rank = (
                    0.0 if len(current) == 1 else exit_lane[index] / (len(current) - 1)
                )
                entry[following_index] = round(rank * (len(following) - 1))
    lane_by_id = {lane.lane_id: lane for lane in lanes}
    public_road_ids = {road.road_id for road in topology.roads}
    parking_access_ids = {access.access_id for access in topology.parking_accesses}
    points: list[np.ndarray] = []
    speeds: list[float] = []
    element_ids: list[str] = []
    for index, road_candidates in enumerate(candidates):
        incoming_lane = road_candidates[int(entry[index] or 0)]
        outgoing_lane = road_candidates[int(exit_lane[index] or 0)]
        count = max(
            len(incoming_lane.centerline_world), len(outgoing_lane.centerline_world), 8
        )
        incoming_points = _resample(incoming_lane.centerline_world, count)
        outgoing_points = _resample(outgoing_lane.centerline_world, count)
        alpha = np.linspace(0.0, 1.0, count)
        smooth = alpha * alpha * alpha * (10.0 + alpha * (-15.0 + 6.0 * alpha))
        road_path = (
            incoming_points * (1.0 - smooth[:, None])
            + outgoing_points * smooth[:, None]
        )
        road_speed = min(incoming_lane.speed_limit_mps, outgoing_lane.speed_limit_mps)
        if speed_cap_mps is not None:
            road_speed = min(road_speed, speed_cap_mps)
        _append_path(
            points,
            speeds,
            element_ids,
            road_path,
            road_speed,
            traversals[index][0],
        )

        following_index = (index + 1) % len(candidates)
        target_lane = candidates[following_index][int(entry[following_index] or 0)]
        connector = _connector_path(
            outgoing_lane.lane_id,
            target_lane.lane_id,
            lane_by_id,
            public_road_ids,
            parking_access_ids,
        )
        for lane in connector[1:-1]:
            connector_speed = lane.speed_limit_mps
            if speed_cap_mps is not None:
                connector_speed = min(connector_speed, speed_cap_mps)
            _append_path(
                points,
                speeds,
                element_ids,
                lane.centerline_world,
                connector_speed,
                lane.element_id,
            )
    closure_distance = float(np.linalg.norm(points[-1][:2] - points[0][:2]))
    if closure_distance > _MIN_ROUTE_POINT_SPACING_M:
        element_ids.append(element_ids[-1])
        points.append(points[0].copy())
        speeds.append(speeds[0])
    elif closure_distance > 1.0e-4:
        points[-1] = points[0].copy()
        speeds[-1] = min(speeds[-1], speeds[0])
    if len(points) < 3 or _polyline_length(np.asarray(points)) <= 1.0:
        raise GameMapError("Traffic route resolves to degenerate geometry")
    if len(element_ids) != len(points) - 1:
        raise GameMapError("Traffic route element metadata is misaligned")
    route_points = np.asarray(points, dtype=np.float32)
    route_speeds = _curve_limited_speeds(
        route_points, np.asarray(speeds, dtype=np.float32)
    )
    return route_points, route_speeds, tuple(element_ids)


def _insert_turnarounds(
    traversals: list[tuple[str, str, str]],
    topology: GameMapTopology,
    directed: dict[tuple[str, str, str], list[GameMapLane]],
) -> list[tuple[str, str, str]]:
    """Route immediate reversals through an incident cul-de-sac arm."""
    node_types = {node.node_id: node.node_type for node in topology.nodes}
    roads = list(topology.roads)
    result: list[tuple[str, str, str]] = []
    for index, current in enumerate(traversals):
        result.append(current)
        following = traversals[(index + 1) % len(traversals)]
        if current[2] != following[1] or current[0] != following[0]:
            continue
        node_id = current[2]
        if node_types[node_id] == "cul_de_sac":
            continue
        candidates: list[tuple[str, str]] = []
        for road in roads:
            if road.road_id == current[0]:
                continue
            if road.from_node_id == node_id:
                remote = road.to_node_id
            elif road.to_node_id == node_id:
                remote = road.from_node_id
            else:
                continue
            if (
                node_types[remote] == "cul_de_sac"
                and (road.road_id, node_id, remote) in directed
                and (road.road_id, remote, node_id) in directed
            ):
                candidates.append((road.road_id, remote))
        if not candidates:
            raise GameMapError(
                f"Traffic route cannot reverse direction at node {node_id!r}; "
                "add a waypoint loop or use a cul-de-sac endpoint"
            )
        road_id, remote = sorted(candidates)[0]
        result.extend(((road_id, node_id, remote), (road_id, remote, node_id)))
    return result


def _compile_waypoint_route(
    node_ids: tuple[str, ...],
    end_behavior: str,
    topology: GameMapTopology,
    lanes: tuple[GameMapLane, ...],
    directed: dict[tuple[str, str, str], list[GameMapLane]],
    speed_mps: float | None,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    waypoint_cycle = list(node_ids)
    if end_behavior == "reverse":
        waypoint_cycle.extend(reversed(node_ids[:-1]))
    legs = list(zip(waypoint_cycle, waypoint_cycle[1:]))
    if end_behavior == "wrap":
        legs.append((waypoint_cycle[-1], waypoint_cycle[0]))
    traversals: list[tuple[str, str, str]] = []
    for source_id, target_id in legs:
        traversals.extend(_shortest_roads(source_id, target_id, topology, directed))
    traversals = _insert_turnarounds(traversals, topology, directed)
    return _compile_route(traversals, topology, lanes, speed_mps)


def _tree_path(
    start_id: str,
    end_id: str,
    adjacency: dict[str, list[tuple[str, str]]],
) -> list[str]:
    queue = deque([start_id])
    previous: dict[str, str | None] = {start_id: None}
    while queue:
        node_id = queue.popleft()
        if node_id == end_id:
            break
        for neighbor_id, _ in adjacency.get(node_id, ()):
            if neighbor_id in previous:
                continue
            previous[neighbor_id] = node_id
            queue.append(neighbor_id)
    if end_id not in previous:
        return []
    path: list[str] = []
    node_id: str | None = end_id
    while node_id is not None:
        path.append(node_id)
        node_id = previous[node_id]
    return list(reversed(path))


def _fundamental_cycles(topology: GameMapTopology) -> list[tuple[str, ...]]:
    parent: dict[str, str] = {}

    def find(node_id: str) -> str:
        parent.setdefault(node_id, node_id)
        while parent[node_id] != node_id:
            parent[node_id] = parent[parent[node_id]]
            node_id = parent[node_id]
        return node_id

    tree: dict[str, list[tuple[str, str]]] = {}
    cycles: list[tuple[str, ...]] = []
    for road in sorted(topology.roads, key=lambda value: value.road_id):
        first = road.from_node_id
        second = road.to_node_id
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root
            tree.setdefault(first, []).append((second, road.road_id))
            tree.setdefault(second, []).append((first, road.road_id))
            continue
        path = _tree_path(second, first, tree)
        if len(path) >= 3:
            cycles.append(tuple([first, *path[:-1]]))
    return cycles


def _nearest_cul_de_sac_pairs(
    topology: GameMapTopology,
    directed: dict[tuple[str, str, str], list[GameMapLane]],
) -> list[tuple[str, str]]:
    cul_de_sacs = sorted(
        node.node_id for node in topology.nodes if node.node_type == "cul_de_sac"
    )
    pairs: set[tuple[str, str]] = set()
    for source_id in cul_de_sacs:
        candidates: list[tuple[float, str]] = []
        for target_id in cul_de_sacs:
            if target_id == source_id:
                continue
            try:
                route = _shortest_roads(source_id, target_id, topology, directed)
            except GameMapError:
                continue
            length = sum(
                min(
                    _polyline_length(lane.centerline_world)
                    for lane in directed[traversal]
                )
                for traversal in route
            )
            candidates.append((length, target_id))
        if candidates:
            target_id = min(candidates)[1]
            pairs.add(tuple(sorted((source_id, target_id))))
    return sorted(pairs)


def _generated_route_templates(
    topology: GameMapTopology,
    lanes: tuple[GameMapLane, ...],
    directed: dict[tuple[str, str, str], list[GameMapLane]],
) -> list[_RouteTemplate]:
    candidates: list[tuple[tuple[str, ...], str]] = []
    for cycle in _fundamental_cycles(topology):
        candidates.append((cycle, "wrap"))
        candidates.append((tuple([cycle[0], *reversed(cycle[1:])]), "wrap"))
    candidates.extend(
        ((pair, "reverse") for pair in _nearest_cul_de_sac_pairs(topology, directed))
    )
    templates: list[_RouteTemplate] = []
    seen_geometry: set[bytes] = set()
    for node_ids, end_behavior in candidates:
        try:
            centerline, speed_limits, route_element_ids = _compile_waypoint_route(
                node_ids,
                end_behavior,
                topology,
                lanes,
                directed,
                None,
            )
        except GameMapError:
            continue
        fingerprint = hashlib.sha256(centerline.tobytes()).digest()
        if fingerprint in seen_geometry:
            continue
        seen_geometry.add(fingerprint)
        templates.append(
            _RouteTemplate(
                node_ids=node_ids,
                end_behavior=end_behavior,
                centerline_world=centerline,
                speed_limits_mps=speed_limits,
                route_element_ids=route_element_ids,
            )
        )
    return templates


def _placement_at_distance(
    centerline: np.ndarray,
    speeds: np.ndarray,
    distance_m: float,
    dimensions_lwh_m: tuple[float, float, float],
) -> _Placement:
    lengths = np.linalg.norm(np.diff(centerline[:, :2], axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    segment = min(
        max(int(np.searchsorted(cumulative, distance_m, side="right") - 1), 0),
        len(lengths) - 1,
    )
    alpha = (distance_m - cumulative[segment]) / max(float(lengths[segment]), 1e-9)
    position = centerline[segment, :2] + alpha * (
        centerline[segment + 1, :2] - centerline[segment, :2]
    )
    forward = centerline[segment + 1, :2] - centerline[segment, :2]
    forward /= max(float(np.linalg.norm(forward)), 1e-9)
    speed = float(speeds[segment] * (1.0 - alpha) + speeds[segment + 1] * alpha)
    return _Placement(
        position_xy=np.asarray(position, dtype=np.float64),
        forward_xy=np.asarray(forward, dtype=np.float64),
        speed_mps=speed,
        half_length_m=dimensions_lwh_m[0] * 0.5,
        half_width_m=dimensions_lwh_m[1] * 0.5,
    )


def _footprints_overlap(first: _Placement, second: _Placement) -> bool:
    first_left = np.asarray([-first.forward_xy[1], first.forward_xy[0]])
    second_left = np.asarray([-second.forward_xy[1], second.forward_xy[0]])
    delta = second.position_xy - first.position_xy
    axes = (first.forward_xy, first_left, second.forward_xy, second_left)
    first_extents = (
        first.half_length_m + _GENERATED_FOOTPRINT_BUFFER_M,
        first.half_width_m + _GENERATED_FOOTPRINT_BUFFER_M,
    )
    second_extents = (
        second.half_length_m + _GENERATED_FOOTPRINT_BUFFER_M,
        second.half_width_m + _GENERATED_FOOTPRINT_BUFFER_M,
    )
    for axis in axes:
        first_radius = first_extents[0] * abs(float(np.dot(first.forward_xy, axis)))
        first_radius += first_extents[1] * abs(float(np.dot(first_left, axis)))
        second_radius = second_extents[0] * abs(float(np.dot(second.forward_xy, axis)))
        second_radius += second_extents[1] * abs(float(np.dot(second_left, axis)))
        if abs(float(np.dot(delta, axis))) >= first_radius + second_radius:
            return False
    return True


def _placement_is_safe(
    candidate: _Placement,
    occupied: list[_Placement],
    spawn_positions: tuple[np.ndarray, ...],
) -> bool:
    if any(
        float(np.linalg.norm(candidate.position_xy - spawn_position))
        < _GENERATED_SPAWN_CLEARANCE_M
        for spawn_position in spawn_positions
    ):
        return False
    for other in occupied:
        if _footprints_overlap(candidate, other):
            return False
        heading_dot = float(np.dot(candidate.forward_xy, other.forward_xy))
        heading_dot = float(np.clip(heading_dot, -1.0, 1.0))
        if math.acos(heading_dot) > _HEADWAY_MAX_ANGLE_RAD:
            continue
        delta = other.position_xy - candidate.position_xy
        lateral = abs(
            float(
                candidate.forward_xy[0] * delta[1] - candidate.forward_xy[1] * delta[0]
            )
        )
        if lateral > _HEADWAY_LANE_CORRIDOR_M:
            continue
        longitudinal = abs(float(np.dot(delta, candidate.forward_xy)))
        required = (
            candidate.half_length_m
            + other.half_length_m
            + _HEADWAY_MIN_CLEARANCE_M
            + _HEADWAY_TIME_S * max(candidate.speed_mps, other.speed_mps)
        )
        if longitudinal < required:
            return False
    return True


def _generate_traffic(
    count: int,
    authored: list[GameMapTrafficVehicle],
    topology: GameMapTopology,
    lanes: tuple[GameMapLane, ...],
    directed: dict[tuple[str, str, str], list[GameMapLane]],
    map_id: str,
    spawns: tuple[GameMapSpawn, ...],
) -> list[GameMapTrafficVehicle]:
    templates = _generated_route_templates(topology, lanes, directed)
    dimensions = _VEHICLE_DIMENSIONS_LWH_M["car"]
    occupied = [
        _placement_at_distance(
            vehicle.centerline_world,
            vehicle.speed_limits_mps,
            vehicle.start_distance_m,
            vehicle.dimensions_lwh_m,
        )
        for vehicle in authored
    ]
    spawn_positions = tuple(
        np.asarray(spawn.position_world[:2], dtype=np.float64) for spawn in spawns
    )
    slots: list[tuple[bytes, _RouteTemplate, float]] = []
    for template in templates:
        route_length = _polyline_length(template.centerline_world)
        signature = "|".join((*template.node_ids, template.end_behavior)).encode()
        for offset_m in np.arange(0.0, route_length, _GENERATED_SLOT_SPACING_M):
            key = hashlib.sha256(
                map_id.encode()
                + b"|"
                + signature
                + b"|"
                + f"{float(offset_m):.3f}".encode()
            ).digest()
            slots.append((key, template, float(offset_m)))
    accepted: list[tuple[_RouteTemplate, float]] = []
    for _, template, offset_m in sorted(slots, key=lambda item: item[0]):
        placement = _placement_at_distance(
            template.centerline_world,
            template.speed_limits_mps,
            offset_m,
            dimensions,
        )
        if not _placement_is_safe(placement, occupied, spawn_positions):
            continue
        occupied.append(placement)
        accepted.append((template, offset_m))
        if len(accepted) == count:
            break
    if len(accepted) < count:
        maximum = len(authored) + len(accepted)
        raise GameMapError(
            f"traffic_count requests {len(authored) + count} vehicles, but this "
            f"map has safe capacity for {maximum}"
        )
    used_ids = {vehicle.vehicle_id for vehicle in authored}
    generated: list[GameMapTrafficVehicle] = []
    next_id = 1
    for template, offset_m in accepted[:count]:
        while True:
            vehicle_id = f"generated-traffic-{next_id:04d}"
            next_id += 1
            if vehicle_id not in used_ids:
                break
        used_ids.add(vehicle_id)
        generated.append(
            GameMapTrafficVehicle(
                vehicle_id=vehicle_id,
                node_ids=template.node_ids,
                end_behavior=template.end_behavior,
                vehicle_type="car",
                dimensions_lwh_m=dimensions,
                speed_mps=None,
                start_distance_m=offset_m,
                centerline_world=template.centerline_world,
                speed_limits_mps=template.speed_limits_mps,
                route_element_ids=template.route_element_ids,
            )
        )
    return generated


def compile_traffic(
    raw_values: object,
    topology: GameMapTopology,
    lanes: tuple[GameMapLane, ...],
    *,
    traffic_count: object = None,
    map_id: str = "",
    spawns: tuple[GameMapSpawn, ...] = (),
) -> tuple[GameMapTrafficVehicle, ...]:
    """Validate and compile optional traffic definitions."""
    if traffic_count is not None and (
        isinstance(traffic_count, bool)
        or not isinstance(traffic_count, int)
        or traffic_count < 0
    ):
        raise GameMapError("traffic_count must be a nonnegative integer")
    if raw_values is None:
        raw_values = []
    if not isinstance(raw_values, list):
        raise GameMapError("traffic must be a sequence")
    nodes = {node.node_id: node for node in topology.nodes}
    directed = _directed_road_lanes(topology, lanes)
    results: list[GameMapTrafficVehicle] = []
    seen_ids: set[str] = set()
    allowed = {
        "id",
        "nodes",
        "end_behavior",
        "vehicle_type",
        "dimensions_lwh_m",
        "speed_mps",
        "start_distance_m",
    }
    for index, raw_value in enumerate(raw_values):
        if not isinstance(raw_value, dict):
            raise GameMapError(f"traffic[{index}] must be a mapping")
        unknown = set(raw_value) - allowed
        missing = {"id", "nodes", "end_behavior"} - set(raw_value)
        if unknown or missing:
            detail = (
                f"unknown fields {sorted(unknown)}"
                if unknown
                else f"missing fields {sorted(missing)}"
            )
            raise GameMapError(f"traffic[{index}] has {detail}")
        vehicle_id = str(raw_value["id"]).strip()
        if not vehicle_id or vehicle_id in seen_ids:
            raise GameMapError(f"Traffic id {vehicle_id!r} is empty or duplicated")
        seen_ids.add(vehicle_id)
        raw_nodes = raw_value["nodes"]
        if not isinstance(raw_nodes, list) or len(raw_nodes) < 2:
            raise GameMapError(
                f"Traffic {vehicle_id!r}.nodes requires at least two nodes"
            )
        node_ids = tuple(str(item).strip() for item in raw_nodes)
        for node_id in node_ids:
            if node_id not in nodes:
                raise GameMapError(
                    f"Traffic {vehicle_id!r} references unknown node {node_id!r}"
                )
            if nodes[node_id].node_type == "parking_lot":
                raise GameMapError(
                    f"Traffic {vehicle_id!r} cannot visit parking-lot node {node_id!r}"
                )
        end_behavior = str(raw_value["end_behavior"]).strip()
        if end_behavior not in {"reverse", "wrap"}:
            raise GameMapError(
                f"Traffic {vehicle_id!r}.end_behavior must be reverse or wrap"
            )
        vehicle_type = str(raw_value.get("vehicle_type", "car")).strip().lower()
        if vehicle_type not in _VEHICLE_DIMENSIONS_LWH_M:
            raise GameMapError(
                f"Traffic {vehicle_id!r}.vehicle_type must be car, truck, or bus"
            )
        dimensions_raw = raw_value.get(
            "dimensions_lwh_m", _VEHICLE_DIMENSIONS_LWH_M[vehicle_type]
        )
        if not isinstance(dimensions_raw, (list, tuple)) or len(dimensions_raw) != 3:
            raise GameMapError(
                f"Traffic {vehicle_id!r}.dimensions_lwh_m requires three values"
            )
        try:
            dimensions = tuple(float(item) for item in dimensions_raw)
        except (TypeError, ValueError) as exc:
            raise GameMapError(
                f"Traffic {vehicle_id!r}.dimensions_lwh_m must be numeric"
            ) from exc
        if any(not math.isfinite(item) or item <= 0.0 for item in dimensions):
            raise GameMapError(
                f"Traffic {vehicle_id!r}.dimensions_lwh_m must be positive and finite"
            )
        speed_value = raw_value.get("speed_mps")
        try:
            speed_mps = None if speed_value is None else float(speed_value)
            start_distance_m = float(raw_value.get("start_distance_m", 0.0))
        except (TypeError, ValueError) as exc:
            raise GameMapError(
                f"Traffic {vehicle_id!r} speed_mps and start_distance_m must be numeric"
            ) from exc
        if speed_mps is not None and (not math.isfinite(speed_mps) or speed_mps <= 0.0):
            raise GameMapError(
                f"Traffic {vehicle_id!r}.speed_mps must be positive and finite"
            )
        if not math.isfinite(start_distance_m) or start_distance_m < 0.0:
            raise GameMapError(
                f"Traffic {vehicle_id!r}.start_distance_m must be nonnegative and finite"
            )

        centerline, speed_limits, route_element_ids = _compile_waypoint_route(
            node_ids,
            end_behavior,
            topology,
            lanes,
            directed,
            speed_mps,
        )
        route_length = _polyline_length(centerline)
        if start_distance_m >= route_length:
            raise GameMapError(
                f"Traffic {vehicle_id!r}.start_distance_m must be less than route length {route_length:.2f} m"
            )
        results.append(
            GameMapTrafficVehicle(
                vehicle_id=vehicle_id,
                node_ids=node_ids,
                end_behavior=end_behavior,
                vehicle_type=vehicle_type,
                dimensions_lwh_m=dimensions,
                speed_mps=speed_mps,
                start_distance_m=start_distance_m,
                centerline_world=centerline,
                speed_limits_mps=speed_limits,
                route_element_ids=route_element_ids,
            )
        )
    if traffic_count is None:
        return tuple(results)
    if traffic_count < len(results):
        raise GameMapError(
            f"traffic_count is {traffic_count}, but traffic defines "
            f"{len(results)} vehicles; remove entries or increase traffic_count"
        )
    generated_count = traffic_count - len(results)
    if generated_count:
        results.extend(
            _generate_traffic(
                generated_count,
                results,
                topology,
                lanes,
                directed,
                map_id,
                spawns,
            )
        )
    return tuple(results)


__all__ = ["compile_traffic"]
