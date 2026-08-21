# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compile authored traffic waypoints onto the directed public-road graph."""

from __future__ import annotations

import heapq
import math
from collections import deque

import numpy as np

from omnidreams_game_engine.game_map._schema import GameMapError
from omnidreams_game_engine.game_map.types import (
    GameMapLane,
    GameMapTopology,
    GameMapTrafficVehicle,
)

_VEHICLE_DIMENSIONS_LWH_M = {
    "car": (4.5, 1.8, 1.5),
    "truck": (7.0, 2.5, 3.0),
    "bus": (12.0, 2.55, 3.2),
}
_TURN_THRESHOLD_RAD = math.radians(35.0)


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
    points: list[np.ndarray], speeds: list[float], path: np.ndarray, speed_mps: float
) -> None:
    for point in path:
        if points and float(np.linalg.norm(points[-1][:2] - point[:2])) <= 1.0e-4:
            speeds[-1] = min(speeds[-1], speed_mps)
            continue
        points.append(np.asarray(point, dtype=np.float64))
        speeds.append(speed_mps)


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
    queue: deque[tuple[str, list[str]]] = deque([(source_id, [source_id])])
    visited = {source_id}
    while queue:
        lane_id, path = queue.popleft()
        if lane_id == target_id:
            return [lane_by_id[item] for item in path]
        for successor in lane_by_id[lane_id].successor_ids:
            if successor in visited or successor not in lane_by_id:
                continue
            lane = lane_by_id[successor]
            if lane.element_id in public_road_ids and successor != target_id:
                continue
            if lane.element_id in parking_access_ids:
                continue
            visited.add(successor)
            queue.append((successor, [*path, successor]))
    raise GameMapError(
        f"Traffic route has no legal lane connection from {source_id!r} to {target_id!r}"
    )


def _compile_route(
    traversals: list[tuple[str, str, str]],
    topology: GameMapTopology,
    lanes: tuple[GameMapLane, ...],
    speed_cap_mps: float | None,
) -> tuple[np.ndarray, np.ndarray]:
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
        _append_path(points, speeds, road_path, road_speed)

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
            _append_path(points, speeds, lane.centerline_world, connector_speed)
    if float(np.linalg.norm(points[-1][:2] - points[0][:2])) > 1.0e-4:
        points.append(points[0].copy())
        speeds.append(speeds[0])
    if len(points) < 3 or _polyline_length(np.asarray(points)) <= 1.0:
        raise GameMapError("Traffic route resolves to degenerate geometry")
    return np.asarray(points, dtype=np.float32), np.asarray(speeds, dtype=np.float32)


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


def compile_traffic(
    raw_values: object,
    topology: GameMapTopology,
    lanes: tuple[GameMapLane, ...],
) -> tuple[GameMapTrafficVehicle, ...]:
    """Validate and compile optional traffic definitions."""
    if raw_values is None:
        return ()
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
        centerline, speed_limits = _compile_route(
            traversals, topology, lanes, speed_mps
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
            )
        )
    return tuple(results)


__all__ = ["compile_traffic"]
