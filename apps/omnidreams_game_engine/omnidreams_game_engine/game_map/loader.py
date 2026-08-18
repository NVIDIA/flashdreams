# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Node-graph game-map loading and geometry compilation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from omnidreams_game_engine.game_map._schema import (
    _SCHEMA_VERSION,
    GameMapError,
    GameMapHeader,
    _bezier,
    _CompilerSettings,
    _lane_edge_markings,
    _LaneBuild,
    _mapping,
    _offset_polyline,
    _parse_compiler_settings,
    _parse_profiles,
    _parse_variants,
    _positive_float,
    _Profile,
    _read_document,
    _segments,
    _sequence,
    _surface_for_road,
    _xyz,
    load_game_map_header,
    resolve_seed_asset,
)
from omnidreams_game_engine.game_map.types import (
    GameMapDirectLink,
    GameMapElement,
    GameMapLane,
    GameMapNode,
    GameMapRoad,
    GameMapRoadAttachment,
    GameMapSpawn,
    GameMapTopology,
    ResolvedGameMap,
)

_POSITION_TOLERANCE_M = 0.05


@dataclass(frozen=True)
class _RoadSpec:
    road: GameMapRoad
    spans_xy: tuple[np.ndarray, ...]


@dataclass
class _LaneIncidence:
    lane: _LaneBuild
    node_id: str
    kind: str
    edge_ref: str


def _finite_float(value: object, context: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise GameMapError(f"{context} must be a number") from exc
    if not math.isfinite(number):
        raise GameMapError(f"{context} must be finite")
    return number


def _point(value: object, context: str) -> np.ndarray:
    raw = _mapping(value, context)
    if set(raw) != {"x_m", "y_m"}:
        raise GameMapError(f"{context} requires exactly x_m and y_m")
    return np.asarray(
        [
            _finite_float(raw["x_m"], f"{context}.x_m"),
            _finite_float(raw["y_m"], f"{context}.y_m"),
        ],
        dtype=np.float64,
    )


def _parse_nodes(
    doc: dict[str, Any], profiles: dict[str, _Profile]
) -> tuple[GameMapNode, ...]:
    nodes: list[GameMapNode] = []
    ids: set[str] = set()
    geometry_keys = {
        "intersection": frozenset({"width_m", "depth_m"}),
        "cul_de_sac": frozenset({"radius_m"}),
        "parking_lot": frozenset({"width_m", "depth_m"}),
        "driveway": frozenset({"width_m"}),
    }
    for index, value in enumerate(_sequence(doc.get("nodes"), "nodes")):
        raw = _mapping(value, f"nodes[{index}]")
        node_type = str(raw.get("type", ""))
        requires_profile = node_type in {"parking_lot", "driveway"}
        expected = {"id", "type", "pose", "geometry"}
        if requires_profile:
            expected.add("profile")
        if node_type not in geometry_keys or set(raw) != expected:
            raise GameMapError(
                f"nodes[{index}] must be a supported node with exactly {sorted(expected)}"
            )
        node_id = str(raw["id"]).strip()
        if not node_id or node_id in ids:
            raise GameMapError(f"Node id {node_id!r} is empty or duplicated")
        ids.add(node_id)
        pose = _mapping(raw["pose"], f"node {node_id!r}.pose")
        if set(pose) != {"x_m", "y_m", "rotation_deg"}:
            raise GameMapError(
                f"Node {node_id!r}.pose requires x_m, y_m, and rotation_deg"
            )
        geometry = _mapping(raw["geometry"], f"node {node_id!r}.geometry")
        allowed_geometry = geometry_keys[node_type]
        required_geometry = (
            frozenset() if node_type == "intersection" else allowed_geometry
        )
        if not required_geometry <= set(geometry) <= allowed_geometry:
            raise GameMapError(
                f"Node {node_id!r}.geometry accepts exactly {sorted(allowed_geometry)}; "
                "intersection dimensions may both be omitted"
            )
        if (
            node_type == "intersection"
            and bool(geometry)
            and set(geometry) != set(allowed_geometry)
        ):
            raise GameMapError(
                f"Node {node_id!r}.geometry must provide both width_m and depth_m"
            )
        dimensions = {
            key: _positive_float(item, f"node {node_id!r}.geometry.{key}")
            for key, item in geometry.items()
        }
        profile_id = str(raw["profile"]) if requires_profile else None
        if profile_id is not None and profile_id not in profiles:
            raise GameMapError(
                f"Node {node_id!r} references unknown profile {profile_id!r}"
            )
        nodes.append(
            GameMapNode(
                node_id=node_id,
                node_type=node_type,
                x_m=_finite_float(pose["x_m"], f"node {node_id!r}.pose.x_m"),
                y_m=_finite_float(pose["y_m"], f"node {node_id!r}.pose.y_m"),
                rotation_deg=_finite_float(
                    pose["rotation_deg"], f"node {node_id!r}.pose.rotation_deg"
                )
                % 360.0,
                profile_id=profile_id,
                geometry=dimensions,
            )
        )
    if not nodes:
        raise GameMapError("Map must define at least one node")
    return tuple(nodes)


def _parse_roads(
    doc: dict[str, Any], nodes: dict[str, GameMapNode], profiles: dict[str, _Profile]
) -> tuple[_RoadSpec, ...]:
    roads: list[_RoadSpec] = []
    ids: set[str] = set()
    for index, value in enumerate(_sequence(doc.get("roads"), "roads")):
        raw = _mapping(value, f"roads[{index}]")
        if set(raw) not in (
            {"id", "from", "to", "profile"},
            {"id", "from", "to", "profile", "path"},
        ):
            raise GameMapError(
                f"roads[{index}] requires id, from, to, profile, and optional path"
            )
        road_id = str(raw["id"]).strip()
        if not road_id or road_id in ids:
            raise GameMapError(f"Road id {road_id!r} is empty or duplicated")
        ids.add(road_id)
        from_id, to_id = str(raw["from"]), str(raw["to"])
        for endpoint in (from_id, to_id):
            if endpoint not in nodes:
                raise GameMapError(
                    f"Road {road_id!r} references unknown node {endpoint!r}"
                )
            if nodes[endpoint].node_type not in {"intersection", "cul_de_sac"}:
                raise GameMapError(
                    f"Road {road_id!r} may connect only intersections and cul-de-sacs"
                )
        profile_id = str(raw["profile"])
        if profile_id not in profiles:
            raise GameMapError(
                f"Road {road_id!r} references unknown profile {profile_id!r}"
            )
        start = np.asarray([nodes[from_id].x_m, nodes[from_id].y_m])
        end = np.asarray([nodes[to_id].x_m, nodes[to_id].y_m])
        spans: list[np.ndarray] = []
        cursor = start
        if "path" in raw:
            path = _sequence(raw["path"], f"road {road_id!r}.path")
            if not path:
                raise GameMapError(f"Road {road_id!r}.path must not be empty")
            for span_index, span_value in enumerate(path):
                span = _mapping(span_value, f"road {road_id!r}.path[{span_index}]")
                if set(span) != {"control_points", "end"}:
                    raise GameMapError(
                        f"Road {road_id!r} path spans require control_points and end"
                    )
                controls = _sequence(
                    span["control_points"],
                    f"road {road_id!r}.path[{span_index}].control_points",
                )
                if len(controls) != 2:
                    raise GameMapError(
                        f"Road {road_id!r} path spans require exactly two control points"
                    )
                control_1 = _point(
                    controls[0],
                    f"road {road_id!r}.path[{span_index}].control_points[0]",
                )
                control_2 = _point(
                    controls[1],
                    f"road {road_id!r}.path[{span_index}].control_points[1]",
                )
                span_end = _point(
                    span["end"], f"road {road_id!r}.path[{span_index}].end"
                )
                if np.linalg.norm(control_1 - cursor) <= _POSITION_TOLERANCE_M:
                    raise GameMapError(
                        f"Road {road_id!r} span {span_index} has a degenerate start tangent"
                    )
                if np.linalg.norm(span_end - control_2) <= _POSITION_TOLERANCE_M:
                    raise GameMapError(
                        f"Road {road_id!r} span {span_index} has a degenerate end tangent"
                    )
                spans.append(np.vstack((cursor, control_1, control_2, span_end)))
                cursor = span_end
            if np.linalg.norm(cursor - end) > _POSITION_TOLERANCE_M:
                raise GameMapError(
                    f"Road {road_id!r} final path endpoint must equal its to-node pose"
                )
        elif np.linalg.norm(start - end) <= _POSITION_TOLERANCE_M:
            raise GameMapError(f"Self-loop road {road_id!r} requires a path")
        runtime_spans = tuple(
            np.column_stack((span, np.zeros(4))).astype(np.float32) for span in spans
        )
        roads.append(
            _RoadSpec(
                GameMapRoad(
                    road_id=road_id,
                    from_node_id=from_id,
                    to_node_id=to_id,
                    profile_id=profile_id,
                    bezier_spans_world=runtime_spans,
                ),
                tuple(spans),
            )
        )
    if not roads:
        raise GameMapError("Map must define at least one road")
    return tuple(roads)


def _parse_links(
    doc: dict[str, Any], nodes: dict[str, GameMapNode]
) -> tuple[GameMapDirectLink, ...]:
    links: list[GameMapDirectLink] = []
    ids: set[str] = set()
    for index, value in enumerate(_sequence(doc.get("links"), "links")):
        raw = _mapping(value, f"links[{index}]")
        if set(raw) != {"id", "a", "b"}:
            raise GameMapError(f"links[{index}] requires exactly id, a, and b")
        link_id = str(raw["id"]).strip()
        a, b = str(raw["a"]), str(raw["b"])
        if not link_id or link_id in ids:
            raise GameMapError(f"Link id {link_id!r} is empty or duplicated")
        ids.add(link_id)
        if a not in nodes or b not in nodes or a == b:
            raise GameMapError(
                f"Link {link_id!r} must reference two distinct existing nodes"
            )
        pair = {nodes[a].node_type, nodes[b].node_type}
        if pair not in ({"intersection", "driveway"}, {"driveway", "parking_lot"}):
            raise GameMapError(
                f"Link {link_id!r} must connect intersection-driveway or driveway-parking_lot"
            )
        links.append(GameMapDirectLink(link_id, a, b))
    return tuple(links)


def _parse_attachments(
    doc: dict[str, Any], nodes: dict[str, GameMapNode], roads: set[str]
) -> tuple[GameMapRoadAttachment, ...]:
    attachments: list[GameMapRoadAttachment] = []
    driveways: set[str] = set()
    for index, value in enumerate(
        _sequence(doc.get("road_attachments"), "road_attachments")
    ):
        raw = _mapping(value, f"road_attachments[{index}]")
        if set(raw) != {"driveway", "road"}:
            raise GameMapError(f"road_attachments[{index}] requires driveway and road")
        driveway, road = str(raw["driveway"]), str(raw["road"])
        if (
            driveway not in nodes
            or nodes[driveway].node_type != "driveway"
            or driveway in driveways
        ):
            raise GameMapError(
                f"Road attachment driveway {driveway!r} is invalid or duplicated"
            )
        if road not in roads:
            raise GameMapError(f"Road attachment references unknown road {road!r}")
        driveways.add(driveway)
        attachments.append(GameMapRoadAttachment(driveway, road))
    return tuple(attachments)


def _validate_topology(
    topology: GameMapTopology, profiles: dict[str, _Profile]
) -> None:
    nodes = {node.node_id: node for node in topology.nodes}
    road_degree = {node_id: 0 for node_id in nodes}
    for road in topology.roads:
        road_degree[road.from_node_id] += 1
        road_degree[road.to_node_id] += 1
    link_neighbors: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    for link in topology.direct_links:
        link_neighbors[link.node_a_id].append(link.node_b_id)
        link_neighbors[link.node_b_id].append(link.node_a_id)
    attached = {item.driveway_node_id for item in topology.road_attachments}
    for node in topology.nodes:
        if node.node_type == "cul_de_sac" and road_degree[node.node_id] != 1:
            raise GameMapError(
                f"Cul-de-sac {node.node_id!r} must terminate exactly one road"
            )
        if node.node_type in {"parking_lot", "driveway"} and road_degree[node.node_id]:
            raise GameMapError(
                f"{node.node_type} {node.node_id!r} cannot be an authored road endpoint"
            )
        if node.node_type == "parking_lot" and not link_neighbors[node.node_id]:
            raise GameMapError(
                f"Parking lot {node.node_id!r} must have at least one driveway"
            )
        if node.node_type == "cul_de_sac" and road_degree[node.node_id] == 1:
            road = next(
                road
                for road in topology.roads
                if node.node_id in {road.from_node_id, road.to_node_id}
            )
            minimum_radius = profiles[road.profile_id].surface_width_m * 0.5
            if node.geometry["radius_m"] <= minimum_radius:
                raise GameMapError(
                    f"Cul-de-sac {node.node_id!r} radius_m must exceed "
                    f"half the incident road width ({minimum_radius:.2f} m)"
                )
        if node.node_type == "driveway":
            profile = profiles[str(node.profile_id)]
            if node.geometry["width_m"] < profile.width_m:
                raise GameMapError(
                    f"Driveway {node.node_id!r} width_m must fit all access lanes"
                )
            neighbors = link_neighbors[node.node_id]
            lots = [
                item for item in neighbors if nodes[item].node_type == "parking_lot"
            ]
            sources = [
                item for item in neighbors if nodes[item].node_type == "intersection"
            ]
            if len(lots) != 1:
                raise GameMapError(
                    f"Driveway {node.node_id!r} must serve exactly one parking lot"
                )
            if len(sources) + int(node.node_id in attached) != 1:
                raise GameMapError(
                    f"Driveway {node.node_id!r} needs exactly one intersection link or road attachment"
                )


def _rotation(node: GameMapNode) -> np.ndarray:
    angle = math.radians(node.rotation_deg)
    return np.asarray(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )


def _rectangle(node: GameMapNode, width: float, depth: float) -> np.ndarray:
    local = np.asarray(
        [
            [-width * 0.5, -depth * 0.5],
            [width * 0.5, -depth * 0.5],
            [width * 0.5, depth * 0.5],
            [-width * 0.5, depth * 0.5],
        ]
    )
    return local @ _rotation(node).T + np.asarray([node.x_m, node.y_m])


def _sample_road(spec: _RoadSpec, spacing_m: float) -> np.ndarray:
    road = spec.road
    if not spec.spans_xy:
        raise AssertionError("Straight road sampling requires node positions")
    groups: list[np.ndarray] = []
    for span in spec.spans_xy:
        estimate = sum(
            float(np.linalg.norm(span[index + 1] - span[index])) for index in range(3)
        )
        samples = max(3, int(math.ceil(estimate / spacing_m)) + 1)
        t = np.linspace(0.0, 1.0, samples)[:, None]
        points = (
            (1.0 - t) ** 3 * span[0]
            + 3.0 * (1.0 - t) ** 2 * t * span[1]
            + 3.0 * (1.0 - t) * t**2 * span[2]
            + t**3 * span[3]
        )
        groups.append(points if not groups else points[1:])
    return np.concatenate(groups, axis=0)


def _line_parts(geometry: BaseGeometry) -> list[np.ndarray]:
    if geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        values = [geometry]
    elif geometry.geom_type == "MultiLineString":
        values = list(geometry.geoms)
    elif geometry.geom_type == "GeometryCollection":
        values = [item for item in geometry.geoms if item.geom_type == "LineString"]
    else:
        return []
    return [np.asarray(item.coords, dtype=np.float64) for item in values if item.length]


def _trim_line(
    points: np.ndarray,
    start: Polygon | None,
    end: Polygon | None,
    context: str,
) -> np.ndarray:
    remaining: BaseGeometry = LineString(points)
    if start is not None:
        remaining = remaining.difference(start.buffer(1.0e-5))
    if end is not None:
        remaining = remaining.difference(end.buffer(1.0e-5))
    parts = _line_parts(remaining)
    if not parts:
        raise GameMapError(
            f"{context} is completely contained by its endpoint footprints"
        )
    result = max(parts, key=lambda item: LineString(item).length)
    original_start = points[0]
    if np.linalg.norm(result[0] - original_start) > np.linalg.norm(
        result[-1] - original_start
    ):
        result = result[::-1]
    return result


def _node_polygons(
    topology: GameMapTopology,
    profiles: dict[str, _Profile],
    raw_roads: dict[str, np.ndarray],
) -> dict[str, Polygon]:
    incidences: dict[str, list[tuple[np.ndarray, float]]] = {
        node.node_id: [] for node in topology.nodes
    }
    for road in topology.roads:
        points = raw_roads[road.road_id]
        profile = profiles[road.profile_id]
        incidences[road.from_node_id].append(
            (points[1] - points[0], profile.surface_width_m)
        )
        incidences[road.to_node_id].append(
            (points[-2] - points[-1], profile.surface_width_m)
        )
    nodes = {node.node_id: node for node in topology.nodes}
    for link in topology.direct_links:
        a, b = nodes[link.node_a_id], nodes[link.node_b_id]
        vector = np.asarray([b.x_m - a.x_m, b.y_m - a.y_m])
        driveway = a if a.node_type == "driveway" else b
        width = driveway.geometry["width_m"]
        incidences[a.node_id].append((vector, width))
        incidences[b.node_id].append((-vector, width))
    polygons: dict[str, Polygon] = {}
    for node in topology.nodes:
        center = np.asarray([node.x_m, node.y_m])
        if node.node_type == "intersection":
            incident = incidences[node.node_id]
            max_width = max((width for _vector, width in incident), default=8.0)
            fit_multiplier = 1.5 if len(incident) >= 4 else 1.35
            width = node.geometry.get("width_m", max(8.0, max_width * fit_multiplier))
            depth = node.geometry.get("depth_m", max(8.0, max_width * fit_multiplier))
            reach = max(width, depth) * 0.55
            if "width_m" in node.geometry:
                core: BaseGeometry = Polygon(_rectangle(node, width, depth))
            else:
                core = Point(center).buffer(max_width * 0.5, quad_segs=12)
            arms: list[BaseGeometry] = [core]
            for vector, opening_width in incident:
                length = float(np.linalg.norm(vector))
                if length <= _POSITION_TOLERANCE_M:
                    continue
                direction = vector / length
                arms.append(
                    LineString((center, center + direction * reach)).buffer(
                        opening_width * 0.5,
                        cap_style=2,
                        join_style=2,
                    )
                )
            polygon = unary_union(arms).buffer(0)
        elif node.node_type == "cul_de_sac":
            radius = node.geometry["radius_m"]
            circle = Point(center).buffer(radius, quad_segs=32)
            vector, opening_width = incidences[node.node_id][0]
            direction = vector / max(float(np.linalg.norm(vector)), 1.0e-9)
            normal = np.asarray([-direction[1], direction[0]])
            chord_distance = math.sqrt(radius**2 - (opening_width * 0.5) ** 2)
            extent = radius * 4.0
            clip = Polygon(
                (
                    center - direction * extent + normal * extent,
                    center + direction * chord_distance + normal * extent,
                    center + direction * chord_distance - normal * extent,
                    center - direction * extent - normal * extent,
                )
            )
            polygon = circle.intersection(clip)
        elif node.node_type == "parking_lot":
            polygon = Polygon(
                _rectangle(node, node.geometry["width_m"], node.geometry["depth_m"])
            )
        else:
            polygon = Polygon(_rectangle(node, node.geometry["width_m"], 1.0))
        if not isinstance(polygon, Polygon) or polygon.area <= 0.0:
            raise GameMapError(f"Node {node.node_id!r} has an invalid footprint")
        polygons[node.node_id] = polygon
    return polygons


def _curbs_with_openings(
    polygon: Polygon, openings: list[tuple[np.ndarray, float]]
) -> list[np.ndarray]:
    boundary: BaseGeometry = polygon.exterior
    for point, width in openings:
        boundary = boundary.difference(Point(point).buffer(width * 0.55))
    return [_xyz(part) for part in _line_parts(boundary)]


def _build_linear_lanes(
    element_id: str,
    points: np.ndarray,
    profile: _Profile,
    allows_taxi_stops: bool,
) -> list[_LaneBuild]:
    lanes: list[_LaneBuild] = []
    for index, direction in enumerate(profile.directions):
        left_marking, right_marking = _lane_edge_markings(profile, index, direction)
        offset = (
            len(profile.directions) - 1
        ) * profile.lane_width_m * 0.5 - index * profile.lane_width_m
        center = _offset_polyline(points, offset)
        start_endpoint, end_endpoint = "from", "to"
        if direction == "backward":
            center = center[::-1]
            start_endpoint, end_endpoint = end_endpoint, start_endpoint
        left = _offset_polyline(center, profile.lane_width_m * 0.5)
        right = _offset_polyline(center, -profile.lane_width_m * 0.5)
        roadside = _offset_polyline(
            center, -(profile.lane_width_m * 0.5 + profile.curb_offset_m)
        )
        lanes.append(
            _LaneBuild(
                lane_id=f"{element_id}:lane:{index}",
                element_id=element_id,
                centerline=_xyz(center),
                left_edge=_xyz(left),
                right_edge=_xyz(right),
                roadside_edge=_xyz(roadside),
                speed_limit_mps=profile.speed_limit_mps,
                marking_style=profile.marking_style,
                marking_color=profile.marking_color,
                start_endpoint=start_endpoint,
                end_endpoint=end_endpoint,
                successors=[],
                allows_taxi_stops=allows_taxi_stops,
                left_marking_style=left_marking[0],
                left_marking_color=left_marking[1],
                right_marking_style=right_marking[0],
                right_marking_color=right_marking[1],
            )
        )
    return lanes


def _incidences_for_lanes(
    lanes: list[_LaneBuild], a: str, b: str, edge_ref: str
) -> list[_LaneIncidence]:
    result: list[_LaneIncidence] = []
    for lane in lanes:
        if lane.start_endpoint == "from":
            result.extend(
                (
                    _LaneIncidence(lane, a, "start", f"{edge_ref}:a"),
                    _LaneIncidence(lane, b, "end", f"{edge_ref}:b"),
                )
            )
        else:
            result.extend(
                (
                    _LaneIncidence(lane, b, "start", f"{edge_ref}:b"),
                    _LaneIncidence(lane, a, "end", f"{edge_ref}:a"),
                )
            )
    return result


def _wire_node(
    node: GameMapNode,
    incidences: list[_LaneIncidence],
    lanes: list[_LaneBuild],
    connector_samples: int,
) -> None:
    incoming = [item for item in incidences if item.kind == "end"]
    outgoing = [item for item in incidences if item.kind == "start"]
    connector_count = 0
    for source in incoming:
        for target in outgoing:
            if source.edge_ref == target.edge_ref and node.node_type != "cul_de_sac":
                continue
            if node.node_type in {"driveway", "parking_lot"}:
                source.lane.successors.append(target.lane.lane_id)
                continue
            center = np.asarray([node.x_m, node.y_m, 0.0], dtype=np.float32)
            centerline = _bezier(
                source.lane.centerline[-1],
                center,
                target.lane.centerline[0],
                connector_samples,
            )
            width = float(
                np.linalg.norm(source.lane.left_edge[-1] - source.lane.right_edge[-1])
            )
            left = _xyz(_offset_polyline(centerline[:, :2], width * 0.5))
            right = _xyz(_offset_polyline(centerline[:, :2], -width * 0.5))
            connector_id = f"{node.node_id}:connector:{connector_count}"
            connector_count += 1
            connector = _LaneBuild(
                lane_id=connector_id,
                element_id=node.node_id,
                centerline=centerline,
                left_edge=left,
                right_edge=right,
                roadside_edge=right,
                speed_limit_mps=source.lane.speed_limit_mps,
                marking_style="VIRTUAL",
                marking_color="WHITE",
                start_endpoint="",
                end_endpoint="",
                successors=[target.lane.lane_id],
                allows_taxi_stops=False,
                conditioning_visible=False,
            )
            lanes.append(connector)
            source.lane.successors.append(connector_id)


def _spawn(
    raw: dict[str, Any],
    source_path: Path,
    lane_by_id: dict[str, _LaneBuild],
) -> GameMapSpawn:
    if set(raw) != {"id", "road", "lane", "distance_m", "variants"}:
        raise GameMapError(
            "Spawns require exactly id, road, lane, distance_m, and variants"
        )
    lane_id = f"{str(raw['road'])}:lane:{raw['lane']}"
    if lane_id not in lane_by_id or not lane_by_id[lane_id].allows_taxi_stops:
        raise GameMapError(f"Spawn references unavailable road lane {lane_id!r}")
    lane = lane_by_id[lane_id]
    distance = _positive_float(raw["distance_m"], "spawn.distance_m")
    points = lane.centerline
    lengths = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    total = float(np.sum(lengths))
    if distance >= total:
        raise GameMapError(
            f"Spawn distance {distance} must be below lane length {total}"
        )
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    segment = min(
        int(np.searchsorted(cumulative, distance, side="right") - 1), len(lengths) - 1
    )
    alpha = (distance - cumulative[segment]) / max(float(lengths[segment]), 1.0e-9)
    position = points[segment] + alpha * (points[segment + 1] - points[segment])
    direction = points[segment + 1] - points[segment]
    return GameMapSpawn(
        spawn_id=str(raw["id"]),
        lane_id=lane_id,
        distance_m=distance,
        position_world=position.astype(np.float32),
        yaw_rad=math.atan2(float(direction[1]), float(direction[0])),
        variants=_parse_variants(raw, source_path),
    )


def load_game_map(path: Path) -> ResolvedGameMap:
    """Parse and compile the current node-graph schema into runtime geometry."""
    source_path = Path(path).expanduser().resolve()
    doc = _read_document(source_path)
    if doc.get("schema_version") != _SCHEMA_VERSION:
        raise GameMapError(
            f"Unsupported schema_version {doc.get('schema_version')!r}; expected {_SCHEMA_VERSION}"
        )
    required = {
        "schema_version",
        "id",
        "name",
        "compiler",
        "profiles",
        "nodes",
        "roads",
        "links",
        "road_attachments",
        "spawns",
    }
    if set(doc) != required:
        raise GameMapError(f"Map must contain exactly {sorted(required)}")
    map_id = str(doc["id"]).strip()
    if not map_id:
        raise GameMapError("Map id must not be empty")
    settings = _parse_compiler_settings(doc)
    profiles = _parse_profiles(doc)
    node_values = _parse_nodes(doc, profiles)
    nodes = {node.node_id: node for node in node_values}
    road_specs = _parse_roads(doc, nodes, profiles)
    links = _parse_links(doc, nodes)
    attachments = _parse_attachments(
        doc, nodes, {spec.road.road_id for spec in road_specs}
    )
    adjacency: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    for spec in road_specs:
        adjacency[spec.road.from_node_id].append(f"road:{spec.road.road_id}")
        adjacency[spec.road.to_node_id].append(f"road:{spec.road.road_id}")
    for link in links:
        adjacency[link.node_a_id].append(f"link:{link.link_id}")
        adjacency[link.node_b_id].append(f"link:{link.link_id}")
    for attachment in attachments:
        adjacency[attachment.driveway_node_id].append(
            f"road_attachment:{attachment.road_id}"
        )
    topology = GameMapTopology(
        nodes=node_values,
        roads=tuple(spec.road for spec in road_specs),
        direct_links=links,
        road_attachments=attachments,
        adjacency=tuple(
            (node_id, tuple(sorted(references)))
            for node_id, references in adjacency.items()
        ),
    )
    _validate_topology(topology, profiles)

    raw_roads: dict[str, np.ndarray] = {}
    for spec in road_specs:
        if spec.spans_xy:
            raw_roads[spec.road.road_id] = _sample_road(spec, settings.sample_spacing_m)
        else:
            start = nodes[spec.road.from_node_id]
            end = nodes[spec.road.to_node_id]
            length = math.hypot(end.x_m - start.x_m, end.y_m - start.y_m)
            samples = max(2, int(math.ceil(length / settings.sample_spacing_m)) + 1)
            raw_roads[spec.road.road_id] = np.linspace(
                [start.x_m, start.y_m], [end.x_m, end.y_m], samples
            )
    polygons = _node_polygons(topology, profiles, raw_roads)
    roads_by_id = {spec.road.road_id: spec.road for spec in road_specs}
    for attachment in attachments:
        driveway = nodes[attachment.driveway_node_id]
        road = roads_by_id[attachment.road_id]
        profile = profiles[road.profile_id]
        point = Point(driveway.x_m, driveway.y_m)
        centerline = LineString(raw_roads[road.road_id])
        distance = point.distance(centerline)
        expected = profile.surface_width_m * 0.5
        if abs(distance - expected) > 0.75:
            raise GameMapError(
                f"Driveway {driveway.node_id!r} pose must be at the curb-opening "
                f"center of road {road.road_id!r}; distance={distance:.2f} m, "
                f"expected={expected:.2f} m"
            )
        nearest_distance = centerline.project(point)
        nearest = np.asarray(centerline.interpolate(nearest_distance).coords[0])
        outward = np.asarray([driveway.x_m, driveway.y_m]) - nearest
        outward /= max(float(np.linalg.norm(outward)), 1.0e-9)
        angle = math.radians(driveway.rotation_deg)
        facing = np.asarray([math.cos(angle), math.sin(angle)])
        if float(np.dot(outward, facing)) < math.cos(math.radians(30.0)):
            raise GameMapError(
                f"Driveway {driveway.node_id!r} rotation must point outward from "
                f"road {road.road_id!r}"
            )
    elements: list[GameMapElement] = []
    lanes: list[_LaneBuild] = []
    incidences: dict[str, list[_LaneIncidence]] = {node_id: [] for node_id in nodes}
    collision_groups: list[np.ndarray] = []
    opening_by_node: dict[str, list[tuple[np.ndarray, float]]] = {
        node_id: [] for node_id in nodes
    }
    trimmed_roads: dict[str, np.ndarray] = {}

    attachments_by_road: dict[str, list[GameMapRoadAttachment]] = {}
    for attachment in attachments:
        attachments_by_road.setdefault(attachment.road_id, []).append(attachment)
    for spec in road_specs:
        road = spec.road
        profile = profiles[road.profile_id]
        points = _trim_line(
            raw_roads[road.road_id],
            polygons[road.from_node_id],
            polygons[road.to_node_id],
            f"Road {road.road_id!r}",
        )
        trimmed_roads[road.road_id] = points
        built = _build_linear_lanes(road.road_id, points, profile, True)
        lanes.extend(built)
        for incidence in _incidences_for_lanes(
            built, road.from_node_id, road.to_node_id, f"road:{road.road_id}"
        ):
            incidences[incidence.node_id].append(incidence)
        surface = _surface_for_road(points, profile.surface_width_m)
        elements.append(
            GameMapElement(
                road.road_id,
                "road",
                road.profile_id,
                surface,
            )
        )
        if profile.curb:
            curb_points = {
                side: _offset_polyline(points, side * profile.surface_width_m * 0.5)
                for side in (-1.0, 1.0)
            }
            attachment_sides = {
                attachment.driveway_node_id: min(
                    curb_points,
                    key=lambda side: LineString(curb_points[side]).distance(
                        Point(
                            nodes[attachment.driveway_node_id].x_m,
                            nodes[attachment.driveway_node_id].y_m,
                        )
                    ),
                )
                for attachment in attachments_by_road.get(road.road_id, ())
            }
            for side in (-1.0, 1.0):
                geometry: BaseGeometry = LineString(curb_points[side])
                for attachment in attachments_by_road.get(road.road_id, ()):
                    if attachment_sides[attachment.driveway_node_id] != side:
                        continue
                    driveway = nodes[attachment.driveway_node_id]
                    geometry = geometry.difference(
                        Point(driveway.x_m, driveway.y_m).buffer(
                            driveway.geometry["width_m"] * 0.55
                        )
                    )
                collision_groups.extend(_xyz(part) for part in _line_parts(geometry))
        opening_by_node[road.from_node_id].append((points[0], profile.surface_width_m))
        opening_by_node[road.to_node_id].append((points[-1], profile.surface_width_m))

    for link in links:
        node_a, node_b = nodes[link.node_a_id], nodes[link.node_b_id]
        driveway = node_a if node_a.node_type == "driveway" else node_b
        profile = profiles[str(driveway.profile_id)]
        centerline = np.linspace([node_a.x_m, node_a.y_m], [node_b.x_m, node_b.y_m], 8)
        points = _trim_line(
            centerline,
            None if node_a.node_type == "driveway" else polygons[node_a.node_id],
            None if node_b.node_type == "driveway" else polygons[node_b.node_id],
            f"Link {link.link_id!r}",
        )
        built = _build_linear_lanes(link.link_id, points, profile, False)
        lanes.extend(built)
        for incidence in _incidences_for_lanes(
            built, node_a.node_id, node_b.node_id, f"link:{link.link_id}"
        ):
            incidences[incidence.node_id].append(incidence)
        elements.append(
            GameMapElement(
                link.link_id,
                "implicit_driveway",
                profile.profile_id,
                _surface_for_road(points, driveway.geometry["width_m"]),
            )
        )
        if profile.curb:
            collision_groups.extend(
                (
                    _xyz(
                        _offset_polyline(
                            points, driveway.geometry["width_m"] * side * 0.5
                        )
                    )
                    for side in (-1.0, 1.0)
                )
            )
        opening_by_node[node_a.node_id].append(
            (points[0], driveway.geometry["width_m"])
        )
        opening_by_node[node_b.node_id].append(
            (points[-1], driveway.geometry["width_m"])
        )

    for node in node_values:
        if node.node_type != "parking_lot":
            continue
        openings = opening_by_node[node.node_id]
        for first_index, (first_point, first_width) in enumerate(openings):
            for second_point, second_width in openings[first_index + 1 :]:
                required_separation = 0.45 * (first_width + second_width)
                if (
                    float(np.linalg.norm(first_point - second_point))
                    < required_separation
                ):
                    raise GameMapError(
                        f"Node {node.node_id!r} has overlapping incident openings"
                    )

    for node in node_values:
        if node.node_type == "driveway":
            continue
        polygon = polygons[node.node_id]
        coords = np.asarray(polygon.exterior.coords, dtype=np.float64)
        incident_profile_ids = [
            roads_by_id[reference.removeprefix("road:")].profile_id
            for reference in adjacency[node.node_id]
            if reference.startswith("road:")
        ]
        profile_id = node.profile_id or (
            incident_profile_ids[0] if incident_profile_ids else next(iter(profiles))
        )
        elements.append(
            GameMapElement(
                node.node_id,
                node.node_type,
                profile_id,
                _xyz(coords),
            )
        )
        collision_groups.extend(
            _curbs_with_openings(polygon, opening_by_node[node.node_id])
        )

    # Parking-lot interior aisles are derived access geometry, not authored roads.
    for node in node_values:
        if node.node_type != "parking_lot":
            continue
        profile = profiles[str(node.profile_id)]
        linked_driveways = [
            nodes[link.node_b_id if link.node_a_id == node.node_id else link.node_a_id]
            for link in links
            if node.node_id in {link.node_a_id, link.node_b_id}
        ]
        for index, driveway in enumerate(linked_driveways):
            center = np.asarray([node.x_m, node.y_m])
            driveway_position = np.asarray([driveway.x_m, driveway.y_m])
            direction = center - driveway_position
            direction /= max(float(np.linalg.norm(direction)), 1.0e-9)
            boundary_crossing = polygons[node.node_id].boundary.intersection(
                LineString((driveway_position, center))
            )
            boundary_points = (
                [np.asarray(boundary_crossing.coords[0])]
                if boundary_crossing.geom_type == "Point"
                else [
                    np.asarray(item.coords[0])
                    for item in getattr(boundary_crossing, "geoms", ())
                    if item.geom_type == "Point"
                ]
            )
            if not boundary_points:
                raise GameMapError(
                    f"Could not place parking aisle for lot {node.node_id!r}"
                )
            start = min(
                boundary_points,
                key=lambda point: float(np.linalg.norm(point - driveway_position)),
            )
            end = center + direction * min(node.geometry["depth_m"] * 0.2, 5.0)
            points = np.linspace(start, end, 8)
            element_id = node.node_id if index == 0 else f"{node.node_id}:aisle:{index}"
            built = _build_linear_lanes(element_id, points, profile, True)
            lanes.extend(built)
            for lane in built:
                incidences[node.node_id].append(
                    _LaneIncidence(
                        lane=lane,
                        node_id=node.node_id,
                        kind="start" if lane.start_endpoint == "from" else "end",
                        edge_ref=f"lot:{element_id}",
                    )
                )
            forward = next(
                lane
                for lane, direction_name in zip(built, profile.directions, strict=True)
                if direction_name == "forward"
            )
            backward = next(
                lane
                for lane, direction_name in zip(built, profile.directions, strict=True)
                if direction_name == "backward"
            )
            turnaround = _LaneBuild(
                lane_id=f"{element_id}:turnaround",
                element_id=node.node_id,
                centerline=_bezier(
                    forward.centerline[-1],
                    np.asarray([node.x_m, node.y_m, 0.0], dtype=np.float32),
                    backward.centerline[0],
                    settings.intersection_connector_samples,
                ),
                left_edge=np.empty((0, 3), dtype=np.float32),
                right_edge=np.empty((0, 3), dtype=np.float32),
                roadside_edge=np.empty((0, 3), dtype=np.float32),
                speed_limit_mps=profile.speed_limit_mps,
                marking_style="VIRTUAL",
                marking_color="WHITE",
                start_endpoint="",
                end_endpoint="",
                successors=[backward.lane_id],
                allows_taxi_stops=False,
                conditioning_visible=False,
            )
            turnaround.left_edge = _xyz(
                _offset_polyline(
                    turnaround.centerline[:, :2], profile.lane_width_m * 0.5
                )
            )
            turnaround.right_edge = _xyz(
                _offset_polyline(
                    turnaround.centerline[:, :2], -profile.lane_width_m * 0.5
                )
            )
            turnaround.roadside_edge = turnaround.right_edge
            forward.successors.append(turnaround.lane_id)
            lanes.append(turnaround)

    for attachment in attachments:
        driveway = nodes[attachment.driveway_node_id]
        road_lanes = [lane for lane in lanes if lane.element_id == attachment.road_id]
        driveway_incidence = incidences[driveway.node_id]
        incoming_access = [item for item in driveway_incidence if item.kind == "end"]
        outgoing_access = [item for item in driveway_incidence if item.kind == "start"]
        for road_lane in road_lanes:
            road_lane.successors.extend(item.lane.lane_id for item in outgoing_access)
        for item in incoming_access:
            item.lane.successors.extend(lane.lane_id for lane in road_lanes)

    for node in node_values:
        _wire_node(
            node,
            incidences[node.node_id],
            lanes,
            settings.intersection_connector_samples,
        )

    lane_by_id = {lane.lane_id: lane for lane in lanes}
    spawn_values = _sequence(doc["spawns"], "spawns")
    if not spawn_values:
        raise GameMapError("Map must define at least one spawn")
    spawns = tuple(
        _spawn(_mapping(value, f"spawns[{index}]"), source_path, lane_by_id)
        for index, value in enumerate(spawn_values)
    )
    runtime_lanes = tuple(
        GameMapLane(
            lane_id=lane.lane_id,
            element_id=lane.element_id,
            centerline_world=lane.centerline,
            left_edge_world=lane.left_edge,
            right_edge_world=lane.right_edge,
            roadside_edge_world=lane.roadside_edge,
            speed_limit_mps=lane.speed_limit_mps,
            marking_style=lane.marking_style,
            marking_color=lane.marking_color,
            left_marking_style=lane.left_marking_style or lane.marking_style,
            left_marking_color=lane.left_marking_color or lane.marking_color,
            right_marking_style=lane.right_marking_style or lane.marking_style,
            right_marking_color=lane.right_marking_color or lane.marking_color,
            successor_ids=tuple(dict.fromkeys(lane.successors)),
            allows_taxi_stops=lane.allows_taxi_stops,
            conditioning_visible=lane.conditioning_visible,
        )
        for lane in lanes
    )
    collision_segments = [
        _segments(group) for group in collision_groups if len(group) >= 2
    ]
    collisions = (
        np.concatenate(collision_segments).astype(np.float32)
        if collision_segments
        else np.empty((0, 2, 3), dtype=np.float32)
    )
    all_points = np.concatenate([element.surface_world for element in elements])
    minimum = np.min(all_points[:, :2], axis=0) - settings.ground_margin_m
    maximum = np.max(all_points[:, :2], axis=0) + settings.ground_margin_m
    ground_vertices = np.asarray(
        [
            [minimum[0], minimum[1], 0.0],
            [maximum[0], minimum[1], 0.0],
            [maximum[0], maximum[1], 0.0],
            [minimum[0], maximum[1], 0.0],
        ],
        dtype=np.float32,
    )
    return ResolvedGameMap(
        schema_version=_SCHEMA_VERSION,
        map_id=map_id,
        name=str(doc["name"]),
        source_path=source_path,
        compiler_settings=settings.as_dict(),
        topology=topology,
        lanes=runtime_lanes,
        elements=tuple(elements),
        collision_segments_world=collisions,
        road_marking_polygons_world=(),
        line_markings=(),
        ground_vertices=ground_vertices,
        ground_faces=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
        spawns=spawns,
    )
