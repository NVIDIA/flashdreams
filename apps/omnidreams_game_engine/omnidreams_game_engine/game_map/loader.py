# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Node-graph game-map loading and geometry compilation."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import substring, unary_union

from omnidreams_game_engine.game_map._schema import (
    _SCHEMA_VERSION,
    GameMapError,
    _bezier,
    _finite_float,
    _lane_edge_markings,
    _LaneBuild,
    _mapping,
    _offset_polyline,
    _parse_attribute_values,
    _parse_compiler_settings,
    _parse_map_identity,
    _parse_profiles,
    _parse_variants,
    _positive_float,
    _Profile,
    _read_document,
    _sequence,
    _xyz,
)
from omnidreams_game_engine.game_map.types import (
    GameMapBoundaryAttributes,
    GameMapCurb,
    GameMapDirectLink,
    GameMapElement,
    GameMapLane,
    GameMapLaneDivider,
    GameMapLinearAttributes,
    GameMapNode,
    GameMapRoad,
    GameMapRoadAttachment,
    GameMapSpawn,
    GameMapTopology,
    ResolvedGameMap,
)

_POSITION_TOLERANCE_M = 0.05
_AREA_TOLERANCE_M2 = 1.0e-4
_LINE_TOLERANCE_M = 1.0e-4

_LINEAR_ATTRIBUTE_FIELDS = frozenset(
    {
        "lane_width_m",
        "curb_offset_m",
        "lanes",
        "speed_limit_mps",
        "curb",
        "lane_marking",
        "divider_markings",
    }
)

_NODE_ATTRIBUTE_FIELDS = {
    "intersection": frozenset(
        {
            "curb",
            "intersection_arm_length_m",
            "intersection_width_m",
            "intersection_depth_m",
        }
    ),
    "cul_de_sac": frozenset({"curb", "culdesac_radius_m"}),
    "parking_lot": frozenset(
        {
            *_LINEAR_ATTRIBUTE_FIELDS,
            "parking_lot_width_m",
            "parking_lot_depth_m",
        }
    ),
    "driveway": _LINEAR_ATTRIBUTE_FIELDS,
}


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


@dataclass(frozen=True)
class _Connection:
    """Exact shared opening between two resolved surface elements."""

    connection_id: str
    """Stable topology-derived connection identifier."""

    first_element_id: str
    """First connected surface element."""

    second_element_id: str
    """Second connected surface element."""

    width_m: float
    """Opening width measured along both element boundaries."""

    center_xy: np.ndarray
    """Opening center with shape ``[2]``."""


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


def _resolve_attribute_values(
    raw: dict[str, Any],
    profiles: dict[str, _Profile],
    *,
    structural_fields: set[str],
    allowed_fields: frozenset[str],
    required_fields: frozenset[str],
    context: str,
) -> tuple[str | None, dict[str, object]]:
    """Resolve direct attributes over optional partial profile defaults."""
    profile_id = None if "profile" not in raw else str(raw["profile"]).strip()
    if profile_id is not None and profile_id not in profiles:
        raise GameMapError(f"{context} references unknown profile {profile_id!r}")
    direct_raw = {
        key: value
        for key, value in raw.items()
        if key not in structural_fields and key != "profile"
    }
    unknown = set(direct_raw) - allowed_fields
    if unknown:
        raise GameMapError(f"{context} has unknown attributes {sorted(unknown)}")
    direct = _parse_attribute_values(direct_raw, context)
    values = {
        key: value
        for key, value in (
            profiles[profile_id].values.items() if profile_id is not None else ()
        )
        if key in allowed_fields
    }
    values.update(direct)
    missing = required_fields - set(values)
    if missing:
        raise GameMapError(f"{context} is missing attributes {sorted(missing)}")
    return profile_id, values


def _linear_attributes(
    values: dict[str, object], context: str
) -> GameMapLinearAttributes:
    """Build a complete linear attribute bundle."""
    directions = tuple(str(value) for value in values["lanes"])
    dividers = tuple(
        (str(value[0]), str(value[1])) for value in values["divider_markings"]
    )
    if len(dividers) != len(directions) - 1:
        raise GameMapError(
            f"{context}.divider_markings must contain one entry per adjacent lane pair"
        )
    marking = tuple(str(value) for value in values["lane_marking"])
    return GameMapLinearAttributes(
        curb=bool(values["curb"]),
        lane_width_m=float(values["lane_width_m"]),
        curb_offset_m=float(values["curb_offset_m"]),
        directions=directions,
        speed_limit_mps=float(values["speed_limit_mps"]),
        marking_style=marking[0],
        marking_color=marking[1],
        divider_markings=dividers,
    )


def _parse_nodes(
    doc: dict[str, Any], profiles: dict[str, _Profile]
) -> tuple[GameMapNode, ...]:
    nodes: list[GameMapNode] = []
    ids: set[str] = set()
    for index, value in enumerate(_sequence(doc.get("nodes"), "nodes")):
        raw = _mapping(value, f"nodes[{index}]")
        if not {"id", "type", "pose"} <= set(raw):
            raise GameMapError(f"nodes[{index}] requires id, type, and pose")
        node_type = str(raw.get("type", ""))
        if node_type not in _NODE_ATTRIBUTE_FIELDS:
            raise GameMapError(f"nodes[{index}] has unsupported type {node_type!r}")
        node_id = str(raw["id"]).strip()
        if not node_id or node_id in ids:
            raise GameMapError(f"Node id {node_id!r} is empty or duplicated")
        ids.add(node_id)
        pose = _mapping(raw["pose"], f"node {node_id!r}.pose")
        if set(pose) != {"x_m", "y_m", "rotation_deg"}:
            raise GameMapError(
                f"Node {node_id!r}.pose requires x_m, y_m, and rotation_deg"
            )
        context = f"node {node_id!r}"
        allowed = _NODE_ATTRIBUTE_FIELDS[node_type]
        required = {
            "intersection": frozenset({"curb", "intersection_arm_length_m"}),
            "cul_de_sac": frozenset({"curb", "culdesac_radius_m"}),
            "parking_lot": frozenset(
                {
                    *_LINEAR_ATTRIBUTE_FIELDS,
                    "parking_lot_width_m",
                    "parking_lot_depth_m",
                }
            ),
            "driveway": _LINEAR_ATTRIBUTE_FIELDS,
        }[node_type]
        profile_id, values = _resolve_attribute_values(
            raw,
            profiles,
            structural_fields={"id", "type", "pose"},
            allowed_fields=allowed,
            required_fields=required,
            context=context,
        )
        if node_type == "intersection":
            core = {
                key
                for key in ("intersection_width_m", "intersection_depth_m")
                if key in values
            }
            if core not in (set(), {"intersection_width_m", "intersection_depth_m"}):
                raise GameMapError(
                    f"Intersection {node_id!r} must provide both "
                    "intersection_width_m and intersection_depth_m"
                )
        geometry = {
            key: float(item)
            for key, item in values.items()
            if key
            in {
                "intersection_arm_length_m",
                "intersection_width_m",
                "intersection_depth_m",
                "culdesac_radius_m",
                "parking_lot_width_m",
                "parking_lot_depth_m",
            }
        }
        attributes: GameMapBoundaryAttributes | GameMapLinearAttributes
        attributes = (
            _linear_attributes(values, context)
            if node_type in {"parking_lot", "driveway"}
            else GameMapBoundaryAttributes(curb=bool(values["curb"]))
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
                attributes=attributes,
                geometry=geometry,
            )
        )
    if not nodes:
        raise GameMapError("Map must define at least one node")
    return tuple(nodes)


def _path_point_spans(
    start: np.ndarray, path_points: list[np.ndarray], end: np.ndarray, road_id: str
) -> tuple[np.ndarray, ...]:
    points = np.asarray([start, *path_points, end], dtype=np.float64)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    for index, length in enumerate(segment_lengths):
        if length <= _POSITION_TOLERANCE_M:
            raise GameMapError(
                f"Road {road_id!r}.path creates a degenerate segment at index {index}"
            )

    tangents = np.empty_like(points)
    is_closed = np.linalg.norm(start - end) <= _POSITION_TOLERANCE_M
    if is_closed:
        if len(path_points) < 2:
            raise GameMapError(
                f"Self-loop road {road_id!r} requires at least two path points"
            )
        loop_tangent = 0.5 * (points[1] - points[-2])
        tangents[0] = loop_tangent
        tangents[-1] = loop_tangent
    else:
        tangents[0] = points[1] - points[0]
        tangents[-1] = points[-1] - points[-2]
    if len(points) > 2:
        tangents[1:-1] = 0.5 * (points[2:] - points[:-2])

    for index, tangent in enumerate(tangents):
        if np.linalg.norm(tangent) <= _POSITION_TOLERANCE_M:
            raise GameMapError(
                f"Road {road_id!r}.path creates a degenerate tangent at point {index}"
            )

    spans: list[np.ndarray] = []
    for index in range(len(points) - 1):
        control_1 = points[index] + tangents[index] / 3.0
        control_2 = points[index + 1] - tangents[index + 1] / 3.0
        spans.append(
            np.vstack((points[index], control_1, control_2, points[index + 1]))
        )
    return tuple(spans)


def _bezier_spans(
    value: object, start: np.ndarray, end: np.ndarray, road_id: str
) -> tuple[np.ndarray, ...]:
    bezier = _sequence(value, f"road {road_id!r}.bezier")
    if not bezier:
        raise GameMapError(f"Road {road_id!r}.bezier must not be empty")
    spans: list[np.ndarray] = []
    cursor = start
    for span_index, span_value in enumerate(bezier):
        span = _mapping(span_value, f"road {road_id!r}.bezier[{span_index}]")
        if set(span) != {"control_points", "end"}:
            raise GameMapError(
                f"Road {road_id!r} Bezier spans require control_points and end"
            )
        controls = _sequence(
            span["control_points"],
            f"road {road_id!r}.bezier[{span_index}].control_points",
        )
        if len(controls) != 2:
            raise GameMapError(
                f"Road {road_id!r} Bezier spans require exactly two control points"
            )
        control_1 = _point(
            controls[0],
            f"road {road_id!r}.bezier[{span_index}].control_points[0]",
        )
        control_2 = _point(
            controls[1],
            f"road {road_id!r}.bezier[{span_index}].control_points[1]",
        )
        span_end = _point(
            span["end"], f"road {road_id!r}.bezier[{span_index}].end"
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
            f"Road {road_id!r} final Bezier endpoint must match its to-node pose "
            f"within {_POSITION_TOLERANCE_M:g}m"
        )
    return tuple(spans)


def _path_spans(
    value: object, start: np.ndarray, end: np.ndarray, road_id: str
) -> tuple[np.ndarray, ...]:
    path = _sequence(value, f"road {road_id!r}.path")
    if not path:
        raise GameMapError(f"Road {road_id!r}.path must not be empty")
    path_points: list[np.ndarray] = []
    for index, item in enumerate(path):
        raw = _mapping(item, f"road {road_id!r}.path[{index}]")
        if "control_points" in raw or "end" in raw:
            raise GameMapError(
                f"Road {road_id!r}.path accepts path points only; "
                "put explicit spans under bezier"
            )
        path_points.append(_point(raw, f"road {road_id!r}.path[{index}]"))
    return _path_point_spans(start, path_points, end, road_id)


def _parse_roads(
    doc: dict[str, Any], nodes: dict[str, GameMapNode], profiles: dict[str, _Profile]
) -> tuple[_RoadSpec, ...]:
    roads: list[_RoadSpec] = []
    ids: set[str] = set()
    for index, value in enumerate(_sequence(doc.get("roads"), "roads")):
        raw = _mapping(value, f"roads[{index}]")
        if not {"id", "from", "to"} <= set(raw):
            raise GameMapError(f"roads[{index}] requires id, from, and to")
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
        context = f"road {road_id!r}"
        profile_id, values = _resolve_attribute_values(
            raw,
            profiles,
            structural_fields={"id", "from", "to", "path", "bezier"},
            allowed_fields=_LINEAR_ATTRIBUTE_FIELDS,
            required_fields=_LINEAR_ATTRIBUTE_FIELDS,
            context=context,
        )
        attributes = _linear_attributes(values, context)
        start = np.asarray([nodes[from_id].x_m, nodes[from_id].y_m])
        end = np.asarray([nodes[to_id].x_m, nodes[to_id].y_m])
        path_spans: tuple[np.ndarray, ...] = ()
        if "path" in raw:
            path_spans = _path_spans(raw["path"], start, end, road_id)
        bezier_spans: tuple[np.ndarray, ...] = ()
        if "bezier" in raw:
            bezier_spans = _bezier_spans(raw["bezier"], start, end, road_id)
        if "bezier" in raw:
            spans = bezier_spans
        elif "path" in raw:
            spans = path_spans
        else:
            spans = ()
            if np.linalg.norm(start - end) <= _POSITION_TOLERANCE_M:
                raise GameMapError(
                    f"Self-loop road {road_id!r} requires path or bezier"
                )
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
                    attributes=attributes,
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


def _validate_element_ids(topology: GameMapTopology) -> None:
    owners: dict[str, str] = {}
    identifiers = (
        *((node.node_id, "node") for node in topology.nodes),
        *((road.road_id, "road") for road in topology.roads),
        *((link.link_id, "link") for link in topology.direct_links),
    )
    for identifier, kind in identifiers:
        previous = owners.setdefault(identifier, kind)
        if previous != kind:
            raise GameMapError(
                f"Map element id {identifier!r} is shared by a {previous} and {kind}"
            )


def _validate_topology(topology: GameMapTopology) -> None:
    _validate_element_ids(topology)
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
            minimum_radius = road.attributes.surface_width_m * 0.5
            if node.geometry["culdesac_radius_m"] <= minimum_radius:
                raise GameMapError(
                    f"Cul-de-sac {node.node_id!r} culdesac_radius_m must exceed "
                    f"half the incident road width ({minimum_radius:.2f} m)"
                )
        if node.node_type == "driveway":
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


def _polyline_prefix(points: np.ndarray, length_m: float) -> np.ndarray:
    """Return the exact prefix of a polyline through ``length_m``."""
    line = LineString(points)
    prefix = substring(line, 0.0, min(length_m, line.length))
    if prefix.geom_type != "LineString" or prefix.length <= 0.0:
        raise GameMapError("Intersection arm path is degenerate")
    return np.asarray(prefix.coords, dtype=np.float64)


def _node_polygons(
    topology: GameMapTopology,
    raw_roads: dict[str, np.ndarray],
) -> dict[str, Polygon]:
    incidences: dict[str, list[tuple[np.ndarray, float]]] = {
        node.node_id: [] for node in topology.nodes
    }
    for road in topology.roads:
        points = raw_roads[road.road_id]
        width = road.attributes.surface_width_m
        incidences[road.from_node_id].append((points, width))
        incidences[road.to_node_id].append((points[::-1], width))
    nodes = {node.node_id: node for node in topology.nodes}
    for link in topology.direct_links:
        a, b = nodes[link.node_a_id], nodes[link.node_b_id]
        path = np.asarray([[a.x_m, a.y_m], [b.x_m, b.y_m]])
        driveway = a if a.node_type == "driveway" else b
        assert isinstance(driveway.attributes, GameMapLinearAttributes)
        width = driveway.attributes.surface_width_m
        incidences[a.node_id].append((path, width))
        incidences[b.node_id].append((path[::-1], width))
    attached_roads = {
        attachment.driveway_node_id: attachment.road_id
        for attachment in topology.road_attachments
    }
    roads = {road.road_id: road for road in topology.roads}
    polygons: dict[str, Polygon] = {}
    for node in topology.nodes:
        center = np.asarray([node.x_m, node.y_m])
        if node.node_type == "driveway":
            arms = []
            for path, opening_width in incidences[node.node_id]:
                arm_path = _polyline_prefix(path, opening_width * 0.25)
                direction = arm_path[1] - arm_path[0]
                direction /= max(float(np.linalg.norm(direction)), 1.0e-9)
                arm_path = np.vstack(
                    (arm_path[0] - direction * opening_width * 0.25, arm_path)
                )
                arms.append(
                    LineString(arm_path).buffer(
                        opening_width * 0.5,
                        cap_style=2,
                        join_style=2,
                    )
                )
            if not arms:
                raise GameMapError(
                    f"Driveway {node.node_id!r} must have an access link"
                )
            polygon = unary_union(arms).buffer(0)
            attached_road_id = attached_roads.get(node.node_id)
            if attached_road_id is not None:
                road = roads[attached_road_id]
                road_surface = LineString(raw_roads[attached_road_id]).buffer(
                    road.attributes.surface_width_m * 0.5,
                    cap_style=2,
                    join_style=2,
                )
                polygon = polygon.difference(road_surface)
        elif node.node_type == "intersection":
            incident = incidences[node.node_id]
            if "intersection_width_m" in node.geometry:
                core: BaseGeometry = Polygon(
                    _rectangle(
                        node,
                        node.geometry["intersection_width_m"],
                        node.geometry["intersection_depth_m"],
                    )
                )
                arms: list[BaseGeometry] = [core]
            else:
                arms = []
            for path, opening_width in incident:
                arm_path = _polyline_prefix(
                    path, node.geometry["intersection_arm_length_m"]
                )
                direction = arm_path[1] - arm_path[0]
                direction /= max(float(np.linalg.norm(direction)), 1.0e-9)
                arm_path = np.vstack(
                    (arm_path[0] - direction * opening_width * 0.5, arm_path)
                )
                arms.append(
                    LineString(arm_path).buffer(
                        opening_width * 0.5,
                        cap_style=2,
                        join_style=2,
                    )
                )
            if not arms:
                raise GameMapError(
                    f"Intersection {node.node_id!r} must have at least one incidence"
                )
            polygon = unary_union(arms).buffer(0)
        elif node.node_type == "cul_de_sac":
            radius = node.geometry["culdesac_radius_m"]
            circle = Point(center).buffer(radius, quad_segs=32)
            path, opening_width = incidences[node.node_id][0]
            vector = path[1] - path[0]
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
                _rectangle(
                    node,
                    node.geometry["parking_lot_width_m"],
                    node.geometry["parking_lot_depth_m"],
                )
            )
        else:
            raise AssertionError(f"Unsupported footprint node {node.node_type!r}")
        if polygon.geom_type == "MultiPolygon":
            parts = [part for part in polygon.geoms if part.area > _AREA_TOLERANCE_M2]
            if len(parts) == 1:
                polygon = parts[0]
        if not isinstance(polygon, Polygon) or polygon.area <= 0.0:
            raise GameMapError(f"Node {node.node_id!r} has an invalid footprint")
        polygons[node.node_id] = polygon
    return polygons


def _corridor_polygon(
    points: np.ndarray,
    width_m: float,
    excluded: tuple[Polygon, ...],
    context: str,
) -> Polygon:
    """Build a paved corridor outside its connected element footprints."""
    geometry: BaseGeometry = LineString(points).buffer(
        width_m * 0.5,
        cap_style=2,
        join_style=2,
    )
    for polygon in excluded:
        geometry = geometry.difference(polygon)
    if geometry.geom_type in {"MultiPolygon", "GeometryCollection"}:
        parts = [
            part
            for part in geometry.geoms
            if isinstance(part, Polygon) and part.area > _AREA_TOLERANCE_M2
        ]
        if len(parts) == 1:
            geometry = parts[0]
        elif parts:
            raise GameMapError(
                f"{context} is split into surfaces with areas "
                f"{[round(part.area, 6) for part in parts]}"
            )
    if not isinstance(geometry, Polygon) or geometry.area <= _AREA_TOLERANCE_M2:
        raise GameMapError(
            f"{context} does not produce one connected surface "
            f"({geometry.geom_type}, area={geometry.area:.6f})"
        )
    if geometry.interiors:
        raise GameMapError(f"{context} surface must not contain holes")
    return geometry


def _surface_array(polygon: Polygon) -> np.ndarray:
    """Convert a resolved surface polygon to world coordinates."""
    points = np.asarray(polygon.exterior.coords, dtype=np.float64)
    return np.column_stack((points, np.zeros(len(points), dtype=np.float64)))


def _boundary_window(
    polygon: Polygon,
    center_xy: np.ndarray,
    width_m: float,
) -> BaseGeometry:
    """Return an exact-width interval centered on a polygon boundary."""
    boundary = LineString(polygon.exterior.coords)
    distance = boundary.project(Point(center_xy))
    start = distance - width_m * 0.5
    end = distance + width_m * 0.5
    parts: list[BaseGeometry] = []
    if start < 0.0:
        parts.append(substring(boundary, boundary.length + start, boundary.length))
        start = 0.0
    if end > boundary.length:
        parts.append(substring(boundary, 0.0, end - boundary.length))
        end = boundary.length
    parts.append(substring(boundary, start, end))
    lines = [part for part in parts if part.length > _LINE_TOLERANCE_M]
    return lines[0] if len(lines) == 1 else unary_union(lines)


def _curbs_for_elements(
    elements: list[GameMapElement],
    connections: list[_Connection],
) -> list[GameMapElement]:
    """Validate element contacts and attach connection-aware curb polylines."""
    polygons = {
        element.element_id: Polygon(element.surface_world[:, :2])
        for element in elements
    }
    for element_id, polygon in polygons.items():
        if not polygon.is_valid:
            raise GameMapError(f"Element {element_id!r} has an invalid surface")
    connection_groups: dict[tuple[str, str], list[_Connection]] = {}
    for connection in connections:
        pair = tuple(
            sorted((connection.first_element_id, connection.second_element_id))
        )
        connection_groups.setdefault(pair, []).append(connection)

    openings: dict[str, list[BaseGeometry]] = {
        element.element_id: [] for element in elements
    }
    element_ids = [element.element_id for element in elements]
    for first_index, first_id in enumerate(element_ids):
        first = polygons[first_id]
        for second_id in element_ids[first_index + 1 :]:
            second = polygons[second_id]
            pair = tuple(sorted((first_id, second_id)))
            overlap_area = first.intersection(second).area
            declared = connection_groups.get(pair)
            if declared is None:
                if overlap_area > _AREA_TOLERANCE_M2:
                    raise GameMapError(
                        f"Unrelated elements {first_id!r} and {second_id!r} overlap"
                    )
                if (
                    first.boundary.intersection(second.boundary).length
                    > _LINE_TOLERANCE_M
                ):
                    raise GameMapError(
                        f"Unrelated elements {first_id!r} and {second_id!r} "
                        "share a boundary"
                    )
                continue
            if overlap_area > _AREA_TOLERANCE_M2:
                labels = ", ".join(item.connection_id for item in declared)
                raise GameMapError(
                    f"Connected elements {first_id!r} and {second_id!r} overlap "
                    f"at {labels}"
                )
            for connection in declared:
                first_opening = _boundary_window(
                    first, connection.center_xy, connection.width_m
                )
                second_opening = _boundary_window(
                    second, connection.center_xy, connection.width_m
                )
                separation = first_opening.hausdorff_distance(second_opening)
                if separation > _POSITION_TOLERANCE_M:
                    raise GameMapError(
                        f"Connection {connection.connection_id!r} between "
                        f"{first_id!r} and {second_id!r} has mismatched openings "
                        f"({separation:.3f} m apart)"
                    )
            openings[first_id].append(
                first.boundary.intersection(second.boundary.buffer(_LINE_TOLERANCE_M))
            )
            openings[second_id].append(
                second.boundary.intersection(first.boundary.buffer(_LINE_TOLERANCE_M))
            )

    resolved: list[GameMapElement] = []
    for element in elements:
        if not element.attributes.curb:
            resolved.append(element)
            continue
        boundary: BaseGeometry = polygons[element.element_id].boundary
        for opening in openings[element.element_id]:
            boundary = boundary.difference(
                opening.buffer(
                    _LINE_TOLERANCE_M,
                    cap_style=2,
                    join_style=2,
                )
            )
        parts = sorted(
            _line_parts(boundary),
            key=lambda points: (
                round(float(np.min(points[:, 0])), 6),
                round(float(np.min(points[:, 1])), 6),
                round(float(np.max(points[:, 0])), 6),
                round(float(np.max(points[:, 1])), 6),
            ),
        )
        curbs = tuple(
            GameMapCurb(
                curb_id=f"{element.element_id}:curb:{index}",
                polyline_world=_xyz(points),
            )
            for index, points in enumerate(parts)
            if len(points) >= 2
        )
        resolved.append(replace(element, curbs=curbs))
    return resolved


def _build_linear_lanes(
    element_id: str,
    points: np.ndarray,
    attributes: GameMapLinearAttributes,
    allows_taxi_stops: bool,
) -> list[_LaneBuild]:
    lanes: list[_LaneBuild] = []
    for index, direction in enumerate(attributes.directions):
        left_marking, right_marking = _lane_edge_markings(attributes, index, direction)
        offset = (
            len(attributes.directions) - 1
        ) * attributes.lane_width_m * 0.5 - index * attributes.lane_width_m
        center = _offset_polyline(points, offset)
        start_endpoint, end_endpoint = "from", "to"
        if direction == "backward":
            center = center[::-1]
            start_endpoint, end_endpoint = end_endpoint, start_endpoint
        left = _offset_polyline(center, attributes.lane_width_m * 0.5)
        right = _offset_polyline(center, -attributes.lane_width_m * 0.5)
        roadside = _offset_polyline(
            center,
            -(attributes.lane_width_m * 0.5 + attributes.curb_offset_m),
        )
        lanes.append(
            _LaneBuild(
                lane_id=f"{element_id}:lane:{index}",
                element_id=element_id,
                centerline=_xyz(center),
                left_edge=_xyz(left),
                right_edge=_xyz(right),
                roadside_edge=_xyz(roadside),
                speed_limit_mps=attributes.speed_limit_mps,
                marking_style=attributes.marking_style,
                marking_color=attributes.marking_color,
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


def _build_lane_dividers(
    lanes: list[_LaneBuild], attributes: GameMapLinearAttributes
) -> list[GameMapLaneDivider]:
    """Resolve authored profile dividers without rediscovering them geometrically."""
    dividers: list[GameMapLaneDivider] = []
    for index, (style, color) in enumerate(attributes.divider_markings):
        if style == "VIRTUAL":
            continue
        first = lanes[index]
        second = lanes[index + 1]
        first_side = "right" if attributes.directions[index] == "forward" else "left"
        second_side = (
            "left" if attributes.directions[index + 1] == "forward" else "right"
        )
        first_edge = first.right_edge if first_side == "right" else first.left_edge
        second_edge = second.right_edge if second_side == "right" else second.left_edge
        if first_edge.shape != second_edge.shape:
            raise GameMapError(
                f"Adjacent lanes in {first.element_id!r} have mismatched samples"
            )
        direct_error = float(np.linalg.norm(first_edge - second_edge, axis=1).max())
        reverse_error = float(
            np.linalg.norm(first_edge - second_edge[::-1], axis=1).max()
        )
        aligned_second = (
            second_edge if direct_error <= reverse_error else second_edge[::-1]
        )
        lane_edges = ((first.lane_id, first_side), (second.lane_id, second_side))
        dividers.append(
            GameMapLaneDivider(
                divider_id=":".join(sorted((first.lane_id, second.lane_id))),
                lane_edges=lane_edges,
                polyline_world=np.mean((first_edge, aligned_second), axis=0).astype(
                    np.float32
                ),
                style=style,
                color=color,
            )
        )
    return dividers


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
    spawn_id = str(raw["id"]).strip()
    if not spawn_id:
        raise GameMapError("Spawn id must not be empty")
    lane_index = raw["lane"]
    if type(lane_index) is not int or lane_index < 0:
        raise GameMapError("spawn.lane must be a nonnegative integer")
    lane_id = f"{str(raw['road'])}:lane:{lane_index}"
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
        spawn_id=spawn_id,
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
    map_id, map_name = _parse_map_identity(doc)
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
    _validate_topology(topology)

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
    polygons = _node_polygons(topology, raw_roads)
    roads_by_id = {spec.road.road_id: spec.road for spec in road_specs}
    for attachment in attachments:
        driveway = nodes[attachment.driveway_node_id]
        road = roads_by_id[attachment.road_id]
        point = Point(driveway.x_m, driveway.y_m)
        centerline = LineString(raw_roads[road.road_id])
        distance = point.distance(centerline)
        expected = road.attributes.surface_width_m * 0.5
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
    connections: list[_Connection] = []
    lanes: list[_LaneBuild] = []
    lane_dividers: list[GameMapLaneDivider] = []
    incidences: dict[str, list[_LaneIncidence]] = {node_id: [] for node_id in nodes}
    parking_openings: dict[str, list[tuple[np.ndarray, float]]] = {
        node.node_id: [] for node in node_values if node.node_type == "parking_lot"
    }

    for spec in road_specs:
        road = spec.road
        attributes = road.attributes
        points = _trim_line(
            raw_roads[road.road_id],
            polygons[road.from_node_id],
            polygons[road.to_node_id],
            f"Road {road.road_id!r}",
        )
        built = _build_linear_lanes(road.road_id, points, attributes, True)
        lanes.extend(built)
        lane_dividers.extend(_build_lane_dividers(built, attributes))
        for incidence in _incidences_for_lanes(
            built, road.from_node_id, road.to_node_id, f"road:{road.road_id}"
        ):
            incidences[incidence.node_id].append(incidence)
        excluded = tuple(
            {
                node_id: polygons[node_id]
                for node_id in (road.from_node_id, road.to_node_id)
            }.values()
        )
        surface = _corridor_polygon(
            raw_roads[road.road_id],
            attributes.surface_width_m,
            excluded,
            f"Road {road.road_id!r}",
        )
        elements.append(
            GameMapElement(
                element_id=road.road_id,
                element_type="road",
                profile_id=road.profile_id,
                attributes=attributes,
                surface_world=_surface_array(surface),
                curbs=(),
            )
        )
        connections.extend(
            _Connection(
                connection_id=f"road:{road.road_id}:{endpoint}",
                first_element_id=road.road_id,
                second_element_id=node_id,
                width_m=attributes.surface_width_m,
                center_xy=center,
            )
            for endpoint, node_id, center in (
                ("from", road.from_node_id, points[0]),
                ("to", road.to_node_id, points[-1]),
            )
        )

    links_by_driveway: dict[str, list[str]] = {}
    for link in links:
        node_a, node_b = nodes[link.node_a_id], nodes[link.node_b_id]
        driveway = node_a if node_a.node_type == "driveway" else node_b
        assert isinstance(driveway.attributes, GameMapLinearAttributes)
        attributes = driveway.attributes
        links_by_driveway.setdefault(driveway.node_id, []).append(link.link_id)
        centerline = np.linspace([node_a.x_m, node_a.y_m], [node_b.x_m, node_b.y_m], 8)
        points = _trim_line(
            centerline,
            polygons[node_a.node_id],
            polygons[node_b.node_id],
            f"Link {link.link_id!r}",
        )
        built = _build_linear_lanes(link.link_id, points, attributes, False)
        lanes.extend(built)
        for incidence in _incidences_for_lanes(
            built, node_a.node_id, node_b.node_id, f"link:{link.link_id}"
        ):
            incidences[incidence.node_id].append(incidence)
        excluded = tuple(polygons[node.node_id] for node in (node_a, node_b))
        surface = _corridor_polygon(
            centerline,
            attributes.surface_width_m,
            excluded,
            f"Link {link.link_id!r}",
        )
        elements.append(
            GameMapElement(
                element_id=link.link_id,
                element_type="implicit_driveway",
                profile_id=driveway.profile_id,
                attributes=attributes,
                surface_world=_surface_array(surface),
                curbs=(),
            )
        )
        for endpoint in (node_a, node_b):
            connections.append(
                _Connection(
                    connection_id=f"link:{link.link_id}:{endpoint.node_id}",
                    first_element_id=link.link_id,
                    second_element_id=endpoint.node_id,
                    width_m=attributes.surface_width_m,
                    center_xy=(
                        points[0] if endpoint.node_id == node_a.node_id else points[-1]
                    ),
                )
            )
        for endpoint, point in ((node_a, points[0]), (node_b, points[-1])):
            if endpoint.node_id in parking_openings:
                parking_openings[endpoint.node_id].append(
                    (point, attributes.surface_width_m)
                )

    attachments_by_driveway = {
        attachment.driveway_node_id: attachment for attachment in attachments
    }
    for driveway_id in links_by_driveway:
        driveway = nodes[driveway_id]
        assert isinstance(driveway.attributes, GameMapLinearAttributes)
        if driveway_id in attachments_by_driveway:
            attachment = attachments_by_driveway[driveway_id]
            connections.append(
                _Connection(
                    connection_id=f"road_attachment:{driveway_id}",
                    first_element_id=attachment.road_id,
                    second_element_id=driveway_id,
                    width_m=driveway.attributes.surface_width_m,
                    center_xy=np.asarray([driveway.x_m, driveway.y_m]),
                )
            )

    for node in node_values:
        if node.node_type != "parking_lot":
            continue
        openings = parking_openings[node.node_id]
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
        polygon = polygons[node.node_id]
        elements.append(
            GameMapElement(
                element_id=node.node_id,
                element_type=node.node_type,
                profile_id=node.profile_id,
                attributes=node.attributes,
                surface_world=_surface_array(polygon),
                curbs=(),
            )
        )

    # Parking-lot interior aisles are derived access geometry, not authored roads.
    for node in node_values:
        if node.node_type != "parking_lot":
            continue
        assert isinstance(node.attributes, GameMapLinearAttributes)
        attributes = node.attributes
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
            end = center + direction * min(
                node.geometry["parking_lot_depth_m"] * 0.2, 5.0
            )
            points = np.linspace(start, end, 8)
            element_id = node.node_id if index == 0 else f"{node.node_id}:aisle:{index}"
            built = _build_linear_lanes(element_id, points, attributes, True)
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
                for lane, direction_name in zip(
                    built, attributes.directions, strict=True
                )
                if direction_name == "forward"
            )
            backward = next(
                lane
                for lane, direction_name in zip(
                    built, attributes.directions, strict=True
                )
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
                speed_limit_mps=attributes.speed_limit_mps,
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
                    turnaround.centerline[:, :2], attributes.lane_width_m * 0.5
                )
            )
            turnaround.right_edge = _xyz(
                _offset_polyline(
                    turnaround.centerline[:, :2], -attributes.lane_width_m * 0.5
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
    spawn_ids = [spawn.spawn_id for spawn in spawns]
    if len(set(spawn_ids)) != len(spawn_ids):
        raise GameMapError("Spawn ids must be non-empty and unique")
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
    elements = _curbs_for_elements(elements, connections)
    elements = [
        replace(
            element,
            surface_world=np.asarray(element.surface_world, dtype=np.float32),
        )
        for element in elements
    ]
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
        name=map_name,
        source_path=source_path,
        compiler_settings=settings.as_dict(),
        topology=topology,
        lanes=runtime_lanes,
        elements=tuple(elements),
        road_marking_polygons_world=(),
        lane_dividers=tuple(lane_dividers),
        line_markings=(),
        ground_vertices=ground_vertices,
        ground_faces=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
        spawns=spawns,
    )
