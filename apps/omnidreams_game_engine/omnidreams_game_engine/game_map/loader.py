# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Node-graph game-map loading and geometry compilation."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from shapely import is_valid_reason
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import polygonize, substring, unary_union

from omnidreams_game_engine.game_map._schema import (
    _SCHEMA_VERSION,
    GameMapError,
    _bezier,
    _finite_float,
    _lane_edge_markings,
    _LaneBuild,
    _mapping,
    _nonnegative_float,
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
from omnidreams_game_engine.game_map.traffic import compile_traffic
from omnidreams_game_engine.game_map.types import (
    GameMapBoundaryAttributes,
    GameMapCurb,
    GameMapElement,
    GameMapLane,
    GameMapLaneDivider,
    GameMapLinearAttributes,
    GameMapNode,
    GameMapParkingAccess,
    GameMapRaceCourse,
    GameMapRoad,
    GameMapRoadBoundary,
    GameMapSpawn,
    GameMapTopology,
    ResolvedGameMap,
)

_POSITION_TOLERANCE_M = 0.05
_AREA_TOLERANCE_M2 = 1.0e-4
_LINE_TOLERANCE_M = 1.0e-4
_OPENING_TOLERANCE_M = 1.0e-2
"""Maximum numeric drift when matching separately materialized seam polylines."""

_BOUNDARY_CLEARANCE_M = 1.0e-2
"""Outward clearance that keeps sampled roadside corners behind their openings."""

_INTERSECTION_TURN_HANDLE_RATIO = 0.4
"""Cubic handle length as a fraction of the connector endpoint chord."""

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
_REQUIRED_LINEAR_ATTRIBUTE_FIELDS = _LINEAR_ATTRIBUTE_FIELDS - {"curb"}

_NODE_ATTRIBUTE_FIELDS = {
    "intersection": frozenset({"curb"}),
    "road_joint": frozenset(),
    "cul_de_sac": frozenset({"curb", "culdesac_radius_m"}),
    "parking_lot": frozenset(),
    "driveway": frozenset(),
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

    opening_xy: np.ndarray
    """Shared boundary polyline with shape ``[N, 2]``."""


@dataclass(frozen=True)
class _RoadArm:
    """One road cross-section oriented outward from an incident node."""

    node_id: str
    """Identifier of the node that owns the arm."""

    road: GameMapRoad
    """Authored road incident to the node."""

    path_xy: np.ndarray
    """Sampled road centerline oriented outward from the node."""

    attributes: GameMapLinearAttributes
    """Authored cross-section oriented outward from the node."""


@dataclass(frozen=True)
class _ArmTransition:
    """One node-owned transition from a local to an authored cross-section."""

    arm: _RoadArm
    """Road arm whose authored cross-section differs from the node."""

    local_attributes: GameMapLinearAttributes
    """Dominant cross-section used at the node opening."""

    length_m: float
    """Distance over which the node cross-section becomes the road cross-section."""


@dataclass(frozen=True)
class _TransitionGeometry:
    """Resolved centerline and cross-sections for one tapered node arm."""

    transition: _ArmTransition
    """Semantic transition resolved for this arm."""

    path_xy: np.ndarray
    """Sampled centerline from the node opening to the authored road."""


@dataclass(frozen=True)
class _BoundaryArmGeometry:
    """Boundary rails from a node core to one connected surface."""

    reference_id: str
    """Identifier of the road or inferred parking access."""

    left_xy: np.ndarray
    """Left roadside rail oriented from the core to the opening."""

    right_xy: np.ndarray
    """Right roadside rail oriented from the core to the opening."""


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
    if "curb" in allowed_fields:
        values.setdefault("curb", True)
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
        node_type = str(raw.get("type", ""))
        if node_type not in _NODE_ATTRIBUTE_FIELDS:
            raise GameMapError(f"nodes[{index}] has unsupported type {node_type!r}")
        node_id = str(raw["id"]).strip()
        if not node_id or node_id in ids:
            raise GameMapError(f"Node id {node_id!r} is empty or duplicated")
        ids.add(node_id)
        context = f"node {node_id!r}"
        if node_type == "parking_lot":
            expected = {
                "id",
                "type",
                "vertices",
                "connected_to",
                "opening_vertex",
            }
            if set(raw) != expected:
                raise GameMapError(
                    f"{context} requires exactly id, type, vertices, "
                    "connected_to, and opening_vertex"
                )
            vertices = tuple(
                tuple(
                    float(item)
                    for item in _point(value, f"{context}.vertices[{vertex_index}]")
                )
                for vertex_index, value in enumerate(
                    _sequence(raw["vertices"], f"{context}.vertices")
                )
            )
            if len(vertices) < 3:
                raise GameMapError(f"{context}.vertices requires at least three points")
            polygon = Polygon(vertices)
            if not polygon.is_valid or polygon.area <= _AREA_TOLERANCE_M2:
                raise GameMapError(f"{context}.vertices must form a simple polygon")
            if polygon.exterior.is_ccw:
                raise GameMapError(f"{context}.vertices must be clockwise")
            if len(set(vertices)) != len(vertices):
                raise GameMapError(f"{context}.vertices contains duplicate points")
            if not str(raw["connected_to"]).strip():
                raise GameMapError(f"{context}.connected_to must not be empty")
            opening_value = raw["opening_vertex"]
            if type(opening_value) is not int:
                raise GameMapError(f"{context}.opening_vertex must be an integer")
            if opening_value < 1 or opening_value > len(vertices):
                raise GameMapError(
                    f"{context}.opening_vertex must be between 1 and {len(vertices)}"
                )
            centroid = polygon.centroid
            nodes.append(
                GameMapNode(
                    node_id=node_id,
                    node_type=node_type,
                    x_m=float(centroid.x),
                    y_m=float(centroid.y),
                    profile_id=None,
                    attributes=GameMapBoundaryAttributes(curb=True),
                    geometry={},
                    polygon_vertices_xy=vertices,
                )
            )
            continue
        if not {"id", "type", "pose"} <= set(raw):
            raise GameMapError(f"nodes[{index}] requires id, type, and pose")
        pose = _mapping(raw["pose"], f"node {node_id!r}.pose")
        if set(pose) != {"x_m", "y_m"}:
            raise GameMapError(f"Node {node_id!r}.pose requires x_m and y_m")
        if node_type in {"road_joint", "driveway"}:
            expected = {"id", "type", "pose"}
            allowed = set(expected)
            if node_type == "road_joint":
                allowed.add("lane_transition_length_m")
            missing = expected - set(raw)
            unknown = set(raw) - allowed
            if missing:
                raise GameMapError(f"{context} is missing attributes {sorted(missing)}")
            if unknown:
                raise GameMapError(
                    f"{context} has unknown attributes {sorted(unknown)}"
                )
            profile_id = None
            geometry = (
                {
                    "lane_transition_length_m": _nonnegative_float(
                        raw.get("lane_transition_length_m", 0.0),
                        f"{context}.lane_transition_length_m",
                    ),
                }
                if node_type == "road_joint"
                else {}
            )
            attributes: GameMapBoundaryAttributes | GameMapLinearAttributes
            attributes = GameMapBoundaryAttributes(curb=False)
        else:
            allowed = _NODE_ATTRIBUTE_FIELDS[node_type]
            required = {
                "intersection": frozenset(),
                "cul_de_sac": frozenset({"culdesac_radius_m"}),
            }[node_type]
            profile_id, values = _resolve_attribute_values(
                raw,
                profiles,
                structural_fields={"id", "type", "pose"}
                | (
                    {"lane_transition_length_m"}
                    if node_type == "intersection"
                    else set()
                ),
                allowed_fields=allowed,
                required_fields=required,
                context=context,
            )
            geometry = {
                key: float(item)
                for key, item in values.items()
                if key in {"culdesac_radius_m"}
            }
            if node_type == "intersection":
                geometry["lane_transition_length_m"] = _nonnegative_float(
                    raw.get("lane_transition_length_m", 0.0),
                    f"{context}.lane_transition_length_m",
                )
            attributes = GameMapBoundaryAttributes(curb=bool(values["curb"]))
        nodes.append(
            GameMapNode(
                node_id=node_id,
                node_type=node_type,
                x_m=_finite_float(pose["x_m"], f"node {node_id!r}.pose.x_m"),
                y_m=_finite_float(pose["y_m"], f"node {node_id!r}.pose.y_m"),
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
        span_end = _point(span["end"], f"road {road_id!r}.bezier[{span_index}].end")
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
            if nodes[endpoint].node_type not in {
                "intersection",
                "road_joint",
                "driveway",
                "cul_de_sac",
            }:
                raise GameMapError(
                    f"Road {road_id!r} may connect only intersections, road joints, "
                    "driveways, and cul-de-sacs"
                )
        context = f"road {road_id!r}"
        profile_id, values = _resolve_attribute_values(
            raw,
            profiles,
            structural_fields={"id", "from", "to", "path", "bezier"},
            allowed_fields=_LINEAR_ATTRIBUTE_FIELDS,
            required_fields=_REQUIRED_LINEAR_ATTRIBUTE_FIELDS,
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


def _reverse_direction(direction: str) -> str:
    return "backward" if direction == "forward" else "forward"


def _oriented_joint_attributes(
    road: GameMapRoad,
    *,
    reverse: bool,
) -> GameMapLinearAttributes:
    """Orient road attributes along the canonical path through a joint."""
    attributes = road.attributes
    return _reversed_attributes(attributes) if reverse else attributes


def _reversed_attributes(
    attributes: GameMapLinearAttributes,
) -> GameMapLinearAttributes:
    """Reverse linear attributes with their physical cross-section order."""
    return replace(
        attributes,
        directions=tuple(
            _reverse_direction(direction)
            for direction in reversed(attributes.directions)
        ),
        divider_markings=tuple(reversed(attributes.divider_markings)),
    )


def _outward_attributes(road: GameMapRoad, node_id: str) -> GameMapLinearAttributes:
    """Orient road attributes along its centerline away from ``node_id``."""
    return _oriented_joint_attributes(
        road,
        reverse=road.to_node_id == node_id,
    )


def _direction_block_count(attributes: GameMapLinearAttributes) -> int:
    return 1 + sum(
        first != second
        for first, second in zip(
            attributes.directions[:-1],
            attributes.directions[1:],
            strict=True,
        )
    )


def _opposing_divider(
    attributes: GameMapLinearAttributes,
) -> tuple[str, str] | None:
    indices = [
        index
        for index, (first, second) in enumerate(
            zip(
                attributes.directions[:-1],
                attributes.directions[1:],
                strict=True,
            )
        )
        if first != second
    ]
    return None if not indices else attributes.divider_markings[indices[0]]


def _dominant_cross_section(
    first: GameMapLinearAttributes,
    second: GameMapLinearAttributes,
    context: str,
) -> GameMapLinearAttributes:
    """Select the node-side profile that can contain both road profiles."""
    direction_set = {"backward", "forward"}
    compatible = (
        set(first.directions) == set(second.directions)
        and set(first.directions) <= direction_set
        and _direction_block_count(first) <= 2
        and _direction_block_count(second) <= 2
        and _opposing_divider(first) == _opposing_divider(second)
    )
    if not compatible:
        raise GameMapError(
            f"{context} requires compatible direction ordering and opposing dividers"
        )

    first_counts = {
        direction: first.directions.count(direction) for direction in direction_set
    }
    second_counts = {
        direction: second.directions.count(direction) for direction in direction_set
    }
    first_dominates = all(
        first_counts[direction] >= second_counts[direction]
        for direction in direction_set
    )
    second_dominates = all(
        second_counts[direction] >= first_counts[direction]
        for direction in direction_set
    )
    if not first_dominates and not second_dominates:
        raise GameMapError(
            f"{context} has conflicting directional lane counts; one road must "
            "have at least as many lanes in both directions"
        )
    if first_dominates and not second_dominates:
        dominant = first
    elif second_dominates and not first_dominates:
        dominant = second
    else:
        dominant = first if first.lane_width_m >= second.lane_width_m else second
    return replace(
        dominant,
        speed_limit_mps=min(first.speed_limit_mps, second.speed_limit_mps),
    )


def _cross_section_changes(
    first: GameMapLinearAttributes,
    second: GameMapLinearAttributes,
) -> bool:
    return (
        first.directions != second.directions
        or first.lane_width_m != second.lane_width_m
    )


def _lane_layout_for_arm(
    layout: GameMapLinearAttributes,
    arm: GameMapLinearAttributes,
) -> GameMapLinearAttributes:
    return replace(
        layout,
        curb_offset_m=arm.curb_offset_m,
        curb=arm.curb,
        speed_limit_mps=arm.speed_limit_mps,
    )


def _resolve_linear_joint_nodes(
    nodes: tuple[GameMapNode, ...],
    road_specs: tuple[_RoadSpec, ...],
) -> tuple[GameMapNode, ...]:
    """Infer linear attributes for every degree-two road joint and driveway."""
    incident: dict[str, list[GameMapRoad]] = {node.node_id: [] for node in nodes}
    for spec in road_specs:
        incident[spec.road.from_node_id].append(spec.road)
        incident[spec.road.to_node_id].append(spec.road)

    resolved: list[GameMapNode] = []
    for node in nodes:
        if node.node_type not in {"road_joint", "driveway"}:
            resolved.append(node)
            continue
        roads = sorted(incident[node.node_id], key=lambda road: road.road_id)
        if len(roads) != 2 or any(
            road.from_node_id == road.to_node_id for road in roads
        ):
            raise GameMapError(
                f"{node.node_type.replace('_', ' ').title()} {node.node_id!r} "
                "must connect exactly two distinct roads"
            )
        first = _oriented_joint_attributes(
            roads[0], reverse=roads[0].from_node_id == node.node_id
        )
        second = _oriented_joint_attributes(
            roads[1], reverse=roads[1].to_node_id == node.node_id
        )
        context = f"{node.node_type.replace('_', ' ').title()} {node.node_id!r}"
        if node.node_type == "driveway":
            compatible = (
                first.lane_width_m == second.lane_width_m
                and first.curb_offset_m == second.curb_offset_m
                and first.directions == second.directions
                and first.curb == second.curb
                and first.marking_style == second.marking_style
                and first.marking_color == second.marking_color
                and first.divider_markings == second.divider_markings
            )
            if not compatible:
                raise GameMapError(
                    f"{context} requires compatible road cross-sections, "
                    "markings, and curb modes"
                )
            dominant = replace(
                first,
                speed_limit_mps=min(first.speed_limit_mps, second.speed_limit_mps),
            )
        else:
            dominant = _dominant_cross_section(first, second, context)
            if (
                _cross_section_changes(first, second)
                and node.geometry["lane_transition_length_m"] <= 0.0
            ):
                raise GameMapError(
                    f"{context} changes lane count or width and requires a "
                    "positive lane_transition_length_m"
                )
        resolved.append(
            replace(
                node,
                attributes=dominant,
            )
        )
    return tuple(resolved)


def _arm_for_road(
    road: GameMapRoad,
    node_id: str,
    raw_roads: dict[str, np.ndarray],
) -> _RoadArm:
    return _RoadArm(
        node_id=node_id,
        road=road,
        path_xy=_road_path_from_node(road, raw_roads[road.road_id], node_id),
        attributes=_outward_attributes(road, node_id),
    )


def _mutual_opposite_pairs(arms: list[_RoadArm]) -> list[tuple[_RoadArm, _RoadArm]]:
    """Pair mutually straightest intersection arms within 45 degrees."""
    if len(arms) < 2:
        return []
    directions: list[np.ndarray] = []
    for arm in arms:
        vector = arm.path_xy[1] - arm.path_xy[0]
        directions.append(vector / max(float(np.linalg.norm(vector)), 1.0e-9))
    best: dict[int, int] = {}
    for first_index, first_direction in enumerate(directions):
        candidates = [
            (float(np.dot(first_direction, second_direction)), second_index)
            for second_index, second_direction in enumerate(directions)
            if second_index != first_index
        ]
        dot, second_index = min(candidates)
        if dot <= -math.cos(math.radians(45.0)):
            best[first_index] = second_index
    return [
        (arms[first_index], arms[second_index])
        for first_index, second_index in sorted(best.items())
        if first_index < second_index and best.get(second_index) == first_index
    ]


def _cross_section_transitions(
    topology: GameMapTopology,
    raw_roads: dict[str, np.ndarray],
) -> dict[tuple[str, str], _ArmTransition]:
    """Plan every node arm that must taper to its authored road profile."""
    incident: dict[str, list[GameMapRoad]] = {
        node.node_id: [] for node in topology.nodes
    }
    for road in topology.roads:
        incident[road.from_node_id].append(road)
        if road.to_node_id != road.from_node_id:
            incident[road.to_node_id].append(road)

    transitions: dict[tuple[str, str], _ArmTransition] = {}
    for node in topology.nodes:
        if node.node_type == "road_joint":
            assert isinstance(node.attributes, GameMapLinearAttributes)
            roads = sorted(incident[node.node_id], key=lambda road: road.road_id)
            arms = [_arm_for_road(road, node.node_id, raw_roads) for road in roads]
            local_attributes = (
                _reversed_attributes(node.attributes),
                node.attributes,
            )
            for arm, local in zip(arms, local_attributes, strict=True):
                local = _lane_layout_for_arm(local, arm.attributes)
                if _cross_section_changes(local, arm.attributes):
                    transitions[(node.node_id, arm.road.road_id)] = _ArmTransition(
                        arm,
                        local,
                        node.geometry["lane_transition_length_m"],
                    )
            continue
        if node.node_type != "intersection":
            continue
        arms = [
            _arm_for_road(road, node.node_id, raw_roads)
            for road in incident[node.node_id]
            if road.from_node_id != road.to_node_id
        ]
        for first, second in _mutual_opposite_pairs(arms):
            second_through = _reversed_attributes(second.attributes)
            if not _cross_section_changes(first.attributes, second_through):
                continue
            context = (
                f"Intersection {node.node_id!r} through roads "
                f"{first.road.road_id!r} and {second.road.road_id!r}"
            )
            dominant = _dominant_cross_section(
                first.attributes,
                second_through,
                context,
            )
            length = node.geometry["lane_transition_length_m"]
            if length <= 0.0:
                raise GameMapError(
                    f"{context} changes lane count or width and requires a "
                    "positive lane_transition_length_m"
                )
            local_values = (
                dominant,
                _reversed_attributes(dominant),
            )
            for arm, local in zip((first, second), local_values, strict=True):
                local = _lane_layout_for_arm(local, arm.attributes)
                if _cross_section_changes(local, arm.attributes):
                    transitions[(node.node_id, arm.road.road_id)] = _ArmTransition(
                        arm,
                        local,
                        length,
                    )
    return transitions


def _parking_accesses_from_nodes(
    doc: dict[str, Any], nodes: dict[str, GameMapNode]
) -> tuple[GameMapParkingAccess, ...]:
    accesses: list[GameMapParkingAccess] = []
    for index, value in enumerate(_sequence(doc.get("nodes"), "nodes")):
        raw = _mapping(value, f"nodes[{index}]")
        if raw.get("type") != "parking_lot":
            continue
        lot_id = str(raw["id"])
        source_id = str(raw["connected_to"])
        opening_value = raw["opening_vertex"]
        if source_id not in nodes or nodes[source_id].node_type not in {
            "intersection",
            "driveway",
        }:
            raise GameMapError(
                f"Parking lot {lot_id!r}.connected_to must reference an "
                "intersection or driveway"
            )
        opening_index = opening_value - 1
        accesses.append(
            GameMapParkingAccess(f"{lot_id}:access", source_id, lot_id, opening_index)
        )
    return tuple(accesses)


def _validate_element_ids(topology: GameMapTopology) -> None:
    owners: dict[str, str] = {}
    identifiers = (
        *((node.node_id, "node") for node in topology.nodes),
        *((road.road_id, "road") for road in topology.roads),
        *((access.access_id, "parking access") for access in topology.parking_accesses),
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
    source_accesses: dict[str, list[GameMapParkingAccess]] = {
        node_id: [] for node_id in nodes
    }
    lot_accesses: dict[str, list[GameMapParkingAccess]] = {
        node_id: [] for node_id in nodes
    }
    for access in topology.parking_accesses:
        source_accesses[access.source_node_id].append(access)
        lot_accesses[access.parking_lot_node_id].append(access)
    for node in topology.nodes:
        if node.node_type == "intersection" and road_degree[node.node_id] < 3:
            raise GameMapError(
                f"Intersection {node.node_id!r} must connect at least three road "
                f"arms (found {road_degree[node.node_id]})"
            )
        if node.node_type == "cul_de_sac" and road_degree[node.node_id] != 1:
            raise GameMapError(
                f"Cul-de-sac {node.node_id!r} must terminate exactly one road"
            )
        if node.node_type == "parking_lot" and road_degree[node.node_id]:
            raise GameMapError(
                f"Parking lot {node.node_id!r} cannot be an authored road endpoint"
            )
        if node.node_type == "parking_lot" and not lot_accesses[node.node_id]:
            raise GameMapError(
                f"Parking lot {node.node_id!r} must have at least one parking access"
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
            if road_degree[node.node_id] != 2:
                raise GameMapError(
                    f"Driveway {node.node_id!r} must connect exactly two roads"
                )
            if len(source_accesses[node.node_id]) != 1:
                raise GameMapError(
                    f"Driveway {node.node_id!r} must have exactly one parking access"
                )


def _parse_race_courses(
    doc: dict[str, Any], topology: GameMapTopology
) -> tuple[GameMapRaceCourse, ...]:
    """Validate ordered race courses against authored nodes and roads."""
    if "race_courses" not in doc:
        return ()
    values = _sequence(doc["race_courses"], "race_courses")
    if not values:
        raise GameMapError("race_courses must contain at least one course")
    valid_elements = {
        *(node.node_id for node in topology.nodes),
        *(road.road_id for road in topology.roads),
    }
    courses: list[GameMapRaceCourse] = []
    course_ids: set[str] = set()
    for index, value in enumerate(values):
        raw = _mapping(value, f"race_courses[{index}]")
        required = {"id", "start", "checkpoints", "lap_count"}
        allowed = required | {"checkpoint_markers"}
        if not required <= set(raw) or not set(raw) <= allowed:
            raise GameMapError(
                f"race_courses[{index}] requires {sorted(required)} and optionally "
                "'checkpoint_markers'"
            )
        course_id = str(raw["id"]).strip()
        if not course_id or course_id in course_ids:
            raise GameMapError(f"Race course id {course_id!r} is empty or duplicated")
        course_ids.add(course_id)
        start = str(raw["start"]).strip()
        if start not in valid_elements:
            raise GameMapError(
                f"Race course {course_id!r} start references unknown node or road "
                f"{start!r}"
            )
        checkpoints = tuple(
            str(item).strip()
            for item in _sequence(
                raw["checkpoints"], f"race course {course_id!r}.checkpoints"
            )
        )
        if not checkpoints:
            raise GameMapError(
                f"Race course {course_id!r} requires at least one checkpoint"
            )
        if any(not checkpoint for checkpoint in checkpoints):
            raise GameMapError(
                f"Race course {course_id!r} checkpoints must not be empty"
            )
        if len(set(checkpoints)) != len(checkpoints):
            raise GameMapError(f"Race course {course_id!r} checkpoints must be unique")
        if start in checkpoints:
            raise GameMapError(
                f"Race course {course_id!r} may not reuse start as a checkpoint"
            )
        unknown = [item for item in checkpoints if item not in valid_elements]
        if unknown:
            raise GameMapError(
                f"Race course {course_id!r} checkpoints reference unknown nodes or "
                f"roads {unknown}"
            )
        lap_count = raw["lap_count"]
        if type(lap_count) is not int or lap_count < 0:
            raise GameMapError(
                f"Race course {course_id!r}.lap_count must be a nonnegative integer"
            )
        checkpoint_markers = raw.get("checkpoint_markers", True)
        if type(checkpoint_markers) is not bool:
            raise GameMapError(
                f"Race course {course_id!r}.checkpoint_markers must be a boolean"
            )
        courses.append(
            GameMapRaceCourse(
                course_id=course_id,
                start_element_id=start,
                checkpoint_element_ids=checkpoints,
                lap_count=lap_count,
                checkpoint_markers=checkpoint_markers,
            )
        )
    return tuple(courses)


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


def _road_path_from_node(
    road: GameMapRoad,
    points: np.ndarray,
    node_id: str,
) -> np.ndarray:
    """Orient a road centerline outward from one endpoint node."""
    return points if road.from_node_id == node_id else points[::-1]


def _trimmed_road_paths_and_joints(
    topology: GameMapTopology,
    raw_roads: dict[str, np.ndarray],
    spacing_m: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Trim incident roads and build compact tangent joint centerlines."""
    nodes = {node.node_id: node for node in topology.nodes}
    incident: dict[str, list[GameMapRoad]] = {
        node.node_id: [] for node in topology.nodes
    }
    for road in topology.roads:
        incident[road.from_node_id].append(road)
        incident[road.to_node_id].append(road)

    joint_trims: dict[tuple[str, str], float] = {}
    for node in topology.nodes:
        if node.node_type != "road_joint":
            continue
        assert isinstance(node.attributes, GameMapLinearAttributes)
        roads = sorted(incident[node.node_id], key=lambda road: road.road_id)
        layouts = (_reversed_attributes(node.attributes), node.attributes)
        arms: list[tuple[np.ndarray, float]] = []
        for road, layout in zip(roads, layouts, strict=True):
            path = _road_path_from_node(road, raw_roads[road.road_id], node.node_id)
            attributes = _lane_layout_for_arm(
                layout,
                _outward_attributes(road, node.node_id),
            )
            arms.append((path, attributes.surface_width_m))
        reaches, _order, _corners = _inferred_intersection_arm_reaches(arms)
        for road, reach in zip(roads, reaches, strict=True):
            joint_trims[(node.node_id, road.road_id)] = reach

    trimmed: dict[str, np.ndarray] = {}
    for road in topology.roads:
        line = LineString(raw_roads[road.road_id])
        start_node = nodes[road.from_node_id]
        end_node = nodes[road.to_node_id]

        def trim_for(node: GameMapNode) -> float:
            if node.node_type == "road_joint":
                return joint_trims[(node.node_id, road.road_id)]
            if node.node_type == "driveway":
                access = next(
                    item
                    for item in topology.parking_accesses
                    if item.source_node_id == node.node_id
                )
                lot = nodes[access.parking_lot_node_id]
                vertices = np.asarray(lot.polygon_vertices_xy)
                return 0.5 * float(
                    np.linalg.norm(
                        vertices[(access.opening_vertex_index + 1) % len(vertices)]
                        - vertices[access.opening_vertex_index]
                    )
                )
            return 0.0

        start_trim = trim_for(start_node)
        end_trim = trim_for(end_node)
        if start_trim + end_trim >= line.length - _POSITION_TOLERANCE_M:
            raise GameMapError(
                f"Road {road.road_id!r} is too short for road-joint trims "
                f"{start_trim:g} m and {end_trim:g} m "
                f"(centerline length {line.length:.3f} m)"
            )
        remaining = substring(line, start_trim, line.length - end_trim)
        if remaining.geom_type != "LineString":
            raise GameMapError(
                f"Road {road.road_id!r} does not retain one centerline after trimming"
            )
        trimmed[road.road_id] = np.asarray(remaining.coords, dtype=np.float64)

    joints: dict[str, np.ndarray] = {}
    for node in topology.nodes:
        if node.node_type not in {"road_joint", "driveway"}:
            continue
        if node.node_type == "driveway":
            access = next(
                item
                for item in topology.parking_accesses
                if item.source_node_id == node.node_id
            )
            lot = nodes[access.parking_lot_node_id]
            vertices = np.asarray(lot.polygon_vertices_xy)
            length = 0.5 * float(
                np.linalg.norm(
                    vertices[(access.opening_vertex_index + 1) % len(vertices)]
                    - vertices[access.opening_vertex_index]
                )
            )
        first_road, second_road = sorted(
            incident[node.node_id], key=lambda road: road.road_id
        )
        if node.node_type == "driveway":
            lengths = (length, length)
        else:
            lengths = (
                joint_trims[(node.node_id, first_road.road_id)],
                joint_trims[(node.node_id, second_road.road_id)],
            )
        first_path = _road_path_from_node(
            first_road, raw_roads[first_road.road_id], node.node_id
        )
        second_path = _road_path_from_node(
            second_road, raw_roads[second_road.road_id], node.node_id
        )
        prefixes = [
            _polyline_prefix(path, trim)
            for path, trim in zip((first_path, second_path), lengths, strict=True)
        ]
        cuts = [prefix[-1] for prefix in prefixes]
        outward_tangents: list[np.ndarray] = []
        for prefix in prefixes:
            tangent = prefix[-1] - prefix[-2]
            tangent /= max(float(np.linalg.norm(tangent)), 1.0e-9)
            outward_tangents.append(tangent)
        if node.node_type == "driveway":
            centerline = np.asarray(
                [cuts[0], [node.x_m, node.y_m], cuts[1]], dtype=np.float64
            )
            if not LineString(centerline).is_simple:
                raise GameMapError(
                    f"Driveway {node.node_id!r} produces a self-intersecting join"
                )
            joints[node.node_id] = centerline
            continue
        incoming_tangent = -outward_tangents[0]
        outgoing_tangent = outward_tangents[1]
        turn_angle = math.acos(
            float(np.clip(np.dot(incoming_tangent, outgoing_tangent), -1.0, 1.0))
        )
        if turn_angle >= math.pi - 1.0e-6:
            raise GameMapError(
                f"Road joint {node.node_id!r} cannot form a tangent U-turn"
            )
        if turn_angle <= 1.0e-6:
            handle_ratio = 2.0 / 3.0
        else:
            handle_ratio = (
                4.0 / 3.0 * math.tan(turn_angle * 0.25) / math.tan(turn_angle * 0.5)
            )
        handle_lengths = [trim * handle_ratio for trim in lengths]
        controls = [
            cut - outward_tangent * handle
            for cut, outward_tangent, handle in zip(
                cuts, outward_tangents, handle_lengths, strict=True
            )
        ]
        span = np.asarray(
            [cuts[0], controls[0], controls[1], cuts[1]], dtype=np.float64
        )
        estimate = sum(
            float(np.linalg.norm(span[index + 1] - span[index])) for index in range(3)
        )
        samples = max(3, int(math.ceil(estimate / spacing_m)) + 1)
        t = np.linspace(0.0, 1.0, samples)[:, None]
        centerline = (
            (1.0 - t) ** 3 * span[0]
            + 3.0 * (1.0 - t) ** 2 * t * span[1]
            + 3.0 * (1.0 - t) * t**2 * span[2]
            + t**3 * span[3]
        )
        tangent_epsilons = [
            min(spacing_m * 0.5, handle * 0.25) for handle in handle_lengths
        ]
        centerline = np.concatenate(
            (
                centerline[:1],
                (cuts[0] - outward_tangents[0] * tangent_epsilons[0])[None, :],
                centerline[1:-1],
                (cuts[1] - outward_tangents[1] * tangent_epsilons[1])[None, :],
                centerline[-1:],
            ),
            axis=0,
        )
        line = LineString(centerline)
        if line.length <= _POSITION_TOLERANCE_M or not line.is_simple:
            raise GameMapError(
                f"Road joint {node.node_id!r} produces a degenerate or "
                "self-intersecting curve"
            )
        joints[node.node_id] = centerline
    return trimmed, joints


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


def _polyline_end_tangent(points: np.ndarray) -> np.ndarray:
    """Return a stable unit tangent at the end of a sampled polyline."""
    for index in range(len(points) - 1, 0, -1):
        vector = points[-1] - points[index - 1]
        length = float(np.linalg.norm(vector))
        if length > _POSITION_TOLERANCE_M:
            return vector / length
    raise GameMapError("Polyline endpoint has no stable tangent")


def _inferred_intersection_arm_reaches(
    incident: list[tuple[np.ndarray, float]],
) -> tuple[list[float], list[int], list[np.ndarray | None]]:
    """Infer arm openings and roadside corners from approach geometry."""
    directions: list[np.ndarray] = []
    left_normals: list[np.ndarray] = []
    centerlines: list[LineString] = []
    center_distances: list[np.ndarray] = []
    left_boundary_distances: list[np.ndarray] = []
    right_boundary_distances: list[np.ndarray] = []
    left_boundaries: list[LineString] = []
    right_boundaries: list[LineString] = []
    for path, _width in incident:
        direction = path[1] - path[0]
        direction /= max(float(np.linalg.norm(direction)), 1.0e-9)
        directions.append(direction)
        left_normals.append(np.asarray([-direction[1], direction[0]]))
    for path, width in incident:
        widths = np.full(len(path), width, dtype=np.float64)
        left = _variable_offset_polyline(path, widths * 0.5)
        right = _variable_offset_polyline(path, -widths * 0.5)
        centerlines.append(LineString(path))
        center_distances.append(
            np.concatenate(
                ([0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
            )
        )
        left_boundary_distances.append(
            np.concatenate(
                ([0.0], np.cumsum(np.linalg.norm(np.diff(left, axis=0), axis=1)))
            )
        )
        right_boundary_distances.append(
            np.concatenate(
                ([0.0], np.cumsum(np.linalg.norm(np.diff(right, axis=0), axis=1)))
            )
        )
        left_boundaries.append(LineString(left))
        right_boundaries.append(LineString(right))

    reaches = [0.0 for _path, _width in incident]
    order = sorted(
        range(len(incident)),
        key=lambda index: math.atan2(directions[index][1], directions[index][0]),
    )
    bearings = [math.atan2(direction[1], direction[0]) for direction in directions]
    corners: list[np.ndarray | None] = []
    for order_index, first_index in enumerate(order):
        second_index = order[(order_index + 1) % len(order)]
        first_direction = directions[first_index]
        second_direction = directions[second_index]
        sector_angle = (bearings[second_index] - bearings[first_index]) % (
            2.0 * math.pi
        )
        crossing: BaseGeometry = Point()
        if sector_angle < math.pi - 1.0e-6:
            crossing = left_boundaries[first_index].intersection(
                right_boundaries[second_index]
            )
        crossing_points: list[np.ndarray] = []
        if crossing.geom_type == "Point" and not crossing.is_empty:
            crossing_points.append(np.asarray(crossing.coords[0], dtype=np.float64))
        elif crossing.geom_type in {"MultiPoint", "GeometryCollection"}:
            crossing_points.extend(
                np.asarray(part.coords[0], dtype=np.float64)
                for part in crossing.geoms
                if part.geom_type == "Point"
            )
        if crossing_points:
            corner = min(
                crossing_points,
                key=lambda point: centerlines[first_index].project(Point(point))
                + centerlines[second_index].project(Point(point)),
            )
            first_boundary_distance = left_boundaries[first_index].project(
                Point(corner)
            )
            second_boundary_distance = right_boundaries[second_index].project(
                Point(corner)
            )
            first_center_distance = float(
                np.interp(
                    first_boundary_distance,
                    left_boundary_distances[first_index],
                    center_distances[first_index],
                )
            )
            second_center_distance = float(
                np.interp(
                    second_boundary_distance,
                    right_boundary_distances[second_index],
                    center_distances[second_index],
                )
            )
            if (
                first_center_distance
                < centerlines[first_index].length - _POSITION_TOLERANCE_M
                and second_center_distance
                < centerlines[second_index].length - _POSITION_TOLERANCE_M
            ):
                corners.append(corner)
                reaches[first_index] = max(reaches[first_index], first_center_distance)
                reaches[second_index] = max(
                    reaches[second_index], second_center_distance
                )
                continue
        if sector_angle >= math.pi - 1.0e-6:
            corners.append(None)
            continue
        matrix = np.column_stack((first_direction, -second_direction))
        if abs(float(np.linalg.det(matrix))) <= 1.0e-9:
            corners.append(None)
            continue
        first_width = incident[first_index][1]
        second_width = incident[second_index][1]
        first_edge = left_normals[first_index] * first_width * 0.5
        second_edge = -left_normals[second_index] * second_width * 0.5
        first_reach, second_reach = np.linalg.solve(matrix, second_edge - first_edge)
        if (
            first_reach < -_POSITION_TOLERANCE_M
            or second_reach < -_POSITION_TOLERANCE_M
            or first_reach >= centerlines[first_index].length - _POSITION_TOLERANCE_M
            or second_reach >= centerlines[second_index].length - _POSITION_TOLERANCE_M
        ):
            corners.append(None)
            continue
        corner = (
            incident[first_index][0][0] + first_edge + (first_direction * first_reach)
        )
        corners.append(corner)
        if first_reach > 0.0:
            reaches[first_index] = max(reaches[first_index], float(first_reach))
        if second_reach > 0.0:
            reaches[second_index] = max(reaches[second_index], float(second_reach))

    if len(incident) == 2 and max(reaches) <= _POSITION_TOLERANCE_M:
        fallback = max(width for _path, width in incident) * 0.5
        reaches = [fallback, fallback]
    reaches = [
        reach + _BOUNDARY_CLEARANCE_M if reach > 0.0 else reach for reach in reaches
    ]
    return reaches, order, corners


def _parking_access_path(
    access: GameMapParkingAccess,
    nodes: dict[str, GameMapNode],
    spacing_m: float,
) -> tuple[np.ndarray, float]:
    """Infer a tangent cubic from a road node to one authored lot edge."""
    source = nodes[access.source_node_id]
    lot = nodes[access.parking_lot_node_id]
    vertices = np.asarray(lot.polygon_vertices_xy, dtype=np.float64)
    first = vertices[access.opening_vertex_index]
    second = vertices[(access.opening_vertex_index + 1) % len(vertices)]
    edge = second - first
    width = float(np.linalg.norm(edge))
    if width <= _POSITION_TOLERANCE_M:
        raise GameMapError(
            f"Parking access {access.access_id!r} has a degenerate opening edge"
        )
    edge_direction = edge / width
    outward = np.asarray([-edge_direction[1], edge_direction[0]])
    end = 0.5 * (first + second)
    start = np.asarray([source.x_m, source.y_m], dtype=np.float64)
    chord = end - start
    chord_length = float(np.linalg.norm(chord))
    if chord_length <= _POSITION_TOLERANCE_M:
        raise GameMapError(f"Parking access {access.access_id!r} is degenerate")
    if float(np.dot(start - end, outward)) <= _POSITION_TOLERANCE_M:
        raise GameMapError(
            f"Parking access {access.access_id!r} source must be outside the lot "
            "on the opening edge's exterior side"
        )
    handle = chord_length / 3.0
    control_1 = start + chord / chord_length * handle
    inward = -outward
    control_2 = end - inward * handle
    span = np.asarray([start, control_1, control_2, end])
    estimate = sum(
        float(np.linalg.norm(span[index + 1] - span[index])) for index in range(3)
    )
    samples = max(8, int(math.ceil(estimate / spacing_m)) + 1)
    t = np.linspace(0.0, 1.0, samples)[:, None]
    path = (
        (1.0 - t) ** 3 * span[0]
        + 3.0 * (1.0 - t) ** 2 * t * span[1]
        + 3.0 * (1.0 - t) * t**2 * span[2]
        + t**3 * span[3]
    )
    if not LineString(path).is_simple:
        raise GameMapError(
            f"Parking access {access.access_id!r} produces a self-intersecting curve"
        )
    return path, width


def _polyline_section(
    points: np.ndarray,
    start_m: float,
    end_m: float,
    context: str,
) -> np.ndarray:
    line = LineString(points)
    if end_m >= line.length - _POSITION_TOLERANCE_M:
        raise GameMapError(
            f"{context} transition length {end_m - start_m:g} m consumes its "
            f"road arm (available length {max(0.0, line.length - start_m):.3f} m)"
        )
    section = substring(line, start_m, end_m)
    if section.geom_type != "LineString" or section.length <= _POSITION_TOLERANCE_M:
        raise GameMapError(f"{context} produces a degenerate lane transition")
    coordinates = np.asarray(section.coords, dtype=np.float64)
    cleaned = [coordinates[0]]
    for index, point in enumerate(coordinates[1:], start=1):
        if np.linalg.norm(point - cleaned[-1]) > _POSITION_TOLERANCE_M:
            cleaned.append(point)
        elif index == len(coordinates) - 1 and len(cleaned) > 1:
            cleaned[-1] = point
    if len(cleaned) < 2:
        raise GameMapError(f"{context} produces a degenerate lane transition")
    return np.asarray(cleaned, dtype=np.float64)


def _variable_offset_polyline(
    points: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    tangents = np.empty_like(points)
    tangents[0] = points[1] - points[0]
    tangents[-1] = points[-1] - points[-2]
    if len(points) > 2:
        tangents[1:-1] = points[2:] - points[:-2]
    lengths = np.linalg.norm(tangents, axis=1)
    if np.any(lengths <= 1.0e-9):
        raise GameMapError("Lane transition has a degenerate centerline tangent")
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0])) / lengths[:, None]
    return points + normals * offsets[:, None]


def _ribbon_sides(
    points: np.ndarray,
    widths_m: np.ndarray,
    context: str,
    start_opening_xy: np.ndarray | None = None,
    end_opening_xy: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Offset both sides of a centerline and optionally pin its openings."""
    if len(points) != len(widths_m):
        raise AssertionError(f"{context} has mismatched path and width samples")
    left = _remove_rail_loops(_variable_offset_polyline(points, widths_m * 0.5))
    right = _remove_rail_loops(_variable_offset_polyline(points, -widths_m * 0.5))
    if start_opening_xy is not None:
        first = start_opening_xy[0].copy()
        second = start_opening_xy[1].copy()
        direct = np.linalg.norm(left[0] - first) + np.linalg.norm(right[0] - second)
        reverse = np.linalg.norm(left[0] - second) + np.linalg.norm(right[0] - first)
        if direct <= reverse:
            left[0], right[0] = first, second
        else:
            left[0], right[0] = second, first
    if end_opening_xy is not None:
        first = end_opening_xy[0].copy()
        second = end_opening_xy[1].copy()
        direct = np.linalg.norm(left[-1] - first) + np.linalg.norm(right[-1] - second)
        reverse = np.linalg.norm(left[-1] - second) + np.linalg.norm(right[-1] - first)
        if direct <= reverse:
            left[-1], right[-1] = first, second
        else:
            left[-1], right[-1] = second, first
    return left, right


def _remove_rail_loops(points: np.ndarray) -> np.ndarray:
    """Trim self-intersecting loops from an offset boundary rail."""
    cleaned: list[np.ndarray] = [points[0], points[1]]
    for point in points[2:]:
        current = LineString((cleaned[-1], point))
        crossing_index: int | None = None
        crossing_point: np.ndarray | None = None
        for index in range(len(cleaned) - 2):
            crossing = current.intersection(
                LineString((cleaned[index], cleaned[index + 1]))
            )
            if crossing.geom_type == "Point" and not crossing.is_empty:
                crossing_index = index
                crossing_point = np.asarray(crossing.coords[0], dtype=np.float64)
                break
        if crossing_index is not None and crossing_point is not None:
            cleaned = [*cleaned[: crossing_index + 1], crossing_point, point]
        else:
            cleaned.append(point)
    return np.asarray(cleaned)


def _polygon_from_ribbon(
    left: np.ndarray,
    right: np.ndarray,
    context: str,
) -> Polygon:
    """Build one explicit surface from paired boundary rails."""
    polygon = Polygon(np.vstack((left, right[::-1])))
    if not polygon.is_valid or polygon.area <= _AREA_TOLERANCE_M2 or polygon.interiors:
        raise GameMapError(
            f"{context} produces an invalid boundary ribbon: {is_valid_reason(polygon)}"
        )
    return polygon


def _road_joint_ribbon(
    points: np.ndarray,
    widths_m: np.ndarray,
    context: str,
) -> tuple[np.ndarray, np.ndarray, Polygon]:
    """Build a compact joint ribbon while preserving its curved outside rail.

    Args:
        points: Sampled joint centerline.
        widths_m: Paved width at each centerline sample.
        context: Element description used in validation errors.

    Returns:
        Left and right roadside rails with their enclosed surface polygon.

    Raises:
        GameMapError: The rails cannot form one valid surface.
    """
    left, right = _ribbon_sides(points, widths_m, context)
    polygon = Polygon(np.vstack((left, right[::-1])))
    if polygon.is_valid and polygon.area > _AREA_TOLERANCE_M2:
        return left, right, polygon

    start_direction = points[1] - points[0]
    end_direction = points[-1] - points[-2]
    turn = float(
        start_direction[0] * end_direction[1] - start_direction[1] * end_direction[0]
    )
    inner = left if turn > 0.0 else right
    first_tangent = inner[1] - inner[0]
    last_tangent = inner[-1] - inner[-2]
    matrix = np.column_stack((first_tangent, -last_tangent))
    if abs(float(np.linalg.det(matrix))) <= 1.0e-9:
        return left, right, _polygon_from_ribbon(left, right, context)
    first_distance, _last_distance = np.linalg.solve(
        matrix,
        inner[-1] - inner[0],
    )
    vertex = inner[0] + first_tangent * first_distance
    mitered = np.asarray((inner[0], vertex, inner[-1]), dtype=np.float64)
    if turn > 0.0:
        left = mitered
    else:
        right = mitered
    return left, right, _polygon_from_ribbon(left, right, context)


def _taper_polygon(
    points: np.ndarray,
    start_width_m: float,
    end_width_m: float,
    context: str,
) -> Polygon:
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    distances = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    alpha = distances / max(float(distances[-1]), 1.0e-9)
    widths = start_width_m + alpha * (end_width_m - start_width_m)
    left, right = _ribbon_sides(points, widths, context)
    return _polygon_from_ribbon(left, right, context)


def _linear_width_samples(
    points: np.ndarray,
    start_width_m: float,
    end_width_m: float,
) -> np.ndarray:
    """Interpolate surface widths by distance along a sampled centerline."""
    distances = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    )
    alpha = distances / max(float(distances[-1]), 1.0e-9)
    return start_width_m + alpha * (end_width_m - start_width_m)


def _multiarm_node_polygon(
    node: GameMapNode,
    incident: list[tuple[np.ndarray, float, str]],
    transitions: dict[tuple[str, str], _ArmTransition],
) -> tuple[
    Polygon,
    dict[str, np.ndarray],
    dict[tuple[str, str], _TransitionGeometry],
]:
    """Trace a multi-arm node from connected roadside boundaries."""
    reaches, order, corners = _inferred_intersection_arm_reaches(
        [(path, width) for path, width, _reference_id in incident]
    )
    arms: dict[int, _BoundaryArmGeometry] = {}
    openings: dict[str, np.ndarray] = {}
    transition_geometry: dict[tuple[str, str], _TransitionGeometry] = {}
    for index, (path, width, reference_id) in enumerate(incident):
        reach = reaches[index]
        path_length = LineString(path).length
        if reach >= path_length - _POSITION_TOLERANCE_M:
            raise GameMapError(
                f"Node {node.node_id!r} opening for {reference_id!r} consumes "
                f"its approach ({reach:.3f} m required, {path_length:.3f} m available)"
            )
        core_path = _polyline_prefix(path, max(reach, _POSITION_TOLERANCE_M * 2.0))
        tangent = _polyline_end_tangent(core_path)
        normal = np.asarray([-tangent[1], tangent[0]])
        core_left = core_path[-1] + normal * width * 0.5
        core_right = core_path[-1] - normal * width * 0.5

        transition = transitions.get((node.node_id, reference_id))
        if transition is None:
            left = core_left[None, :]
            right = core_right[None, :]
        else:
            context = f"Node {node.node_id!r} road {reference_id!r}"
            transition_path = _polyline_section(
                path,
                reach,
                reach + transition.length_m,
                context,
            )
            widths = _linear_width_samples(
                transition_path,
                transition.local_attributes.surface_width_m,
                transition.arm.attributes.surface_width_m,
            )
            left, right = _ribbon_sides(transition_path, widths, context)
            left[0] = core_left
            right[0] = core_right
            transition_geometry[(node.node_id, reference_id)] = _TransitionGeometry(
                transition,
                transition_path,
            )
        arms[index] = _BoundaryArmGeometry(reference_id, left, right)
        opening = np.asarray([right[-1], left[-1]])
        if np.linalg.norm(opening[1] - opening[0]) <= _LINE_TOLERANCE_M:
            raise GameMapError(
                f"Node {node.node_id!r} produces a degenerate opening for "
                f"{reference_id!r}"
            )
        openings[reference_id] = opening

    first = arms[order[0]]
    perimeter: list[np.ndarray] = [first.right_xy[-1], first.left_xy[-1]]
    perimeter.extend(first.left_xy[-2::-1])
    for order_index in range(len(order)):
        corner = corners[order_index]
        if corner is not None:
            perimeter.append(corner)
        next_index = order[(order_index + 1) % len(order)]
        next_arm = arms[next_index]
        perimeter.extend(next_arm.right_xy)
        if next_index == order[0]:
            break
        perimeter.append(next_arm.left_xy[-1])
        perimeter.extend(next_arm.left_xy[-2::-1])

    cleaned = [perimeter[0]]
    for point in perimeter[1:]:
        if np.linalg.norm(point - cleaned[-1]) > _POSITION_TOLERANCE_M:
            cleaned.append(point)
    if len(cleaned) > 1 and np.linalg.norm(cleaned[0] - cleaned[-1]) <= (
        _POSITION_TOLERANCE_M
    ):
        cleaned.pop()
    polygon = Polygon(cleaned)
    if not polygon.is_valid:
        linework = unary_union(LineString(np.vstack((cleaned, cleaned[0]))))
        candidates = list(polygonize(linework))
        resolved = unary_union(candidates)
        if isinstance(resolved, Polygon):
            polygon = resolved
    extent = max(
        100.0,
        max(float(np.linalg.norm(point - [node.x_m, node.y_m])) for point in cleaned)
        * 4.0,
    )
    for opening in openings.values():
        opening_center = np.mean(opening, axis=0)
        opening_tangent = opening[1] - opening[0]
        opening_tangent /= float(np.linalg.norm(opening_tangent))
        outward = np.asarray([opening_tangent[1], -opening_tangent[0]])
        if np.dot(outward, opening_center - [node.x_m, node.y_m]) < 0.0:
            outward *= -1.0
        clip = Polygon(
            (
                opening_center + opening_tangent * extent,
                opening_center - opening_tangent * extent,
                opening_center - opening_tangent * extent - outward * extent,
                opening_center + opening_tangent * extent - outward * extent,
            )
        )
        polygon = polygon.intersection(clip)
        support = Polygon(
            (
                opening[0],
                opening[1],
                opening[1] - outward * _POSITION_TOLERANCE_M,
                opening[0] - outward * _POSITION_TOLERANCE_M,
            )
        )
        supported = polygon.union(support)
        if isinstance(supported, Polygon):
            polygon = supported
    if (
        not isinstance(polygon, Polygon)
        or not polygon.is_valid
        or polygon.area <= _AREA_TOLERANCE_M2
        or polygon.interiors
    ):
        raise GameMapError(
            f"Node {node.node_id!r} produces an invalid boundary-driven footprint: "
            f"{is_valid_reason(polygon)}"
        )
    return polygon, openings, transition_geometry


def _node_polygons(
    topology: GameMapTopology,
    raw_roads: dict[str, np.ndarray],
    road_joint_centerlines: dict[str, np.ndarray],
    parking_access_paths: dict[str, tuple[np.ndarray, float]],
    transitions: dict[tuple[str, str], _ArmTransition],
) -> tuple[
    dict[str, Polygon],
    dict[tuple[str, str], _TransitionGeometry],
    dict[tuple[str, str], np.ndarray],
]:
    incidences: dict[str, list[tuple[np.ndarray, float, str | None]]] = {
        node.node_id: [] for node in topology.nodes
    }
    for road in topology.roads:
        points = raw_roads[road.road_id]
        for endpoint, node_id, path in (
            ("from", road.from_node_id, points),
            ("to", road.to_node_id, points[::-1]),
        ):
            transition = transitions.get((node_id, road.road_id))
            width = (
                transition.local_attributes.surface_width_m
                if transition is not None
                else road.attributes.surface_width_m
            )
            reference_id = (
                f"{road.road_id}:{endpoint}"
                if road.from_node_id == road.to_node_id
                else road.road_id
            )
            incidences[node_id].append((path, width, reference_id))
    for access in topology.parking_accesses:
        path, width = parking_access_paths[access.access_id]
        incidences[access.source_node_id].append((path, width, access.access_id))
    polygons: dict[str, Polygon] = {}
    transition_geometry: dict[tuple[str, str], _TransitionGeometry] = {}
    node_openings: dict[tuple[str, str], np.ndarray] = {}
    for node in topology.nodes:
        center = np.asarray([node.x_m, node.y_m])
        if node.node_type == "driveway":
            assert isinstance(node.attributes, GameMapLinearAttributes)
            joint_roads = sorted(
                (
                    road
                    for road in topology.roads
                    if node.node_id in {road.from_node_id, road.to_node_id}
                ),
                key=lambda road: road.road_id,
            )
            centerline = road_joint_centerlines[node.node_id]
            driveway_incident: list[tuple[np.ndarray, float, str]] = []
            for road, cut in zip(
                joint_roads, (centerline[0], centerline[-1]), strict=True
            ):
                outward = _road_path_from_node(
                    road, raw_roads[road.road_id], node.node_id
                )
                branch = np.vstack((center, cut, outward[1:]))
                driveway_incident.append(
                    (branch, node.attributes.surface_width_m, road.road_id)
                )
            access = next(
                access
                for access in topology.parking_accesses
                if access.source_node_id == node.node_id
            )
            access_path, access_width = parking_access_paths[access.access_id]
            driveway_incident.append((access_path, access_width, access.access_id))
            polygon, openings, resolved_transitions = _multiarm_node_polygon(
                node,
                driveway_incident,
                transitions,
            )
            node_openings.update(
                ((node.node_id, reference_id), opening)
                for reference_id, opening in openings.items()
            )
            transition_geometry.update(resolved_transitions)
        elif node.node_type == "intersection":
            incident = incidences[node.node_id]
            if not incident:
                raise GameMapError(
                    f"Node {node.node_id!r} must have at least one incidence"
                )
            polygon, openings, resolved_transitions = _multiarm_node_polygon(
                node,
                [
                    (path, width, reference_id)
                    for path, width, reference_id in incident
                    if reference_id is not None
                ],
                transitions,
            )
            node_openings.update(
                ((node.node_id, reference_id), opening)
                for reference_id, opening in openings.items()
            )
            transition_geometry.update(resolved_transitions)
        elif node.node_type == "road_joint":
            assert isinstance(node.attributes, GameMapLinearAttributes)
            centerline = road_joint_centerlines[node.node_id]
            joint_roads = sorted(
                (
                    road
                    for road in topology.roads
                    if node.node_id in {road.from_node_id, road.to_node_id}
                ),
                key=lambda road: road.road_id,
            )
            endpoint_layouts = (
                _reversed_attributes(node.attributes),
                node.attributes,
            )
            endpoint_widths = [
                _lane_layout_for_arm(
                    layout,
                    _outward_attributes(road, node.node_id),
                ).surface_width_m
                for road, layout in zip(
                    joint_roads,
                    endpoint_layouts,
                    strict=True,
                )
            ]
            path_parts: list[np.ndarray] = []
            width_parts: list[np.ndarray] = []
            first_transition = transitions.get((node.node_id, joint_roads[0].road_id))
            if first_transition is not None:
                context = f"Road joint {node.node_id!r} road {joint_roads[0].road_id!r}"
                outward = _road_path_from_node(
                    joint_roads[0], raw_roads[joint_roads[0].road_id], node.node_id
                )
                transition_path = _polyline_section(
                    outward, 0.0, first_transition.length_m, context
                )
                transition_geometry[(node.node_id, joint_roads[0].road_id)] = (
                    _TransitionGeometry(first_transition, transition_path)
                )
                path_parts.append(transition_path[::-1])
                width_parts.append(
                    _linear_width_samples(
                        transition_path,
                        first_transition.local_attributes.surface_width_m,
                        first_transition.arm.attributes.surface_width_m,
                    )[::-1]
                )
            path_parts.append(centerline)
            width_parts.append(
                _linear_width_samples(
                    centerline, endpoint_widths[0], endpoint_widths[1]
                )
            )
            second_transition = transitions.get((node.node_id, joint_roads[1].road_id))
            if second_transition is not None:
                context = f"Road joint {node.node_id!r} road {joint_roads[1].road_id!r}"
                outward = _road_path_from_node(
                    joint_roads[1], raw_roads[joint_roads[1].road_id], node.node_id
                )
                transition_path = _polyline_section(
                    outward, 0.0, second_transition.length_m, context
                )
                transition_geometry[(node.node_id, joint_roads[1].road_id)] = (
                    _TransitionGeometry(second_transition, transition_path)
                )
                path_parts.append(transition_path)
                width_parts.append(
                    _linear_width_samples(
                        transition_path,
                        second_transition.local_attributes.surface_width_m,
                        second_transition.arm.attributes.surface_width_m,
                    )
                )
            combined_path = path_parts[0]
            combined_widths = width_parts[0]
            for path_part, width_part in zip(
                path_parts[1:], width_parts[1:], strict=True
            ):
                combined_path = np.vstack((combined_path, path_part[1:]))
                combined_widths = np.concatenate((combined_widths, width_part[1:]))
            context = f"Road joint {node.node_id!r}"
            left, right, polygon = _road_joint_ribbon(
                combined_path,
                combined_widths,
                context,
            )
            node_openings[(node.node_id, joint_roads[0].road_id)] = np.asarray(
                [right[0], left[0]]
            )
            node_openings[(node.node_id, joint_roads[1].road_id)] = np.asarray(
                [right[-1], left[-1]]
            )
        elif node.node_type == "cul_de_sac":
            radius = node.geometry["culdesac_radius_m"]
            path, opening_width, road_id = incidences[node.node_id][0]
            assert road_id is not None
            vector = path[1] - path[0]
            direction = vector / max(float(np.linalg.norm(vector)), 1.0e-9)
            normal = np.asarray([-direction[1], direction[0]])
            chord_distance = math.sqrt(radius**2 - (opening_width * 0.5) ** 2)
            opening_center = center + direction * chord_distance
            right = opening_center - normal * opening_width * 0.5
            left = opening_center + normal * opening_width * 0.5
            bearing = math.atan2(direction[1], direction[0])
            half_angle = math.asin(opening_width * 0.5 / radius)
            angles = np.linspace(
                bearing + half_angle,
                bearing + 2.0 * math.pi - half_angle,
                129,
            )
            arc = center + radius * np.column_stack((np.cos(angles), np.sin(angles)))
            polygon = Polygon(np.vstack((right, left, arc[1:-1])))
            node_openings[(node.node_id, road_id)] = np.asarray([right, left])
        elif node.node_type == "parking_lot":
            polygon = Polygon(node.polygon_vertices_xy)
            vertices = np.asarray(node.polygon_vertices_xy, dtype=np.float64)
            for access in topology.parking_accesses:
                if access.parking_lot_node_id != node.node_id:
                    continue
                first = vertices[access.opening_vertex_index]
                second = vertices[(access.opening_vertex_index + 1) % len(vertices)]
                node_openings[(node.node_id, access.access_id)] = np.asarray(
                    [first, second]
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
    return polygons, transition_geometry, node_openings


def _surface_array(polygon: Polygon) -> np.ndarray:
    """Convert a resolved surface polygon to world coordinates."""
    points = np.asarray(polygon.exterior.coords, dtype=np.float64)
    return np.column_stack((points, np.zeros(len(points), dtype=np.float64)))


def _exclude_connected_footprints(
    surface: Polygon,
    excluded: tuple[Polygon, ...],
    context: str,
) -> Polygon:
    """Trim numeric seam overlap from an explicit corridor ribbon."""
    geometry: BaseGeometry = surface
    for footprint in excluded:
        geometry = geometry.difference(footprint)
    if isinstance(geometry, Polygon):
        return geometry
    parts = [
        part
        for part in getattr(geometry, "geoms", ())
        if isinstance(part, Polygon) and part.area > _AREA_TOLERANCE_M2
    ]
    if len(parts) != 1:
        raise GameMapError(f"{context} does not retain one connected surface")
    return parts[0]


def _boundaries_for_elements(
    elements: list[GameMapElement],
    connections: list[_Connection],
    permitted_boundary_contacts: set[tuple[str, str]] | None = None,
    curb_regions: dict[str, list[tuple[BaseGeometry, bool]]] | None = None,
) -> list[GameMapElement]:
    """Validate contacts and attach semantic boundaries and physical curbs."""
    polygons = {
        element.element_id: Polygon(element.surface_world[:, :2])
        for element in elements
    }
    for element_id, polygon in polygons.items():
        if not polygon.is_valid:
            raise GameMapError(
                f"Element {element_id!r} has an invalid surface: "
                f"{is_valid_reason(polygon)}"
            )
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
                        f"Unrelated elements {first_id!r} and {second_id!r} overlap "
                        f"by {overlap_area:.6f} m^2"
                    )
                if pair not in (permitted_boundary_contacts or set()) and (
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
                    f"by {overlap_area:.6f} m^2 at {labels}"
                )
            for connection in declared:
                opening = LineString(connection.opening_xy)
                first_error = opening.difference(
                    first.boundary.buffer(_OPENING_TOLERANCE_M)
                ).length
                second_error = opening.difference(
                    second.boundary.buffer(_OPENING_TOLERANCE_M)
                ).length
                if (
                    opening.length <= _LINE_TOLERANCE_M
                    or first_error > _OPENING_TOLERANCE_M
                    or second_error > _OPENING_TOLERANCE_M
                ):
                    raise GameMapError(
                        f"Connection {connection.connection_id!r} between "
                        f"{first_id!r} and {second_id!r} has mismatched openings "
                        f"({first_error:.9f}/{second_error:.9f} m outside boundaries, "
                        f"{opening.length:.9f} m long)"
                    )
                openings[first_id].append(opening)
                openings[second_id].append(opening)

    resolved: list[GameMapElement] = []
    for element in elements:
        boundary: BaseGeometry = polygons[element.element_id].boundary
        for opening in openings[element.element_id]:
            boundary = boundary.difference(
                opening.buffer(
                    _OPENING_TOLERANCE_M,
                    cap_style=2,
                    join_style=2,
                )
            )
        parts = sorted(
            (
                points
                for points in _line_parts(boundary)
                if LineString(points).length > _OPENING_TOLERANCE_M * 2.0
            ),
            key=lambda points: (
                round(float(np.min(points[:, 0])), 6),
                round(float(np.min(points[:, 1])), 6),
                round(float(np.max(points[:, 0])), 6),
                round(float(np.max(points[:, 1])), 6),
            ),
        )
        road_boundaries = tuple(
            GameMapRoadBoundary(
                boundary_id=f"{element.element_id}:road_boundary:{index}",
                polyline_world=_xyz(points),
            )
            for index, points in enumerate(parts)
            if len(points) >= 2
        )
        remaining_curb_boundary = boundary
        selected_curb_parts: list[np.ndarray] = []
        for region, enabled in (curb_regions or {}).get(element.element_id, []):
            selected = remaining_curb_boundary.intersection(region)
            if enabled:
                selected_curb_parts.extend(_line_parts(selected))
            remaining_curb_boundary = remaining_curb_boundary.difference(region)
        if element.attributes.curb:
            selected_curb_parts.extend(_line_parts(remaining_curb_boundary))
        selected_curb_parts.sort(
            key=lambda points: (
                round(float(np.min(points[:, 0])), 6),
                round(float(np.min(points[:, 1])), 6),
                round(float(np.max(points[:, 0])), 6),
                round(float(np.max(points[:, 1])), 6),
            )
        )
        curbs = tuple(
            GameMapCurb(
                curb_id=f"{element.element_id}:curb:{index}",
                polyline_world=_xyz(points),
            )
            for index, points in enumerate(selected_curb_parts)
            if len(points) >= 2
            and LineString(points).length > _OPENING_TOLERANCE_M * 2.0
        )
        resolved.append(replace(element, road_boundaries=road_boundaries, curbs=curbs))
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


def _lane_boundary_offsets(attributes: GameMapLinearAttributes) -> np.ndarray:
    lane_count = len(attributes.directions)
    return np.linspace(
        lane_count * attributes.lane_width_m * 0.5,
        -lane_count * attributes.lane_width_m * 0.5,
        lane_count + 1,
    )


def _direction_groups(directions: tuple[str, ...]) -> list[tuple[str, int, int]]:
    groups: list[tuple[str, int, int]] = []
    start = 0
    for index in range(1, len(directions) + 1):
        if index == len(directions) or directions[index] != directions[start]:
            groups.append((directions[start], start, index - start))
            start = index
    return groups


def _transition_boundary_mapping(
    local: GameMapLinearAttributes,
    road: GameMapLinearAttributes,
) -> list[int]:
    local_groups = _direction_groups(local.directions)
    road_groups = _direction_groups(road.directions)
    if [group[0] for group in local_groups] != [group[0] for group in road_groups]:
        raise GameMapError("Lane transition changes directional lane ordering")
    mapping = [0] * (len(local.directions) + 1)
    for group_index, (local_group, road_group) in enumerate(
        zip(local_groups, road_groups, strict=True)
    ):
        _direction, local_start, local_count = local_group
        _road_direction, road_start, road_count = road_group
        extra = local_count - road_count
        if extra < 0:
            raise GameMapError("Lane transition local profile is not dominant")
        for boundary in range(local_count + 1):
            if group_index == 0:
                road_boundary = max(0, boundary - extra)
            else:
                road_boundary = min(boundary, road_count)
            mapping[local_start + boundary] = road_start + road_boundary
    return mapping


def _build_transition_lanes(
    geometry: _TransitionGeometry,
) -> list[_LaneBuild]:
    transition = geometry.transition
    local = transition.local_attributes
    road = transition.arm.attributes
    path = geometry.path_xy
    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    distances = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    alpha = distances / max(float(distances[-1]), 1.0e-9)
    local_offsets = _lane_boundary_offsets(local)
    road_offsets = _lane_boundary_offsets(road)
    mapping = _transition_boundary_mapping(local, road)
    boundaries = [
        _variable_offset_polyline(
            path,
            local_offsets[index]
            + alpha * (road_offsets[remote_index] - local_offsets[index]),
        )
        for index, remote_index in enumerate(mapping)
    ]

    lanes: list[_LaneBuild] = []
    for index, direction in enumerate(local.directions):
        upper = boundaries[index]
        lower = boundaries[index + 1]
        center = 0.5 * (upper + lower)
        left, right = upper, lower
        roadside_offsets = (
            local_offsets[index + 1]
            + alpha * (road_offsets[mapping[index + 1]] - local_offsets[index + 1])
            - road.curb_offset_m
        )
        roadside = _variable_offset_polyline(path, roadside_offsets)
        kind = "start"
        if direction == "backward":
            center = center[::-1]
            left, right = lower[::-1], upper[::-1]
            roadside_offsets = (
                local_offsets[index]
                + alpha * (road_offsets[mapping[index]] - local_offsets[index])
                + road.curb_offset_m
            )
            roadside = _variable_offset_polyline(path, roadside_offsets)[::-1]
            kind = "end"
        left_marking, right_marking = _lane_edge_markings(local, index, direction)
        lanes.append(
            _LaneBuild(
                lane_id=(
                    f"{transition.arm.node_id}:transition:"
                    f"{transition.arm.road.road_id}:lane:{index}"
                ),
                element_id=transition.arm.node_id,
                centerline=_xyz(center),
                left_edge=_xyz(left),
                right_edge=_xyz(right),
                roadside_edge=_xyz(roadside),
                speed_limit_mps=road.speed_limit_mps,
                marking_style=local.marking_style,
                marking_color=local.marking_color,
                start_endpoint="from" if kind == "start" else "to",
                end_endpoint="to" if kind == "start" else "from",
                successors=[],
                allows_taxi_stops=False,
                left_marking_style=left_marking[0],
                left_marking_color=left_marking[1],
                right_marking_style=right_marking[0],
                right_marking_color=right_marking[1],
            )
        )
    return lanes


def _splice_transition_lanes(
    transition_geometry: dict[tuple[str, str], _TransitionGeometry],
    incidences: dict[str, list[_LaneIncidence]],
    lanes: list[_LaneBuild],
    lane_dividers: list[GameMapLaneDivider],
) -> None:
    """Replace narrow road incidences with visible node transition lanes."""
    for (node_id, road_id), geometry in sorted(transition_geometry.items()):
        road_incidences = [
            incidence
            for incidence in incidences[node_id]
            if incidence.lane.element_id == road_id
        ]
        if not road_incidences:
            raise AssertionError(f"Missing road incidences for {node_id!r}/{road_id!r}")
        incidences[node_id] = [
            incidence
            for incidence in incidences[node_id]
            if incidence.lane.element_id != road_id
        ]
        built = _build_transition_lanes(geometry)
        for lane, direction in zip(
            built,
            geometry.transition.local_attributes.directions,
            strict=True,
        ):
            kind = "start" if direction == "forward" else "end"
            candidates = [item for item in road_incidences if item.kind == kind]
            transition_far = (
                lane.centerline[-1, :2]
                if direction == "forward"
                else lane.centerline[0, :2]
            )
            target = min(
                candidates,
                key=lambda incidence: float(
                    np.linalg.norm(
                        (
                            incidence.lane.centerline[0, :2]
                            if kind == "start"
                            else incidence.lane.centerline[-1, :2]
                        )
                        - transition_far
                    )
                ),
            )
            if direction == "forward":
                lane.successors.append(target.lane.lane_id)
            else:
                target.lane.successors.append(lane.lane_id)
            incidences[node_id].append(
                _LaneIncidence(lane, node_id, kind, target.edge_ref)
            )
        lanes.extend(built)
        lane_dividers.extend(
            _build_lane_dividers(built, geometry.transition.local_attributes)
        )


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
    *,
    access_turns_only: bool = False,
) -> None:
    incoming = [item for item in incidences if item.kind == "end"]
    outgoing = [item for item in incidences if item.kind == "start"]
    connector_count = 0
    for source in incoming:
        for target in outgoing:
            source_is_access = source.edge_ref.startswith("parking_access:")
            target_is_access = target.edge_ref.startswith("parking_access:")
            if access_turns_only and source_is_access == target_is_access:
                continue
            if source.edge_ref == target.edge_ref and node.node_type != "cul_de_sac":
                continue
            if node.node_type == "parking_lot":
                source.lane.successors.append(target.lane.lane_id)
                continue
            center = np.asarray([node.x_m, node.y_m, 0.0], dtype=np.float32)
            if node.node_type == "cul_de_sac":
                centerline = _bezier(
                    source.lane.centerline[-1],
                    center,
                    target.lane.centerline[0],
                    connector_samples,
                )
            else:
                start = source.lane.centerline[-1]
                end = target.lane.centerline[0]
                incoming_tangent = start - source.lane.centerline[-2]
                outgoing_tangent = target.lane.centerline[1] - end
                incoming_tangent /= max(float(np.linalg.norm(incoming_tangent)), 1.0e-9)
                outgoing_tangent /= max(float(np.linalg.norm(outgoing_tangent)), 1.0e-9)
                chord_length = float(np.linalg.norm(end - start))
                handle_length = chord_length * _INTERSECTION_TURN_HANDLE_RATIO
                first_control = start + incoming_tangent * handle_length
                second_control = end - outgoing_tangent * handle_length
                t = np.linspace(0.0, 1.0, connector_samples, dtype=np.float32)[:, None]
                centerline = (
                    (1.0 - t) ** 3 * start
                    + 3.0 * (1.0 - t) ** 2 * t * first_control
                    + 3.0 * (1.0 - t) * t**2 * second_control
                    + t**3 * end
                ).astype(np.float32)
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


def _wire_road_joint(
    node: GameMapNode,
    incidences: list[_LaneIncidence],
    centerline_xy: np.ndarray,
    lanes: list[_LaneBuild],
    lane_dividers: list[GameMapLaneDivider],
) -> None:
    """Build conditioning-visible lanes through one road joint."""
    assert isinstance(node.attributes, GameMapLinearAttributes)
    joint_lanes = _build_linear_lanes(
        node.node_id,
        centerline_xy,
        node.attributes,
        False,
    )
    incoming = [incidence for incidence in incidences if incidence.kind == "end"]
    outgoing = [incidence for incidence in incidences if incidence.kind == "start"]
    if len(incoming) != len(joint_lanes) or len(outgoing) != len(joint_lanes):
        raise GameMapError(
            f"Road joint {node.node_id!r} cannot pair its directed road lanes"
        )

    unused_incoming = list(incoming)
    unused_outgoing = list(outgoing)
    for joint_lane in joint_lanes:
        source = min(
            unused_incoming,
            key=lambda incidence: float(
                np.linalg.norm(
                    incidence.lane.centerline[-1, :2] - joint_lane.centerline[0, :2]
                )
            ),
        )
        target = min(
            unused_outgoing,
            key=lambda incidence: float(
                np.linalg.norm(
                    incidence.lane.centerline[0, :2] - joint_lane.centerline[-1, :2]
                )
            ),
        )
        unused_incoming.remove(source)
        unused_outgoing.remove(target)
        joint_lane.speed_limit_mps = source.lane.speed_limit_mps
        joint_lane.successors.append(target.lane.lane_id)
        source.lane.successors.append(joint_lane.lane_id)

    lanes.extend(joint_lanes)
    lane_dividers.extend(_build_lane_dividers(joint_lanes, node.attributes))


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
    node_values = _resolve_linear_joint_nodes(node_values, road_specs)
    nodes = {node.node_id: node for node in node_values}
    parking_accesses = _parking_accesses_from_nodes(doc, nodes)
    adjacency: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    for spec in road_specs:
        adjacency[spec.road.from_node_id].append(f"road:{spec.road.road_id}")
        adjacency[spec.road.to_node_id].append(f"road:{spec.road.road_id}")
    for access in parking_accesses:
        reference = f"parking_access:{access.access_id}"
        adjacency[access.source_node_id].append(reference)
        adjacency[access.parking_lot_node_id].append(reference)
    topology = GameMapTopology(
        nodes=node_values,
        roads=tuple(spec.road for spec in road_specs),
        parking_accesses=parking_accesses,
        adjacency=tuple(
            (node_id, tuple(sorted(references)))
            for node_id, references in adjacency.items()
        ),
    )
    _validate_topology(topology)
    race_courses = _parse_race_courses(doc, topology)

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
    parking_access_paths = {
        access.access_id: _parking_access_path(access, nodes, settings.sample_spacing_m)
        for access in parking_accesses
    }
    transitions = _cross_section_transitions(topology, raw_roads)
    raw_roads, road_joint_centerlines = _trimmed_road_paths_and_joints(
        topology, raw_roads, settings.sample_spacing_m
    )
    polygons, transition_geometry, node_openings = _node_polygons(
        topology,
        raw_roads,
        road_joint_centerlines,
        parking_access_paths,
        transitions,
    )
    elements: list[GameMapElement] = []
    connections: list[_Connection] = []
    lanes: list[_LaneBuild] = []
    lane_dividers: list[GameMapLaneDivider] = []
    incidences: dict[str, list[_LaneIncidence]] = {node_id: [] for node_id in nodes}

    for spec in road_specs:
        road = spec.road
        attributes = road.attributes
        from_reference = (
            f"{road.road_id}:from"
            if road.from_node_id == road.to_node_id
            else road.road_id
        )
        to_reference = (
            f"{road.road_id}:to"
            if road.from_node_id == road.to_node_id
            else road.road_id
        )
        try:
            points = _trim_line(
                raw_roads[road.road_id],
                polygons[road.from_node_id],
                polygons[road.to_node_id],
                f"Road {road.road_id!r}",
            )
        except GameMapError as error:
            transition_nodes = [
                node_id
                for node_id in (road.from_node_id, road.to_node_id)
                if (node_id, road.road_id) in transition_geometry
            ]
            if transition_nodes and "completely contained" in str(error):
                raise GameMapError(
                    f"Node {transition_nodes[0]!r} transition consumes its road arm "
                    f"on {road.road_id!r}"
                ) from error
            raise
        built = _build_linear_lanes(road.road_id, points, attributes, True)
        lanes.extend(built)
        lane_dividers.extend(_build_lane_dividers(built, attributes))
        for incidence in _incidences_for_lanes(
            built, road.from_node_id, road.to_node_id, f"road:{road.road_id}"
        ):
            incidences[incidence.node_id].append(incidence)
        context = f"Road {road.road_id!r}"
        widths = np.full(len(points), attributes.surface_width_m, dtype=np.float64)
        left, right = _ribbon_sides(
            points,
            widths,
            context,
            start_opening_xy=node_openings[(road.from_node_id, from_reference)],
            end_opening_xy=node_openings[(road.to_node_id, to_reference)],
        )
        surface = _polygon_from_ribbon(left, right, context)
        surface = _exclude_connected_footprints(
            surface,
            (polygons[road.from_node_id], polygons[road.to_node_id]),
            context,
        )
        elements.append(
            GameMapElement(
                element_id=road.road_id,
                element_type="road",
                profile_id=road.profile_id,
                attributes=attributes,
                surface_world=_surface_array(surface),
                road_boundaries=(),
                curbs=(),
            )
        )
        connections.extend(
            _Connection(
                connection_id=f"road:{road.road_id}:{endpoint}",
                first_element_id=road.road_id,
                second_element_id=node_id,
                opening_xy=node_openings[
                    (
                        node_id,
                        f"{road.road_id}:{endpoint}"
                        if road.from_node_id == road.to_node_id
                        else road.road_id,
                    )
                ],
            )
            for endpoint, node_id in (
                ("from", road.from_node_id),
                ("to", road.to_node_id),
            )
        )
    for access in parking_accesses:
        source = nodes[access.source_node_id]
        lot = nodes[access.parking_lot_node_id]
        centerline, opening_width = parking_access_paths[access.access_id]
        attributes = GameMapLinearAttributes(
            curb=True,
            lane_width_m=opening_width * 0.5,
            curb_offset_m=0.0,
            directions=("forward", "backward"),
            speed_limit_mps=5.5,
            marking_style="VIRTUAL",
            marking_color="WHITE",
            divider_markings=(("VIRTUAL", "WHITE"),),
        )
        points = _trim_line(
            centerline,
            polygons[source.node_id],
            polygons[lot.node_id],
            f"Parking access {access.access_id!r}",
        )
        built = _build_linear_lanes(access.access_id, points, attributes, False)
        lanes.extend(built)
        lane_dividers.extend(_build_lane_dividers(built, attributes))
        for incidence in _incidences_for_lanes(
            built,
            source.node_id,
            lot.node_id,
            f"parking_access:{access.access_id}",
        ):
            if incidence.node_id == source.node_id:
                incidences[incidence.node_id].append(incidence)
        context = f"Parking access {access.access_id!r}"
        widths = np.full(len(points), opening_width, dtype=np.float64)
        left, right = _ribbon_sides(
            points,
            widths,
            context,
            start_opening_xy=node_openings[(source.node_id, access.access_id)],
            end_opening_xy=node_openings[(lot.node_id, access.access_id)],
        )
        surface = _polygon_from_ribbon(left, right, context)
        surface = _exclude_connected_footprints(
            surface,
            (polygons[source.node_id], polygons[lot.node_id]),
            context,
        )
        elements.append(
            GameMapElement(
                element_id=access.access_id,
                element_type="parking_access",
                profile_id=None,
                attributes=attributes,
                surface_world=_surface_array(surface),
                road_boundaries=(),
                curbs=(),
            )
        )
        connections.extend(
            (
                _Connection(
                    connection_id=f"parking_access:{access.access_id}:source",
                    first_element_id=access.access_id,
                    second_element_id=source.node_id,
                    opening_xy=node_openings[(source.node_id, access.access_id)],
                ),
                _Connection(
                    connection_id=f"parking_access:{access.access_id}:lot",
                    first_element_id=access.access_id,
                    second_element_id=lot.node_id,
                    opening_xy=node_openings[(lot.node_id, access.access_id)],
                ),
            )
        )

    _splice_transition_lanes(
        transition_geometry,
        incidences,
        lanes,
        lane_dividers,
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
                road_boundaries=(),
                curbs=(),
            )
        )

    for node in node_values:
        if node.node_type in {"road_joint", "driveway"}:
            centerline = road_joint_centerlines[node.node_id]
            _wire_road_joint(
                node,
                [
                    incidence
                    for incidence in incidences[node.node_id]
                    if incidence.edge_ref.startswith("road:")
                ],
                centerline,
                lanes,
                lane_dividers,
            )
            if node.node_type == "driveway":
                _wire_node(
                    node,
                    incidences[node.node_id],
                    lanes,
                    settings.intersection_connector_samples,
                    access_turns_only=True,
                )
        else:
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
    traffic = compile_traffic(
        doc.get("traffic"),
        topology,
        runtime_lanes,
        traffic_count=doc.get("traffic_count"),
        map_id=map_id,
        spawns=spawns,
    )
    permitted_boundary_contacts = {
        tuple(sorted((access.access_id, road.road_id)))
        for access in parking_accesses
        for road in topology.roads
        if access.source_node_id in {road.from_node_id, road.to_node_id}
    }
    transition_curb_regions: dict[str, list[tuple[BaseGeometry, bool]]] = {}
    for (node_id, _road_id), geometry in transition_geometry.items():
        transition = geometry.transition
        region = _taper_polygon(
            geometry.path_xy,
            transition.local_attributes.surface_width_m,
            transition.arm.attributes.surface_width_m,
            f"Node {node_id!r} curb region",
        ).buffer(_POSITION_TOLERANCE_M)
        transition_curb_regions.setdefault(node_id, []).append(
            (region, transition.arm.attributes.curb)
        )
    elements = _boundaries_for_elements(
        elements,
        connections,
        permitted_boundary_contacts,
        transition_curb_regions,
    )
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
        race_courses=race_courses,
        traffic=traffic,
    )
