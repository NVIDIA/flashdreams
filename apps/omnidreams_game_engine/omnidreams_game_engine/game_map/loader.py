# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""YAML parsing and geometry resolution for semantic game maps."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from omnidreams_game_engine.game_map.types import (
    GameMapElement,
    GameMapLane,
    GameMapSpawn,
    GameMapVisualVariant,
    ResolvedGameMap,
)

_SCHEMA_VERSION = 1
_PLACEMENT_TOLERANCE_M = 1.0e-3
_ANGLE_TOLERANCE_DEG = 1.0e-3
_SAMPLE_SPACING_M = 2.0


class GameMapError(ValueError):
    """Invalid semantic game-map definition."""


@dataclass(frozen=True)
class GameMapHeader:
    """Cheap scene-picker metadata read without compiling geometry."""

    map_id: str
    name: str
    variants: tuple[GameMapVisualVariant, ...]
    source_path: Path


@dataclass(frozen=True)
class _Profile:
    profile_id: str
    lane_width_m: float
    directions: tuple[str, ...]
    speed_limit_mps: float
    curb: bool
    marking_style: str
    marking_color: str

    @property
    def width_m(self) -> float:
        return self.lane_width_m * len(self.directions)


@dataclass(frozen=True)
class _Pose:
    x_m: float
    y_m: float
    heading_deg: float


@dataclass(frozen=True)
class _Port:
    name: str
    x_m: float
    y_m: float
    heading_deg: float


@dataclass(frozen=True)
class _ElementSpec:
    element_id: str
    element_type: str
    profile_id: str
    geometry: dict[str, Any]
    pose: _Pose | None
    attach_port: str | None
    attach_to: str | None


@dataclass
class _LaneBuild:
    lane_id: str
    element_id: str
    centerline: np.ndarray
    left_edge: np.ndarray
    right_edge: np.ndarray
    speed_limit_mps: float
    marking_style: str
    marking_color: str
    start_port: str
    end_port: str
    successors: list[str]
    allows_taxi_stops: bool


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GameMapError(f"{context} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _sequence(value: object, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise GameMapError(f"{context} must be a list")
    return value


def _positive_float(value: object, context: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise GameMapError(f"{context} must be a number") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise GameMapError(f"{context} must be positive")
    return number


def _read_document(path: Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise GameMapError(f"Game-map path does not exist or is not a file: {path}")
    if not path.name.endswith(".robotaxi.yaml"):
        raise GameMapError("Crazy Robotaxi maps must use the .robotaxi.yaml suffix")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise GameMapError(f"Could not parse {path}: {exc}") from exc
    return _mapping(raw, "map document")


def _parse_variants(
    raw_spawn: dict[str, Any], source_path: Path
) -> tuple[GameMapVisualVariant, ...]:
    variants_raw = _mapping(raw_spawn.get("variants"), "spawn.variants")
    if "default" not in variants_raw:
        raise GameMapError("Every spawn must define a default visual variant")
    variants: list[GameMapVisualVariant] = []
    for name, raw_variant in variants_raw.items():
        variant = _mapping(raw_variant, f"variant {name!r}")
        image = str(variant.get("image", "")).strip()
        prompt = str(variant.get("prompt", "")).strip()
        if not image or not prompt:
            raise GameMapError(f"Variant {name!r} requires image and prompt")
        resolve_seed_asset(source_path, image)
        variants.append(GameMapVisualVariant(name=name, image=image, prompt=prompt))
    variants.sort(key=lambda item: (item.name != "default", item.name))
    return tuple(variants)


def load_game_map_header(path: Path) -> GameMapHeader:
    """Load map name and default-spawn variants without resolving geometry."""
    source_path = Path(path).expanduser().resolve()
    doc = _read_document(source_path)
    version = doc.get("schema_version")
    if version != _SCHEMA_VERSION:
        raise GameMapError(
            f"Unsupported schema_version {version!r}; expected {_SCHEMA_VERSION}"
        )
    spawns = _sequence(doc.get("spawns"), "spawns")
    if not spawns:
        raise GameMapError("Map must define at least one spawn")
    first_spawn = _mapping(spawns[0], "spawns[0]")
    return GameMapHeader(
        map_id=str(doc.get("id", "")).strip(),
        name=str(doc.get("name", doc.get("id", source_path.stem))).strip(),
        variants=_parse_variants(first_spawn, source_path),
        source_path=source_path,
    )


def resolve_seed_asset(source_path: Path, reference: str) -> Path:
    """Resolve a map-relative or package seed-image reference."""
    if reference.startswith("package://"):
        location = reference.removeprefix("package://")
        package, separator, resource = location.partition("/")
        if not separator or not package or not resource:
            raise GameMapError(
                "Package assets must use package://package/path/to/resource"
            )
        traversable = resources.files(package).joinpath(resource)
        if not traversable.is_file():
            raise GameMapError(f"Seed image does not exist: {reference}")
        return Path(str(traversable))
    path = Path(reference).expanduser()
    if not path.is_absolute():
        path = source_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise GameMapError(f"Seed image does not exist: {path}")
    return path


def _parse_profiles(doc: dict[str, Any]) -> dict[str, _Profile]:
    raw_profiles = _mapping(doc.get("profiles"), "profiles")
    profiles: dict[str, _Profile] = {}
    for profile_id, raw_value in raw_profiles.items():
        raw = _mapping(raw_value, f"profile {profile_id!r}")
        marking = _mapping(
            raw.get(
                "lane_marking",
                {"style": "DASHED_SINGLE", "color": "WHITE"},
            ),
            f"profile {profile_id!r}.lane_marking",
        )
        directions = tuple(
            str(value).lower()
            for value in _sequence(raw.get("lanes"), f"profile {profile_id!r}.lanes")
        )
        if not directions or any(
            value not in {"forward", "backward"} for value in directions
        ):
            raise GameMapError(
                f"Profile {profile_id!r} lanes must contain forward/backward values"
            )
        profiles[profile_id] = _Profile(
            profile_id=profile_id,
            lane_width_m=_positive_float(
                raw.get("lane_width_m"), f"profile {profile_id!r}.lane_width_m"
            ),
            directions=directions,
            speed_limit_mps=_positive_float(
                raw.get("speed_limit_mps", 13.4),
                f"profile {profile_id!r}.speed_limit_mps",
            ),
            curb=bool(raw.get("curb", True)),
            marking_style=str(marking.get("style", "DASHED_SINGLE")).upper(),
            marking_color=str(marking.get("color", "WHITE")).upper(),
        )
    if not profiles:
        raise GameMapError("Map must define at least one road profile")
    return profiles


def _parse_elements(
    doc: dict[str, Any], profiles: dict[str, _Profile]
) -> tuple[_ElementSpec, ...]:
    result: list[_ElementSpec] = []
    ids: set[str] = set()
    for index, raw_value in enumerate(_sequence(doc.get("elements"), "elements")):
        raw = _mapping(raw_value, f"elements[{index}]")
        element_id = str(raw.get("id", "")).strip()
        if not element_id or element_id in ids:
            raise GameMapError(f"Element id {element_id!r} is empty or duplicated")
        ids.add(element_id)
        element_type = str(raw.get("type", ""))
        if element_type not in {"road_segment", "intersection"}:
            raise GameMapError(
                f"Element {element_id!r} has unsupported type {element_type!r}"
            )
        profile_id = str(raw.get("profile", ""))
        if profile_id not in profiles:
            raise GameMapError(
                f"Element {element_id!r} references unknown profile {profile_id!r}"
            )
        pose = None
        if "pose" in raw:
            raw_pose = _mapping(raw["pose"], f"element {element_id!r}.pose")
            pose = _Pose(
                float(raw_pose.get("x_m", 0.0)),
                float(raw_pose.get("y_m", 0.0)),
                float(raw_pose.get("heading_deg", 0.0)),
            )
        attach_port = attach_to = None
        if "attach" in raw:
            attach = _mapping(raw["attach"], f"element {element_id!r}.attach")
            attach_port = str(attach.get("port", ""))
            attach_to = str(attach.get("to", ""))
            if "." not in attach_to:
                raise GameMapError(
                    f"Element {element_id!r} attach.to must be element.port"
                )
        if (pose is None) == (attach_to is None):
            raise GameMapError(
                f"Element {element_id!r} must define exactly one of pose or attach"
            )
        result.append(
            _ElementSpec(
                element_id,
                element_type,
                profile_id,
                _mapping(raw.get("geometry", {}), f"element {element_id!r}.geometry"),
                pose,
                attach_port,
                attach_to,
            )
        )
    if sum(spec.pose is not None for spec in result) != 1:
        raise GameMapError("Map must contain exactly one pose-anchored element")
    return tuple(result)


def _local_ports(spec: _ElementSpec, profile: _Profile) -> tuple[_Port, ...]:
    if spec.element_type == "road_segment":
        kind = str(spec.geometry.get("kind", "straight"))
        if kind == "straight":
            length = _positive_float(
                spec.geometry.get("length_m"), f"element {spec.element_id!r}.length_m"
            )
            return (_Port("start", 0.0, 0.0, 180.0), _Port("end", length, 0.0, 0.0))
        if kind == "arc":
            radius = _positive_float(
                spec.geometry.get("radius_m"), f"element {spec.element_id!r}.radius_m"
            )
            sweep = float(spec.geometry.get("sweep_deg", 0.0))
            if not 1.0 <= abs(sweep) <= 180.0:
                raise GameMapError(
                    f"Element {spec.element_id!r} arc sweep must be within 1..180 degrees"
                )
            theta = math.radians(sweep)
            sign = 1.0 if sweep > 0.0 else -1.0
            x_m = radius * math.sin(abs(theta))
            y_m = sign * radius * (1.0 - math.cos(abs(theta)))
            return (_Port("start", 0.0, 0.0, 180.0), _Port("end", x_m, y_m, sweep))
        raise GameMapError(
            f"Element {spec.element_id!r} has unsupported road geometry {kind!r}"
        )
    kind = str(spec.geometry.get("kind", "t"))
    if kind not in {"t", "four_way"}:
        raise GameMapError(
            f"Element {spec.element_id!r} has unsupported intersection kind {kind!r}"
        )
    half = profile.width_m
    ports = [
        _Port("east", half, 0.0, 0.0),
        _Port("west", -half, 0.0, 180.0),
        _Port("north", 0.0, half, 90.0),
    ]
    if kind == "four_way":
        ports.append(_Port("south", 0.0, -half, -90.0))
    return tuple(ports)


def _transform_xy(points: np.ndarray, pose: _Pose) -> np.ndarray:
    angle = math.radians(pose.heading_deg)
    rotation = np.asarray(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float64,
    )
    return points @ rotation.T + np.asarray([pose.x_m, pose.y_m])


def _world_ports(
    spec: _ElementSpec, profile: _Profile, pose: _Pose
) -> dict[str, _Port]:
    result: dict[str, _Port] = {}
    for port in _local_ports(spec, profile):
        xy = _transform_xy(np.asarray([[port.x_m, port.y_m]], dtype=np.float64), pose)[
            0
        ]
        result[port.name] = _Port(
            port.name,
            float(xy[0]),
            float(xy[1]),
            (port.heading_deg + pose.heading_deg) % 360.0,
        )
    return result


def _opposite_heading_delta(a_deg: float, b_deg: float) -> float:
    return abs(((a_deg - b_deg - 180.0 + 180.0) % 360.0) - 180.0)


def _resolve_poses(
    specs: tuple[_ElementSpec, ...], profiles: dict[str, _Profile]
) -> tuple[dict[str, _Pose], set[frozenset[str]]]:
    by_id = {spec.element_id: spec for spec in specs}
    poses = {spec.element_id: spec.pose for spec in specs if spec.pose is not None}
    connections: set[frozenset[str]] = set()
    unresolved = {spec.element_id for spec in specs if spec.pose is None}
    while unresolved:
        progressed = False
        for element_id in tuple(unresolved):
            spec = by_id[element_id]
            assert spec.attach_to is not None and spec.attach_port is not None
            target_id, target_port_name = spec.attach_to.rsplit(".", 1)
            if target_id not in by_id:
                raise GameMapError(
                    f"Element {element_id!r} attaches to unknown element {target_id!r}"
                )
            if target_id not in poses:
                continue
            local_ports = {
                port.name: port
                for port in _local_ports(spec, profiles[spec.profile_id])
            }
            if spec.attach_port not in local_ports:
                raise GameMapError(
                    f"Element {element_id!r} has no port {spec.attach_port!r}"
                )
            target_ports = _world_ports(
                by_id[target_id],
                profiles[by_id[target_id].profile_id],
                poses[target_id],
            )
            if target_port_name not in target_ports:
                raise GameMapError(
                    f"Element {target_id!r} has no port {target_port_name!r}"
                )
            if spec.profile_id != by_id[target_id].profile_id:
                raise GameMapError(
                    f"Connection {element_id}.{spec.attach_port} -> {spec.attach_to} uses incompatible profiles"
                )
            local = local_ports[spec.attach_port]
            target = target_ports[target_port_name]
            heading = target.heading_deg + 180.0 - local.heading_deg
            local_xy = _transform_xy(
                np.asarray([[local.x_m, local.y_m]], dtype=np.float64),
                _Pose(0.0, 0.0, heading),
            )[0]
            poses[element_id] = _Pose(
                target.x_m - float(local_xy[0]),
                target.y_m - float(local_xy[1]),
                heading,
            )
            connections.add(
                frozenset((f"{element_id}.{spec.attach_port}", spec.attach_to))
            )
            unresolved.remove(element_id)
            progressed = True
        if not progressed:
            raise GameMapError(
                "Element attachments are cyclic or disconnected from the anchor"
            )
    return {
        key: value for key, value in poses.items() if value is not None
    }, connections


def _validate_extra_connections(
    doc: dict[str, Any],
    specs: tuple[_ElementSpec, ...],
    profiles: dict[str, _Profile],
    poses: dict[str, _Pose],
    connections: set[frozenset[str]],
) -> None:
    by_id = {spec.element_id: spec for spec in specs}
    all_ports = {
        element_id: _world_ports(
            by_id[element_id], profiles[by_id[element_id].profile_id], pose
        )
        for element_id, pose in poses.items()
    }
    for index, raw_value in enumerate(doc.get("connections", []) or []):
        raw = _mapping(raw_value, f"connections[{index}]")
        endpoints = (str(raw.get("a", "")), str(raw.get("b", "")))
        resolved: list[tuple[str, _Port]] = []
        for endpoint in endpoints:
            if "." not in endpoint:
                raise GameMapError(
                    f"Connection endpoint {endpoint!r} must be element.port"
                )
            element_id, port_name = endpoint.rsplit(".", 1)
            if element_id not in all_ports or port_name not in all_ports[element_id]:
                raise GameMapError(f"Connection references unknown port {endpoint!r}")
            resolved.append((element_id, all_ports[element_id][port_name]))
        first, second = resolved[0][1], resolved[1][1]
        gap = math.hypot(first.x_m - second.x_m, first.y_m - second.y_m)
        angle_error = _opposite_heading_delta(first.heading_deg, second.heading_deg)
        if gap > _PLACEMENT_TOLERANCE_M or angle_error > _ANGLE_TOLERANCE_DEG:
            raise GameMapError(
                f"Connection {endpoints[0]} -> {endpoints[1]} does not close: gap={gap:.4f}m, angle_error={angle_error:.4f}deg"
            )
        if by_id[resolved[0][0]].profile_id != by_id[resolved[1][0]].profile_id:
            raise GameMapError(
                f"Connection {endpoints[0]} -> {endpoints[1]} uses incompatible profiles"
            )
        connections.add(frozenset(endpoints))
    used: set[str] = set()
    for connection in connections:
        for endpoint in connection:
            if endpoint in used:
                raise GameMapError(f"Port {endpoint!r} is connected more than once")
            used.add(endpoint)


def _sample_centerline(spec: _ElementSpec) -> np.ndarray:
    kind = str(spec.geometry.get("kind", "straight"))
    if kind == "straight":
        length = float(spec.geometry["length_m"])
        count = max(2, int(math.ceil(length / _SAMPLE_SPACING_M)) + 1)
        return np.column_stack((np.linspace(0.0, length, count), np.zeros(count)))
    radius = float(spec.geometry["radius_m"])
    sweep = math.radians(float(spec.geometry["sweep_deg"]))
    arc_length = radius * abs(sweep)
    count = max(3, int(math.ceil(arc_length / _SAMPLE_SPACING_M)) + 1)
    angles = np.linspace(0.0, abs(sweep), count)
    sign = 1.0 if sweep > 0.0 else -1.0
    return np.column_stack(
        (radius * np.sin(angles), sign * radius * (1.0 - np.cos(angles)))
    )


def _offset_polyline(points: np.ndarray, offset_m: float) -> np.ndarray:
    tangents = np.gradient(points, axis=0)
    lengths = np.linalg.norm(tangents, axis=1)
    tangents = tangents / np.maximum(lengths[:, None], 1.0e-9)
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    return points + normals * offset_m


def _xyz(points_xy: np.ndarray) -> np.ndarray:
    return np.column_stack((points_xy, np.zeros(len(points_xy)))).astype(np.float32)


def _surface_for_road(centerline: np.ndarray, width_m: float) -> np.ndarray:
    left = _offset_polyline(centerline, width_m * 0.5)
    right = _offset_polyline(centerline, -width_m * 0.5)
    return _xyz(np.concatenate((left, right[::-1], left[:1]), axis=0))


def _segments(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.empty((0, 2, 3), dtype=np.float32)
    return np.stack((points[:-1], points[1:]), axis=1).astype(np.float32)


def _connected_endpoints(connections: set[frozenset[str]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for connection in connections:
        first, second = tuple(connection)
        result[first] = second
        result[second] = first
    return result


def _road_geometry(
    spec: _ElementSpec, profile: _Profile, pose: _Pose, connected: dict[str, str]
) -> tuple[GameMapElement, list[_LaneBuild], list[np.ndarray]]:
    local_center = _sample_centerline(spec)
    world_center = _transform_xy(local_center, pose)
    surface = _surface_for_road(world_center, profile.width_m)
    ports = _world_ports(spec, profile, pose)
    lane_builds: list[_LaneBuild] = []
    for index, direction in enumerate(profile.directions):
        offset = (
            len(profile.directions) - 1
        ) * profile.lane_width_m * 0.5 - index * profile.lane_width_m
        lane_center = _offset_polyline(world_center, offset)
        start_port, end_port = "start", "end"
        if direction == "backward":
            lane_center = lane_center[::-1]
            start_port, end_port = end_port, start_port
        left = _offset_polyline(lane_center, profile.lane_width_m * 0.5)
        right = _offset_polyline(lane_center, -profile.lane_width_m * 0.5)
        lane_builds.append(
            _LaneBuild(
                f"{spec.element_id}:lane:{index}",
                spec.element_id,
                _xyz(lane_center),
                _xyz(left),
                _xyz(right),
                profile.speed_limit_mps,
                profile.marking_style,
                profile.marking_color,
                start_port,
                end_port,
                [],
                True,
            )
        )
    curb_segments: list[np.ndarray] = []
    if profile.curb:
        left_side = _xyz(_offset_polyline(world_center, profile.width_m * 0.5))
        right_side = _xyz(_offset_polyline(world_center, -profile.width_m * 0.5))
        curb_segments.extend((_segments(left_side), _segments(right_side)))
        for port_name, endpoint_index in (("start", 0), ("end", -1)):
            endpoint = f"{spec.element_id}.{port_name}"
            if endpoint not in connected:
                curb_segments.append(
                    np.asarray(
                        [[left_side[endpoint_index], right_side[endpoint_index]]],
                        dtype=np.float32,
                    )
                )
    element_ports = tuple(
        (
            name,
            port.x_m,
            port.y_m,
            port.heading_deg,
            f"{spec.element_id}.{name}" in connected,
        )
        for name, port in ports.items()
    )
    return (
        GameMapElement(
            spec.element_id, spec.element_type, spec.profile_id, surface, element_ports
        ),
        lane_builds,
        curb_segments,
    )


def _intersection_geometry(
    spec: _ElementSpec, profile: _Profile, pose: _Pose, connected: dict[str, str]
) -> tuple[GameMapElement, list[np.ndarray]]:
    half = profile.width_m
    local = np.asarray(
        [[-half, -half], [half, -half], [half, half], [-half, half], [-half, -half]],
        dtype=np.float64,
    )
    surface = _xyz(_transform_xy(local, pose))
    ports = _world_ports(spec, profile, pose)
    curbs: list[np.ndarray] = []
    if profile.curb:
        corners = surface[:-1]
        for index in range(4):
            start = corners[index]
            end = corners[(index + 1) % 4]
            midpoint = 0.5 * (start + end)
            side_heading = (
                math.degrees(
                    math.atan2(
                        float(midpoint[1] - pose.y_m), float(midpoint[0] - pose.x_m)
                    )
                )
                + 360.0
            ) % 360.0
            opening = next(
                (
                    name
                    for name, port in ports.items()
                    if abs(((port.heading_deg - side_heading + 180.0) % 360.0) - 180.0)
                    < 1.0
                ),
                None,
            )
            if opening is not None and f"{spec.element_id}.{opening}" in connected:
                direction = end - start
                length = float(np.linalg.norm(direction[:2]))
                unit = direction / max(length, 1.0e-9)
                gap = profile.width_m * 0.5
                curbs.append(
                    np.asarray(
                        [[start, midpoint - unit * gap], [midpoint + unit * gap, end]],
                        dtype=np.float32,
                    )
                )
            else:
                curbs.append(np.asarray([[start, end]], dtype=np.float32))
    element_ports = tuple(
        (
            name,
            port.x_m,
            port.y_m,
            port.heading_deg,
            f"{spec.element_id}.{name}" in connected,
        )
        for name, port in ports.items()
    )
    return GameMapElement(
        spec.element_id, spec.element_type, spec.profile_id, surface, element_ports
    ), curbs


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:2] - b[:2]))


def _bezier(start: np.ndarray, control: np.ndarray, end: np.ndarray) -> np.ndarray:
    t = np.linspace(0.0, 1.0, 8, dtype=np.float32)[:, None]
    return ((1.0 - t) ** 2 * start + 2.0 * (1.0 - t) * t * control + t**2 * end).astype(
        np.float32
    )


def _wire_lane_successors(
    lanes: list[_LaneBuild],
    specs: tuple[_ElementSpec, ...],
    poses: dict[str, _Pose],
    connections: dict[str, str],
) -> None:
    by_id = {spec.element_id: spec for spec in specs}
    lane_by_id = {lane.lane_id: lane for lane in lanes}
    endpoints: dict[str, list[tuple[_LaneBuild, str]]] = {}
    for lane in lanes:
        endpoints.setdefault(f"{lane.element_id}.{lane.start_port}", []).append(
            (lane, "start")
        )
        endpoints.setdefault(f"{lane.element_id}.{lane.end_port}", []).append(
            (lane, "end")
        )
    intersection_ports: dict[str, dict[str, list[tuple[_LaneBuild, str]]]] = {}
    for endpoint, other in connections.items():
        element_id, port_name = endpoint.rsplit(".", 1)
        other_element, _ = other.rsplit(".", 1)
        if (
            by_id[element_id].element_type == "intersection"
            and by_id[other_element].element_type == "road_segment"
        ):
            intersection_ports.setdefault(element_id, {})[port_name] = endpoints.get(
                other, []
            )
    connector_count = 0
    for intersection_id, ports in intersection_ports.items():
        center = np.asarray(
            [poses[intersection_id].x_m, poses[intersection_id].y_m, 0.0],
            dtype=np.float32,
        )
        incoming = [
            (port, lane)
            for port, values in ports.items()
            for lane, end_kind in values
            if end_kind == "end"
        ]
        outgoing = [
            (port, lane)
            for port, values in ports.items()
            for lane, end_kind in values
            if end_kind == "start"
        ]
        for in_port, source in incoming:
            for out_port, target in outgoing:
                if in_port == out_port:
                    continue
                centerline = _bezier(
                    source.centerline[-1], center, target.centerline[0]
                )
                width = _distance(source.left_edge[-1], source.right_edge[-1])
                left = _xyz(_offset_polyline(centerline[:, :2], width * 0.5))
                right = _xyz(_offset_polyline(centerline[:, :2], -width * 0.5))
                connector_id = f"{intersection_id}:connector:{connector_count}"
                connector_count += 1
                connector = _LaneBuild(
                    connector_id,
                    intersection_id,
                    centerline,
                    left,
                    right,
                    source.speed_limit_mps,
                    source.marking_style,
                    source.marking_color,
                    "",
                    "",
                    [target.lane_id],
                    False,
                )
                lanes.append(connector)
                lane_by_id[connector_id] = connector
                source.successors.append(connector_id)
    for endpoint, other in connections.items():
        element_id, _ = endpoint.rsplit(".", 1)
        other_id, _ = other.rsplit(".", 1)
        if (
            by_id[element_id].element_type != "road_segment"
            or by_id[other_id].element_type != "road_segment"
        ):
            continue
        for source, source_kind in endpoints.get(endpoint, []):
            if source_kind != "end":
                continue
            candidates = [
                lane for lane, kind in endpoints.get(other, []) if kind == "start"
            ]
            if candidates:
                target = min(
                    candidates,
                    key=lambda lane: _distance(
                        source.centerline[-1], lane.centerline[0]
                    ),
                )
                source.successors.append(target.lane_id)


def _spawn_on_lane(
    raw: dict[str, Any], source_path: Path, lane_by_id: dict[str, _LaneBuild]
) -> GameMapSpawn:
    spawn_id = str(raw.get("id", "")).strip()
    element_id = str(raw.get("element", "")).strip()
    lane_index = int(raw.get("lane", 0))
    lane_id = f"{element_id}:lane:{lane_index}"
    if lane_id not in lane_by_id:
        raise GameMapError(f"Spawn {spawn_id!r} references unknown lane {lane_id!r}")
    lane = lane_by_id[lane_id]
    distance_m = max(0.0, float(raw.get("distance_m", 0.0)))
    segment_lengths = np.linalg.norm(np.diff(lane.centerline[:, :2], axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    distance_m = min(distance_m, float(cumulative[-1]))
    index = min(
        int(np.searchsorted(cumulative, distance_m, side="right") - 1),
        len(segment_lengths) - 1,
    )
    index = max(0, index)
    span = max(1.0e-9, float(segment_lengths[index]))
    alpha = (distance_m - float(cumulative[index])) / span
    position = (1.0 - alpha) * lane.centerline[index] + alpha * lane.centerline[
        index + 1
    ]
    vector = lane.centerline[index + 1] - lane.centerline[index]
    yaw = math.atan2(float(vector[1]), float(vector[0]))
    return GameMapSpawn(
        spawn_id,
        lane_id,
        distance_m,
        position.astype(np.float32),
        yaw,
        _parse_variants(raw, source_path),
    )


def load_game_map(path: Path) -> ResolvedGameMap:
    """Parse, validate, and resolve a semantic game map."""
    source_path = Path(path).expanduser().resolve()
    doc = _read_document(source_path)
    if doc.get("schema_version") != _SCHEMA_VERSION:
        raise GameMapError(
            f"Unsupported schema_version {doc.get('schema_version')!r}; expected {_SCHEMA_VERSION}"
        )
    map_id = str(doc.get("id", "")).strip()
    if not map_id:
        raise GameMapError("Map id must not be empty")
    profiles = _parse_profiles(doc)
    specs = _parse_elements(doc, profiles)
    poses, connection_pairs = _resolve_poses(specs, profiles)
    _validate_extra_connections(doc, specs, profiles, poses, connection_pairs)
    connections = _connected_endpoints(connection_pairs)
    elements: list[GameMapElement] = []
    lane_builds: list[_LaneBuild] = []
    collision_groups: list[np.ndarray] = []
    for spec in specs:
        profile = profiles[spec.profile_id]
        if spec.element_type == "road_segment":
            element, built_lanes, curbs = _road_geometry(
                spec, profile, poses[spec.element_id], connections
            )
            elements.append(element)
            lane_builds.extend(built_lanes)
            collision_groups.extend(curbs)
        else:
            element, curbs = _intersection_geometry(
                spec, profile, poses[spec.element_id], connections
            )
            elements.append(element)
            collision_groups.extend(curbs)
    _wire_lane_successors(lane_builds, specs, poses, connections)
    lane_by_id = {lane.lane_id: lane for lane in lane_builds}
    spawns_raw = _sequence(doc.get("spawns"), "spawns")
    if not spawns_raw:
        raise GameMapError("Map must define at least one spawn")
    spawns = tuple(
        _spawn_on_lane(_mapping(raw, f"spawns[{index}]"), source_path, lane_by_id)
        for index, raw in enumerate(spawns_raw)
    )
    lanes = tuple(
        GameMapLane(
            lane_id=lane.lane_id,
            element_id=lane.element_id,
            centerline_world=lane.centerline,
            left_edge_world=lane.left_edge,
            right_edge_world=lane.right_edge,
            speed_limit_mps=lane.speed_limit_mps,
            marking_style=lane.marking_style,
            marking_color=lane.marking_color,
            successor_ids=tuple(dict.fromkeys(lane.successors)),
            allows_taxi_stops=lane.allows_taxi_stops,
        )
        for lane in lane_builds
    )
    nonempty = [group.reshape(-1, 2, 3) for group in collision_groups if group.size]
    collisions = (
        np.concatenate(nonempty, axis=0).astype(np.float32)
        if nonempty
        else np.empty((0, 2, 3), dtype=np.float32)
    )
    all_points = np.concatenate([element.surface_world for element in elements], axis=0)
    x_min, y_min = np.min(all_points[:, :2], axis=0) - 20.0
    x_max, y_max = np.max(all_points[:, :2], axis=0) + 20.0
    ground_vertices = np.asarray(
        [
            [x_min, y_min, 0.0],
            [x_max, y_min, 0.0],
            [x_max, y_max, 0.0],
            [x_min, y_max, 0.0],
        ],
        dtype=np.float32,
    )
    ground_faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return ResolvedGameMap(
        _SCHEMA_VERSION,
        map_id,
        str(doc.get("name", map_id)),
        source_path,
        lanes,
        tuple(elements),
        collisions,
        ground_vertices,
        ground_faces,
        spawns,
    )
