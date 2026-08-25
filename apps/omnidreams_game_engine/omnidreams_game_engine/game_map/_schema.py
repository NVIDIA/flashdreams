# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Strict field, profile, and shared configuration parsing for game maps."""

from __future__ import annotations

import math
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from omnidreams_game_engine.game_map.types import (
    GameMapLinearAttributes,
    GameMapVisualVariant,
)

GAME_MAP_SUFFIX = ".robotaxi.yaml"
"""Filename suffix for authored node-graph game maps."""

_SCHEMA_VERSION = 1
_REQUIRED_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "id",
        "name",
        "compiler",
        "nodes",
        "roads",
        "spawns",
    }
)
_OPTIONAL_ROOT_FIELDS = frozenset(
    {"profiles", "race_courses", "traffic", "traffic_count"}
)

_PROFILE_ATTRIBUTE_FIELDS = frozenset(
    {
        "lane_width_m",
        "curb_offset_m",
        "lanes",
        "speed_limit_mps",
        "curb",
        "lane_marking",
        "divider_markings",
        "culdesac_radius_m",
    }
)


class GameMapError(ValueError):
    """Invalid semantic game-map definition."""


@dataclass(frozen=True)
class GameMapHeader:
    """Game-map metadata read without compiling geometry."""

    map_id: str
    name: str
    variants: tuple[GameMapVisualVariant, ...]
    source_path: Path
    race_course_ids: tuple[str, ...] = ()


def _parse_race_course_ids(doc: dict[str, Any]) -> tuple[str, ...]:
    """Read stable course identifiers without compiling map geometry."""
    if "race_courses" not in doc:
        return ()
    raw_courses = _sequence(doc["race_courses"], "race_courses")
    if not raw_courses:
        raise GameMapError("race_courses must contain at least one course")
    course_ids: list[str] = []
    for index, value in enumerate(raw_courses):
        raw = _mapping(value, f"race_courses[{index}]")
        course_id = str(raw.get("id", "")).strip()
        if not course_id or course_id in course_ids:
            raise GameMapError(f"Race course id {course_id!r} is empty or duplicated")
        course_ids.append(course_id)
    return tuple(course_ids)


@dataclass(frozen=True)
class _CompilerSettings:
    sample_spacing_m: float
    ground_margin_m: float
    intersection_connector_samples: int

    def as_dict(self) -> dict[str, object]:
        """Return settings as stable cache metadata."""
        return dict(self.__dict__)


@dataclass(frozen=True)
class _Profile:
    """Partial reusable defaults for resolved element attributes."""

    profile_id: str
    """Stable author-defined profile identifier."""

    values: dict[str, object]
    """Validated partial attribute values."""


@dataclass
class _LaneBuild:
    lane_id: str
    element_id: str
    centerline: np.ndarray
    left_edge: np.ndarray
    right_edge: np.ndarray
    roadside_edge: np.ndarray
    speed_limit_mps: float
    marking_style: str
    marking_color: str
    start_endpoint: str
    end_endpoint: str
    successors: list[str]
    allows_taxi_stops: bool
    left_marking_style: str | None = None
    left_marking_color: str | None = None
    right_marking_style: str | None = None
    right_marking_color: str | None = None
    conditioning_visible: bool = True


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GameMapError(f"{context} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise GameMapError(f"{context} keys must be strings")
    return dict(value)


def _sequence(value: object, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise GameMapError(f"{context} must be a sequence")
    return value


def _positive_float(value: object, context: str) -> float:
    number = _finite_float(value, context)
    if number <= 0.0:
        raise GameMapError(f"{context} must be positive")
    return number


def _nonnegative_float(value: object, context: str) -> float:
    number = _finite_float(value, context)
    if number < 0.0:
        raise GameMapError(f"{context} must be nonnegative")
    return number


def _finite_float(value: object, context: str) -> float:
    if isinstance(value, bool):
        raise GameMapError(f"{context} must be a number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise GameMapError(f"{context} must be a number") from exc
    if not math.isfinite(number):
        raise GameMapError(f"{context} must be finite")
    return number


def _read_document(path: Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise GameMapError(f"Game-map path does not exist or is not a file: {path}")
    if not path.name.endswith(GAME_MAP_SUFFIX):
        raise GameMapError(f"Game maps must use the {GAME_MAP_SUFFIX} suffix")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise GameMapError(f"Could not parse {path}: {exc}") from exc
    return _mapping(raw, "map document")


def _parse_map_identity(doc: dict[str, Any]) -> tuple[str, str]:
    version = doc.get("schema_version")
    if version != _SCHEMA_VERSION:
        raise GameMapError(
            f"Unsupported schema_version {version!r}; expected {_SCHEMA_VERSION}"
        )
    fields = set(doc)
    missing = _REQUIRED_ROOT_FIELDS - fields
    if missing:
        raise GameMapError(f"Map is missing required fields {sorted(missing)}")
    unknown = fields - (_REQUIRED_ROOT_FIELDS | _OPTIONAL_ROOT_FIELDS)
    if unknown:
        raise GameMapError(f"Map has unknown fields {sorted(unknown)}")
    map_id = str(doc["id"]).strip()
    name = str(doc["name"]).strip()
    if not map_id:
        raise GameMapError("Map id must not be empty")
    if not name:
        raise GameMapError("Map name must not be empty")
    return map_id, name


def _parse_variants(
    raw_spawn: dict[str, Any], source_path: Path
) -> tuple[GameMapVisualVariant, ...]:
    variants_raw = _mapping(raw_spawn.get("variants"), "spawn.variants")
    if "default" not in variants_raw:
        raise GameMapError("Every spawn must define a default visual variant")
    variants: list[GameMapVisualVariant] = []
    for name, raw_variant in variants_raw.items():
        variant = _mapping(raw_variant, f"variant {name!r}")
        unknown = set(variant) - {"image", "prompt"}
        if unknown:
            raise GameMapError(f"Variant {name!r} has unknown fields {sorted(unknown)}")
        image_value = variant.get("image")
        image = None if image_value is None else str(image_value).strip()
        prompt = str(variant.get("prompt", "")).strip()
        if not prompt:
            raise GameMapError(f"Variant {name!r} requires a non-empty prompt")
        if image_value is not None and not image:
            raise GameMapError(f"Variant {name!r} image must not be empty")
        if image is not None:
            resolve_seed_asset(source_path, image)
        variants.append(GameMapVisualVariant(name=name, image=image, prompt=prompt))
    variants.sort(key=lambda item: (item.name != "default", item.name))
    return tuple(variants)


def load_game_map_header(path: Path) -> GameMapHeader:
    """Load map name and default-spawn variants without resolving geometry."""
    source_path = Path(path).expanduser().resolve()
    doc = _read_document(source_path)
    map_id, name = _parse_map_identity(doc)
    spawns = _sequence(doc.get("spawns"), "spawns")
    if not spawns:
        raise GameMapError("Map must define at least one spawn")
    first_spawn = _mapping(spawns[0], "spawns[0]")
    return GameMapHeader(
        map_id=map_id,
        name=name,
        variants=_parse_variants(first_spawn, source_path),
        source_path=source_path,
        race_course_ids=_parse_race_course_ids(doc),
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


def _parse_attribute_values(raw: dict[str, Any], context: str) -> dict[str, object]:
    """Validate and normalize partial profile-compatible attributes."""
    unknown = set(raw) - _PROFILE_ATTRIBUTE_FIELDS
    if unknown:
        raise GameMapError(f"{context} has unknown attributes {sorted(unknown)}")
    result: dict[str, object] = {}
    for key in (
        "lane_width_m",
        "speed_limit_mps",
        "culdesac_radius_m",
    ):
        if key in raw:
            result[key] = _positive_float(raw[key], f"{context}.{key}")
    if "curb_offset_m" in raw:
        result["curb_offset_m"] = _nonnegative_float(
            raw["curb_offset_m"], f"{context}.curb_offset_m"
        )
    if "curb" in raw:
        if type(raw["curb"]) is not bool:
            raise GameMapError(f"{context}.curb must be a boolean")
        result["curb"] = raw["curb"]
    if "lanes" in raw:
        directions = tuple(
            str(value).lower() for value in _sequence(raw["lanes"], f"{context}.lanes")
        )
        if not directions or any(
            value not in {"forward", "backward"} for value in directions
        ):
            raise GameMapError(f"{context}.lanes must contain forward/backward values")
        result["lanes"] = directions
    if "lane_marking" in raw:
        marking = _mapping(raw["lane_marking"], f"{context}.lane_marking")
        if set(marking) != {"style", "color"}:
            raise GameMapError(f"{context}.lane_marking requires style and color")
        result["lane_marking"] = (
            str(marking["style"]).upper(),
            str(marking["color"]).upper(),
        )
    if "divider_markings" in raw:
        dividers: list[tuple[str, str]] = []
        for index, value in enumerate(
            _sequence(raw["divider_markings"], f"{context}.divider_markings")
        ):
            divider = _mapping(value, f"{context}.divider_markings[{index}]")
            if set(divider) != {"style", "color"}:
                raise GameMapError(
                    f"{context}.divider_markings[{index}] requires style and color"
                )
            dividers.append(
                (str(divider["style"]).upper(), str(divider["color"]).upper())
            )
        result["divider_markings"] = tuple(dividers)
    return result


def _parse_profiles(doc: dict[str, Any]) -> dict[str, _Profile]:
    """Parse optional partial profile defaults."""
    raw_profiles = _mapping(doc.get("profiles", {}), "profiles")
    profiles: dict[str, _Profile] = {}
    for profile_id, raw_value in raw_profiles.items():
        if not profile_id:
            raise GameMapError("Profile ids must not be empty")
        raw = _mapping(raw_value, f"profile {profile_id!r}")
        profiles[profile_id] = _Profile(
            profile_id=profile_id,
            values=_parse_attribute_values(raw, f"profile {profile_id!r}"),
        )
    return profiles


def _parse_compiler_settings(doc: dict[str, Any]) -> _CompilerSettings:
    raw = _mapping(doc.get("compiler"), "compiler")
    expected = {
        "sample_spacing_m",
        "ground_margin_m",
        "intersection_connector_samples",
    }
    if set(raw) != expected:
        raise GameMapError(f"compiler must contain exactly {sorted(expected)}")
    samples = raw["intersection_connector_samples"]
    if type(samples) is not int or samples < 2:
        raise GameMapError(
            "compiler.intersection_connector_samples must be an integer >= 2"
        )
    return _CompilerSettings(
        sample_spacing_m=_positive_float(
            raw["sample_spacing_m"], "compiler.sample_spacing_m"
        ),
        ground_margin_m=_nonnegative_float(
            raw["ground_margin_m"], "compiler.ground_margin_m"
        ),
        intersection_connector_samples=samples,
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


def _lane_edge_markings(
    attributes: GameMapLinearAttributes, index: int, direction: str
) -> tuple[tuple[str, str], tuple[str, str]]:
    virtual = ("VIRTUAL", "WHITE")
    above = attributes.divider_markings[index - 1] if index > 0 else virtual
    below = (
        attributes.divider_markings[index]
        if index < len(attributes.directions) - 1
        else virtual
    )
    return (below, above) if direction == "backward" else (above, below)


def _bezier(
    start: np.ndarray, control: np.ndarray, end: np.ndarray, samples: int
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, samples, dtype=np.float32)[:, None]
    return ((1.0 - t) ** 2 * start + 2.0 * (1.0 - t) * t * control + t**2 * end).astype(
        np.float32
    )
