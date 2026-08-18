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

from omnidreams_game_engine.game_map.types import GameMapVisualVariant

_SCHEMA_VERSION = 1


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
class _CompilerSettings:
    sample_spacing_m: float
    ground_margin_m: float
    intersection_connector_samples: int
    parking_turnaround_width_multiplier: float
    parking_turnaround_min_depth_m: float
    parking_turnaround_control_inset_m: float

    def as_dict(self) -> dict[str, object]:
        """Return settings as stable cache metadata."""
        return dict(self.__dict__)


@dataclass(frozen=True)
class _Profile:
    profile_id: str
    lane_width_m: float
    curb_offset_m: float
    directions: tuple[str, ...]
    speed_limit_mps: float
    curb: bool
    marking_style: str
    marking_color: str
    divider_markings: tuple[tuple[str, str], ...]

    @property
    def width_m(self) -> float:
        return self.lane_width_m * len(self.directions)

    @property
    def surface_width_m(self) -> float:
        return self.width_m + 2.0 * self.curb_offset_m


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
    start_port: str
    end_port: str
    successors: list[str]
    allows_taxi_stops: bool
    left_marking_style: str | None = None
    left_marking_color: str | None = None
    right_marking_style: str | None = None
    right_marking_color: str | None = None


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GameMapError(f"{context} must be a mapping")
    return {str(key): item for key, item in value.items()}


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
        if set(variant) != {"image", "prompt"}:
            raise GameMapError(f"Variant {name!r} requires exactly image and prompt")
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
        required = {
            "lane_width_m",
            "curb_offset_m",
            "lanes",
            "speed_limit_mps",
            "curb",
            "lane_marking",
            "divider_markings",
        }
        if set(raw) != required:
            raise GameMapError(
                f"Profile {profile_id!r} must contain exactly {sorted(required)}"
            )
        marking = _mapping(raw["lane_marking"], f"profile {profile_id!r}.lane_marking")
        if set(marking) != {"style", "color"}:
            raise GameMapError(
                f"Profile {profile_id!r}.lane_marking requires style and color"
            )
        directions = tuple(
            str(value).lower()
            for value in _sequence(raw["lanes"], f"profile {profile_id!r}.lanes")
        )
        if not directions or any(
            value not in {"forward", "backward"} for value in directions
        ):
            raise GameMapError(
                f"Profile {profile_id!r} lanes must contain forward/backward values"
            )
        divider_values = _sequence(
            raw["divider_markings"], f"profile {profile_id!r}.divider_markings"
        )
        if len(divider_values) != len(directions) - 1:
            raise GameMapError(
                f"Profile {profile_id!r}.divider_markings must contain one entry per adjacent lane pair"
            )
        dividers: list[tuple[str, str]] = []
        for index, value in enumerate(divider_values):
            divider = _mapping(
                value, f"profile {profile_id!r}.divider_markings[{index}]"
            )
            if set(divider) != {"style", "color"}:
                raise GameMapError(
                    f"Profile {profile_id!r}.divider_markings[{index}] requires style and color"
                )
            dividers.append(
                (str(divider["style"]).upper(), str(divider["color"]).upper())
            )
        if type(raw["curb"]) is not bool:
            raise GameMapError(f"Profile {profile_id!r}.curb must be a boolean")
        profiles[profile_id] = _Profile(
            profile_id=profile_id,
            lane_width_m=_positive_float(
                raw["lane_width_m"], f"profile {profile_id!r}.lane_width_m"
            ),
            curb_offset_m=_nonnegative_float(
                raw["curb_offset_m"], f"profile {profile_id!r}.curb_offset_m"
            ),
            directions=directions,
            speed_limit_mps=_positive_float(
                raw["speed_limit_mps"], f"profile {profile_id!r}.speed_limit_mps"
            ),
            curb=raw["curb"],
            marking_style=str(marking["style"]).upper(),
            marking_color=str(marking["color"]).upper(),
            divider_markings=tuple(dividers),
        )
    if not profiles:
        raise GameMapError("Map must define at least one road profile")
    return profiles


def _parse_compiler_settings(doc: dict[str, Any]) -> _CompilerSettings:
    raw = _mapping(doc.get("compiler"), "compiler")
    expected = {
        "sample_spacing_m",
        "ground_margin_m",
        "intersection_connector_samples",
        "parking_lot",
    }
    if set(raw) != expected:
        raise GameMapError(f"compiler must contain exactly {sorted(expected)}")
    parking = _mapping(raw["parking_lot"], "compiler.parking_lot")
    parking_expected = {
        "turnaround_width_multiplier",
        "turnaround_min_depth_m",
        "turnaround_control_inset_m",
    }
    if set(parking) != parking_expected:
        raise GameMapError(
            f"compiler.parking_lot must contain exactly {sorted(parking_expected)}"
        )
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
        parking_turnaround_width_multiplier=_positive_float(
            parking["turnaround_width_multiplier"],
            "compiler.parking_lot.turnaround_width_multiplier",
        ),
        parking_turnaround_min_depth_m=_positive_float(
            parking["turnaround_min_depth_m"],
            "compiler.parking_lot.turnaround_min_depth_m",
        ),
        parking_turnaround_control_inset_m=_nonnegative_float(
            parking["turnaround_control_inset_m"],
            "compiler.parking_lot.turnaround_control_inset_m",
        ),
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
    profile: _Profile, index: int, direction: str
) -> tuple[tuple[str, str], tuple[str, str]]:
    virtual = ("VIRTUAL", "WHITE")
    above = profile.divider_markings[index - 1] if index > 0 else virtual
    below = (
        profile.divider_markings[index]
        if index < len(profile.directions) - 1
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
