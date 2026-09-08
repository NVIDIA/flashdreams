# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Content-addressed ClipGT compilation for semantic game maps."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from filelock import FileLock
from PIL import Image
from shapely.geometry import LineString, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import substring, unary_union

from omnidreams_game_engine import camera_defaults
from omnidreams_game_engine.camera_defaults import DEFAULT_FRONT_CAMERA_LOGICAL_NAME
from omnidreams_game_engine.game_map import spawn_render
from omnidreams_game_engine.game_map._schema import resolve_seed_asset
from omnidreams_game_engine.game_map.loader import load_game_map
from omnidreams_game_engine.game_map.types import (
    ResolvedGameMap,
    game_map_to_dict,
)
from omnidreams_game_engine.math3d import rig_pose_from_state
from omnidreams_game_engine.ply_io import save_mesh_vf
from omnidreams_game_engine.scene_fixture import _calibration_row

# Pre-release maps and compiler output stay at version 1. Do not increment this
# during development; a future release process owns version changes.
_COMPILER_VERSION = "1"
_START_TIMESTAMP_US = 1_700_000_000_000_000
_BOUNDARY_CHUNK_CORE_LENGTH_M = 80.0
"""Maximum non-overlapping span represented by one BEV boundary record."""

_BOUNDARY_CHUNK_OVERLAP_M = 5.0
"""Per-side overlap that hides record endpoints without leaving the perimeter."""


@dataclass(frozen=True)
class CompiledGameMap:
    """Resolved map and its private renderer archive."""

    source_path: Path
    """Canonical semantic YAML path."""

    archive_path: Path
    """Content-addressed private USDZ/ClipGT archive."""

    game_map: ResolvedGameMap
    """Resolved semantic runtime map."""

    cache_hit: bool
    """Whether compilation reused an existing archive."""


def _cache_root() -> Path:
    return (
        Path(
            os.path.expanduser(
                os.environ.get("FLASHDREAMS_CACHE_DIR", "~/.cache/flashdreams")
            )
        )
        / "omnidreams-game-engine"
        / "game-maps"
    )


def _digest(game_map: ResolvedGameMap) -> str:
    hasher = hashlib.sha256()
    hasher.update(_COMPILER_VERSION.encode())
    hasher.update(Path(__file__).read_bytes())
    hasher.update(game_map.source_path.read_bytes())
    resolved = game_map_to_dict(game_map)
    resolved.pop("source_path", None)
    hasher.update(json.dumps(resolved, sort_keys=True, separators=(",", ":")).encode())
    for spawn in game_map.spawns:
        for variant in spawn.variants:
            hasher.update(variant.name.encode())
            hasher.update(variant.prompt.encode())
            if variant.image is None:
                hasher.update(b"generated-spawn-first-frame")
                hasher.update(spawn_render.SPAWN_RENDERER_VERSION.encode())
                hasher.update(Path(spawn_render.__file__).read_bytes())
                hasher.update(Path(camera_defaults.__file__).read_bytes())
            else:
                asset = resolve_seed_asset(game_map.source_path, variant.image)
                hasher.update(asset.read_bytes())
    return hasher.hexdigest()


def _point(point: np.ndarray) -> dict[str, float]:
    return {"x": float(point[0]), "y": float(point[1]), "z": float(point[2])}


def _key(game_map: ResolvedGameMap, label: str) -> dict[str, str]:
    return {
        "clip_id": game_map.map_id,
        "label_class_id": label,
        "map_id": game_map.map_id,
        "map_id_version": f"v{game_map.schema_version}",
    }


def _lane_rows(game_map: ResolvedGameMap) -> list[dict[str, object]]:
    shared_edges = {
        lane_edge
        for divider in game_map.lane_dividers
        for lane_edge in divider.lane_edges
    }
    rows: list[dict[str, object]] = []
    for lane in game_map.lanes:
        if not lane.conditioning_visible:
            continue
        left_shared = (lane.lane_id, "left") in shared_edges
        right_shared = (lane.lane_id, "right") in shared_edges
        left_style, left_color = lane.left_marking_style, lane.left_marking_color
        right_style, right_color = lane.right_marking_style, lane.right_marking_color
        rows.append(
            {
                "key": _key(game_map, lane.lane_id),
                "lane": {
                    "left_rail": [_point(point) for point in lane.left_edge_world],
                    "right_rail": [_point(point) for point in lane.right_edge_world],
                    "vehicle_types": ["CAR"],
                    "map_end": "NONE",
                    "use_types": [],
                    "left_edge_styles": (
                        [left_style if left_shared else "VIRTUAL"]
                        if lane.allows_taxi_stops
                        else []
                    ),
                    "right_edge_styles": (
                        [right_style if right_shared else "VIRTUAL"]
                        if lane.allows_taxi_stops
                        else []
                    ),
                    "left_edge_colors": [left_color if left_shared else "WHITE"],
                    "right_edge_colors": [right_color if right_shared else "WHITE"],
                    "egomotion_label_class_id": "ego",
                },
                "version": 1,
            }
        )
    return rows


def _lane_line_rows(game_map: ResolvedGameMap) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for divider in game_map.lane_dividers:
        rows.append(
            {
                "key": _key(game_map, f"lane_line:{divider.divider_id}"),
                "lane_line": {
                    "line_rail": [_point(point) for point in divider.polyline_world],
                    "styles": [divider.style],
                    "colors": [divider.color],
                    "left_driving_direction": ["FORWARD"],
                    "right_driving_direction": ["FORWARD"],
                    "is_first_point_physical_end": "false",
                    "is_last_point_physical_end": "false",
                    "egomotion_label_class_id": "ego",
                },
                "version": 1,
            }
        )
    for marking in game_map.line_markings:
        rows.append(
            {
                "key": _key(game_map, f"lane_line:{marking.marking_id}"),
                "lane_line": {
                    "line_rail": [_point(point) for point in marking.polyline_world],
                    "styles": [marking.style],
                    "colors": [marking.color],
                    "left_driving_direction": ["FORWARD"],
                    "right_driving_direction": ["FORWARD"],
                    "is_first_point_physical_end": "true",
                    "is_last_point_physical_end": "true",
                    "egomotion_label_class_id": "ego",
                },
                "version": 1,
            }
        )
    return rows


def _line_geometries(geometry: BaseGeometry) -> list[LineString]:
    """Return every line component nested in a Shapely geometry."""
    if isinstance(geometry, LineString):
        return [geometry]
    return [
        line
        for part in getattr(geometry, "geoms", ())
        for line in _line_geometries(part)
    ]


def _cyclic_substring(
    line: LineString, start_distance: float, end_distance: float
) -> np.ndarray:
    """Extract a wrapping interval from a closed perimeter line.

    Args:
        line: Closed surface-boundary ring.
        start_distance: Possibly negative start distance along ``line``.
        end_distance: Possibly over-length end distance along ``line``.

    Returns:
        Ordered XY points for the requested cyclic interval.
    """
    length = float(line.length)
    ranges = (
        ((length + start_distance, length), (0.0, end_distance))
        if start_distance < 0.0
        else (
            ((start_distance, length), (0.0, end_distance - length))
            if end_distance > length
            else ((start_distance, end_distance),)
        )
    )
    parts: list[np.ndarray] = []
    for range_start, range_end in ranges:
        points = np.asarray(
            substring(line, range_start, range_end).coords,
            dtype=np.float32,
        )
        if parts and np.linalg.norm(parts[-1][-1] - points[0]) <= 1.0e-6:
            points = points[1:]
        if len(points):
            parts.append(points)
    return np.concatenate(parts, axis=0)


def _boundary_chunks(game_map: ResolvedGameMap) -> tuple[np.ndarray, ...]:
    """Build overlapping local chunks along the true road-surface perimeter.

    Args:
        game_map: Resolved semantic map whose surfaces define the BEV boundary.

    Returns:
        Deterministically ordered XYZ boundary chunks. Every point remains on
        the surface-union perimeter, including at intersections and openings.
    """
    surfaces = [Polygon(element.surface_world[:, :2]) for element in game_map.elements]
    lines = _line_geometries(unary_union(surfaces).boundary)
    lines.sort(key=lambda line: tuple(round(value, 6) for value in line.bounds))
    z_m = float(game_map.elements[0].surface_world[0, 2])
    chunks: list[np.ndarray] = []
    for line in lines:
        length = float(line.length)
        chunk_count = max(1, math.ceil(length / _BOUNDARY_CHUNK_CORE_LENGTH_M))
        if chunk_count == 1:
            chunks.append(
                np.column_stack(
                    (
                        np.asarray(line.coords, dtype=np.float32),
                        np.full(len(line.coords), z_m, dtype=np.float32),
                    )
                )
            )
            continue
        core_length = length / chunk_count
        for index in range(chunk_count):
            points_xy = _cyclic_substring(
                line,
                index * core_length - _BOUNDARY_CHUNK_OVERLAP_M,
                (index + 1) * core_length + _BOUNDARY_CHUNK_OVERLAP_M,
            )
            chunks.append(
                np.column_stack(
                    (
                        points_xy,
                        np.full(len(points_xy), z_m, dtype=np.float32),
                    )
                )
            )
    return tuple(chunks)


def _boundary_rows(game_map: ResolvedGameMap) -> list[dict[str, object]]:
    return [
        {
            "key": _key(game_map, f"road_boundary:{index}"),
            "road_boundary": {
                "location": [_point(point) for point in chunk],
                "category": "road_boundary",
                "egomotion_label_class_id": "ego",
            },
            "version": 1,
        }
        for index, chunk in enumerate(_boundary_chunks(game_map))
    ]


def _intersection_rows(game_map: ResolvedGameMap) -> list[dict[str, object]]:
    return [
        {
            "key": _key(game_map, f"intersection:{element.element_id}"),
            "intersection_area": {
                "location": [_point(point) for point in element.surface_world],
                "category": "intersection",
                "egomotion_label_class_id": "ego",
            },
            "version": 1,
        }
        for element in game_map.elements
        if element.element_type == "intersection"
    ]


def _road_marking_rows(game_map: ResolvedGameMap) -> list[dict[str, object]]:
    roadnet_masks = [
        {
            "key": _key(game_map, f"roadnet_mask:{element.element_id}"),
            "road_marking": {
                "location": [_point(point) for point in element.surface_world],
                "category": "ROI_POLYGON_ROADNET_MASK",
                "egomotion_label_class_id": "ego",
            },
            "version": 1,
        }
        for element in game_map.elements
        if element.element_type == "parking_lot"
    ]
    parking_space_markings = [
        {
            "key": _key(game_map, f"road_marking:{index}"),
            "road_marking": {
                "location": [_point(point) for point in polygon],
                "category": "ROI_POLYGON_ROAD_MARKING",
                "egomotion_label_class_id": "ego",
            },
            "version": 1,
        }
        for index, polygon in enumerate(game_map.road_marking_polygons_world)
    ]
    return roadnet_masks + parking_space_markings


def _write_parquet(
    archive: zipfile.ZipFile, name: str, rows: list[dict[str, object]]
) -> None:
    if not rows:
        return
    buffer = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows), buffer)
    archive.writestr(name, buffer.getvalue())


def _write_image(archive: zipfile.ZipFile, name: str, source: Path) -> None:
    buffer = io.BytesIO()
    with Image.open(source) as image:
        image.convert("RGB").save(buffer, format="PNG")
    archive.writestr(name, buffer.getvalue())


def _write_image_array(
    archive: zipfile.ZipFile, name: str, image_array: np.ndarray
) -> None:
    buffer = io.BytesIO()
    Image.fromarray(image_array).save(buffer, format="PNG")
    archive.writestr(name, buffer.getvalue())


def _metadata(game_map: ResolvedGameMap) -> dict[str, object]:
    return {
        "scene_id": game_map.map_id,
        "dataset_hash": "semantic-game-map",
        "is_resumable": False,
        "sensors": {
            "camera_ids": [DEFAULT_FRONT_CAMERA_LOGICAL_NAME],
            "lidar_ids": [],
        },
        "time_range": {
            "start": _START_TIMESTAMP_US,
            "end": _START_TIMESTAMP_US + 33_333,
        },
        "version_string": f"omnidreams-game-map-{_COMPILER_VERSION}",
    }


def _trajectory(game_map: ResolvedGameMap) -> dict[str, object]:
    spawn = game_map.default_spawn
    pose = rig_pose_from_state(
        float(spawn.position_world[0]),
        float(spawn.position_world[1]),
        float(spawn.position_world[2]),
        spawn.yaw_rad,
    ).tolist()
    return {
        "rig_trajectories": [
            {
                "T_rig_worlds": [pose, pose],
                "T_rig_world_timestamps_us": [
                    _START_TIMESTAMP_US,
                    _START_TIMESTAMP_US + 33_333,
                ],
            }
        ]
    }


def _write_archive(path: Path, game_map: ResolvedGameMap) -> None:
    spawn = game_map.default_spawn
    generated_image: np.ndarray | None = None
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(
            "metadata.yaml", yaml.safe_dump(_metadata(game_map), sort_keys=True)
        )
        archive.writestr("rig_trajectories.json", json.dumps(_trajectory(game_map)))
        archive.writestr(
            "game_map.json",
            json.dumps(game_map_to_dict(game_map), separators=(",", ":")),
        )
        archive.writestr(
            "mesh_ground.ply",
            save_mesh_vf(game_map.ground_vertices, game_map.ground_faces),
        )
        for variant in spawn.variants:
            suffix = "" if variant.name == "default" else f"_{variant.name}"
            archive.writestr(f"prompt{suffix}.txt", variant.prompt)
            image_name = f"first_image{suffix}.png"
            if variant.image is None:
                if generated_image is None:
                    generated_image = spawn_render.render_spawn_first_frame(
                        game_map, spawn
                    )
                _write_image_array(
                    archive,
                    image_name,
                    generated_image,
                )
            else:
                _write_image(
                    archive,
                    image_name,
                    resolve_seed_asset(game_map.source_path, variant.image),
                )
        _write_parquet(
            archive, "clipgt/calibration_estimate.parquet", _calibration_row()
        )
        _write_parquet(archive, "clipgt/lane.parquet", _lane_rows(game_map))
        _write_parquet(archive, "clipgt/lane_line.parquet", _lane_line_rows(game_map))
        _write_parquet(
            archive, "clipgt/road_boundary.parquet", _boundary_rows(game_map)
        )
        _write_parquet(
            archive, "clipgt/intersection_area.parquet", _intersection_rows(game_map)
        )
        _write_parquet(
            archive, "clipgt/road_marking.parquet", _road_marking_rows(game_map)
        )


def compile_game_map(
    path: Path,
    *,
    cache_root: Path | None = None,
    force: bool = False,
) -> CompiledGameMap:
    """Compile a map, optionally replacing its valid cached archive."""
    game_map = load_game_map(path)
    digest = _digest(game_map)
    root = _cache_root() if cache_root is None else Path(cache_root)
    output_dir = root / digest
    archive_path = output_dir / f"{game_map.map_id}.usdz"
    lock = FileLock(str(root / f"{digest}.lock"))
    root.mkdir(parents=True, exist_ok=True)
    with lock:
        if archive_path.is_file() and not force:
            try:
                with zipfile.ZipFile(archive_path, "r") as archive:
                    if "game_map.json" in archive.namelist():
                        return CompiledGameMap(
                            game_map.source_path, archive_path, game_map, True
                        )
            except (OSError, zipfile.BadZipFile):
                pass
        output_dir.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=output_dir, prefix=".map-", suffix=".usdz"
        )
        os.close(file_descriptor)
        temporary = Path(temporary_name)
        try:
            _write_archive(temporary, game_map)
            temporary.replace(archive_path)
        finally:
            temporary.unlink(missing_ok=True)
    return CompiledGameMap(game_map.source_path, archive_path, game_map, False)
