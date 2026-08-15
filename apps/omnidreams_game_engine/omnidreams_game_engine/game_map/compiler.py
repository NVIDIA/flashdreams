# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Content-addressed ClipGT compilation for semantic game maps."""

from __future__ import annotations

import hashlib
import io
import json
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

from omnidreams_game_engine.game_map.loader import load_game_map, resolve_seed_asset
from omnidreams_game_engine.game_map.types import (
    GameMapLane,
    ResolvedGameMap,
    game_map_to_dict,
)
from omnidreams_game_engine.math3d import rig_pose_from_state
from omnidreams_game_engine.ply_io import save_mesh_vf
from omnidreams_game_engine.scene_fixture import _calibration_row

_COMPILER_VERSION = "3"
_START_TIMESTAMP_US = 1_700_000_000_000_000
_CAMERA_NAME = "camera_front_wide_120fov"
_SHARED_EDGE_TOLERANCE_M = 0.01


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
        / "crazy-robotaxi"
        / "maps"
    )


def _digest(game_map: ResolvedGameMap) -> str:
    hasher = hashlib.sha256()
    hasher.update(_COMPILER_VERSION.encode())
    hasher.update(game_map.source_path.read_bytes())
    for spawn in game_map.spawns:
        for variant in spawn.variants:
            asset = resolve_seed_asset(game_map.source_path, variant.image)
            hasher.update(variant.name.encode())
            hasher.update(variant.prompt.encode())
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


def _aligned_polyline(
    reference: np.ndarray, candidate: np.ndarray
) -> np.ndarray | None:
    """Align a coincident candidate to a reference edge, or return ``None``."""
    if reference.shape != candidate.shape:
        return None
    direct_error = float(np.linalg.norm(reference - candidate, axis=1).max())
    reverse_error = float(np.linalg.norm(reference - candidate[::-1], axis=1).max())
    if min(direct_error, reverse_error) > _SHARED_EDGE_TOLERANCE_M:
        return None
    return candidate if direct_error <= reverse_error else candidate[::-1]


def _lane_edge_groups(
    game_map: ResolvedGameMap,
) -> list[list[tuple[GameMapLane, str, np.ndarray]]]:
    """Group coincident road-lane edges within each authored element."""
    groups: list[list[tuple[GameMapLane, str, np.ndarray]]] = []
    for lane in game_map.lanes:
        if not lane.allows_taxi_stops:
            continue
        for side, points in (
            ("left", lane.left_edge_world),
            ("right", lane.right_edge_world),
        ):
            group = next(
                (
                    members
                    for members in groups
                    if members[0][0].element_id == lane.element_id
                    and _aligned_polyline(members[0][2], points) is not None
                ),
                None,
            )
            if group is None:
                groups.append([(lane, side, points)])
            else:
                group.append((lane, side, points))
    return groups


def _lane_rows(game_map: ResolvedGameMap) -> list[dict[str, object]]:
    shared_edges = {
        (lane.lane_id, side)
        for members in _lane_edge_groups(game_map)
        if len(members) > 1
        for lane, side, _points in members
    }
    rows: list[dict[str, object]] = []
    for lane in game_map.lanes:
        left_shared = (lane.lane_id, "left") in shared_edges
        right_shared = (lane.lane_id, "right") in shared_edges
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
                        [lane.marking_style if left_shared else "VIRTUAL"]
                        if lane.allows_taxi_stops
                        else []
                    ),
                    "right_edge_styles": (
                        [lane.marking_style if right_shared else "VIRTUAL"]
                        if lane.allows_taxi_stops
                        else []
                    ),
                    "left_edge_colors": [
                        lane.marking_color if left_shared else "WHITE"
                    ],
                    "right_edge_colors": [
                        lane.marking_color if right_shared else "WHITE"
                    ],
                    "egomotion_label_class_id": "ego",
                },
                "version": 1,
            }
        )
    return rows


def _lane_line_rows(game_map: ResolvedGameMap) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for members in _lane_edge_groups(game_map):
        if len(members) < 2:
            continue
        lane, _side, reference = members[0]
        aligned_points = [reference]
        for _member_lane, _member_side, points in members[1:]:
            aligned = _aligned_polyline(reference, points)
            assert aligned is not None
            aligned_points.append(aligned)
        divider = np.mean(aligned_points, axis=0)
        member_ids = ":".join(
            sorted(member.lane_id for member, _side, _points in members)
        )
        rows.append(
            {
                "key": _key(game_map, f"lane_line:{member_ids}"),
                "lane_line": {
                    "line_rail": [_point(point) for point in divider],
                    "styles": [lane.marking_style],
                    "colors": [lane.marking_color],
                    "left_driving_direction": ["FORWARD"],
                    "right_driving_direction": ["FORWARD"],
                    "is_first_point_physical_end": "false",
                    "is_last_point_physical_end": "false",
                    "egomotion_label_class_id": "ego",
                },
                "version": 1,
            }
        )
    return rows


def _boundary_rows(game_map: ResolvedGameMap) -> list[dict[str, object]]:
    return [
        {
            "key": _key(game_map, f"curb:{index}"),
            "road_boundary": {
                "location": [_point(segment[0]), _point(segment[1])],
                "category": "curb",
                "egomotion_label_class_id": "ego",
            },
            "version": 1,
        }
        for index, segment in enumerate(game_map.collision_segments_world)
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


def _metadata(game_map: ResolvedGameMap) -> dict[str, object]:
    return {
        "scene_id": game_map.map_id,
        "dataset_hash": "semantic-game-map",
        "is_resumable": False,
        "sensors": {"camera_ids": [_CAMERA_NAME], "lidar_ids": []},
        "time_range": {
            "start": _START_TIMESTAMP_US,
            "end": _START_TIMESTAMP_US + 33_333,
        },
        "version_string": f"robotaxi-map-{_COMPILER_VERSION}",
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
            _write_image(
                archive,
                f"first_image{suffix}.png",
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


def compile_game_map(path: Path, *, cache_root: Path | None = None) -> CompiledGameMap:
    """Compile a semantic YAML map into a cached private renderer archive."""
    game_map = load_game_map(path)
    digest = _digest(game_map)
    root = _cache_root() if cache_root is None else Path(cache_root)
    output_dir = root / digest
    archive_path = output_dir / f"{game_map.map_id}.usdz"
    lock = FileLock(str(root / f"{digest}.lock"))
    root.mkdir(parents=True, exist_ok=True)
    with lock:
        if archive_path.is_file():
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
