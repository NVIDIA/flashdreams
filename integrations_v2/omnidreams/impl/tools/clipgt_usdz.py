# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Checkpoint-free scene USDZ archives built from ClipGT caches.

The resulting archive is intentionally an application bundle rather than a
Pixar USD package. It contains the recorded ClipGT map/calibration data plus
the scene metadata consumed by FlashDreams and AlpaSim. No
NuRec checkpoint or neural reconstruction assets are required.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tempfile
import uuid
import zipfile
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from interactive_drive.math3d import (
    quaternion_to_matrix_xyzw,
    transform_from_rt,
)
from interactive_drive.ply_io import save_mesh_vf
from PIL import Image

_CALIBRATION_FILE = "calibration_estimate.parquet"
_EGOMOTION_FILE = "egomotion_estimate.parquet"
_ALPASIM_MAP_FILES = frozenset(
    {
        "lane.parquet",
        "road_boundary.parquet",
        "traffic_sign.parquet",
        "wait_line.parquet",
    }
)
"""Map tables read unconditionally by AlpaSim's MADS vector-map importer."""

_CAMERA_CLIPGT_NAME = "camera:front:wide:120fov"
_CAMERA_LOGICAL_NAME = "camera_front_wide_120fov"
_DEFAULT_PROMPT = (
    "A forward-facing driving scene with realistic roads, traffic, and lighting."
)
_CHECKPOINT_FREE_TRAINING_DATE = "1970-01-01"
"""Stable metadata sentinel used because this path has no trained checkpoint."""

_DYNAMIC_LABEL_TOKENS = frozenset(
    {
        "automobile",
        "bicycle",
        "bus",
        "car",
        "cyclist",
        "motorcycle",
        "person",
        "pedestrian",
        "rider",
        "trailer",
        "truck",
        "vehicle",
    }
)
_IDENTITY_MATRIX = np.eye(4, dtype=np.float32)
_GROUND_POINT_FIELDS = {
    "buffer_zone.parquet": ("location",),
    "crosswalk.parquet": ("location",),
    "gore_area.parquet": ("location",),
    "intersection_area.parquet": ("location",),
    "lane.parquet": ("left_rail", "right_rail"),
    "lane_line.parquet": ("line_rail",),
    "obstacle.parquet": ("center",),
    "pole.parquet": ("location",),
    "road_boundary.parquet": ("location",),
    "road_island.parquet": ("location",),
    "road_marking.parquet": ("location",),
    "traffic_light.parquet": ("center",),
    "traffic_sign.parquet": ("center",),
    "wait_line.parquet": ("location",),
}
"""ClipGT geometry used to size the checkpoint-free ground mesh."""


def _canonical_parquet_name(path: Path) -> str:
    """Return the unprefixed ClipGT filename used inside the archive."""
    if path.suffix != ".parquet":
        raise ValueError(f"Expected a .parquet file, got {path.name!r}")
    payload_name = path.stem.rsplit(".", 1)[-1]
    if not payload_name or any(ch in payload_name for ch in "/\\"):
        raise ValueError(f"Invalid ClipGT parquet name: {path.name!r}")
    return f"{payload_name}.parquet"


def _discover_parquets(clipgt_dir: Path) -> dict[str, Path]:
    if not clipgt_dir.is_dir():
        raise FileNotFoundError(f"ClipGT directory does not exist: {clipgt_dir}")

    files: dict[str, Path] = {}
    for path in sorted(clipgt_dir.iterdir()):
        if not path.is_file() or path.suffix != ".parquet":
            continue
        canonical_name = _canonical_parquet_name(path)
        if canonical_name in files:
            raise ValueError(
                f"Duplicate normalized ClipGT filename {canonical_name!r}: "
                f"{files[canonical_name].name!r}, {path.name!r}"
            )
        files[canonical_name] = path

    missing = {
        _CALIBRATION_FILE,
        _EGOMOTION_FILE,
        *_ALPASIM_MAP_FILES,
    } - files.keys()
    if missing:
        raise FileNotFoundError(
            f"ClipGT directory is missing required parquet(s): {sorted(missing)}"
        )
    return files


def _validate_table(name: str, path: Path) -> pa.Table:
    try:
        table = pq.read_table(path)
    except Exception as exc:
        raise ValueError(f"Unable to read ClipGT parquet {path}: {exc}") from exc

    payload_name = Path(name).stem
    required_columns = {"key", payload_name, "version"}
    missing = required_columns - set(table.column_names)
    if missing:
        raise ValueError(
            f"{path.name} is missing expected column(s) {sorted(missing)}; "
            f"found {table.column_names}"
        )
    if name == _CALIBRATION_FILE and table.num_rows < 1:
        raise ValueError("calibration_estimate.parquet must contain at least one row")
    if name == _EGOMOTION_FILE and table.num_rows < 2:
        raise ValueError("egomotion_estimate.parquet must contain at least two rows")
    if name == "lane.parquet" and table.num_rows < 1:
        raise ValueError("lane.parquet must contain at least one road lane")
    return table


def _table_clip_ids(name: str, table: pa.Table) -> set[str]:
    clip_ids: set[str] = set()
    for row_idx, row in enumerate(table.select(["key"]).to_pylist()):
        key = row.get("key") or {}
        clip_id = key.get("clip_id")
        if not clip_id:
            raise ValueError(f"{name} row {row_idx} has no key.clip_id")
        clip_ids.add(str(clip_id))
    return clip_ids


def _load_and_validate_parquets(
    files: dict[str, Path],
) -> tuple[str, dict[str, pa.Table]]:
    tables: dict[str, pa.Table] = {}
    clip_ids: set[str] = set()
    for name, path in files.items():
        table = _validate_table(name, path)
        tables[name] = table
        clip_ids.update(_table_clip_ids(name, table))

    if len(clip_ids) != 1:
        raise ValueError(
            "All non-empty ClipGT tables must have one identical key.clip_id; "
            f"found {sorted(clip_ids)}"
        )
    return next(iter(clip_ids)), tables


def _normalize_quaternion(value: dict[str, Any], *, context: str) -> np.ndarray:
    quat = np.asarray(
        [value.get("x"), value.get("y"), value.get("z"), value.get("w")],
        dtype=np.float64,
    )
    if quat.shape != (4,) or not np.all(np.isfinite(quat)):
        raise ValueError(f"{context} has a non-finite xyzw quaternion: {quat}")
    norm = float(np.linalg.norm(quat))
    if norm < 1e-8:
        raise ValueError(f"{context} has a zero quaternion")
    return (quat / norm).astype(np.float32)


def _point(value: dict[str, Any], *, context: str) -> np.ndarray:
    point = np.asarray(
        [value.get("x"), value.get("y"), value.get("z")], dtype=np.float64
    )
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError(f"{context} has non-finite xyz coordinates: {point}")
    return point.astype(np.float32)


def _egomotion_records(
    table: pa.Table,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[list[float]]]]:
    records: list[tuple[int, np.ndarray, np.ndarray]] = []
    for row_idx, row in enumerate(table.to_pylist()):
        key = row.get("key") or {}
        payload = row.get("egomotion_estimate") or {}
        timestamp_us = int(key["timestamp_micros"])
        position = _point(
            payload.get("location") or {}, context=f"egomotion row {row_idx}"
        )
        quat = _normalize_quaternion(
            payload.get("orientation") or {}, context=f"egomotion row {row_idx}"
        )
        records.append((timestamp_us, position, quat))

    records.sort(key=lambda record: record[0])
    timestamps_us = np.asarray([record[0] for record in records], dtype=np.int64)
    if np.any(np.diff(timestamps_us) <= 0):
        raise ValueError("ClipGT egomotion timestamps must be strictly increasing")

    positions = np.stack([record[1] for record in records]).astype(np.float32)
    quaternions = np.stack([record[2] for record in records]).astype(np.float32)
    matrices = [
        transform_from_rt(quaternion_to_matrix_xyzw(quat.tolist()), position.tolist())
        .astype(np.float32)
        .tolist()
        for position, quat in zip(positions, quaternions, strict=True)
    ]
    return timestamps_us, positions, quaternions, matrices


def _find_camera_sensor(
    calibration_table: pa.Table, camera_clipgt_name: str
) -> dict[str, Any]:
    for row_idx, row in enumerate(calibration_table.to_pylist()):
        calibration = row.get("calibration_estimate") or {}
        rig_json = calibration.get("rig_json")
        try:
            rig_data = json.loads(rig_json) if isinstance(rig_json, str) else rig_json
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"calibration_estimate row {row_idx} has invalid rig_json"
            ) from exc
        if not isinstance(rig_data, dict):
            raise ValueError(
                f"calibration_estimate row {row_idx} rig_json must be an object"
            )

        rig = rig_data.get("rig", rig_data)
        if not isinstance(rig, dict) or not isinstance(rig.get("sensors", []), list):
            raise ValueError(f"calibration_estimate row {row_idx} has no sensor list")
        for sensor in rig.get("sensors", []):
            if not isinstance(sensor, dict) or sensor.get("name") != camera_clipgt_name:
                continue
            properties = sensor.get("properties") or {}
            if not isinstance(properties, dict):
                raise ValueError(
                    f"Camera {camera_clipgt_name!r} properties must be an object"
                )
            required = {
                "width",
                "height",
                "cx",
                "cy",
                "polynomial",
                "polynomial-type",
            }
            missing = required - properties.keys()
            if properties.get("Model") != "ftheta":
                raise ValueError(
                    f"Camera {camera_clipgt_name!r} must use the FTheta model"
                )
            if missing:
                raise ValueError(
                    f"Camera {camera_clipgt_name!r} is missing calibration "
                    f"properties {sorted(missing)}"
                )
            if "nominalSensor2Rig_FLU" not in sensor:
                raise ValueError(
                    f"Camera {camera_clipgt_name!r} has no nominalSensor2Rig_FLU"
                )
            return sensor
    raise ValueError(
        f"Calibration does not contain requested camera {camera_clipgt_name!r}"
    )


def _frame_ranges(timestamps_us: np.ndarray) -> list[list[int]]:
    deltas = np.diff(timestamps_us)
    frame_interval_us = int(round(float(np.median(deltas))))
    if frame_interval_us <= 0:
        raise ValueError("Unable to derive a positive camera frame interval")
    return [
        [max(0, int(timestamp_us) - frame_interval_us), int(timestamp_us)]
        for timestamp_us in timestamps_us
    ]


def _rig_document(
    *,
    scene_id: str,
    camera_logical_name: str,
    timestamps_us: np.ndarray,
    pose_matrices: list[list[list[float]]],
) -> dict[str, Any]:
    identity = _IDENTITY_MATRIX.tolist()
    return {
        "world_to_nre": {"matrix": identity},
        "T_world_base": identity,
        "camera_calibrations": {
            camera_logical_name: {"logical_sensor_name": camera_logical_name}
        },
        "rig_trajectories": [
            {
                "sequence_id": scene_id,
                "T_rig_world_timestamps_us": timestamps_us.tolist(),
                "T_rig_worlds": pose_matrices,
                "cameras_frame_timestamps_us": {
                    camera_logical_name: _frame_ranges(timestamps_us)
                },
                # AlpaSim's default Hyperion vehicle geometry expressed in the
                # NRE rig-bbox schema.  Supplying it avoids a runtime-only
                # vehicle override and is sufficient for collision/physics.
                "rig_bbox": {
                    "centroid": [1.3965, 0.0, 0.7515],
                    "dim": [5.393, 2.109, 1.503],
                    "rot": [0.0, 0.0, 0.0],
                },
            }
        ],
    }


def _is_dynamic_label(label: str) -> bool:
    normalized = label.lower().replace("-", "_").replace(" ", "_")
    return bool(set(normalized.split("_")) & _DYNAMIC_LABEL_TOKENS)


def _sequence_tracks_document(
    *, scene_id: str, obstacle_table: pa.Table | None
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if obstacle_table is not None:
        for row in obstacle_table.to_pylist():
            obstacle = row.get("obstacle") or {}
            track_id = obstacle.get("trackline_id")
            if track_id is not None:
                grouped[str(track_id)].append(row)

    tracks_id: list[str] = []
    tracks_label_class: list[str] = []
    tracks_flags: list[list[str]] = []
    tracks_timestamps_us: list[list[int]] = []
    tracks_poses: list[list[list[float]]] = []
    cuboids_dims: list[list[float]] = []

    for track_id in sorted(grouped):
        rows = sorted(grouped[track_id], key=lambda row: row["key"]["timestamp_micros"])
        deduplicated: list[dict[str, Any]] = []
        last_timestamp_us: int | None = None
        for row in rows:
            timestamp_us = int(row["key"]["timestamp_micros"])
            if timestamp_us == last_timestamp_us:
                continue
            deduplicated.append(row)
            last_timestamp_us = timestamp_us
        if len(deduplicated) < 2:
            continue

        timestamps: list[int] = []
        poses: list[list[float]] = []
        dimensions: list[np.ndarray] = []
        labels: list[str] = []
        for row_idx, row in enumerate(deduplicated):
            obstacle = row.get("obstacle") or {}
            timestamp_us = int(row["key"]["timestamp_micros"])
            center = _point(
                obstacle.get("center") or {},
                context=f"obstacle track {track_id} row {row_idx}",
            )
            quat = _normalize_quaternion(
                obstacle.get("orientation") or {},
                context=f"obstacle track {track_id} row {row_idx}",
            )
            dimensions.append(
                _point(
                    obstacle.get("size") or {},
                    context=f"obstacle track {track_id} size row {row_idx}",
                )
            )
            timestamps.append(timestamp_us)
            poses.append([*center.tolist(), *quat.tolist()])
            labels.append(str(obstacle.get("category") or "Other"))

        label = min(set(labels), key=lambda value: (-labels.count(value), value))
        dim = np.median(np.stack(dimensions), axis=0)
        if np.any(dim <= 0):
            raise ValueError(f"Obstacle track {track_id} has non-positive dimensions")

        tracks_id.append(track_id)
        tracks_label_class.append(label)
        tracks_flags.append(["CONTROLLABLE"] if _is_dynamic_label(label) else [])
        tracks_timestamps_us.append(timestamps)
        tracks_poses.append(poses)
        cuboids_dims.append(dim.astype(float).tolist())

    return {
        scene_id: {
            "tracks_data": {
                "tracks_id": tracks_id,
                "tracks_label_class": tracks_label_class,
                "tracks_flags": tracks_flags,
                "tracks_timestamps_us": tracks_timestamps_us,
                "tracks_poses": tracks_poses,
            },
            "cuboidtracks_data": {"cuboids_dims": cuboids_dims},
        }
    }


def _xyz_values(value: Any) -> list[np.ndarray]:
    if isinstance(value, list):
        return [point for item in value for point in _xyz_values(item)]
    if not isinstance(value, dict) or not {"x", "y", "z"} <= value.keys():
        return []
    point = np.asarray([value["x"], value["y"], value["z"]], dtype=np.float64)
    return [point] if np.all(np.isfinite(point)) else []


def _ground_coverage_points(
    tables: dict[str, pa.Table], positions: np.ndarray
) -> np.ndarray:
    points = [point.astype(np.float64) for point in positions]
    for name, fields in _GROUND_POINT_FIELDS.items():
        table = tables.get(name)
        if table is None:
            continue
        payload_name = Path(name).stem
        for row in table.select([payload_name]).to_pylist():
            payload = row.get(payload_name) or {}
            for field in fields:
                points.extend(_xyz_values(payload.get(field)))
    return np.stack(points)


def _ground_ribbon_mesh(
    positions: np.ndarray,
    quaternions: np.ndarray,
    coverage_points: np.ndarray,
    *,
    minimum_half_width_m: float = 50.0,
    coverage_margin_m: float = 10.0,
) -> bytes:
    """Create an oriented ground ribbon covering the recorded scene geometry."""
    if len(positions) < 2:
        raise ValueError("At least two egomotion points are needed for a ground mesh")
    if quaternions.shape != (len(positions), 4):
        raise ValueError("Ground mesh quaternions must have shape (N, 4)")
    if coverage_points.ndim != 2 or coverage_points.shape[1] != 3:
        raise ValueError("Ground mesh coverage points must have shape (N, 3)")

    xy = positions[:, :2].astype(np.float64)
    tangent = np.stack(
        [
            np.asarray(quaternion_to_matrix_xyzw(quat.tolist()))[:2, 0]
            for quat in quaternions
        ]
    )
    tangent_norm = np.linalg.norm(tangent, axis=1)
    valid = tangent_norm > 1e-6
    position_tangent = np.gradient(xy, axis=0)
    position_tangent_norm = np.linalg.norm(position_tangent, axis=1)
    position_valid = position_tangent_norm > 1e-6
    tangent[position_valid & ~valid] = position_tangent[position_valid & ~valid]
    tangent_norm = np.linalg.norm(tangent, axis=1)
    valid = tangent_norm > 1e-6
    tangent[~valid] = [1.0, 0.0]
    tangent_norm[~valid] = 1.0
    tangent /= tangent_norm[:, None]
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])

    maximum_nearest_distance_m = 0.0
    for start in range(0, len(coverage_points), 4096):
        coverage_xy = coverage_points[start : start + 4096, :2]
        squared_distances = np.sum(
            (coverage_xy[:, None, :] - xy[None, :, :]) ** 2, axis=2
        )
        maximum_nearest_distance_m = max(
            maximum_nearest_distance_m,
            float(np.sqrt(np.min(squared_distances, axis=1)).max()),
        )
    half_width_m = max(
        minimum_half_width_m,
        maximum_nearest_distance_m + coverage_margin_m,
    )

    extended_positions = np.concatenate(
        [
            positions[:1].copy(),
            positions,
            positions[-1:].copy(),
        ]
    )
    extended_positions[0, :2] -= (tangent[0] * half_width_m).astype(np.float32)
    extended_positions[-1, :2] += (tangent[-1] * half_width_m).astype(np.float32)
    extended_normal = np.concatenate([normal[:1], normal, normal[-1:]])

    left = extended_positions.copy()
    right = extended_positions.copy()
    left[:, :2] += (extended_normal * half_width_m).astype(np.float32)
    right[:, :2] -= (extended_normal * half_width_m).astype(np.float32)
    vertices = np.empty((len(extended_positions) * 2, 3), dtype=np.float32)
    vertices[0::2] = left
    vertices[1::2] = right

    faces = np.empty(((len(extended_positions) - 1) * 2, 3), dtype=np.int32)
    for idx in range(len(extended_positions) - 1):
        left0, right0 = 2 * idx, 2 * idx + 1
        left1, right1 = 2 * (idx + 1), 2 * (idx + 1) + 1
        faces[2 * idx] = [left0, right0, left1]
        faces[2 * idx + 1] = [right0, right1, left1]
    return save_mesh_vf(vertices, faces)


def _read_jpeg(path: Path) -> bytes:
    data = path.read_bytes()
    if not data.startswith(b"\xff\xd8\xff"):
        raise ValueError(f"Initial frame must contain JPEG bytes: {path}")
    with Image.open(io.BytesIO(data)) as image:
        image.verify()
    return data


def _validate_frame_aspect_ratio(jpeg: bytes, camera_sensor: dict[str, Any]) -> None:
    properties = camera_sensor["properties"]
    calibration_width = int(properties["width"])
    calibration_height = int(properties["height"])
    if calibration_width <= 0 or calibration_height <= 0:
        raise ValueError(
            "Camera calibration dimensions must be positive: "
            f"{calibration_width}x{calibration_height}"
        )
    with Image.open(io.BytesIO(jpeg)) as image:
        frame_width, frame_height = image.size
    if not np.isclose(
        frame_width / frame_height,
        calibration_width / calibration_height,
        rtol=1e-3,
        atol=0.0,
    ):
        raise ValueError(
            "Initial frame aspect ratio does not match the requested camera "
            f"calibration: frame={frame_width}x{frame_height}, "
            f"calibration={calibration_width}x{calibration_height}"
        )


def _first_video_frame_as_jpeg(path: Path, *, quality: int = 95) -> bytes:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - package dependency documents this
        raise RuntimeError("OpenCV is required to extract the first MP4 frame") from exc

    capture = cv2.VideoCapture(str(path))
    try:
        ok, bgr = capture.read()
    finally:
        capture.release()
    if not ok or bgr is None:
        raise ValueError(f"Unable to decode the first frame from {path}")
    ok, encoded = cv2.imencode(
        ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if not ok:
        raise ValueError(f"Unable to encode the first frame from {path} as JPEG")
    return encoded.tobytes()


def _png_from_jpeg(jpeg: bytes) -> bytes:
    with Image.open(io.BytesIO(jpeg)) as image:
        output = io.BytesIO()
        image.convert("RGB").save(output, format="PNG")
    return output.getvalue()


def _dataset_hash(
    files: Iterable[tuple[str, Path]],
    jpeg: bytes,
    *,
    camera_logical_name: str,
    prompt: str,
) -> str:
    digest = hashlib.sha256()
    for canonical_name, path in sorted(files):
        digest.update(canonical_name.encode("utf-8"))
        digest.update(path.read_bytes())
    digest.update(jpeg)
    digest.update(camera_logical_name.encode("utf-8"))
    digest.update(prompt.encode("utf-8"))
    return digest.hexdigest()


def _scene_id_for_clip_id(clip_id: str, camera_logical_name: str) -> str:
    scene_uuid = uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"flashdreams:clipgt:{clip_id}:camera:{camera_logical_name}",
    )
    return f"clipgt-{scene_uuid}"


def _write_zip_entry(archive: zipfile.ZipFile, name: str, data: str | bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    archive.writestr(info, data)


def _parquet_bytes(rows: list[dict[str, Any]]) -> bytes:
    buffer = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows), buffer)
    return buffer.getvalue()


def _map_compatibility_entries(
    *, clip_id: str, tables: dict[str, pa.Table]
) -> dict[str, bytes]:
    """Synthesize metadata tables required by trajdata's MADS map importer.

    Production ClipGT caches used by FlashDreams omit ``clip.parquet`` and
    ``association.parquet`` because Ludus does not consume them.  AlpaSim's
    trajdata importer opens both unconditionally.  Empty lane relationships
    are sufficient to build ROAD_LANE geometry from ``lane.parquet``; the
    recorded trajectory remains the authoritative route.
    """
    if "lane.parquet" not in tables:
        return {}

    generated: dict[str, bytes] = {}
    if "clip.parquet" not in tables:
        generated["clip.parquet"] = _parquet_bytes(
            [
                {
                    "key": {"clip_id": clip_id},
                    "clip": {"name": clip_id},
                    "version": 1,
                }
            ]
        )

    if "association.parquet" not in tables:
        lane_ids = [
            str((row.get("key") or {})["map_id"])
            for row in tables["lane.parquet"].select(["key"]).to_pylist()
        ]
        generated["association.parquet"] = _parquet_bytes(
            [
                {
                    "key": {
                        "clip_id": clip_id,
                        "kind": "NEXT_LANE",
                    },
                    "association": {"subjects": lane_id, "objects": []},
                    "version": 1,
                }
                for lane_id in lane_ids
            ]
        )
    return generated


def _metadata_document(
    *,
    scene_id: str,
    source_clip_id: str,
    camera_logical_name: str,
    timestamps_us: np.ndarray,
    dataset_hash: str,
    training_date: str,
) -> dict[str, Any]:
    try:
        date.fromisoformat(training_date)
    except ValueError as exc:
        raise ValueError("training_date must use YYYY-MM-DD format") from exc
    stable_uuid = uuid.uuid5(uuid.NAMESPACE_URL, f"flashdreams:{dataset_hash}")
    return {
        "scene_id": scene_id,
        "source_clip_id": source_clip_id,
        "version_string": "clipgt-scene-fixture-1.0.0",
        "training_date": training_date,
        "dataset_hash": dataset_hash,
        "uuid": str(stable_uuid),
        "is_resumable": False,
        "sensors": {"camera_ids": [camera_logical_name], "lidar_ids": []},
        "logger": {},
        "time_range": {
            "start": int(timestamps_us[0]),
            "end": int(timestamps_us[-1]),
        },
        "training_step_outputs": {},
    }


def build_alpasim_clipgt_usdz(
    output_path: Path,
    *,
    clipgt_dir: Path,
    initial_frame_path: Path | None = None,
    initial_video_path: Path | None = None,
    camera_clipgt_name: str = _CAMERA_CLIPGT_NAME,
    prompt: str = _DEFAULT_PROMPT,
    training_date: str | None = None,
) -> Path:
    """Build a FlashDreams + AlpaSim scene bundle from recorded ClipGT.

    Exactly one of ``initial_frame_path`` and ``initial_video_path`` must be
    supplied. The source Parquets are validated and copied byte-for-byte;
    ego trajectory, traffic tracks, metadata, and a coarse ground mesh are
    generated from the same ClipGT coordinate frame.

    Args:
        output_path: Destination path ending in ``.usdz``.
        clipgt_dir: Directory containing one ClipGT cache.
        initial_frame_path: JPEG seed frame; mutually exclusive with
            ``initial_video_path``.
        initial_video_path: Video whose first frame seeds generation; mutually
            exclusive with ``initial_frame_path``.
        camera_clipgt_name: ClipGT camera sensor to package.
        prompt: Text conditioning stored in the scene bundle.
        training_date: Informational ISO date; ``None`` uses a stable sentinel.

    Returns:
        Path to the atomically written scene bundle.

    Raises:
        FileNotFoundError: A required ClipGT input is missing.
        ValueError: The inputs or ClipGT payload fail validation.
    """
    if (initial_frame_path is None) == (initial_video_path is None):
        raise ValueError(
            "Exactly one of initial_frame_path or initial_video_path must be supplied"
        )
    if camera_clipgt_name != _CAMERA_CLIPGT_NAME:
        raise ValueError(
            "The checkpoint-free AlpaSim path currently supports only "
            f"{_CAMERA_CLIPGT_NAME!r}; got {camera_clipgt_name!r}"
        )

    output_path = Path(output_path)
    if output_path.suffix != ".usdz":
        raise ValueError(f"Output path must end in .usdz: {output_path}")
    clipgt_dir = Path(clipgt_dir)
    files = _discover_parquets(clipgt_dir)
    clip_id, tables = _load_and_validate_parquets(files)
    camera_sensor = _find_camera_sensor(tables[_CALIBRATION_FILE], camera_clipgt_name)

    camera_logical_name = _CAMERA_LOGICAL_NAME
    timestamps_us, positions, quaternions, pose_matrices = _egomotion_records(
        tables[_EGOMOTION_FILE]
    )
    scene_id = _scene_id_for_clip_id(clip_id, camera_logical_name)

    if initial_frame_path is not None:
        jpeg = _read_jpeg(Path(initial_frame_path))
    else:
        assert initial_video_path is not None
        jpeg = _first_video_frame_as_jpeg(Path(initial_video_path))
    _validate_frame_aspect_ratio(jpeg, camera_sensor)

    digest = _dataset_hash(
        files.items(),
        jpeg,
        camera_logical_name=camera_logical_name,
        prompt=prompt,
    )
    metadata = _metadata_document(
        scene_id=scene_id,
        source_clip_id=clip_id,
        camera_logical_name=camera_logical_name,
        timestamps_us=timestamps_us,
        dataset_hash=digest,
        training_date=training_date or _CHECKPOINT_FREE_TRAINING_DATE,
    )
    rig = _rig_document(
        scene_id=scene_id,
        camera_logical_name=camera_logical_name,
        timestamps_us=timestamps_us,
        pose_matrices=pose_matrices,
    )
    tracks = _sequence_tracks_document(
        scene_id=scene_id, obstacle_table=tables.get("obstacle.parquet")
    )
    ground_mesh = _ground_ribbon_mesh(
        positions,
        quaternions,
        _ground_coverage_points(tables, positions),
    )
    generated_map_entries = _map_compatibility_entries(clip_id=clip_id, tables=tables)
    camera_timestamps = [
        {"timestamp": int(timestamp_us)} for timestamp_us in timestamps_us
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)

    try:
        with zipfile.ZipFile(
            temporary_path, mode="w", compression=zipfile.ZIP_STORED
        ) as archive:
            _write_zip_entry(
                archive, "metadata.yaml", yaml.safe_dump(metadata, sort_keys=True)
            )
            _write_zip_entry(archive, "rig_trajectories.json", json.dumps(rig))
            _write_zip_entry(archive, "sequence_tracks.json", json.dumps(tracks))
            _write_zip_entry(archive, "mesh_ground.ply", ground_mesh)
            _write_zip_entry(archive, "prompt.txt", prompt)
            _write_zip_entry(archive, "first_image.png", _png_from_jpeg(jpeg))
            _write_zip_entry(
                archive,
                f"frames/{camera_logical_name}/{int(timestamps_us[0])}.jpeg",
                jpeg,
            )
            _write_zip_entry(
                archive,
                f"clipgt/{camera_logical_name}.json",
                json.dumps(camera_timestamps),
            )
            for name, path in sorted(files.items()):
                _write_zip_entry(archive, f"clipgt/{name}", path.read_bytes())
            for name, data in sorted(generated_map_entries.items()):
                _write_zip_entry(archive, f"clipgt/{name}", data)

        with zipfile.ZipFile(temporary_path, "r") as archive:
            corrupt_member = archive.testzip()
            if corrupt_member is not None:
                raise RuntimeError(
                    f"Generated archive has a corrupt member: {corrupt_member}"
                )
        temporary_path.chmod(0o644)
        temporary_path.replace(output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a checkpoint-free FlashDreams/AlpaSim USDZ from one "
            "ClipGT cache directory."
        )
    )
    parser.add_argument("--clipgt-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--initial-frame", type=Path)
    source.add_argument("--initial-video", type=Path)
    parser.add_argument("--camera", default=_CAMERA_CLIPGT_NAME)
    parser.add_argument("--prompt", default=_DEFAULT_PROMPT)
    parser.add_argument("--training-date", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Build a scene bundle from command-line arguments.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.
    """
    args = _build_parser().parse_args(argv)
    output = build_alpasim_clipgt_usdz(
        args.output,
        clipgt_dir=args.clipgt_dir,
        initial_frame_path=args.initial_frame,
        initial_video_path=args.initial_video,
        camera_clipgt_name=args.camera,
        prompt=args.prompt,
        training_date=args.training_date,
    )
    print(output)


if __name__ == "__main__":
    main()
