# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import io
import json
import zipfile
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import yaml
from loguru import logger
from omnidreams.interactive_drive.colors import (
    BBOX_V3_COLORS,
    HDMAP_V3_COLORS,
    LANE_LINE_STYLE_CONFIG,
)
from omnidreams.interactive_drive.config import RasterConfig
from omnidreams.interactive_drive.math3d import (
    euler_xyz_degrees_to_matrix,
    extract_yaw_from_transform,
    normalize_camera_name,
    quaternion_to_matrix_xyzw,
    transform_from_rt,
)
from omnidreams.interactive_drive.patterns import (
    apply_pattern,
    concatenate_segments,
    resample_polyline,
    segments_from_polyline,
    split_segment_runs,
    subdivide_polyline,
    triangulate_polygon_fan,
)
from omnidreams.interactive_drive.ply_io import load_mesh_vf
from omnidreams.interactive_drive.types import (
    CameraCalibration,
    SceneBundle,
    WorldLineSegments,
    WorldPolygonList,
    WorldTriangleList,
    WorldVehicleBBoxTrack,
)
from omnidreams.scenes import (
    SCENE_FRAME_SUFFIXES,
    SCENE_FRAMES_DIRNAME,
    prompt_variant_for_scene_variant,
    resolve_variant_archive,
    variant_from_stem,
)
from PIL import Image

_GROUND_MESH_NAME = "mesh_ground.ply"


@dataclass(frozen=True)
class _PromptEntry:
    archive_name: str
    text: str


def _read_yaml(zf: zipfile.ZipFile, name: str) -> dict[str, Any]:
    return yaml.safe_load(zf.read(name))


def _read_json(zf: zipfile.ZipFile, name: str) -> dict[str, Any]:
    return json.loads(zf.read(name))


def _read_parquet_records(zf: zipfile.ZipFile, name: str) -> list[dict[str, Any]]:
    with zf.open(name) as handle:
        return pq.read_table(handle).to_pylist()


def _points_from_records(points: list[dict[str, float]]) -> np.ndarray:
    return np.array(
        [[point["x"], point["y"], point["z"]] for point in points], dtype=np.float32
    )


def _load_initial_image(
    zf: zipfile.ZipFile, camera_name: str, variant: str, raster: RasterConfig
) -> np.ndarray:
    """Seed frame: the GT first camera frame, else the ``first_image`` render.

    Prefers ``frames/<camera>/<ts>.jpeg`` so generation starts from the real
    capture; falls back to ``first_image[_<variant>].png`` for older /
    synthetic scenes with no per-camera frames.
    """
    name = _discover_initial_frame(zf, camera_name)
    if name is None:
        images = _discover_first_images(zf)
        name = images.get(variant) or images.get("default")
    if name is None:
        raise FileNotFoundError(
            "No frames/<camera>/*.jpeg or first_image*.png found in the USDZ archive"
        )
    _log_initial_frame_selection(
        zf,
        variant=variant,
        camera_name=camera_name,
        source=name,
    )
    with Image.open(io.BytesIO(zf.read(name))) as image:
        rgb = image.convert("RGB")
        resized = rgb.resize(raster.resolution_wh, resample=Image.Resampling.BILINEAR)
        return np.asarray(resized, dtype=np.uint8)


def _discover_initial_frame(zf: zipfile.ZipFile, camera_name: str) -> str | None:
    """Earliest GT frame for ``camera_name`` (``None`` if the archive has none).

    Frames are ``frames/<camera>/<ts_us>.jpeg``; the smallest timestamp is
    the first frame. Accepts colon / underscore camera-name spellings.
    """
    clipgt_name, logical_name = normalize_camera_name(camera_name)
    wanted_prefixes = tuple(
        {
            f"{SCENE_FRAMES_DIRNAME}/{name}/"
            for name in (camera_name, logical_name, clipgt_name)
        }
    )
    candidates = [
        name
        for name in zf.namelist()
        if name.startswith(wanted_prefixes)
        and Path(name).suffix.lower() in SCENE_FRAME_SUFFIXES
    ]
    if not candidates:
        return None

    def _frame_sort_key(name: str) -> tuple[int, str]:
        stem = Path(name).stem
        return (int(stem), name) if stem.isdigit() else (2**63 - 1, name)

    return sorted(candidates, key=_frame_sort_key)[0]


def _load_prompt(zf: zipfile.ZipFile, variant: str, prompt_override: str | None) -> str:
    if prompt_override is not None:
        _log_prompt_selection(
            zf,
            variant=variant,
            selected_variant="override",
            source="--prompt",
            prompt=prompt_override,
            available_variants=(),
            ignored_files=(),
        )
        return prompt_override
    prompt_entries, ignored_files = _discover_prompt_entries(zf)
    selected_variant = _select_prompt_variant(prompt_entries, variant)
    prompt_entry = (
        prompt_entries[selected_variant] if selected_variant is not None else None
    )
    prompt = "" if prompt_entry is None else prompt_entry.text
    _log_prompt_selection(
        zf,
        variant=variant,
        selected_variant=selected_variant,
        source="<none>" if prompt_entry is None else prompt_entry.archive_name,
        prompt=prompt,
        available_variants=tuple(sorted(prompt_entries.keys())),
        ignored_files=ignored_files,
    )
    return prompt


def _select_prompt_variant(
    prompt_entries: dict[str, _PromptEntry], variant: str
) -> str | None:
    """Pick the in-archive prompt key for the requested scene variant.

    Exact match wins first (legacy in-zip ``default`` / ``1`` / ``2``), else
    the weather->prompt mapping, else ``"default"``, else ``None``.
    """
    if variant in prompt_entries:
        return variant
    mapped = prompt_variant_for_scene_variant(variant)
    if mapped in prompt_entries:
        return mapped
    if "default" in prompt_entries:
        return "default"
    return None


def _discover_prompts(zf: zipfile.ZipFile) -> dict[str, str]:
    prompt_entries, _ = _discover_prompt_entries(zf)
    return {variant: entry.text for variant, entry in prompt_entries.items()}


def _discover_prompt_entries(
    zf: zipfile.ZipFile,
) -> tuple[dict[str, _PromptEntry], tuple[str, ...]]:
    prompts: dict[str, _PromptEntry] = {}
    ignored_files: list[str] = []
    for name in zf.namelist():
        if "/" in name or not name.startswith("prompt") or not name.endswith(".txt"):
            continue
        variant = variant_from_stem(Path(name).stem, "prompt")
        if variant is None:
            ignored_files.append(name)
            continue
        prompts[variant] = _PromptEntry(
            archive_name=name,
            text=zf.read(name).decode("utf-8").strip(),
        )
    if "default" not in prompts and prompts:
        first_key = sorted(prompts.keys())[0]
        prompts["default"] = prompts[first_key]
    return prompts, tuple(sorted(ignored_files))


def _log_prompt_selection(
    zf: zipfile.ZipFile,
    *,
    variant: str,
    selected_variant: str | None,
    source: str,
    prompt: str,
    available_variants: tuple[str, ...],
    ignored_files: tuple[str, ...],
) -> None:
    scene_name = Path(str(zf.filename)).name if zf.filename is not None else "<archive>"
    prompt_text = " ".join(prompt.split())
    logger.info(
        "[scene_loader] prompt "
        f"scene={scene_name!r} "
        f"requested_variant={variant!r} "
        f"selected_variant={selected_variant or '<none>'!r} "
        f"source={source!r} "
        f"available_variants={available_variants or '<none>'!r} "
        f"ignored_files={ignored_files or '<none>'!r} "
        f"length={len(prompt)} "
        f"text={prompt_text!r}",
    )


def _log_initial_frame_selection(
    zf: zipfile.ZipFile,
    *,
    variant: str,
    camera_name: str,
    source: str,
) -> None:
    scene_name = Path(str(zf.filename)).name if zf.filename is not None else "<archive>"
    logger.info(
        "[scene_loader] initial_frame "
        f"scene={scene_name!r} "
        f"requested_variant={variant!r} "
        f"camera={camera_name!r} "
        f"source={source!r}",
    )


def _discover_first_images(zf: zipfile.ZipFile) -> dict[str, str]:
    images: dict[str, str] = {}
    for name in zf.namelist():
        if (
            "/" in name
            or not name.startswith("first_image")
            or not name.endswith(".png")
        ):
            continue
        variant = variant_from_stem(Path(name).stem, "first_image")
        if variant is None:
            continue
        images[variant] = name
    if "default" not in images and images:
        first_key = sorted(images.keys())[0]
        images["default"] = images[first_key]
    return images


def _select_camera(
    sensor_records: list[dict[str, Any]], requested_name: str
) -> dict[str, Any]:
    requested_clipgt, requested_logical = normalize_camera_name(requested_name)
    for sensor in sensor_records:
        if sensor["name"] in {requested_clipgt, requested_logical}:
            return sensor
    raise KeyError(f"Camera {requested_name!r} was not found in the calibration rig")


def _load_camera_calibration(
    zf: zipfile.ZipFile, camera_name: str
) -> CameraCalibration:
    calibration_row = _read_parquet_records(zf, "clipgt/calibration_estimate.parquet")[
        0
    ]["calibration_estimate"]
    rig = json.loads(calibration_row["rig_json"])["rig"]
    sensor = _select_camera(rig["sensors"], camera_name)

    props = sensor["properties"]
    poly_type = props["polynomial-type"]
    is_backward = poly_type == "pixeldistance-to-angle"
    polynomial = np.array(
        [float(value) for value in props["polynomial"].split()], dtype=np.float32
    )
    linear_cde = np.array(
        [
            float(props.get("linear-c", 1.0)),
            float(props.get("linear-d", 0.0)),
            float(props.get("linear-e", 0.0)),
        ],
        dtype=np.float32,
    )

    nominal_rotation = euler_xyz_degrees_to_matrix(
        sensor["nominalSensor2Rig_FLU"]["roll-pitch-yaw"]
    )
    correction_rotation = euler_xyz_degrees_to_matrix(
        sensor.get("correction_sensor_R_FLU", {"roll-pitch-yaw": [0.0, 0.0, 0.0]})[
            "roll-pitch-yaw"
        ]
    )
    sensor_to_rig_rotation = (nominal_rotation @ correction_rotation).astype(np.float32)

    nominal_translation = np.asarray(
        sensor["nominalSensor2Rig_FLU"]["t"], dtype=np.float32
    )
    correction_translation = np.asarray(
        sensor.get("correction_rig_T", [0.0, 0.0, 0.0]), dtype=np.float32
    )
    sensor_to_rig_translation = (nominal_translation + correction_translation).astype(
        np.float32
    )

    clipgt_name, logical_name = normalize_camera_name(sensor["name"])
    return CameraCalibration(
        clipgt_name=clipgt_name,
        logical_name=logical_name,
        width=int(props["width"]),
        height=int(props["height"]),
        cx=float(props["cx"]),
        cy=float(props["cy"]),
        polynomial=polynomial,
        is_backward_polynomial=is_backward,
        linear_cde=linear_cde,
        sensor_to_rig_flu=transform_from_rt(
            sensor_to_rig_rotation, sensor_to_rig_translation.tolist()
        ),
    )


def _load_initial_state(zf: zipfile.ZipFile) -> tuple[np.ndarray, int, float, float]:
    trajectory_doc = _read_json(zf, "rig_trajectories.json")
    rig_trajectory = trajectory_doc["rig_trajectories"][0]
    poses = np.asarray(rig_trajectory["T_rig_worlds"], dtype=np.float32)
    timestamps = np.asarray(rig_trajectory["T_rig_world_timestamps_us"], dtype=np.int64)

    initial_pose = poses[0].astype(np.float32)
    initial_timestamp = int(timestamps[0])
    initial_yaw = extract_yaw_from_transform(initial_pose)

    if len(poses) > 1:
        delta_t_s = max(1e-6, (int(timestamps[1]) - int(timestamps[0])) / 1_000_000.0)
        delta_xy = poses[1, :2, 3] - poses[0, :2, 3]
        initial_speed = float(np.linalg.norm(delta_xy) / delta_t_s)
    else:
        initial_speed = 0.0

    return initial_pose, initial_timestamp, initial_yaw, initial_speed


def _sanitize_layer_suffix(name: str) -> str:
    return name.lower().replace(" ", "_")


def _coarsen_segment_group(segments_world: np.ndarray, interval_m: float) -> np.ndarray:
    coarsened_runs: list[np.ndarray] = []
    for run in split_segment_runs(segments_world):
        if len(run) <= 2:
            coarsened_runs.append(segments_from_polyline(run))
            continue
        sampled = resample_polyline(run, interval_m=interval_m)
        coarsened_runs.append(segments_from_polyline(sampled))
    return concatenate_segments(coarsened_runs)


def _build_lane_segments(
    rows: list[dict[str, Any]], raster: RasterConfig
) -> tuple[WorldLineSegments, ...]:
    grouped_segments: dict[
        tuple[tuple[float, float, float, float], float], list[np.ndarray]
    ] = defaultdict(list)
    grouped_names: dict[tuple[tuple[float, float, float, float], float], set[str]] = (
        defaultdict(set)
    )

    for row in rows:
        payload = row["lane_line"]
        polyline = _points_from_records(payload["line_rail"])
        if len(polyline) < 2:
            continue
        color_name = next(
            (value for value in payload.get("colors", []) if value), "OTHER"
        )
        style_name = next(
            (value for value in payload.get("styles", []) if value), "OTHER"
        )
        lane_type = f"{color_name} {style_name}".strip()
        config = LANE_LINE_STYLE_CONFIG.get(lane_type, LANE_LINE_STYLE_CONFIG["OTHER"])

        subdivided = subdivide_polyline(polyline, raster.lane_segment_interval_m)
        base_segments = segments_from_polyline(subdivided)
        patterned_segments = apply_pattern(
            base_segments,
            pattern=str(config.get("pattern", "solid")),
            dual_pattern=config.get("dual_pattern"),  # type: ignore[arg-type]
            dual_offset_m=raster.dual_line_offset_m,
        )
        color_rgba = tuple(float(value) for value in config["color"])  # type: ignore[arg-type]
        width_px = raster.line_width_px * float(config.get("width_scale", 1.0))
        group_key = (color_rgba, width_px)
        for group in patterned_segments:
            if len(group) == 0:
                continue
            grouped_segments[group_key].append(
                _coarsen_segment_group(
                    group, interval_m=raster.polyline_segment_interval_m
                )
            )
            grouped_names[group_key].add(lane_type)

    if not grouped_segments:
        return tuple()

    layers: list[WorldLineSegments] = []
    for key, segment_groups in grouped_segments.items():
        color_rgba, width_px = key
        style_names = "+".join(
            sorted(_sanitize_layer_suffix(name) for name in grouped_names[key])
        )
        layers.append(
            WorldLineSegments(
                segments_world=concatenate_segments(segment_groups),
                color_rgba=color_rgba,
                width_px=width_px,
                layer_name=f"lanelines_{style_names}",
            )
        )
    return tuple(layers)


def _build_cuboid_corners(
    center_xyz: np.ndarray,
    dimensions_xyz: np.ndarray,
    orientation_xyzw: tuple[float, float, float, float],
) -> np.ndarray:
    rotation = quaternion_to_matrix_xyzw(orientation_xyzw)
    half = dimensions_xyz * 0.5
    corners = np.array(
        [
            [-half[0], -half[1], -half[2]],
            [half[0], -half[1], -half[2]],
            [half[0], half[1], -half[2]],
            [-half[0], half[1], -half[2]],
            [-half[0], -half[1], half[2]],
            [half[0], -half[1], half[2]],
            [half[0], half[1], half[2]],
            [-half[0], half[1], half[2]],
        ],
        dtype=np.float32,
    )
    return (corners @ rotation.T) + center_xyz


def _build_cuboid_plate_faces(
    center_xyz: np.ndarray,
    dimensions_xyz: np.ndarray,
    orientation_xyzw: tuple[float, float, float, float],
) -> np.ndarray:
    corners = _build_cuboid_corners(center_xyz, dimensions_xyz, orientation_xyzw)
    thinnest_axis = int(np.argmin(dimensions_xyz))
    face_indices_by_axis = {
        0: ((0, 3, 7, 4), (1, 2, 6, 5)),
        1: ((0, 1, 5, 4), (3, 2, 6, 7)),
        2: ((0, 1, 2, 3), (4, 5, 6, 7)),
    }
    quads = [
        corners[np.array(indices, dtype=np.int32)]
        for indices in face_indices_by_axis[thinnest_axis]
    ]
    return np.concatenate(
        [triangulate_polygon_fan(quad) for quad in quads], axis=0
    ).astype(np.float32)


def _build_sign_face_layer(
    rows: list[dict[str, Any]], payload_key: str, layer_name: str
) -> WorldTriangleList:
    triangles: list[np.ndarray] = []
    for row in rows:
        payload = row[payload_key]
        center = np.array(
            [payload["center"]["x"], payload["center"]["y"], payload["center"]["z"]],
            dtype=np.float32,
        )
        dims = np.array(
            [
                payload["dimensions"]["x"],
                payload["dimensions"]["y"],
                payload["dimensions"]["z"],
            ],
            dtype=np.float32,
        )
        orientation = (
            float(payload["orientation"]["x"]),
            float(payload["orientation"]["y"]),
            float(payload["orientation"]["z"]),
            float(payload["orientation"]["w"]),
        )
        triangles.append(_build_cuboid_plate_faces(center, dims, orientation))
    triangles_world = (
        np.concatenate(triangles, axis=0).astype(np.float32)
        if triangles
        else np.empty((0, 3, 3), dtype=np.float32)
    )
    return WorldTriangleList(
        triangles_world=triangles_world,
        color_rgba=HDMAP_V3_COLORS[layer_name],
        layer_name=layer_name,
    )


def _build_cuboid_edges(
    center_xyz: np.ndarray,
    dimensions_xyz: np.ndarray,
    orientation_xyzw: tuple[float, float, float, float],
) -> np.ndarray:
    corners = _build_cuboid_corners(center_xyz, dimensions_xyz, orientation_xyzw)
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    return np.array([[corners[a], corners[b]] for a, b in edges], dtype=np.float32)


def _build_cuboid_layer(
    rows: list[dict[str, Any]], payload_key: str, layer_name: str, width_px: float
) -> WorldLineSegments:
    groups: list[np.ndarray] = []
    null_orientation_rows = 0
    for row in rows:
        payload = row[payload_key]
        center = np.array(
            [payload["center"]["x"], payload["center"]["y"], payload["center"]["z"]],
            dtype=np.float32,
        )
        dims = np.array(
            [
                payload["dimensions"]["x"],
                payload["dimensions"]["y"],
                payload["dimensions"]["z"],
            ],
            dtype=np.float32,
        )
        # Some ClipGT scenes publish a null orientation quaternion (upstream
        # pose-fit failure); default to identity so the cuboid still loads
        # axis-aligned instead of dropping out of the HDMap view.
        orient = payload.get("orientation") or {}
        qx = orient.get("x")
        qy = orient.get("y")
        qz = orient.get("z")
        qw = orient.get("w")
        if qx is None or qy is None or qz is None or qw is None:
            qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
            null_orientation_rows += 1
        orientation = (float(qx), float(qy), float(qz), float(qw))
        groups.append(_build_cuboid_edges(center, dims, orientation))
    if null_orientation_rows:
        logger.info(
            f"[scene_loader] {layer_name}: {null_orientation_rows} of "
            f"{len(rows)} cuboid(s) had null orientation in the parquet; "
            f"rendered as identity-rotated. Upstream dataset bug "
            f"(missing pose-fit for some features).",
        )
    return WorldLineSegments(
        segments_world=concatenate_segments(groups),
        color_rgba=HDMAP_V3_COLORS[layer_name],
        width_px=width_px,
        layer_name=layer_name,
    )


def _build_polyline_layer(
    rows: list[dict[str, Any]],
    payload_key: str,
    points_key: str,
    layer_name: str,
    raster: RasterConfig,
    width_px: float,
) -> WorldLineSegments:
    segments: list[np.ndarray] = []
    for row in rows:
        polyline = _points_from_records(row[payload_key][points_key])
        subdivided = subdivide_polyline(polyline, raster.polyline_segment_interval_m)
        line_segments = segments_from_polyline(subdivided)
        if len(line_segments) > 0:
            segments.append(line_segments)

    return WorldLineSegments(
        segments_world=concatenate_segments(segments),
        color_rgba=HDMAP_V3_COLORS[layer_name],
        width_px=width_px,
        layer_name=layer_name,
    )


def _build_polygon_loop_layer(
    rows: list[dict[str, Any]],
    payload_key: str,
    points_key: str,
    layer_name: str,
    raster: RasterConfig,
) -> WorldPolygonList:
    polygons_world: list[np.ndarray] = []
    for row in rows:
        polygon = _points_from_records(row[payload_key][points_key])
        if len(polygon) < 3:
            continue
        if np.linalg.norm(polygon[0] - polygon[-1]) <= 1e-4:
            polygon = polygon[:-1]
        if len(polygon) < 3:
            continue
        polygons_world.append(polygon.astype(np.float32))

    return WorldPolygonList(
        polygons_world=tuple(polygons_world),
        color_rgba=HDMAP_V3_COLORS[layer_name],
        layer_name=layer_name,
    )


def _map_obstacle_category_to_bbox_type(category: str) -> str:
    normalized = category.replace("_", " ").replace("-", " ").title().replace(" ", "_")
    if normalized in {
        "Bus",
        "Heavy_Truck",
        "Train_Or_Tram_Car",
        "Trolley_Bus",
        "Trailer",
        "Truck",
    }:
        return "Truck"
    if normalized in {"Vehicle", "Automobile", "Other_Vehicle", "Car"}:
        return "Car"
    if normalized in {"Person", "Pedestrian"}:
        return "Pedestrian"
    if normalized in {"Rider", "Cyclist", "Motorcycle", "Bicycle"}:
        return "Cyclist"
    return "Others"


def _load_vehicle_bbox_tracks(zf: zipfile.ZipFile) -> tuple[WorldVehicleBBoxTrack, ...]:
    if "clipgt/obstacle.parquet" not in zf.namelist():
        return tuple()

    obstacle_rows = _read_parquet_records(zf, "clipgt/obstacle.parquet")
    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in obstacle_rows:
        obstacle_payload = row["obstacle"]
        track_id = str(obstacle_payload.get("trackline_id", ""))
        if track_id == "":
            continue
        grouped_rows[track_id].append(row)

    tracks: list[WorldVehicleBBoxTrack] = []
    for track_id in sorted(grouped_rows.keys()):
        observations = sorted(
            grouped_rows[track_id], key=lambda obs: int(obs["key"]["timestamp_micros"])
        )
        timestamps_us: list[int] = []
        centers_world: list[list[float]] = []
        dimensions_lwh: list[list[float]] = []
        orientations_xyzw: list[list[float]] = []
        for observation in observations:
            obstacle_payload = observation["obstacle"]
            center = obstacle_payload["center"]
            size = obstacle_payload["size"]
            orientation = obstacle_payload["orientation"]
            quaternion = np.array(
                [
                    float(orientation["x"]),
                    float(orientation["y"]),
                    float(orientation["z"]),
                    float(orientation["w"]),
                ],
                dtype=np.float32,
            )
            if float(np.linalg.norm(quaternion)) <= 1e-8:
                continue
            timestamps_us.append(int(observation["key"]["timestamp_micros"]))
            centers_world.append(
                [float(center["x"]), float(center["y"]), float(center["z"])]
            )
            dimensions_lwh.append(
                [float(size["x"]), float(size["y"]), float(size["z"])]
            )
            orientations_xyzw.append(quaternion.tolist())

        if len(timestamps_us) < 2:
            continue
        object_type = _map_obstacle_category_to_bbox_type(
            str(observations[0]["obstacle"].get("category", "Others"))
        )
        if object_type not in BBOX_V3_COLORS:
            object_type = "Others"
        tracks.append(
            WorldVehicleBBoxTrack(
                track_id=track_id,
                object_type=object_type,
                timestamps_us=np.asarray(timestamps_us, dtype=np.int64),
                centers_world=np.asarray(centers_world, dtype=np.float32),
                dimensions_lwh=np.asarray(dimensions_lwh, dtype=np.float32),
                orientations_xyzw=np.asarray(orientations_xyzw, dtype=np.float32),
                # TODO: Excluding ego obstacle requires metadata not currently parsed here.
                max_extrapolation_us=500_000.0,
            )
        )
    return tuple(tracks)


def _load_map_layers(
    zf: zipfile.ZipFile,
    raster: RasterConfig,
) -> tuple[
    tuple[WorldLineSegments, ...],
    tuple[WorldTriangleList, ...],
    tuple[WorldPolygonList, ...],
]:
    def has(name: str) -> bool:
        return name in zf.namelist()

    line_layers: list[WorldLineSegments] = []
    triangle_layers: list[WorldTriangleList] = []
    polygon_layers: list[WorldPolygonList] = []

    if has("clipgt/lane_line.parquet"):
        line_layers.extend(
            _build_lane_segments(
                _read_parquet_records(zf, "clipgt/lane_line.parquet"), raster
            )
        )
    if has("clipgt/road_boundary.parquet"):
        line_layers.append(
            _build_polyline_layer(
                _read_parquet_records(zf, "clipgt/road_boundary.parquet"),
                payload_key="road_boundary",
                points_key="location",
                layer_name="road_boundaries",
                raster=raster,
                width_px=raster.line_width_px,
            )
        )
    if has("clipgt/wait_line.parquet"):
        line_layers.append(
            _build_polyline_layer(
                _read_parquet_records(zf, "clipgt/wait_line.parquet"),
                payload_key="wait_line",
                points_key="location",
                layer_name="wait_lines",
                raster=raster,
                width_px=raster.line_width_px,
            )
        )
    if has("clipgt/pole.parquet"):
        line_layers.append(
            _build_polyline_layer(
                _read_parquet_records(zf, "clipgt/pole.parquet"),
                payload_key="pole",
                points_key="location",
                layer_name="poles",
                raster=raster,
                width_px=raster.pole_width_px,
            )
        )
    if has("clipgt/traffic_sign.parquet"):
        triangle_layers.append(
            _build_sign_face_layer(
                _read_parquet_records(zf, "clipgt/traffic_sign.parquet"),
                "traffic_sign",
                "traffic_signs",
            )
        )
    if has("clipgt/traffic_light.parquet"):
        line_layers.append(
            _build_cuboid_layer(
                _read_parquet_records(zf, "clipgt/traffic_light.parquet"),
                "traffic_light",
                "traffic_lights",
                raster.line_width_px,
            )
        )

    if has("clipgt/crosswalk.parquet"):
        polygon_layers.append(
            _build_polygon_loop_layer(
                _read_parquet_records(zf, "clipgt/crosswalk.parquet"),
                "crosswalk",
                "location",
                "crosswalks",
                raster,
            )
        )
    if has("clipgt/road_marking.parquet"):
        polygon_layers.append(
            _build_polygon_loop_layer(
                _read_parquet_records(zf, "clipgt/road_marking.parquet"),
                "road_marking",
                "location",
                "road_markings",
                raster,
            )
        )
    if has("clipgt/intersection_area.parquet"):
        polygon_layers.append(
            _build_polygon_loop_layer(
                _read_parquet_records(zf, "clipgt/intersection_area.parquet"),
                "intersection_area",
                "location",
                "intersection_areas",
                raster,
            )
        )
    if has("clipgt/road_island.parquet"):
        polygon_layers.append(
            _build_polygon_loop_layer(
                _read_parquet_records(zf, "clipgt/road_island.parquet"),
                "road_island",
                "location",
                "road_islands",
                raster,
            )
        )

    return tuple(line_layers), tuple(triangle_layers), tuple(polygon_layers)


def _load_ground_mesh(
    zf: zipfile.ZipFile,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Read ``mesh_ground.ply`` from the USDZ archive if present.

    Returns ``(vertices, faces)`` for use by
    :class:`omnidreams.interactive_drive.simulation.ground_snap.GroundSnapper`, or ``(None, None)`` when the
    archive ships no ground mesh (e.g. legacy fixtures), in which case
    ground-snap silently no-ops at runtime.
    """
    if _GROUND_MESH_NAME not in zf.namelist():
        return None, None
    try:
        vertices, faces = load_mesh_vf(zf.read(_GROUND_MESH_NAME))
    except (ValueError, TypeError) as exc:
        logger.info(
            f"[scene_loader] failed to parse {_GROUND_MESH_NAME}: {exc}; "
            "ground-snap will no-op for this scene.",
        )
        return None, None
    return vertices.astype(np.float32), faces.astype(np.int32)


def load_scene_bundle(
    scene_path: Path,
    camera_name: str,
    variant: str,
    prompt_override: str | None,
    raster: RasterConfig,
) -> SceneBundle:
    # Swap to the requested variant's sibling archive when present; legacy
    # single-archive scenes resolve to the same path (variant picked in-zip).
    scene_path = resolve_variant_archive(Path(scene_path), variant)
    with zipfile.ZipFile(scene_path, "r") as zf:
        metadata = _read_yaml(zf, "metadata.yaml")
        camera = _load_camera_calibration(zf, camera_name)
        initial_pose, initial_timestamp, initial_yaw, initial_speed = (
            _load_initial_state(zf)
        )
        initial_rgb = _load_initial_image(zf, camera_name, variant, raster)
        prompt = _load_prompt(zf, variant, prompt_override)
        line_layers, triangle_layers, polygon_layers = _load_map_layers(zf, raster)
        vehicle_bbox_tracks = _load_vehicle_bbox_tracks(zf)
        ground_mesh_vertices, ground_mesh_faces = _load_ground_mesh(zf)

    return SceneBundle(
        scene_path=scene_path,
        scene_id=str(metadata.get("scene_id", scene_path.stem)),
        metadata=metadata,
        selected_camera=camera,
        initial_rig_to_world=initial_pose,
        initial_timestamp_us=initial_timestamp,
        initial_yaw_rad=initial_yaw,
        initial_speed_mps=initial_speed,
        initial_rgb=initial_rgb,
        prompt=prompt,
        line_layers=line_layers,
        triangle_layers=triangle_layers,
        polygon_layers=polygon_layers,
        vehicle_bbox_tracks=vehicle_bbox_tracks,
        ground_mesh_vertices=ground_mesh_vertices,
        ground_mesh_faces=ground_mesh_faces,
    )


def reseed_scene_bundle(
    bundle: SceneBundle,
    scene_path: Path,
    camera_name: str,
    variant: str,
    prompt_override: str | None,
    raster: RasterConfig,
) -> SceneBundle:
    """Re-seed an already-parsed ``bundle`` for a different weather variant.

    Variants share all geometry; only the initial frame and prompt differ, so
    this reads just those from the variant's archive and reuses the rest,
    skipping the full re-parse and bounds/snapper rebuild.
    """
    scene_path = resolve_variant_archive(Path(scene_path), variant)
    with zipfile.ZipFile(scene_path, "r") as zf:
        initial_rgb = _load_initial_image(zf, camera_name, variant, raster)
        prompt = _load_prompt(zf, variant, prompt_override)
    return replace(
        bundle, scene_path=scene_path, initial_rgb=initial_rgb, prompt=prompt
    )
