# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from omnidreams.interactive_drive.math3d import rig_pose_from_state
from omnidreams.interactive_drive.ply_io import save_mesh_vf
from PIL import Image

_SCENE_ID = "synthetic-test-scene"
_CAMERA_CLIPGT_NAME = "camera:front:wide:120fov"
_CAMERA_LOGICAL_NAME = "camera_front_wide_120fov"
_FPS = 30
# Trajectory length used by the test fixture and as the default for the
# runtime synthetic-scene helper. 180 frames (6 s at 30 fps, ~60 m at the
# default speed) is enough for the scene-loader unit tests; the runtime
# helper bumps this to ~10 minutes of road via ``length_frames``.
_DEFAULT_TRAJECTORY_FRAMES = 180
_START_TIMESTAMP_US = 1_700_000_000_000_000
# Centerline shape. A pure sine at one wavelength reads as visually
# monotonous over a multi-km drive, so we superpose two waves: a long
# highway-style sweep (1200 m wavelength) and a short drift (200 m
# wavelength). The combined min turn radius is ~600 m -- a clearly
# visible curve from inside the cab without snaking, and the differing
# periods stop the road from feeling like the same kilometre repeated.
_WAVE_LONG_AMPLITUDE_M = 7.0
_WAVE_LONG_PERIOD_S = 120.0
_WAVE_SHORT_AMPLITUDE_M = 2.0
_WAVE_SHORT_PERIOD_S = 20.0
_FORWARD_SPEED_MPS = 10.0
# Lateral half-widths used by the synthetic geometry (meters from the
# centerline).
#   - lane lines at ~1.8 m matches a standard US lane (3.6 m wide).
#   - road boundaries at 9 m make the visible road 18 m wide, i.e. a
#     2-lane carriageway plus shoulders, which is the look the
#     synthetic demo defaults to.
_LANE_LINE_OFFSET_M = 1.8
_ROAD_BOUNDARY_OFFSET_M = 9.0
_POLE_OFFSET_M = 9.5
# Periodic roadside furniture. Poles spaced like streetlamps (~50 m on
# real highways), parked cars on alternating shoulders, and traffic
# signs on alternating shoulders. Each helper respects the wavy
# centerline so the geometry stays anchored to the lane regardless of
# drive distance.
_POLE_PERIOD_M = 50.0
_PARKED_CAR_PERIOD_M = 150.0
_PARKED_CAR_LATERAL_M = 5.0
_TRAFFIC_SIGN_PERIOD_M = 200.0
_TRAFFIC_SIGN_LATERAL_M = 7.0
_TRAFFIC_SIGN_HEIGHT_M = 2.5
# Off-road clutter ("trees / telephone poles") scattered far from the road
# in a wide lateral band. Without these the HDMap goes empty as soon as
# the ego strays past the road boundary -- the world model gets a black
# conditioning frame and either freezes or drifts. Spacing is dense
# (one pair every 15 m forward) and lateral position + height are
# randomized in fixed seeded ranges so the scene reproduces across
# runs while still looking varied. Both sides get a pole per period
# but at INDEPENDENT random lateral offsets so it doesn't read as a
# parallel two-line corridor.
_OFF_ROAD_POLE_PERIOD_M = 15.0
_OFF_ROAD_POLE_LATERAL_MIN_M = 15.0
_OFF_ROAD_POLE_LATERAL_MAX_M = 100.0
_OFF_ROAD_POLE_HEIGHT_MIN_M = 3.0
_OFF_ROAD_POLE_HEIGHT_MAX_M = 8.0
_OFF_ROAD_POLE_FORWARD_JITTER_M = 4.0
_OFF_ROAD_POLE_RNG_SEED = 42
# Long lane-line / road-boundary polylines render as a single subdivided
# strip; very long strips can fade out at distance because the rasterizer
# coarsens the whole run. Splitting into ~80 m chunks keeps each render
# group short and self-contained, which mitigates the distance fade and
# also avoids any per-polyline soft caps in the line uploader.
_LANE_LINE_CHUNK_FRAMES = 240
_IMAGE_WIDTH = 1280
_IMAGE_HEIGHT = 704


def _point_xyz(x_m: float, y_m: float, z_m: float) -> dict[str, float]:
    return {"x": float(x_m), "y": float(y_m), "z": float(z_m)}


def _orientation_from_yaw(yaw_rad: float) -> dict[str, float]:
    half = 0.5 * yaw_rad
    return {"x": 0.0, "y": 0.0, "z": float(math.sin(half)), "w": float(math.cos(half))}


def _trajectory_arrays(
    num_frames: int = _DEFAULT_TRAJECTORY_FRAMES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dt_s = 1.0 / float(_FPS)
    frame_ids = np.arange(int(num_frames), dtype=np.float32)
    t_s = frame_ids * np.float32(dt_s)

    x_m = _FORWARD_SPEED_MPS * t_s
    # Sum of two sines at different periods reads as a more naturally
    # varying road than a single sine at a fixed period.
    long_omega = 2.0 * np.pi / _WAVE_LONG_PERIOD_S
    short_omega = 2.0 * np.pi / _WAVE_SHORT_PERIOD_S
    long_phase = long_omega * t_s
    short_phase = short_omega * t_s
    y_m = _WAVE_LONG_AMPLITUDE_M * np.sin(
        long_phase
    ) + _WAVE_SHORT_AMPLITUDE_M * np.sin(short_phase)
    dydt = _WAVE_LONG_AMPLITUDE_M * long_omega * np.cos(
        long_phase
    ) + _WAVE_SHORT_AMPLITUDE_M * short_omega * np.cos(short_phase)
    yaw_rad = np.arctan2(dydt, _FORWARD_SPEED_MPS).astype(np.float32)
    timestamps_us = _START_TIMESTAMP_US + np.round(t_s * 1_000_000.0).astype(np.int64)
    return x_m.astype(np.float32), y_m.astype(np.float32), yaw_rad, timestamps_us


def _resample_centerline(
    x_m: np.ndarray,
    y_m: np.ndarray,
    yaw_rad: np.ndarray,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return x_m[::stride], y_m[::stride], yaw_rad[::stride]


def _offset_polyline(
    x_m: np.ndarray,
    y_m: np.ndarray,
    yaw_rad: np.ndarray,
    lateral_offset_m: float,
) -> list[dict[str, float]]:
    nx = -np.sin(yaw_rad) * lateral_offset_m
    ny = np.cos(yaw_rad) * lateral_offset_m
    return [
        _point_xyz(px + ox, py + oy, 0.0)
        for px, py, ox, oy in zip(x_m, y_m, nx, ny, strict=True)
    ]


def _chunk_polyline(
    points: list[dict[str, float]],
    *,
    chunk_size: int,
) -> list[list[dict[str, float]]]:
    """Split a long polyline into overlapping chunks of ~``chunk_size`` points.

    The chunks share their endpoint with the previous chunk so the
    rasterizer's per-row coarsening produces a continuous line on screen
    instead of visible gaps at the chunk boundaries.

    Splitting matters for the synthetic scene because lane lines and
    road boundaries each cover the full ~10-20 km trajectory; a single
    polyline of 30 000+ points runs into the rasterizer's distance-fade
    behaviour for very long subdivided strips.
    """
    if chunk_size < 2:
        raise ValueError(f"chunk_size must be >= 2, got {chunk_size}")
    if len(points) <= chunk_size:
        return [points]
    chunks: list[list[dict[str, float]]] = []
    start = 0
    while start < len(points) - 1:
        end = min(start + chunk_size, len(points))
        chunks.append(points[start:end])
        if end == len(points):
            break
        start = end - 1  # share endpoint with the next chunk
    return chunks


def _make_key_record(label_class_id: str) -> dict[str, str]:
    return {
        "clip_id": _SCENE_ID,
        "label_class_id": label_class_id,
        "map_id": "synthetic-map",
        "map_id_version": "v1",
    }


def _lane_line_rows(
    x_m: np.ndarray, y_m: np.ndarray, yaw_rad: np.ndarray
) -> list[dict[str, object]]:
    center_x, center_y, center_yaw = _resample_centerline(x_m, y_m, yaw_rad, stride=3)
    left_polyline = _offset_polyline(
        center_x, center_y, center_yaw, lateral_offset_m=_LANE_LINE_OFFSET_M
    )
    right_polyline = _offset_polyline(
        center_x, center_y, center_yaw, lateral_offset_m=-_LANE_LINE_OFFSET_M
    )
    chunk_size = max(2, _LANE_LINE_CHUNK_FRAMES // 3)  # / 3 for the stride=3 resample
    rows: list[dict[str, object]] = []
    for side, points, style, color in (
        ("left", left_polyline, "SOLID_SINGLE", "WHITE"),
        ("right", right_polyline, "DASHED_SOLID", "YELLOW"),
    ):
        for idx, chunk in enumerate(_chunk_polyline(points, chunk_size=chunk_size)):
            rows.append(
                {
                    "key": _make_key_record(f"lane_line_{side}_{idx}"),
                    "lane_line": {
                        "line_rail": chunk,
                        "styles": [style],
                        "colors": [color],
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


def _polyline_rows(
    key_name: str,
    payload_name: str,
    points_name: str,
    point_sets: list[list[dict[str, float]]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx, points in enumerate(point_sets):
        rows.append(
            {
                "key": _make_key_record(f"{key_name}_{idx}"),
                payload_name: {
                    points_name: points,
                    "category": key_name,
                    "egomotion_label_class_id": "ego",
                },
                "version": 1,
            }
        )
    return rows


def _polygon_rows(
    key_name: str,
    payload_name: str,
    points_name: str,
    polygons: list[list[dict[str, float]]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx, points in enumerate(polygons):
        rows.append(
            {
                "key": _make_key_record(f"{key_name}_{idx}"),
                payload_name: {
                    "category": key_name,
                    points_name: points,
                    "egomotion_label_class_id": "ego",
                },
                "version": 1,
            }
        )
    return rows


def _rectangle_polygon(
    center_x_m: float,
    center_y_m: float,
    half_width_m: float,
    half_height_m: float,
) -> list[dict[str, float]]:
    return [
        _point_xyz(center_x_m - half_width_m, center_y_m - half_height_m, 0.0),
        _point_xyz(center_x_m + half_width_m, center_y_m - half_height_m, 0.0),
        _point_xyz(center_x_m + half_width_m, center_y_m + half_height_m, 0.0),
        _point_xyz(center_x_m - half_width_m, center_y_m + half_height_m, 0.0),
        _point_xyz(center_x_m - half_width_m, center_y_m - half_height_m, 0.0),
    ]


def _traffic_sign_rows() -> list[dict[str, object]]:
    # Dimensions use local (length, width, height). `_build_cuboid_plate_faces`
    # in the loader picks the thinnest local axis as the plate normal, so a thin
    # length keeps the plate's normal along world +x (facing ego) at yaw=0.
    return [
        {
            "key": _make_key_record("traffic_sign_right"),
            "traffic_sign": {
                "center": _point_xyz(25.0, -4.0, 2.0),
                "dimensions": _point_xyz(0.12, 1.0, 1.2),
                "orientation": _orientation_from_yaw(0.0),
                "category": "speed_limit",
                "egomotion_label_class_id": "ego",
            },
            "version": 1,
        },
        {
            "key": _make_key_record("traffic_sign_left"),
            "traffic_sign": {
                "center": _point_xyz(32.0, 5.5, 2.5),
                "dimensions": _point_xyz(0.12, 1.4, 0.8),
                "orientation": _orientation_from_yaw(0.0),
                "category": "warning",
                "egomotion_label_class_id": "ego",
            },
            "version": 1,
        },
    ]


def _traffic_light_rows() -> list[dict[str, object]]:
    return [
        {
            "key": _make_key_record("traffic_light_0"),
            "traffic_light": {
                "center": _point_xyz(29.0, -5.5, 4.0),
                "dimensions": _point_xyz(0.4, 0.6, 1.0),
                "orientation": _orientation_from_yaw(0.0),
                "category": "signal_head",
                "egomotion_label_class_id": "ego",
            },
            "version": 1,
        }
    ]


_OBSTACLE_TRACKS: tuple[dict[str, object], ...] = (
    {
        "trackline_id": "car-001",
        "category": "Automobile",  # -> Car bbox color
        "center": (25.0, -1.8, 0.8),
        "size": (4.7, 2.0, 1.6),
    },
    {
        "trackline_id": "truck-001",
        "category": "Heavy_Truck",  # -> Truck bbox color
        "center": (48.0, 2.0, 1.75),
        "size": (9.0, 2.6, 3.5),
    },
    {
        "trackline_id": "pedestrian-001",
        "category": "Pedestrian",
        "center": (18.0, 4.0, 0.9),
        "size": (0.6, 0.6, 1.8),
    },
    {
        "trackline_id": "cyclist-001",
        "category": "Cyclist",
        "center": (23.0, -3.5, 0.9),
        "size": (1.8, 0.5, 1.5),
    },
    {
        "trackline_id": "debris-001",
        "category": "Debris",  # -> Others bbox color
        "center": (43.0, -5.0, 0.35),
        "size": (0.8, 0.8, 0.7),
    },
)


def _periodic_poles(
    x_m: np.ndarray,
    y_m: np.ndarray,
    yaw_rad: np.ndarray,
    *,
    period_m: float = _POLE_PERIOD_M,
    lateral_offset_m: float = _POLE_OFFSET_M,
    height_m: float = 5.0,
) -> list[list[dict[str, float]]]:
    """Emit ``[base, top]`` line segments for streetlamp-style poles spaced
    every ``period_m`` along both shoulders of the trajectory.

    Each pole is anchored to the centerline at its placement frame and
    offset laterally by ``lateral_offset_m`` (positive = left side,
    negative = right side), so poles follow the wavy road instead of a
    straight world-frame line.
    """
    if len(x_m) == 0:
        return []
    poles: list[list[dict[str, float]]] = []
    next_x = period_m
    for i in range(len(x_m)):
        if x_m[i] < next_x:
            continue
        next_x += period_m
        cx = float(x_m[i])
        cy = float(y_m[i])
        cyaw = float(yaw_rad[i])
        # Lateral basis at this trajectory point: rotated by yaw, offset
        # in the body +Y direction. Same convention as ``_offset_polyline``.
        for side in (1.0, -1.0):
            dx = -math.sin(cyaw) * lateral_offset_m * side
            dy = math.cos(cyaw) * lateral_offset_m * side
            base = _point_xyz(cx + dx, cy + dy, 0.0)
            top = _point_xyz(cx + dx, cy + dy, height_m)
            poles.append([base, top])
    return poles


def _off_road_poles(
    x_m: np.ndarray,
    y_m: np.ndarray,
    yaw_rad: np.ndarray,
    *,
    period_m: float = _OFF_ROAD_POLE_PERIOD_M,
    lateral_min_m: float = _OFF_ROAD_POLE_LATERAL_MIN_M,
    lateral_max_m: float = _OFF_ROAD_POLE_LATERAL_MAX_M,
    height_min_m: float = _OFF_ROAD_POLE_HEIGHT_MIN_M,
    height_max_m: float = _OFF_ROAD_POLE_HEIGHT_MAX_M,
    forward_jitter_m: float = _OFF_ROAD_POLE_FORWARD_JITTER_M,
    seed: int = _OFF_ROAD_POLE_RNG_SEED,
) -> list[list[dict[str, float]]]:
    """Scatter "tree / telephone pole" clutter in a wide off-road band.

    Unlike :func:`_periodic_poles` (streetlamp-style poles at a fixed
    offset just outside the road boundary), these are randomly placed in
    a configurable [lateral_min_m, lateral_max_m] strip on both sides of
    the road. Heights vary so the rendered HDMap reads as a forest /
    telephone-pole field instead of a uniform corridor.

    The point of the helper is to keep the HDMap non-empty when the ego
    strays off the road -- without this, driving past the road boundary
    sees nothing but black conditioning frames, which the world model
    has no idea how to extend. Output is deterministic via the seeded
    RNG so the scene reproduces across runs.
    """
    if len(x_m) == 0:
        return []
    rng = np.random.default_rng(seed)
    poles: list[list[dict[str, float]]] = []
    next_x = period_m
    while next_x < float(x_m[-1]):
        # Randomise the forward position a little so the off-road poles
        # don't form a perfectly periodic grid alongside the periodic
        # streetlamp poles -- looks more natural.
        adjusted_x = next_x + float(rng.uniform(-forward_jitter_m, forward_jitter_m))
        i = int(np.searchsorted(x_m, adjusted_x))
        if i >= len(x_m):
            break
        cx = float(x_m[i])
        cy = float(y_m[i])
        cyaw = float(yaw_rad[i])
        # One pole per side with INDEPENDENT random lateral offsets so the
        # two sides don't read as a parallel two-line corridor.
        for side in (1.0, -1.0):
            lateral = float(rng.uniform(lateral_min_m, lateral_max_m))
            height = float(rng.uniform(height_min_m, height_max_m))
            dx = -math.sin(cyaw) * lateral * side
            dy = math.cos(cyaw) * lateral * side
            base = _point_xyz(cx + dx, cy + dy, 0.0)
            top = _point_xyz(cx + dx, cy + dy, height)
            poles.append([base, top])
        next_x += period_m
    return poles


def _periodic_traffic_signs(
    x_m: np.ndarray,
    y_m: np.ndarray,
    yaw_rad: np.ndarray,
    *,
    period_m: float = _TRAFFIC_SIGN_PERIOD_M,
    lateral_offset_m: float = _TRAFFIC_SIGN_LATERAL_M,
    height_m: float = _TRAFFIC_SIGN_HEIGHT_M,
) -> list[dict[str, object]]:
    """Emit traffic-sign records on alternating shoulders along the trajectory.

    Each entry has the same shape as ``_traffic_sign_rows`` but is
    anchored to the trajectory at ``period_m`` intervals so the model
    sees signage continuing alongside the road instead of just at the
    start. The plate's thinnest dimension is along x (length=0.12) so
    ``_build_cuboid_plate_faces`` in the loader picks the world +x axis
    as the plate normal at yaw=0 -- the periodic signs face the local
    centerline tangent.
    """
    if len(x_m) == 0:
        return []
    rows: list[dict[str, object]] = []
    next_x = period_m
    side_idx = 0
    while next_x < float(x_m[-1]):
        i = int(np.searchsorted(x_m, next_x))
        if i >= len(x_m):
            break
        side = 1.0 if side_idx % 2 == 0 else -1.0
        cx = float(x_m[i])
        cy = float(y_m[i])
        cyaw = float(yaw_rad[i])
        dx = -math.sin(cyaw) * lateral_offset_m * side
        dy = math.cos(cyaw) * lateral_offset_m * side
        rows.append(
            {
                "key": _make_key_record(f"traffic_sign_periodic_{side_idx:04d}"),
                "traffic_sign": {
                    "center": _point_xyz(cx + dx, cy + dy, height_m),
                    "dimensions": _point_xyz(0.12, 1.0, 1.2),
                    "orientation": _orientation_from_yaw(cyaw),
                    "category": "speed_limit",
                    "egomotion_label_class_id": "ego",
                },
                "version": 1,
            }
        )
        next_x += period_m
        side_idx += 1
    return rows


def _periodic_parked_cars(
    x_m: np.ndarray,
    y_m: np.ndarray,
    yaw_rad: np.ndarray,
    *,
    period_m: float = _PARKED_CAR_PERIOD_M,
    lateral_offset_m: float = _PARKED_CAR_LATERAL_M,
) -> list[dict[str, object]]:
    """Emit parked-car obstacle definitions on alternating shoulders along
    the trajectory.

    Each entry has the same shape as ``_OBSTACLE_TRACKS`` so the existing
    ``_obstacle_rows`` plumbing can append them. The car's yaw is set to
    the centerline yaw so cars look parked parallel to the road, not
    randomly oriented.
    """
    if len(x_m) == 0:
        return []
    tracks: list[dict[str, object]] = []
    next_x = period_m
    side_idx = 0
    while next_x < float(x_m[-1]):
        # Find the trajectory frame closest to ``next_x``.
        i = int(np.searchsorted(x_m, next_x))
        if i >= len(x_m):
            break
        side = 1.0 if side_idx % 2 == 0 else -1.0
        cx = float(x_m[i])
        cy = float(y_m[i])
        cyaw = float(yaw_rad[i])
        dx = -math.sin(cyaw) * lateral_offset_m * side
        dy = math.cos(cyaw) * lateral_offset_m * side
        tracks.append(
            {
                "trackline_id": f"parked-car-{side_idx:04d}",
                "category": "Automobile",
                "center": (cx + dx, cy + dy, 0.8),
                "size": (4.7, 2.0, 1.6),
                "yaw_rad": cyaw,
            }
        )
        next_x += period_m
        side_idx += 1
    return tracks


def _obstacle_rows(
    timestamps_us: np.ndarray, extra_tracks: tuple[dict[str, object], ...] = ()
) -> list[dict[str, object]]:
    """Emit two stationary samples per track at the first and last trajectory
    timestamps so ``WorldVehicleBBoxTrack.interpolate_at_timestamp`` resolves at
    every render frame without needing extrapolation."""
    sample_frames = (0, int(len(timestamps_us)) - 1)
    rows: list[dict[str, object]] = []
    for track in (*_OBSTACLE_TRACKS, *extra_tracks):
        center_xyz = track["center"]
        size_xyz = track["size"]
        assert isinstance(center_xyz, tuple) and isinstance(size_xyz, tuple)
        track_yaw = float(track.get("yaw_rad", 0.0))  # type: ignore[arg-type]
        for sample_idx, frame_idx in enumerate(sample_frames):
            rows.append(
                {
                    "key": {
                        "clip_id": _SCENE_ID,
                        "timestamp_micros": int(timestamps_us[frame_idx]),
                        "label_class_id": f"{track['trackline_id']}_s{sample_idx}",
                    },
                    "obstacle": {
                        "trackline_id": track["trackline_id"],
                        "center": _point_xyz(*center_xyz),
                        "size": _point_xyz(*size_xyz),
                        "orientation": _orientation_from_yaw(track_yaw),
                        "category": track["category"],
                    },
                    "version": 1,
                }
            )
    return rows


def _calibration_row() -> list[dict[str, object]]:
    # Pinned to the production `camera:front:wide:120fov` calibration extracted
    # from the clipgt sample scene so synthetic-scene renders exercise the same
    # ftheta polynomial and mounting pose as real CI data.
    rig = {
        "rig": {
            "properties": {},
            "vehicle": {},
            "vehicleio": {},
            "sensors": [
                {
                    "name": _CAMERA_CLIPGT_NAME,
                    "protocol": "camera.virtual",
                    "parameter": "video=synthetic/camera_front_wide_120fov.mp4",
                    "nominalSensor2Rig_FLU": {
                        "roll-pitch-yaw": [
                            0.292217969894409,
                            0.464194804430008,
                            -0.191304489970207,
                        ],
                        "t": [
                            1.69035196304321,
                            0.00553808081895113,
                            1.45306670665741,
                        ],
                    },
                    "correction_sensor_R_FLU": {
                        "roll-pitch-yaw": [
                            -0.1592078059911728,
                            0.11539523303508759,
                            0.5026581287384033,
                        ],
                    },
                    "correction_rig_T": [
                        -0.057110343128442764,
                        -0.0032010308932513,
                        0.008508340455591679,
                    ],
                    "properties": {
                        "width": "3848",
                        "height": "2168",
                        "cx": "1921.318705874846",
                        "cy": "1076.978854184438",
                        "Model": "ftheta",
                        "polynomial-type": "pixeldistance-to-angle",
                        "polynomial": (
                            "0 0.0005385247479413695 -1.598462177407655e-09 "
                            "6.250864794463573e-12 -2.194585699335322e-15 "
                            "4.525222700710391e-19"
                        ),
                        "linear-c": "1.000000",
                        "linear-d": "0.000000",
                        "linear-e": "0.000000",
                    },
                }
            ],
        }
    }
    return [
        {
            "key": {
                "clip_id": _SCENE_ID,
                "timestamp_micros": int(_START_TIMESTAMP_US),
                "label_class_id": "calibration",
            },
            "calibration_estimate": {"name": "default", "rig_json": json.dumps(rig)},
            "version": 1,
        }
    ]


def _initial_rgb() -> np.ndarray:
    x_gradient = np.linspace(0.0, 1.0, _IMAGE_WIDTH, dtype=np.float32)
    y_gradient = np.linspace(0.0, 1.0, _IMAGE_HEIGHT, dtype=np.float32)[:, None]
    red = np.clip(
        50.0
        + 140.0 * np.broadcast_to(x_gradient[None, :], (_IMAGE_HEIGHT, _IMAGE_WIDTH)),
        0.0,
        255.0,
    )
    green = np.clip(
        90.0 + 90.0 * np.broadcast_to(y_gradient, (_IMAGE_HEIGHT, _IMAGE_WIDTH)),
        0.0,
        255.0,
    )
    blue = np.clip(120.0 + 60.0 * (1.0 - x_gradient[None, :] * y_gradient), 0.0, 255.0)
    stacked = np.stack([red, green, blue], axis=2).astype(np.uint8)
    return stacked


def _first_image_variant(base: np.ndarray, shift_px: int) -> np.ndarray:
    shifted = np.roll(base, shift=shift_px, axis=1)
    return shifted.astype(np.uint8)


def _normalise_rgb(rgb: np.ndarray) -> np.ndarray:
    """Coerce a caller-supplied initial frame to a ``(H, W, 3) uint8`` array.

    The scene loader will resize to ``RasterConfig.resolution_wh`` at load
    time, so we don't enforce a specific resolution here - we only require
    that the dtype/shape are something Pillow can ``Image.fromarray(...)``
    on without raising.
    """
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"initial_rgb must be (H, W, 3); got shape {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _write_parquet_entry(
    zf: zipfile.ZipFile, name: str, rows: list[dict[str, object]]
) -> None:
    table = pa.Table.from_pylist(rows)
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    zf.writestr(name, buffer.getvalue())


def _write_png_entry(zf: zipfile.ZipFile, name: str, rgb: np.ndarray) -> None:
    buffer = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buffer, format="PNG")
    zf.writestr(name, buffer.getvalue())


def _metadata_doc(num_frames: int = _DEFAULT_TRAJECTORY_FRAMES) -> dict[str, object]:
    return {
        "scene_id": _SCENE_ID,
        "dataset_hash": "synthetic-dataset-hash",
        "is_resumable": False,
        "sensors": {"camera_ids": [_CAMERA_LOGICAL_NAME], "lidar_ids": []},
        "time_range": {
            "start": int(_START_TIMESTAMP_US),
            "end": int(
                _START_TIMESTAMP_US + (int(num_frames) - 1) * (1_000_000 // _FPS)
            ),
        },
        "version_string": "synthetic-1.0.0",
    }


def _rig_trajectory_doc(
    poses: list[list[list[float]]], timestamps_us: list[int]
) -> dict[str, object]:
    return {
        "rig_trajectories": [
            {"T_rig_worlds": poses, "T_rig_world_timestamps_us": timestamps_us}
        ]
    }


def _synthetic_ground_mesh_ply() -> bytes:
    """Build a flat ground-plane PLY at z=0 covering the synthetic route.

    Two triangles forming a 200m x 80m quad — generously larger than the
    ~60m forward / +-3m lateral trajectory so any rolled or zoomed-out test
    still hits the mesh. Real scenes ship a much denser ``mesh_ground.ply``;
    this fixture only needs to exercise the load + snap code path.
    """
    vertices = np.array(
        [
            [-100.0, -40.0, 0.0],
            [100.0, -40.0, 0.0],
            [100.0, 40.0, 0.0],
            [-100.0, 40.0, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return save_mesh_vf(vertices, faces)


def build_synthetic_scene_usdz(
    path: Path,
    *,
    initial_rgb: np.ndarray | None = None,
    prompt: str | None = None,
    length_frames: int = _DEFAULT_TRAJECTORY_FRAMES,
) -> Path:
    """Build a procedural USDZ that the scene loader can ingest unchanged.

    The geometry (trajectory, lane lines, road boundary, intersection,
    crosswalk, poles, signs, lights, obstacles) is fixed and deterministic.
    Three optional overrides exist for the runtime "synthetic-scene" mode
    where we want a real-looking demo without shipping any HD-map data:

    Args:
        path: Destination USDZ file.
        initial_rgb: ``(H, W, 3)`` ``uint8`` RGB image to embed as
            ``first_image.png``. Defaults to a debug colour gradient that's
            fine for tests but visually unhelpful for an actual demo. The
            scene loader resizes this to ``RasterConfig.resolution_wh`` at
            load time, so the shape doesn't need to match exactly.
        prompt: Default text prompt embedded as ``prompt.txt``. Defaults to a
            generic forward-driving description.
        length_frames: How many trajectory frames the synthetic road carries.
            Lane lines, road boundaries, and obstacle tracks are all spec'd
            along this trajectory, so larger values produce more drivable
            road. Default 180 (~6 s, 60 m at the default 10 m/s) keeps the
            test fixture small; runtime callers typically pass 18 000
            (~10 minutes, ~6 km) so a demo never runs out of road. The
            single intersection / crosswalk / road-island stay anchored at
            their original near-start coordinates regardless of length.
    """
    if length_frames < 2:
        raise ValueError(
            f"length_frames must be >= 2 (got {length_frames}); the trajectory "
            "needs at least two samples for the loader to compute initial speed."
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    x_m, y_m, yaw_rad, timestamps_us = _trajectory_arrays(num_frames=length_frames)
    poses = [
        rig_pose_from_state(float(px), float(py), 0.0, float(pyaw)).tolist()
        for px, py, pyaw in zip(x_m, y_m, yaw_rad, strict=True)
    ]

    center_x, center_y, center_yaw = _resample_centerline(x_m, y_m, yaw_rad, stride=3)
    road_boundary_chunk_size = max(2, _LANE_LINE_CHUNK_FRAMES // 3)
    road_left_chunks = _chunk_polyline(
        _offset_polyline(
            center_x, center_y, center_yaw, lateral_offset_m=_ROAD_BOUNDARY_OFFSET_M
        ),
        chunk_size=road_boundary_chunk_size,
    )
    road_right_chunks = _chunk_polyline(
        _offset_polyline(
            center_x, center_y, center_yaw, lateral_offset_m=-_ROAD_BOUNDARY_OFFSET_M
        ),
        chunk_size=road_boundary_chunk_size,
    )
    # Stop/wait line across the travel lanes just before the crosswalk.
    wait_line = [_point_xyz(17.0, -3.0, 0.0), _point_xyz(17.0, 3.0, 0.0)]
    # Periodic streetlamp-style poles every ~50 m on both shoulders. Each
    # pole is a short vertical line segment whose base sits on the
    # centerline-rotated +Y axis at the configured offset, so they follow
    # the wavy road instead of a straight world-frame line. The
    # near-start poles at x=20 m stay where they were so the existing
    # demo fixture still has the original "intersection-approach" pair.
    pole_polylines: list[list[dict[str, float]]] = [
        [_point_xyz(20.0, _POLE_OFFSET_M, 0.0), _point_xyz(20.0, _POLE_OFFSET_M, 5.0)],
        [
            _point_xyz(20.0, -_POLE_OFFSET_M, 0.0),
            _point_xyz(20.0, -_POLE_OFFSET_M, 5.0),
        ],
    ]
    pole_polylines.extend(_periodic_poles(x_m, y_m, yaw_rad))
    # Off-road clutter ("trees / telephone poles") scattered up to 100 m
    # laterally so the HDMap stays non-empty even when the ego strays
    # past the road boundary. The world model needs *some* visible
    # structure to condition on; black HDMap frames produce drift.
    pole_polylines.extend(_off_road_poles(x_m, y_m, yaw_rad))
    # Periodic parked cars on alternating shoulders so the sides of the
    # road don't read as empty over multi-km drives.
    extra_obstacles = tuple(_periodic_parked_cars(x_m, y_m, yaw_rad))

    base_rgb = _initial_rgb() if initial_rgb is None else _normalise_rgb(initial_rgb)
    variant1_rgb = _first_image_variant(base_rgb, shift_px=16)
    variant2_rgb = _first_image_variant(base_rgb, shift_px=48)

    default_prompt = (
        "Synthetic default prompt for loader testing." if prompt is None else prompt
    )
    variant1_prompt = "Synthetic prompt variant 1." if prompt is None else prompt
    variant2_prompt = "Synthetic prompt variant 2." if prompt is None else prompt

    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(
            "metadata.yaml",
            yaml.safe_dump(_metadata_doc(num_frames=length_frames), sort_keys=True),
        )
        zf.writestr(
            "rig_trajectories.json",
            json.dumps(_rig_trajectory_doc(poses, timestamps_us.tolist())),
        )
        zf.writestr("mesh_ground.ply", _synthetic_ground_mesh_ply())
        # Match the naming convention used by the real HF scenes (and by
        # ``first_image_<N>.png`` above): underscore-separated variant
        # numbers. Without the underscore, ``_discover_prompts`` rejects
        # the file and the variant selector silently falls back to default.
        zf.writestr("prompt.txt", default_prompt)
        zf.writestr("prompt_1.txt", variant1_prompt)
        zf.writestr("prompt_2.txt", variant2_prompt)

        _write_png_entry(zf, "first_image.png", base_rgb)
        _write_png_entry(zf, "first_image_1.png", variant1_rgb)
        _write_png_entry(zf, "first_image_2.png", variant2_rgb)

        _write_parquet_entry(
            zf, "clipgt/calibration_estimate.parquet", _calibration_row()
        )
        _write_parquet_entry(
            zf, "clipgt/lane_line.parquet", _lane_line_rows(x_m, y_m, yaw_rad)
        )
        _write_parquet_entry(
            zf,
            "clipgt/road_boundary.parquet",
            _polyline_rows(
                "road_boundary",
                "road_boundary",
                "location",
                [*road_left_chunks, *road_right_chunks],
            ),
        )
        _write_parquet_entry(
            zf,
            "clipgt/wait_line.parquet",
            _polyline_rows("wait_line", "wait_line", "location", [wait_line]),
        )
        _write_parquet_entry(
            zf,
            "clipgt/pole.parquet",
            _polyline_rows("pole", "pole", "location", pole_polylines),
        )
        _write_parquet_entry(
            zf,
            "clipgt/traffic_sign.parquet",
            [*_traffic_sign_rows(), *_periodic_traffic_signs(x_m, y_m, yaw_rad)],
        )
        _write_parquet_entry(zf, "clipgt/traffic_light.parquet", _traffic_light_rows())
        _write_parquet_entry(
            zf,
            "clipgt/crosswalk.parquet",
            _polygon_rows(
                "crosswalk",
                "crosswalk",
                "location",
                [
                    _rectangle_polygon(
                        center_x_m=15.0,
                        center_y_m=0.0,
                        half_width_m=1.5,
                        half_height_m=3.0,
                    )
                ],
            ),
        )
        _write_parquet_entry(
            zf,
            "clipgt/road_marking.parquet",
            _polygon_rows(
                "road_marking",
                "road_marking",
                "location",
                [
                    _rectangle_polygon(
                        center_x_m=22.0,
                        center_y_m=-1.8,
                        half_width_m=1.5,
                        half_height_m=0.25,
                    )
                ],
            ),
        )
        _write_parquet_entry(
            zf,
            "clipgt/intersection_area.parquet",
            _polygon_rows(
                "intersection_area",
                "intersection_area",
                "location",
                [
                    _rectangle_polygon(
                        center_x_m=38.0,
                        center_y_m=0.0,
                        half_width_m=7.5,
                        half_height_m=6.0,
                    )
                ],
            ),
        )
        _write_parquet_entry(
            zf,
            "clipgt/road_island.parquet",
            _polygon_rows(
                "road_island",
                "road_island",
                "location",
                [
                    _rectangle_polygon(
                        center_x_m=33.0,
                        center_y_m=4.0,
                        half_width_m=1.5,
                        half_height_m=0.8,
                    )
                ],
            ),
        )
        _write_parquet_entry(
            zf,
            "clipgt/obstacle.parquet",
            _obstacle_rows(timestamps_us, extra_tracks=extra_obstacles),
        )

    return path
