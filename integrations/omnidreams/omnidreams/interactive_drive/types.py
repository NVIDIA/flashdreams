# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]
UInt8Array = npt.NDArray[np.uint8]
Int32Array = npt.NDArray[np.int32]


def _normalized_quaternion_xyzw(quaternion_xyzw: FloatArray) -> FloatArray:
    norm = float(np.linalg.norm(quaternion_xyzw))
    if norm <= 1e-8:
        raise ValueError("Quaternion must have non-zero norm")
    return (quaternion_xyzw / norm).astype(np.float32)


def _slerp_quaternion_xyzw(
    q0_xyzw: FloatArray, q1_xyzw: FloatArray, alpha: float
) -> FloatArray:
    q0 = _normalized_quaternion_xyzw(q0_xyzw)
    q1 = _normalized_quaternion_xyzw(q1_xyzw)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(1.0, max(-1.0, dot))

    if dot > 0.9995:
        mixed = q0 + np.float32(alpha) * (q1 - q0)
        return _normalized_quaternion_xyzw(mixed.astype(np.float32))

    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * alpha
    sin_theta = math.sin(theta)
    s0 = math.cos(theta) - dot * sin_theta / max(sin_theta_0, 1e-8)
    s1 = sin_theta / max(sin_theta_0, 1e-8)
    return (np.float32(s0) * q0 + np.float32(s1) * q1).astype(np.float32)


@dataclass(frozen=True)
class CameraCalibration:
    clipgt_name: str
    logical_name: str
    width: int
    height: int
    cx: float
    cy: float
    polynomial: FloatArray
    is_backward_polynomial: bool
    linear_cde: FloatArray
    sensor_to_rig_flu: FloatArray


@dataclass(frozen=True)
class WorldLineSegments:
    segments_world: FloatArray
    color_rgba: tuple[float, float, float, float]
    width_px: float
    layer_name: str


@dataclass(frozen=True)
class WorldTriangleList:
    triangles_world: FloatArray
    color_rgba: tuple[float, float, float, float]
    layer_name: str


@dataclass(frozen=True)
class WorldPolygonList:
    polygons_world: tuple[FloatArray, ...]
    color_rgba: tuple[float, float, float, float]
    layer_name: str


@dataclass(frozen=True)
class WorldVehicleBBoxTrack:
    track_id: str
    object_type: str
    timestamps_us: npt.NDArray[np.int64]
    centers_world: FloatArray
    dimensions_lwh: FloatArray
    orientations_xyzw: FloatArray
    max_extrapolation_us: float

    def interpolate_at_timestamp(
        self, timestamp_us: int
    ) -> tuple[FloatArray, FloatArray, FloatArray] | None:
        if len(self.timestamps_us) < 2:
            return None
        first_timestamp_us = int(self.timestamps_us[0])
        last_timestamp_us = int(self.timestamps_us[-1])
        if timestamp_us < first_timestamp_us:
            if float(first_timestamp_us - timestamp_us) > self.max_extrapolation_us:
                return None
            left_index = 0
            right_index = 1
        elif timestamp_us > last_timestamp_us:
            if float(timestamp_us - last_timestamp_us) > self.max_extrapolation_us:
                return None
            right_index = len(self.timestamps_us) - 1
            left_index = right_index - 1
        else:
            right_index = int(
                np.searchsorted(self.timestamps_us, np.int64(timestamp_us), side="left")
            )
            if right_index == 0:
                right_index = 1
            if right_index >= len(self.timestamps_us):
                right_index = len(self.timestamps_us) - 1
            left_index = right_index - 1

        t0 = int(self.timestamps_us[left_index])
        t1 = int(self.timestamps_us[right_index])
        alpha = 0.0 if t1 == t0 else float(timestamp_us - t0) / float(t1 - t0)

        center = (1.0 - alpha) * self.centers_world[
            left_index
        ] + alpha * self.centers_world[right_index]
        dims = (1.0 - alpha) * self.dimensions_lwh[
            left_index
        ] + alpha * self.dimensions_lwh[right_index]
        orientation = _slerp_quaternion_xyzw(
            self.orientations_xyzw[left_index],
            self.orientations_xyzw[right_index],
            alpha,
        )
        return center.astype(np.float32), dims.astype(np.float32), orientation


@dataclass(frozen=True)
class SceneBundle:
    scene_path: Path
    scene_id: str
    metadata: dict[str, Any]
    selected_camera: CameraCalibration
    initial_rig_to_world: FloatArray
    initial_timestamp_us: int
    initial_yaw_rad: float
    initial_speed_mps: float
    initial_rgb: UInt8Array
    prompt: str
    line_layers: tuple[WorldLineSegments, ...]
    triangle_layers: tuple[WorldTriangleList, ...]
    polygon_layers: tuple[WorldPolygonList, ...] = ()
    vehicle_bbox_tracks: tuple[WorldVehicleBBoxTrack, ...] = ()
    # Ground-plane mesh from ``mesh_ground.ply`` in the USDZ. Used by
    # :class:`omnidreams.interactive_drive.physics.GroundSnapper` to keep the ego on the
    # ground after each kinematic integration step. ``None`` when the USDZ
    # ships no ground mesh (e.g. legacy fixtures); ground-snap then no-ops.
    ground_mesh_vertices: FloatArray | None = None
    ground_mesh_faces: Int32Array | None = None


@dataclass(frozen=True)
class DriverCommand:
    throttle: float = 0.0
    brake: float = 0.0
    steer: float = 0.0
    stop: bool = False
    reverse: bool = False
    steer_is_direct: bool = False
    manual_control: bool = False


@dataclass(frozen=True)
class DriveControls:
    throttle: float = 0.0
    brake: float = 0.0
    steer: float = 0.0


@dataclass
class VehicleState:
    x_m: float
    y_m: float
    z_m: float
    yaw_rad: float
    speed_mps: float
    steer_rad: float
    pitch_rad: float = 0.0
    roll_rad: float = 0.0


@dataclass(frozen=True)
class TrajectoryChunk:
    timestamps_us: npt.NDArray[np.int64]
    rig_poses_world: FloatArray
    boundary_state_after_chunk: VehicleState


@dataclass
class PresentedFrame:
    timestamp_us: int
    rgb_host_uint8: Any
    depth_host_f32: FloatArray | None
    rgb_native: Any | None = None
    depth_native: Any | None = None
    model_rgb_host_uint8: Any | None = None
    # Top-down BEV map rendered from the same scene with a synthetic camera
    # 25m above the rig (configured by :class:`BevConfig`). Carried alongside
    # the main camera frame so the demo HUD can show a minimap panel without
    # needing a second rasterizer instance. ``None`` when BEV rendering is
    # disabled, or when the world-model backend's first chunk replays the
    # debug HDMap override (BEV is not in that override set).
    bev_host_uint8: Any | None = None
    status_message: str | None = None


@dataclass(frozen=True)
class FrameChunk:
    frames: tuple[PresentedFrame, ...]
    boundary_state_after_chunk: VehicleState
    source_name: str


@dataclass(frozen=True)
class RasterChunk:
    frames: tuple[PresentedFrame, ...]


@dataclass
class ControlSnapshot:
    pressed: set[str] = field(default_factory=set)
    view_mode: str = "rgb"
