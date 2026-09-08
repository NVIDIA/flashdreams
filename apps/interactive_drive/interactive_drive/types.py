# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from flashdreams.serving.realtime.timing import VideoModelTimings

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
    # Ground-plane mesh from ``mesh_ground.ply``; used by
    # :class:`~interactive_drive.simulation.ground_snap.GroundSnapper`
    # to keep the ego on the ground. ``None`` when the USDZ ships no ground
    # mesh, in which case ground-snap no-ops.
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
    velocity_x_mps: float | None = None
    velocity_y_mps: float | None = None
    yaw_rate_radps: float = 0.0
    suspension_pitch_rad: float = 0.0
    suspension_roll_rad: float = 0.0
    suspension_pitch_rate_radps: float = 0.0
    suspension_roll_rate_radps: float = 0.0
    ragdoll_active: bool = False


@dataclass(frozen=True)
class DynamicActorTrajectory:
    """Per-frame rigid-body track consumed by game engines and Ludus."""

    entity_id: str
    """Stable scene entity identifier."""

    object_type: str
    """Semantic actor category used for mass and rendering style."""

    timestamps_us: npt.NDArray[np.int64]
    """Frame timestamps shared with the containing ego trajectory."""

    translations_world: FloatArray
    """Actor centers in world coordinates with shape ``[frames, 3]``."""

    orientations_xyzw: FloatArray
    """World orientations as normalized quaternions with shape ``[frames, 4]``."""

    dimensions_lwh: FloatArray
    """Full box extents in metres."""

    detached_from_track: bool = False
    """Whether rigid-body physics, rather than the recorded track, owns the actor."""

    is_simulated: bool = False
    """Whether this chunk carries mutable PhysX samples for the actor."""

    def to_game_engine_dict(self) -> dict[str, Any]:
        """Return JSON-compatible identity, collider, and transform keyframes."""
        keyframes = [
            {
                "timestamp_us": int(timestamp_us),
                "transform": {
                    "position_m": translation.tolist(),
                    "orientation_xyzw": orientation.tolist(),
                },
            }
            for timestamp_us, translation, orientation in zip(
                self.timestamps_us,
                self.translations_world,
                self.orientations_xyzw,
                strict=True,
            )
        ]
        return {
            "entity_id": self.entity_id,
            "object_type": self.object_type,
            "components": {
                "box_collider": {
                    "half_extents_m": (self.dimensions_lwh * 0.5).tolist()
                },
                "trajectory": {
                    "detached_from_track": self.detached_from_track,
                    "keyframes": keyframes,
                },
            },
        }


@dataclass(frozen=True)
class PhysicsDebugFrame:
    """Collider topology visible to PhysX for one simulated frame."""

    ego_position_m: FloatArray
    """World-space ego chassis center."""

    ego_orientation_xyzw: FloatArray
    """World-space ego chassis orientation."""

    ego_dimensions_lwh: FloatArray
    """Full ego collider dimensions in metres."""

    actor_positions_m: FloatArray
    """Active actor centers with shape ``[actors, 3]``."""

    actor_orientations_xyzw: FloatArray
    """Active actor orientations with shape ``[actors, 4]``."""

    actor_dimensions_lwh: FloatArray
    """Active actor collider dimensions with shape ``[actors, 3]``."""

    barrier_segments_xy_m: FloatArray
    """Active invisible-wall segments with shape ``[barriers, 2, 2]``."""

    barrier_thicknesses_m: FloatArray
    """Active invisible-wall thicknesses with shape ``[barriers]``."""

    barrier_heights_m: FloatArray
    """Active invisible-wall heights with shape ``[barriers]``."""

    actor_ids: tuple[str, ...] = ()
    """Stable actor identities corresponding to the actor arrays."""

    barrier_ids: tuple[str, ...] = ()
    """Stable wall identities corresponding to the barrier arrays."""


@dataclass(frozen=True)
class PhysXChunkTimings:
    """Auditable timing split for all physics frames in one generated chunk."""

    total_ms: float = 0.0
    synchronize_ms: float = 0.0
    actor_update_ms: float = 0.0
    solver_ms: float = 0.0
    readback_ms: float = 0.0
    bridge_ms: float = 0.0
    step_count: int = 0
    max_visible_actors: int = 0
    max_detached_actors: int = 0


@dataclass(frozen=True)
class TrajectoryChunk:
    timestamps_us: npt.NDArray[np.int64]
    rig_poses_world: FloatArray
    boundary_state_after_chunk: VehicleState
    dynamic_actors: tuple[DynamicActorTrajectory, ...] = ()
    physics_debug_frames: tuple[PhysicsDebugFrame, ...] = ()
    """Per-frame active collider snapshots for the optional debug view."""

    actor_collision_detected: bool = False
    """Whether the ego struck another scene actor anywhere in this chunk."""

    actor_collision_frame_index: int | None = None
    """First frame in this chunk whose physics step reported actor contact."""

    physx_elapsed_s: float | None = None
    """Wall time spent in PhysX synchronization and stepping for this chunk."""

    physx_timings: PhysXChunkTimings | None = None
    """Detailed synchronization, native update, solve, and readback timings."""


@dataclass
class PresentedFrame:
    timestamp_us: int
    rgb_host_uint8: Any
    depth_host_f32: FloatArray | None
    rgb_native: Any | None = None
    depth_native: Any | None = None
    model_rgb_host_uint8: Any | None = None
    # Top-down BEV minimap rendered with a synthetic overhead camera (see
    # :class:`BevConfig`). ``None`` when BEV is disabled or the world-model
    # first chunk replays the debug HDMap override.
    bev_host_uint8: Any | None = None
    physx_debug: PhysicsDebugFrame | None = None
    """Collider snapshot rendered lazily when view ``3`` is active."""

    physx_rgb_host_uint8: Any | None = None
    """Lazy Ludus CUDA debug raster, materialized only by host presenters."""

    status_message: str | None = None


@dataclass(frozen=True)
class FrameChunk:
    frames: tuple[PresentedFrame, ...]
    boundary_state_after_chunk: VehicleState
    source_name: str
    video_model_timings: VideoModelTimings | None = None


@dataclass(frozen=True)
class RasterChunk:
    frames: tuple[PresentedFrame, ...]


@dataclass
class ControlSnapshot:
    pressed: set[str] = field(default_factory=set)
    view_mode: str = "rgb"
