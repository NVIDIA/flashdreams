# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutable scene data and frame-aligned simulation values."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from torch import Tensor

from omnidreams_game_engine.game_map.types import ResolvedGameMap

FloatArray = npt.NDArray[np.float32]
UInt8Array = npt.NDArray[np.uint8]
Int32Array = npt.NDArray[np.int32]


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
class WorldLineSegments:
    segments_world: FloatArray
    color_rgba: tuple[float, float, float, float]
    width_px: float
    layer_name: str


@dataclass(frozen=True, slots=True)
class WorldTriangleList:
    triangles_world: FloatArray
    color_rgba: tuple[float, float, float, float]
    layer_name: str


@dataclass(frozen=True, slots=True)
class WorldPolygonList:
    polygons_world: tuple[FloatArray, ...]
    color_rgba: tuple[float, float, float, float]
    layer_name: str


@dataclass(frozen=True, slots=True)
class SceneDefinition:
    """Immutable scene data shared with one model-thread rollout."""

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
    ground_mesh_vertices: FloatArray | None = None
    ground_mesh_faces: Int32Array | None = None
    game_map: ResolvedGameMap | None = None


@dataclass(frozen=True, slots=True)
class DriverCommand:
    throttle: float = 0.0
    brake: float = 0.0
    steer: float = 0.0
    stop: bool = False
    handbrake: bool = False
    reverse: bool = False
    steer_is_direct: bool = False
    manual_control: bool = False


@dataclass(slots=True)
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


@dataclass(frozen=True, slots=True)
class DynamicActorTrajectory:
    entity_id: str
    object_type: str
    timestamps_us: npt.NDArray[np.int64]
    translations_world: FloatArray
    orientations_xyzw: FloatArray
    dimensions_lwh: FloatArray
    detached_from_track: bool = False
    is_simulated: bool = False

    def to_game_engine_dict(self) -> dict[str, Any]:
        """Return the identity, collider, and transform keyframes for Ludus."""
        keyframes = [
            {
                "timestamp_us": int(timestamp),
                "transform": {
                    "position_m": translation.tolist(),
                    "orientation_xyzw": orientation.tolist(),
                },
            }
            for timestamp, translation, orientation in zip(
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


@dataclass(frozen=True, slots=True)
class PhysicsDebugFrame:
    ego_position_m: FloatArray
    ego_orientation_xyzw: FloatArray
    ego_dimensions_lwh: FloatArray
    actor_positions_m: FloatArray
    actor_orientations_xyzw: FloatArray
    actor_dimensions_lwh: FloatArray
    barrier_segments_xy_m: FloatArray
    barrier_thicknesses_m: FloatArray
    barrier_heights_m: FloatArray
    actor_ids: tuple[str, ...] = ()
    barrier_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PhysXChunkTimings:
    total_ms: float = 0.0
    synchronize_ms: float = 0.0
    actor_update_ms: float = 0.0
    solver_ms: float = 0.0
    readback_ms: float = 0.0
    bridge_ms: float = 0.0
    traffic_prepare_ms: float = 0.0
    """Time spent preparing tracked traffic before native simulation."""

    barrier_rebound_ms: float = 0.0
    """Time spent detecting and reinforcing static-barrier contacts."""

    traffic_update_ms: float = 0.0
    """Time spent consuming native actor states and updating traffic controls."""

    state_materialize_ms: float = 0.0
    """Time spent publishing simulated ego and actor state to engine objects."""

    bridge_other_ms: float = 0.0
    """Remaining adapter time outside the named bridge stages."""

    step_count: int = 0
    max_visible_actors: int = 0
    max_detached_actors: int = 0


@dataclass(frozen=True, slots=True)
class TrajectoryChunk:
    timestamps_us: npt.NDArray[np.int64]
    rig_poses_world: FloatArray
    vehicle_states: tuple[VehicleState, ...]
    boundary_state_after_chunk: VehicleState
    applied_commands: tuple[DriverCommand, ...] = ()
    dynamic_actors: tuple[DynamicActorTrajectory, ...] = ()
    physics_debug_frames: tuple[PhysicsDebugFrame, ...] = ()
    actor_collision_detected: bool = False
    actor_collision_frame_index: int | None = None
    static_collision_detected: bool = False
    static_collision_frame_index: int | None = None
    physx_elapsed_s: float | None = None
    physx_timings: PhysXChunkTimings | None = None

    def __post_init__(self) -> None:
        frame_count = len(self.timestamps_us)
        if frame_count <= 0:
            raise ValueError("TrajectoryChunk requires at least one frame")
        if not self.applied_commands:
            object.__setattr__(
                self,
                "applied_commands",
                tuple(DriverCommand() for _ in range(frame_count)),
            )
        aligned = (
            self.rig_poses_world.shape == (frame_count, 4, 4)
            and len(self.vehicle_states) == frame_count
            and len(self.applied_commands) == frame_count
            and (
                not self.physics_debug_frames
                or len(self.physics_debug_frames) == frame_count
            )
        )
        if not aligned:
            raise ValueError("TrajectoryChunk fields must describe the same frames")


@dataclass(frozen=True, slots=True)
class ConditionBatch:
    """Model conditioning and optional HUD data for one engine step."""

    hdmap_bvtchw: Tensor
    """Semantic main-camera frames in ``[B,V,T,C,H,W]`` and ``[-1,1]``."""

    bev_tchw: Tensor | None = None
    """Optional uint8 top-down UI frames in ``[T,C,H,W]``."""
