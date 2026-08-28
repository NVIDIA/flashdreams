# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for simulation and conditioning components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ComputeDeviceName = Literal["automatic", "cuda", "vulkan"]


@dataclass(frozen=True, slots=True)
class ChunkConfig:
    """Frame cadence used by low-level trajectory helpers."""

    fps: int = 30
    initial_chunk_frames: int = 5
    chunk_frames: int = 8

    @property
    def frame_interval_s(self) -> float:
        return 1.0 / self.fps

    @property
    def frame_interval_us(self) -> int:
        return round(1_000_000 / self.fps)


@dataclass(frozen=True, slots=True)
class RasterConfig:
    """Main-camera semantic raster settings."""

    width: int = 1280
    height: int = 704
    compute_device: ComputeDeviceName = "cuda"
    sync_gpu_timing: bool = False
    perf_log_interval_frames: int = 20
    near_plane_m: float = 0.1
    far_plane_m: float = 200.0
    fog_start_m: float = 40.0
    fog_end_m: float = 140.0
    fog_power: float = 1.5
    triangle_raytrace_distance_m: float = 25.0
    triangle_raytrace_edge_samples: int = 8
    lane_segment_interval_m: float = 0.05
    polyline_segment_interval_m: float = 0.8
    line_width_px: float = 12.0
    pole_width_px: float = 5.0
    dual_line_offset_m: float = 0.10
    depth_clear_m: float = 1.0e6

    @property
    def resolution_wh(self) -> tuple[int, int]:
        """Return width and height in image-library order."""
        return self.width, self.height


@dataclass(frozen=True, slots=True)
class BevConfig:
    """Top-down semantic view used by the taxi HUD."""

    enabled: bool = True
    width: int = 1024
    height: int = 1024
    height_m: float = 75.0
    fov_deg: float = 60.0
    tilt_deg: float = 0.0


@dataclass(frozen=True, slots=True)
class VehicleConfig:
    """Generic vehicle and rigid-body tuning."""

    wheel_base_m: float = 2.8
    max_steer_rad: float = 0.5
    steer_rate_rad_per_s: float = 0.55
    steer_return_rate_rad_per_s: float = 0.9
    speed_limit_enabled: bool = True
    max_speed_mps: float = 31.2928
    max_reverse_speed_mps: float = 6.0
    max_accel_mps2: float = 3.5
    max_brake_mps2: float = 6.0
    max_lateral_accel_mps2: float = 6.2
    drag_mps2: float = 0.7
    mass_kg: float = 1_550.0
    tire_grip: float = 1.35
    rolling_resistance: float = 0.015
    aero_drag_coefficient: float = 0.42
    collision_restitution: float = 0.22
    collision_friction: float = 0.65
    max_collision_yaw_rate_radps: float = 0.35
    suspension_stiffness: float = 42.0
    suspension_damping: float = 9.0
    suspension_travel_m: float = 0.22
    suspension_visual_gain: float = 0.15
    max_body_roll_rad: float = 0.5
    max_body_pitch_rad: float = 0.5
    actor_collision_enabled: bool = True
    static_collision_enabled: bool = True
    aabb_length_m: float = 4.8
    aabb_width_m: float = 2.0
    aabb_height_m: float = 1.6
