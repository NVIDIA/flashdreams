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
    """Main raster width in pixels; must be positive."""

    height: int = 704
    """Main raster height in pixels; must be positive."""

    compute_device: ComputeDeviceName = "cuda"
    """Device used for raster computation."""

    sync_gpu_timing: bool = False
    """Synchronizes GPU work for timing measurements."""

    perf_log_interval_frames: int = 20
    """Interval for raster performance logs."""

    near_plane_m: float = 0.1
    """Nearest camera clipping distance; must be less than Far Plane M."""

    far_plane_m: float = 200.0
    """Farthest camera clipping distance; must exceed Near Plane M."""

    fog_start_m: float = 40.0
    """Distance where fog begins; must be less than Fog End M."""

    fog_end_m: float = 140.0
    """Distance where fog reaches full strength; must exceed Fog Start M."""

    fog_power: float = 1.5
    """Exponent controlling the fog transition curve."""

    triangle_raytrace_distance_m: float = 25.0
    """Maximum distance for triangle ray tracing."""

    triangle_raytrace_edge_samples: int = 8
    """Number of edge samples for triangle ray tracing."""

    lane_segment_interval_m: float = 0.05
    """Segment spacing when rasterizing lanes."""

    polyline_segment_interval_m: float = 0.8
    """Segment spacing when rasterizing other polylines."""

    line_width_px: float = 12.0
    """Rendered road-line width in pixels."""

    pole_width_px: float = 5.0
    """Rendered pole width in pixels."""

    dual_line_offset_m: float = 0.10
    """Separation of paired road lines."""

    depth_clear_m: float = 1.0e6
    """Initial depth-buffer distance."""

    @property
    def resolution_wh(self) -> tuple[int, int]:
        """Return width and height in image-library order."""
        return self.width, self.height


@dataclass(frozen=True, slots=True)
class BevConfig:
    """Top-down semantic view used by the taxi HUD."""

    enabled: bool = True
    """Displays the top-down view."""

    width: int = 1024
    """Top-down image width in pixels; must be positive."""

    height: int = 1024
    """Top-down image height in pixels; must be positive."""

    height_m: float = 75.0
    """Camera height above the scene; must be positive."""

    fov_deg: float = 60.0
    """Top-down camera field of view in degrees."""

    tilt_deg: float = 0.0
    """Top-down camera tilt in degrees."""


@dataclass(frozen=True, slots=True)
class VehicleConfig:
    """Generic vehicle and rigid-body tuning."""

    wheel_base_m: float = 2.8
    """Distance between front and rear axles in the vehicle model."""

    max_steer_rad: float = 0.5
    steer_rate_rad_per_s: float = 0.55
    steer_return_rate_rad_per_s: float = 0.9
    speed_limit_enabled: bool = True
    max_speed_mps: float = 31.2928
    """Normal forward speed cap."""

    max_reverse_speed_mps: float = 6.0
    """Reverse speed cap."""

    max_accel_mps2: float = 3.5
    max_brake_mps2: float = 6.0
    max_lateral_accel_mps2: float = 6.2
    drag_mps2: float = 0.7
    """Base speed loss from drag."""

    mass_kg: float = 1_550.0
    """Vehicle mass used by the physical simulation."""

    tire_grip: float = 1.35
    """Tire traction factor."""

    rolling_resistance: float = 0.015
    """Resistance from rolling contact."""

    aero_drag_coefficient: float = 0.42
    """Aerodynamic drag factor."""

    collision_restitution: float = 0.22
    """General collision bounce."""

    collision_friction: float = 0.65
    """Friction during collisions."""

    max_collision_yaw_rate_radps: float = 0.35
    """Cap on rotation caused by collisions."""

    suspension_stiffness: float = 42.0
    """Suspension spring strength."""

    suspension_damping: float = 9.0
    """How quickly suspension motion settles."""

    suspension_travel_m: float = 0.22
    """Maximum suspension movement."""

    suspension_visual_gain: float = 0.15
    """Amount of visible suspension response."""

    max_body_roll_rad: float = 0.5
    max_body_pitch_rad: float = 0.5
    """Limit on forward/back body tilt."""

    actor_collision_enabled: bool = True
    static_collision_enabled: bool = True
    aabb_length_m: float = 4.8
    """Length of the taxi's axis-aligned collision box."""

    aabb_width_m: float = 2.0
    """Width of the taxi's axis-aligned collision box."""

    aabb_height_m: float = 1.6
    """Height of the taxi's axis-aligned collision box."""
