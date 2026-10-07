# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for simulation and conditioning components."""

from __future__ import annotations

from dataclasses import dataclass, field
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

    width: int = field(
        default=1280,
        metadata={
            "description": "Main raster width in pixels; must be positive.",
        },
    )
    height: int = field(
        default=704,
        metadata={
            "description": "Main raster height in pixels; must be positive.",
        },
    )
    compute_device: ComputeDeviceName = field(
        default="cuda",
        metadata={
            "description": "Device used for raster computation.",
            "user_setting": False,
        },
    )
    sync_gpu_timing: bool = field(
        default=False,
        metadata={
            "description": "Synchronizes GPU work for timing measurements.",
            "user_setting": False,
        },
    )
    perf_log_interval_frames: int = field(
        default=20,
        metadata={
            "description": "Interval for raster performance logs.",
            "user_setting": False,
        },
    )
    near_plane_m: float = field(
        default=0.1,
        metadata={
            "description": "Nearest camera clipping distance; must be less than Far Plane M.",
            "user_setting": False,
        },
    )
    far_plane_m: float = field(
        default=200.0,
        metadata={
            "description": "Farthest camera clipping distance; must exceed Near Plane M.",
            "user_setting": False,
        },
    )
    fog_start_m: float = field(
        default=40.0,
        metadata={
            "description": "Distance where fog begins; must be less than Fog End M.",
            "user_setting": False,
        },
    )
    fog_end_m: float = field(
        default=140.0,
        metadata={
            "description": "Distance where fog reaches full strength; must exceed Fog Start M.",
            "user_setting": False,
        },
    )
    fog_power: float = field(
        default=1.5,
        metadata={
            "description": "Exponent controlling the fog transition curve.",
            "user_setting": False,
        },
    )
    triangle_raytrace_distance_m: float = field(
        default=25.0,
        metadata={
            "description": "Maximum distance for triangle ray tracing.",
            "user_setting": False,
        },
    )
    triangle_raytrace_edge_samples: int = field(
        default=8,
        metadata={
            "description": "Number of edge samples for triangle ray tracing.",
            "user_setting": False,
        },
    )
    lane_segment_interval_m: float = field(
        default=0.05,
        metadata={
            "description": "Segment spacing when rasterizing lanes.",
        },
    )
    polyline_segment_interval_m: float = field(
        default=0.8,
        metadata={
            "description": "Segment spacing when rasterizing other polylines.",
        },
    )
    line_width_px: float = field(
        default=12.0,
        metadata={
            "description": "Rendered road-line width in pixels.",
        },
    )
    pole_width_px: float = field(
        default=5.0,
        metadata={
            "description": "Rendered pole width in pixels.",
        },
    )
    dual_line_offset_m: float = field(
        default=0.10,
        metadata={
            "description": "Separation of paired road lines.",
        },
    )
    depth_clear_m: float = field(
        default=1.0e6,
        metadata={
            "description": "Initial depth-buffer distance.",
            "user_setting": False,
        },
    )

    @property
    def resolution_wh(self) -> tuple[int, int]:
        """Return width and height in image-library order."""
        return self.width, self.height


@dataclass(frozen=True, slots=True)
class BevConfig:
    """Top-down semantic view used by the taxi HUD."""

    enabled: bool = field(
        default=True,
        metadata={"description": "Displays the top-down view."},
    )
    width: int = field(
        default=1024,
        metadata={
            "description": "Top-down image width in pixels; must be positive.",
        },
    )
    height: int = field(
        default=1024,
        metadata={
            "description": "Top-down image height in pixels; must be positive.",
        },
    )
    height_m: float = field(
        default=75.0,
        metadata={
            "description": "Camera height above the scene; must be positive.",
        },
    )
    fov_deg: float = field(
        default=60.0,
        metadata={
            "description": "Top-down camera field of view in degrees.",
        },
    )
    tilt_deg: float = field(
        default=0.0,
        metadata={
            "description": "Top-down camera tilt in degrees.",
        },
    )


@dataclass(frozen=True, slots=True)
class VehicleConfig:
    """Generic vehicle and rigid-body tuning."""

    wheel_base_m: float = field(
        default=2.8,
        metadata={
            "description": "Distance between front and rear axles in the vehicle model.",
        },
    )
    max_steer_rad: float = 0.5
    steer_rate_rad_per_s: float = 0.55
    steer_return_rate_rad_per_s: float = 0.9
    speed_limit_enabled: bool = True
    max_speed_mps: float = field(
        default=31.2928,
        metadata={
            "description": "Normal forward speed cap.",
        },
    )
    max_reverse_speed_mps: float = field(
        default=6.0,
        metadata={
            "description": "Reverse speed cap.",
        },
    )
    max_accel_mps2: float = 3.5
    max_brake_mps2: float = 6.0
    max_lateral_accel_mps2: float = 6.2
    drag_mps2: float = field(
        default=0.7,
        metadata={"description": "Base speed loss from drag."},
    )
    mass_kg: float = field(
        default=1_550.0,
        metadata={
            "description": "Vehicle mass used by the physical simulation.",
        },
    )
    tire_grip: float = field(
        default=1.35,
        metadata={"description": "Tire traction factor."},
    )
    rolling_resistance: float = field(
        default=0.015,
        metadata={
            "description": "Resistance from rolling contact.",
        },
    )
    aero_drag_coefficient: float = field(
        default=0.42,
        metadata={
            "description": "Aerodynamic drag factor.",
        },
    )
    collision_restitution: float = field(
        default=0.22,
        metadata={
            "description": "General collision bounce.",
        },
    )
    collision_friction: float = field(
        default=0.65,
        metadata={
            "description": "Friction during collisions.",
        },
    )
    max_collision_yaw_rate_radps: float = field(
        default=0.35,
        metadata={
            "description": "Cap on rotation caused by collisions.",
        },
    )
    suspension_stiffness: float = field(
        default=42.0,
        metadata={
            "description": "Suspension spring strength.",
        },
    )
    suspension_damping: float = field(
        default=9.0,
        metadata={
            "description": "How quickly suspension motion settles.",
        },
    )
    suspension_travel_m: float = field(
        default=0.22,
        metadata={
            "description": "Maximum suspension movement.",
        },
    )
    suspension_visual_gain: float = field(
        default=0.15,
        metadata={
            "description": "Amount of visible suspension response.",
        },
    )
    max_body_roll_rad: float = 0.5
    max_body_pitch_rad: float = field(
        default=0.5,
        metadata={
            "description": "Limit on forward/back body tilt.",
        },
    )
    actor_collision_enabled: bool = True
    static_collision_enabled: bool = True
    aabb_length_m: float = field(
        default=4.8,
        metadata={
            "description": "Length of the taxi's axis-aligned collision box.",
        },
    )
    aabb_width_m: float = field(
        default=2.0,
        metadata={
            "description": "Width of the taxi's axis-aligned collision box.",
        },
    )
    aabb_height_m: float = field(
        default=1.6,
        metadata={
            "description": "Height of the taxi's axis-aligned collision box.",
        },
    )
