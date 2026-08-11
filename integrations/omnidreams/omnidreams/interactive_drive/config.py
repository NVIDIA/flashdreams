# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from flashdreams.infra.postprocess import VideoPostprocessChainConfig

BackendName = Literal["raster", "omnidreams"]
ViewMode = Literal["rgb", "model_rgb", "physx"]
ComputeDeviceName = Literal["automatic", "cuda", "vulkan"]


@dataclass(frozen=True)
class ChunkConfig:
    fps: int = 30
    initial_chunk_frames: int = 5
    chunk_frames: int = 8

    @property
    def frame_interval_s(self) -> float:
        return 1.0 / float(self.fps)

    @property
    def frame_interval_us(self) -> int:
        return round(1_000_000 / float(self.fps))


@dataclass(frozen=True)
class RasterConfig:
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
        return (self.width, self.height)


@dataclass(frozen=True)
class VehicleConfig:
    wheel_base_m: float = 2.8
    max_steer_rad: float = 0.5
    steer_rate_rad_per_s: float = 0.55
    steer_return_rate_rad_per_s: float = 0.9
    speed_limit_enabled: bool = True
    # 70 mph, expressed in the simulation's SI units.
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
    # Bound impact-induced camera rotation.  Normal steering keeps its full
    # response; this only filters single-frame PhysX yaw impulses that would
    # turn the conditioning view away from the struck actor.
    # This prevents the cache from forgetting whom you hit.
    max_collision_yaw_rate_radps: float = 0.35
    suspension_stiffness: float = 42.0
    suspension_damping: float = 9.0
    suspension_travel_m: float = 0.22
    suspension_visual_gain: float = 0.15
    max_body_roll_rad: float = 0.5
    max_body_pitch_rad: float = 0.5
    actor_collision_enabled: bool = True
    static_collision_enabled: bool = True
    # Ego AABB used by :class:`omnidreams.interactive_drive.simulation.ground_snap.GroundSnapper` to decide
    # which area of the ground mesh to query when snapping z + pitch + roll.
    # Defaults match a typical sedan; the alpasim test data uses
    # 5.393 x 2.109 x 1.503 m.
    aabb_length_m: float = 4.8
    aabb_width_m: float = 2.0
    aabb_height_m: float = 1.6


@dataclass(frozen=True)
class WorldModelProfileConfig:
    enabled: bool = False


@dataclass(frozen=True)
class BevConfig:
    """Straight-down HD-map view rendered for the HUD mini-map."""

    enabled: bool = True
    # 1024x1024 = ~2x SSAA at the HUD's ~470x400 BEV panel; dominant lever
    # for BEV quality (under-sampling bakes in unrecoverable aliasing).
    width: int = 1024
    height: int = 1024
    # 75 m altitude + 60° vertical FOV covers ~87 m of ground. With the
    # straight-down camera this is an orthographic-style view of the HD-map
    # plane, centered on the ego rig.
    height_m: float = 75.0
    fov_deg: float = 60.0
    # Keep the mini-map straight down. Non-zero values remain available for
    # explicit CLI overrides, but they no longer define the default HUD view.
    tilt_deg: float = 0.0


@dataclass(frozen=True)
class AppConfig:
    scene_path: Path
    backend: BackendName = "raster"
    game_mode: bool = False
    camera_name: str = "camera_front_wide_120fov"
    variant: str = "default"
    prompt_override: str | None = None
    manifest_path: Path | None = None
    chunk: ChunkConfig = ChunkConfig()
    raster: RasterConfig = RasterConfig()
    vehicle: VehicleConfig = VehicleConfig()
    world_model_profile: WorldModelProfileConfig = WorldModelProfileConfig()
    world_model_offload_text_encoder: bool = False
    postprocess: VideoPostprocessChainConfig = field(
        default_factory=VideoPostprocessChainConfig
    )
    bev: BevConfig = BevConfig()
    # OOB thresholds plumbed to LoopConfig (overridable via CLI --oob-*).
    # Match alpasim's driver-side proximity: warn > 0.6, respawn >= 2.0
    # against the AABB-distance proximity.
    oob_warn_proximity: float = 0.6
    oob_respawn_proximity: float = 2.0
    oob_respawn_debounce_chunks: int = 1
    # OOB AABB geometry: oob_margin_m (50 m, matching alpasim) expands the
    # scene's spatial-content AABB before any in-bounds check;
    # oob_warning_zone_m is the depth of the linear warning ramp inside it.
    oob_margin_m: float = 50.0
    oob_warning_zone_m: float = 100.0
    # When set ("HOST:PORT" or bare ":PORT"), swap the Vulkan presenter for
    # the MJPEG streaming presenter (HTTP frames + keyboard) -- needed on
    # compute-only boxes with no Vulkan-capable GPU.
    stream_mjpeg_bind: str | None = None
    # When set, the main loop exits cleanly after that many distinct
    # chunk indices have been consumed off the present queue. Used by the
    # internal LAG upload helper to produce deterministic, warmup-aware
    # trace runs across machines instead of timing the run with a
    # wall-clock sleep.
    stop_after_consumed_chunks: int | None = None
    # Substring matched against the Vulkan adapter name to force the
    # presenter onto a specific GPU (e.g. "RTX PRO"); None lets SlangPy pick
    # the first enumerated adapter.
    presenter_adapter: str | None = None
    # None follows game_mode; an explicit bool remains a fine-grained override.
    visual_flare_enabled: bool | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "vehicle",
            replace(
                self.vehicle,
                speed_limit_enabled=self.game_mode,
                actor_collision_enabled=self.game_mode,
                static_collision_enabled=self.game_mode,
            ),
        )
        if self.visual_flare_enabled is None:
            object.__setattr__(self, "visual_flare_enabled", self.game_mode)
