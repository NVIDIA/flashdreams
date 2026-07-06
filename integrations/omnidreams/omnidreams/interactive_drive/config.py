# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

BackendName = Literal["raster", "omnidreams"]
ViewMode = Literal["rgb", "model_rgb"]
ComputeDeviceName = Literal["automatic", "cuda", "vulkan"]
LudusBackendName = Literal["cuda", "vulkan"]


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
        return int(round(1_000_000 / float(self.fps)))


@dataclass(frozen=True)
class RasterConfig:
    width: int = 1280
    height: int = 704
    compute_device: ComputeDeviceName = "cuda"
    ludus_backend: LudusBackendName = "cuda"
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
    steer_rate_rad_per_s: float = 0.4
    steer_return_rate_rad_per_s: float = 0.7
    max_speed_mps: float = 18.0
    max_reverse_speed_mps: float = 6.0
    max_accel_mps2: float = 3.5
    max_brake_mps2: float = 6.0
    drag_mps2: float = 0.7
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
    """Synthetic top-down BEV camera rendered alongside the main view, published as a separate stream for the HUD minimap."""

    enabled: bool = True
    # 1024x1024 = ~2x SSAA at the HUD's ~470x400 BEV panel; dominant lever
    # for BEV quality (under-sampling bakes in unrecoverable aliasing).
    width: int = 1024
    height: int = 1024
    # 75 m altitude + 60° vertical FOV covers ~87 m of ground (with the
    # forward tilt, ~90 m ahead / ~10 m behind the rig): a navigation zoom.
    height_m: float = 75.0
    fov_deg: float = 60.0
    # Forward pitch: 0 is pure top-down, positive tilts forward for a
    # Google-Maps feel. 28 stays just under the fov_deg/2 = 30 ceiling above
    # which the image bottom would cross the horizon.
    tilt_deg: float = 28.0


@dataclass(frozen=True)
class AppConfig:
    scene_path: Path
    backend: BackendName = "raster"
    camera_name: str = "camera_front_wide_120fov"
    variant: str = "default"
    prompt_override: str | None = None
    manifest_path: Path | None = None
    chunk: ChunkConfig = ChunkConfig()
    raster: RasterConfig = RasterConfig()
    vehicle: VehicleConfig = VehicleConfig()
    world_model_profile: WorldModelProfileConfig = WorldModelProfileConfig()
    world_model_offload_text_encoder: bool = False
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
