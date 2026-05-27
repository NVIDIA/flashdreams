# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

BackendName = Literal["raster", "omnidreams"]
ViewMode = Literal["rgb", "model_rgb"]
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
        return int(round(1_000_000 / float(self.fps)))


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
    steer_rate_rad_per_s: float = 0.4
    steer_return_rate_rad_per_s: float = 0.7
    max_speed_mps: float = 18.0
    max_reverse_speed_mps: float = 6.0
    max_accel_mps2: float = 3.5
    max_brake_mps2: float = 6.0
    drag_mps2: float = 0.7
    # Ego AABB used by :class:`interactive_drive.physics.GroundSnapper` to decide
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
    """Synthetic top-down bird's-eye-view rendered alongside the main camera.

    Mirrors AlpaSim's ``video_model.return_bev_map`` configuration in
    ``alpasim_runtime`` (cfg paths in
    ``references/alpasim-human-driver/src/wizard/configs/cameras/bev.yaml``):
    a virtual pinhole camera rendered ``height_m`` metres above the rig
    looking straight down with ``fov_deg`` vertical field of view.

    The BEV stream is a tiny extra rasterizer dispatch and is published as
    a separate MJPEG endpoint so the demo HUD can show it as a small map
    panel under the steering / pedal controls.
    """

    enabled: bool = True
    # 1024x1024 gives ~2x SSAA per axis at the HUD's ~470x400 BEV panel,
    # which the LANCZOS cover-fit resize then bandlimits cleanly. This
    # is the dominant lever for BEV image quality — under-sampling here
    # bakes aliasing into the source frame that no downstream filter
    # can recover. The producer-side decode + GoogleMaps filter cost is
    # roughly 4x of 512x512 but it runs on the supervisor's stream
    # consumer thread, not the render thread, so it doesn't compete
    # with the main camera path. Drop this if you're rasterizer-bound
    # on the backend; quality degrades smoothly.
    width: int = 1024
    height: int = 1024
    # 75 m altitude with 60° vertical FOV covers roughly 87 m of ground.
    # Combined with the 20° forward tilt the top of the image looks ~90 m
    # ahead of the rig and the bottom shows ~10 m behind, which is a
    # comfortable navigation-style minimap zoom (AlpaSim's
    # ``return_bev_map`` defaults are ``height_m=40`` / ``fov_deg=50``,
    # ~37 m coverage, but their panel is much smaller than ours).
    height_m: float = 75.0
    fov_deg: float = 60.0
    # Forward pitch in degrees. ``0`` is pure top-down (AlpaSim's default
    # BEV); a positive value tilts the camera forward to give the
    # Google-Maps-navigation feel where ahead-of-rig fills more of the
    # image and the rig sits low in the frame. ``28`` puts us just
    # under the ``fov_deg / 2 = 30`` ceiling above which the bottom of
    # the image would cross the horizon (no rendered geometry above
    # the ground plane to fill it). The HUD's ego marker placement
    # follows this automatically via :func:`_bev_marker_y_rel`.
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
    # When non-None, the app swaps the Vulkan presenter out for
    # :class:`interactive_drive.streaming_presenter.MJPEGStreamingPresenter`
    # which serves frames to a browser over HTTP and reads keyboard
    # events back from it. Format: "HOST:PORT" (e.g. "0.0.0.0:8080"),
    # or bare ":PORT" to bind on all interfaces.
    stream_mjpeg_bind: str | None = None
    # Substring to match against the Vulkan adapter name for the presenter.
    # When None, SlangPy picks whichever Vulkan adapter it enumerates first.
    # When set (e.g. "RTX PRO"), we call ``spy.Device.enumerate_adapters``
    # and pass ``adapter_luid`` so Vulkan is forced onto that adapter even
    # when the default NVIDIA Vulkan ICD would pick a different GPU (e.g.
    # a compute-only GB300 that has no graphics queue).
    presenter_adapter: str | None = None
