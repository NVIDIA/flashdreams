# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Ludus-based HD map rasterizer.

This module provides the LudusConditionRasterizer class, which wraps the
ludus_renderer library to render HD map scenes for conditioning video generation.

Based on imaginaire4's projects/cosmos/sil/world_scenario/ludus_renderer.py
(commit c2071960fb81 from dev/grpc-updates branch), adapted to work with
interactive_drive' USDZ scene bundles. gRPC dynamic object override functionality
removed as we run in-process.

When :class:`BevConfig` is enabled the rasterizer also renders a top-down
bird's-eye-view via a synthetic ``FThetaCamera`` mounted above the rig. The
math is the same as the imaginaire ``ludus_renderer.render_utils`` BEV
helpers: a pinhole-projection camera + a fixed sensor-to-rig matrix that
points the optical axis straight down. The BEV bytes ride alongside the
main RGB on each :class:`PresentedFrame`.
"""

import concurrent.futures
import contextlib
import math
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from ludus_renderer import (
    FThetaCamera,
    load_clipgt_scene,
)
from ludus_renderer.clipgt import ClipgtGpuScene
from ludus_renderer.render_utils import SceneAdapter
from ludus_renderer.torch import LudusCudaTimestampedContext
from ludus_renderer.torch.ops import CAMERA_TYPE_BEV, CAMERA_TYPE_REGULAR
from torch import Tensor

from interactive_drive.config import BevConfig, RasterConfig
from interactive_drive.types import PresentedFrame, RasterChunk, SceneBundle

_BEV_CAMERA_NAME = "interactive_drive_bev"


def _extract_clipgt_from_usdz(usdz_path: Path, dest_dir: Path) -> Path:
    """Extract clipgt parquet files from USDZ archive.

    Our USDZ bundles contain clipgt/ subdirectory with parquet files.
    This extracts them to a directory compatible with load_clipgt_scene.

    Returns:
        Path to the clipgt directory with extracted parquets.
    """
    clipgt_dir = dest_dir / "clipgt"
    clipgt_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(usdz_path, "r") as zf:
        for name in zf.namelist():
            if not name.startswith("clipgt/") or name.endswith("/"):
                continue
            relative = Path(name).relative_to("clipgt")
            if relative.suffix in {".parquet", ".json"}:
                target_name = f"clipgt.{relative.name}"
            else:
                target_name = relative.name
            (clipgt_dir / target_name).write_bytes(zf.read(name))

    return clipgt_dir


@dataclass
class _LoadedSceneData:
    """Container for loaded scene data, analogous to imaginaire's SceneData."""

    clipgt_scene: ClipgtGpuScene
    scene_adapter: SceneAdapter


class _LudusConditionRasterizerImpl:
    """Single-threaded implementation backing :class:`LudusConditionRasterizer`.

    This is the actual rasterizer; do not construct it directly. The public
    :class:`LudusConditionRasterizer` thread-pins this implementation to a
    dedicated worker because NVIDIA EGL on Blackwell + driver 595.58.03 cannot
    migrate a headless surfaceless GL context across threads.
    """

    def __init__(self, raster: RasterConfig, bev: BevConfig | None = None) -> None:
        """Initialize the rasterizer.

        Args:
            raster: Raster configuration specifying resolution and rendering params.
            bev: Optional BEV configuration. When ``enabled``, the rasterizer
                appends a synthetic top-down camera to the scene's camera list
                on :meth:`load_scene` and ``render_chunk`` populates
                :attr:`PresentedFrame.bev_host_uint8`.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for LudusConditionRasterizer.")

        self._raster = raster
        self._bev = bev
        self._device = torch.device("cuda:0")

        self.ctx = LudusCudaTimestampedContext(device=self._device)
        self.ctx.set_depth_scaling(True)
        self.ctx.set_msaa_samples(4)
        self.ctx.set_max_tessellation_levels(cube=0)
        # Use thinner BEV linework so the small map panel doesn't get
        # swallowed by 12-pixel polylines designed for the 1280x704 main view.
        bev_line_width = max(2.0, float(raster.line_width_px) * 0.4)
        bev_pole_width = max(2.0, float(raster.pole_width_px) * 0.6)
        self.ctx.set_line_widths(
            polyline_regular=float(raster.line_width_px),
            polyline_bev=bev_line_width,
            ego_traj_regular=float(raster.pole_width_px),
            ego_traj_bev=bev_pole_width,
            wireframe=4.0,
        )

        self._scene_data: _LoadedSceneData | None = None
        self._scene_id: int | None = None
        self._all_cameras: list[FThetaCamera] = []
        self._all_camera_map: dict[str, int] = {}
        self._sensor_to_rig: dict[str, Tensor] = {}
        self._selected_camera_name: str | None = None
        self._bev_camera_id: int | None = None
        self._bev_sensor_to_rig: Tensor | None = None
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None

    def _to_ludus_camera_pose(self, camera_poses: Tensor) -> Tensor:
        """Convert sensor-to-world camera poses to Ludus' world-to-sensor format."""
        return torch.linalg.inv(camera_poses)

    def load_scene(self, scene: SceneBundle) -> None:
        """Load a scene from the USDZ bundle.

        Args:
            scene: Scene bundle containing path to USDZ and camera selection.
        """
        self.ctx.clear_scenes()

        if self._temp_dir is not None:
            self._temp_dir.cleanup()
        self._temp_dir = tempfile.TemporaryDirectory()

        clipgt_dir = _extract_clipgt_from_usdz(scene.scene_path, Path(self._temp_dir.name))

        clipgt_scene = load_clipgt_scene(
            clipgt_dir,
            device=self._device,
            target_resolution=(self._raster.width, self._raster.height),
            include_ego_trajectory=False,
            include_ego_obstacle=False,
        )

        scene_adapter = SceneAdapter(clipgt_scene)
        self._scene_data = _LoadedSceneData(
            clipgt_scene=clipgt_scene,
            scene_adapter=scene_adapter,
        )

        # Copy the scene's camera list so we can append our synthetic BEV
        # camera without mutating ``clipgt_scene.cameras`` (the loader returns
        # a shared list and downstream consumers expect stable indices).
        self._all_cameras = list(clipgt_scene.cameras)
        self._all_camera_map = dict(clipgt_scene.camera_name_to_id)
        self._sensor_to_rig = dict(clipgt_scene.sensor_to_rig)
        self._selected_camera_name = scene.selected_camera.clipgt_name

        if self._bev is not None and self._bev.enabled:
            bev_camera = _build_bev_camera(self._bev, self._device)
            self._bev_camera_id = len(self._all_cameras)
            self._all_cameras.append(bev_camera)
            self._all_camera_map[_BEV_CAMERA_NAME] = self._bev_camera_id
            self._bev_sensor_to_rig = _bev_sensor_to_rig(
                height_m=self._bev.height_m,
                tilt_deg=self._bev.tilt_deg,
                device=self._device,
            )
            self._sensor_to_rig[_BEV_CAMERA_NAME] = self._bev_sensor_to_rig
        else:
            self._bev_camera_id = None
            self._bev_sensor_to_rig = None

        self.ctx.upload_cameras(self._all_cameras)

        # Single scene upload shared by both the main camera and the BEV
        # minimap. The earlier BEV-specific upload existed solely to
        # clear ``CUBE_FLAG_WIREFRAME`` from cube pools so obstacle
        # outlines wouldn't read as halos on the GoogleMaps-filtered
        # minimap; with the BEV now rendered at 1024x1024 + LANCZOS
        # downsample the outlines anti-alias cleanly and look like the
        # roadway / vehicle borders Google Maps actually draws, so the
        # divergence is no longer worth the extra GPU upload + the
        # second scene-id bookkeeping.
        self._scene_id = self.ctx.upload_scene(clipgt_scene.timestamped_scene)

    def render_chunk(
        self,
        rig_poses_world: npt.NDArray[np.float32],
        timestamps_us: npt.NDArray[np.int64],
    ) -> RasterChunk:
        """Render a chunk of frames from the scene's selected camera.

        When BEV is enabled (see :class:`BevConfig`) the rasterizer also
        renders a top-down map for each frame and attaches it to
        :attr:`PresentedFrame.bev_host_uint8`.

        Args:
            rig_poses_world: Rig-to-world poses [num_frames, 4, 4].
            timestamps_us: Frame timestamps in microseconds [num_frames].

        Returns:
            RasterChunk containing rendered frames.
        """
        if self._scene_data is None or self._scene_id is None or self._selected_camera_name is None:
            raise RuntimeError("load_scene() must be called before render_chunk().")

        camera_name = self._selected_camera_name
        if camera_name not in self._all_camera_map:
            available = sorted(self._all_camera_map.keys())
            raise RuntimeError(f"Camera {camera_name!r} not found. Available: {available}")

        rig_poses_torch = torch.from_numpy(
            np.ascontiguousarray(rig_poses_world, dtype=np.float32)
        ).to(device=self._device)
        timestamps_batch = torch.from_numpy(np.ascontiguousarray(timestamps_us, dtype=np.int64)).to(
            device=self._device
        )

        rgb_numpy = self._render_one_camera(
            rig_poses=rig_poses_torch,
            timestamps_batch=timestamps_batch,
            scene_id=self._scene_id,
            camera_id=self._all_camera_map[camera_name],
            sensor_to_rig=self._sensor_to_rig[camera_name],
            camera_type=CAMERA_TYPE_REGULAR,
            resolution=(self._raster.height, self._raster.width),
        )

        bev_numpy: np.ndarray | None = None
        if (
            self._bev is not None
            and self._bev.enabled
            and self._bev_camera_id is not None
            and self._bev_sensor_to_rig is not None
        ):
            bev_numpy = self._render_one_camera(
                rig_poses=rig_poses_torch,
                timestamps_batch=timestamps_batch,
                scene_id=self._scene_id,
                camera_id=self._bev_camera_id,
                sensor_to_rig=self._bev_sensor_to_rig,
                camera_type=CAMERA_TYPE_BEV,
                resolution=(self._bev.height, self._bev.width),
            )

        frames = [
            PresentedFrame(
                timestamp_us=int(timestamps_us[idx]),
                rgb_host_uint8=np.ascontiguousarray(rgb_numpy[idx], dtype=np.uint8),
                depth_host_f32=None,
                bev_host_uint8=(
                    np.ascontiguousarray(bev_numpy[idx], dtype=np.uint8)
                    if bev_numpy is not None
                    else None
                ),
            )
            for idx in range(len(timestamps_us))
        ]
        return RasterChunk(frames=tuple(frames))

    def _render_one_camera(
        self,
        *,
        rig_poses: Tensor,
        timestamps_batch: Tensor,
        scene_id: int,
        camera_id: int,
        sensor_to_rig: Tensor,
        camera_type: int,
        resolution: tuple[int, int],
    ) -> np.ndarray:
        """Single-camera rasterizer dispatch, shared by the main view and BEV.

        Both code paths build identical camera/timestamp batches and only
        differ in the camera id, sensor-to-rig, camera-type id, and
        target resolution; the scene id is shared. Keeping them in one
        helper keeps the GPU bookkeeping (vflip, dtype-cast, host copy)
        consistent across paths.
        """
        n_frames = timestamps_batch.shape[0]
        camera_poses_world = torch.einsum("nij,jk->nik", rig_poses, sensor_to_rig.to(self._device))
        camera_poses_ludus = self._to_ludus_camera_pose(camera_poses_world)
        scene_id_batch = torch.full((n_frames,), scene_id, dtype=torch.int32, device=self._device)
        camera_id_batch = torch.full((n_frames,), camera_id, dtype=torch.int32, device=self._device)
        camera_type_id_batch = torch.full(
            (n_frames,), camera_type, dtype=torch.int32, device=self._device
        )

        height, width = resolution
        images = self.ctx.render(
            scene_id_batch,
            camera_id_batch,
            timestamps_batch,
            camera_type_id_batch,
            camera_poses_ludus,
            resolution=(height, width),
        )

        rgb = images[:, :, :, :3]
        if self.ctx.needs_vflip:
            rgb = rgb.flip(1)
        rendered_host = rgb.detach().cpu()
        if rendered_host.dtype != torch.uint8:
            rendered_host = (rendered_host.clamp(0.0, 1.0) * 255.0 + 0.5).to(torch.uint8)
        return rendered_host.numpy()

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None

    def __del__(self) -> None:
        self.cleanup()


class LudusConditionRasterizer:
    """Thread-pinned facade over :class:`_LudusConditionRasterizerImpl`.

    NVIDIA EGL on the Blackwell + driver 595.58.03 stack used for the demo
    does not allow a headless surfaceless GL context to migrate across
    threads (``eglMakeCurrent`` returns ``EGL_FALSE`` on any thread other
    than the one that called ``ludusTimestampedInit``). To stay
    single-threaded from EGL's point of view while still letting the
    surrounding demo run multi-threaded pipelines, every public entry
    point on this class runs synchronously on a single dedicated worker
    thread that owns the GL context for its lifetime.

    Behaves exactly like the underlying implementation; consumers should
    not need to care that work is dispatched to the worker.
    """

    def __init__(self, raster: RasterConfig, bev: BevConfig | None = None) -> None:
        self._exec = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ludus-render"
        )
        self._impl: _LudusConditionRasterizerImpl | None = self._exec.submit(
            _LudusConditionRasterizerImpl, raster, bev
        ).result()

    def load_scene(self, scene: SceneBundle) -> None:
        exec_, impl = self._require_alive()
        return exec_.submit(impl.load_scene, scene).result()

    def render_chunk(
        self,
        rig_poses_world: npt.NDArray[np.float32],
        timestamps_us: npt.NDArray[np.int64],
    ) -> "RasterChunk":
        exec_, impl = self._require_alive()
        return exec_.submit(impl.render_chunk, rig_poses_world, timestamps_us).result()

    def _require_alive(
        self,
    ) -> tuple[concurrent.futures.ThreadPoolExecutor, _LudusConditionRasterizerImpl]:
        exec_ = self._exec
        impl = self._impl
        assert exec_ is not None and impl is not None, "rasterizer has been cleaned up"
        return exec_, impl

    def cleanup(self) -> None:
        exec_ = getattr(self, "_exec", None)
        if exec_ is None:
            return
        impl = self._impl
        self._impl = None
        if impl is not None:
            with contextlib.suppress(Exception):
                exec_.submit(impl.cleanup).result()
        exec_.shutdown(wait=True)
        self._exec = None

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.cleanup()


def _build_bev_camera(bev: BevConfig, device: torch.device) -> FThetaCamera:
    """Construct a synthetic pinhole-as-FTheta camera for BEV rendering.

    Mirrors ``ludus_renderer.render_utils.create_bev_camera`` (in the
    omni-dreams-ludus reference), which reproduces a perfect pinhole
    projection by feeding the Taylor expansion of ``f * tan(theta)`` into
    the F-theta forward polynomial. ``height_m`` plus ``fov_deg`` together
    set how much ground (in metres) the BEV covers around the rig.
    """
    cx = float(bev.width) / 2.0
    cy = float(bev.height) / 2.0
    half_fov = math.radians(float(bev.fov_deg)) / 2.0
    focal = (float(bev.height) / 2.0) / math.tan(half_fov)
    diagonal = math.sqrt((float(bev.width) / 2.0) ** 2 + (float(bev.height) / 2.0) ** 2)
    max_ray_angle = math.atan(diagonal / focal)
    poly_coeffs = torch.tensor(
        [0.0, focal, 0.0, focal / 3.0, 0.0, 2.0 * focal / 15.0],
        device=device,
        dtype=torch.float32,
    )
    return FThetaCamera(
        principal_point=torch.tensor([cx, cy], device=device, dtype=torch.float32),
        image_size=torch.tensor(
            [float(bev.width), float(bev.height)], device=device, dtype=torch.float32
        ),
        fw_poly=poly_coeffs,
        max_ray_angle=max_ray_angle,
        depth_max=max(150.0, float(bev.height_m) * 4.0),
    )


def _bev_sensor_to_rig(*, height_m: float, tilt_deg: float, device: torch.device) -> Tensor:
    """Sensor-to-rig transform for a top-down (or forward-tilted) BEV camera.

    Sensor (FLU): X=forward (optical axis), Y=left, Z=up
    Rig (FLU):    X=forward, Y=left, Z=up

    With ``tilt_deg = 0`` the matrix is the AlpaSim straight-down BEV:
      Sensor X (depth)    -> Rig -Z (points down)
      Sensor Y (left)     -> Rig +Y (unchanged)
      Sensor Z (up image) -> Rig +X (forward)

    For ``tilt_deg > 0`` we apply an additional pitch around the rig's
    lateral (Y) axis so the optical axis leans forward by that angle.
    The result is a Google-Maps-navigation-style chase view: rig stays
    roughly centered horizontally, road ahead occupies most of the
    image, and the rig itself sits in the lower part of the frame.
    Camera position stays at ``(0, 0, height_m)`` -- only orientation
    changes -- so we don't have to retune ``height_m`` when adjusting
    tilt.
    """
    theta = math.radians(float(tilt_deg))
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    # Rotation columns express where the sensor axes land in rig FLU:
    #   col 0 (sensor X / optical axis) -> ( sin θ,  0, -cos θ)
    #   col 1 (sensor Y / image left)   -> (     0,  1,       0)
    #   col 2 (sensor Z / image up)     -> ( cos θ,  0,  sin θ)
    # At θ = 0 this collapses to the straight-down matrix above.
    return torch.tensor(
        [
            [sin_t, 0.0, cos_t, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-cos_t, 0.0, sin_t, float(height_m)],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )
