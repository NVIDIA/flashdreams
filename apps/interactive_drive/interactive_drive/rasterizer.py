# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Ludus-based HD map rasterizer wrapping ``ludus_renderer`` for conditioning.

When :class:`BevConfig` is enabled it also renders an orthographic top-down BEV
above the rig. The public facade pipelines BEV one chunk behind RGB so
presentation almost never waits for a current-chunk BEV.
"""

import concurrent.futures
import contextlib
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from loguru import logger
from ludus_renderer import (
    CUBE_FLAG_WIREFRAME,
    PRIM_EGO_OBSTACLE,
    CubePool,
    FThetaCamera,
    LudusCudaTimestampedContext,
    MutableObjectSceneBuffer,
    OrthographicCamera,
    TimestampedScene,
)
from ludus_renderer import (
    load_scene as load_ludus_scene,
)
from ludus_renderer.clipgt import ClipgtGpuScene
from ludus_renderer.render_utils import SceneAdapter, create_bev_camera
from ludus_renderer.torch.ops import CAMERA_TYPE_BEV, CAMERA_TYPE_REGULAR
from torch import Tensor

from flashdreams.infra.acceleration.frame_prefetch import LazyCudaFrame
from interactive_drive.config import BevConfig, RasterConfig
from interactive_drive.cuda_env import DISABLE_CUDA_INTEROP_ENV, env_truthy
from interactive_drive.physx_debug import LudusPhysxDebugSceneBuffer
from interactive_drive.types import (
    DynamicActorTrajectory,
    PhysicsDebugFrame,
    PresentedFrame,
    RasterChunk,
    SceneBundle,
)

_BEV_CAMERA_NAME = "interactive_drive_bev"


@dataclass
class _LoadedSceneData:
    """Loaded clipgt scene + its adapter."""

    clipgt_scene: ClipgtGpuScene
    scene_adapter: SceneAdapter


@dataclass(frozen=True)
class _RenderedCameraFrames:
    frames_hwc_uint8: Tensor
    ready_event: object | None


class _LazyRasterFrame(LazyCudaFrame):
    """Expose a rendered HDMap frame as CUDA first, NumPy only on fallback."""

    def __init__(
        self,
        frames_hwc_uint8: Tensor,
        frame_index: int,
        *,
        source_event: object | None = None,
    ) -> None:
        super().__init__(
            frames_hwc_uint8,
            frame_index,
            source_event=source_event,
            lost_source_message="Lazy raster frame lost its source tensor before materialization.",
            already_materialized_message="Lazy raster frame was already materialized on the host.",
            synchronize_source_event_on_host_copy=True,
        )


class _LudusConditionRasterizerImpl:
    """Single-threaded implementation backing :class:`LudusConditionRasterizer`.

    Do not construct directly; the public facade thread-pins it to one worker
    (see :class:`LudusConditionRasterizer` for the EGL rationale).
    """

    def __init__(
        self,
        raster: RasterConfig,
        bev: BevConfig | None = None,
        *,
        ego_dimensions_lwh: tuple[float, float, float] | None = None,
        max_chunk_frames: int = 1,
    ) -> None:
        """Initialize the rasterizer.

        Args:
            raster: Raster configuration specifying resolution and rendering params.
            bev: Optional BEV configuration. When ``enabled``, the rasterizer
                appends a synthetic top-down camera to the scene's camera list
                on :meth:`load_scene` and ``render_chunk`` populates
                :attr:`PresentedFrame.bev_host_uint8`.
            ego_dimensions_lwh: Controlled car box dimensions used when
                ``bev.show_ego_car`` is enabled.
            max_chunk_frames: Maximum frame count stored in the reusable ego track.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for LudusConditionRasterizer.")
        if bev is not None and bev.show_ego_car and ego_dimensions_lwh is None:
            raise ValueError("show_ego_car requires ego_dimensions_lwh")
        if max_chunk_frames <= 0:
            raise ValueError("max_chunk_frames must be positive")

        self._raster = raster
        self._bev = bev
        self._ego_dimensions_lwh = ego_dimensions_lwh
        self._max_chunk_frames = int(max_chunk_frames)
        self._device = torch.device("cuda:0")
        self._use_cuda_frames = not env_truthy(DISABLE_CUDA_INTEROP_ENV)
        if self._use_cuda_frames:
            logger.info(
                "[rasterizer] cuda_backend=enabled; returning lazy CUDA raster frames",
            )
        else:
            logger.info(
                f"[rasterizer] cuda_backend=disabled by {DISABLE_CUDA_INTEROP_ENV}; "
                "using host raster frames",
            )

        logger.info("[rasterizer] ludus_backend=cuda")
        self.ctx = LudusCudaTimestampedContext(device=self._device)
        self.ctx.set_depth_scaling(True)
        self.ctx.set_msaa_samples(4)
        # Keep adaptive cube tessellation enabled. F-theta projection curves
        # box faces near the image boundary; forcing level zero turns each face
        # into two long screen-space triangles and visibly warps edge colliders.
        self.ctx.set_max_tessellation_levels(cube=3)
        # Use thinner BEV linework so the small map panel doesn't get
        # swallowed by the heavier polylines designed for the main view.
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
        self._dynamic_scene: MutableObjectSceneBuffer | None = None
        self._all_cameras: list[FThetaCamera | OrthographicCamera] = []
        self._all_camera_map: dict[str, int] = {}
        self._sensor_to_rig: dict[str, Tensor] = {}
        self._selected_camera_name: str | None = None
        self._bev_camera_id: int | None = None
        self._bev_sensor_to_rig: Tensor | None = None
        self._physx_debug_scene = LudusPhysxDebugSceneBuffer(
            self.ctx, device=self._device
        )

    def _to_ludus_camera_pose(self, camera_poses: Tensor) -> Tensor:
        """Convert sensor-to-world camera poses to Ludus' world-to-sensor format."""
        return torch.linalg.inv(camera_poses)

    def load_scene(self, scene: SceneBundle) -> None:
        """Load a scene from the USDZ bundle.

        Args:
            scene: Scene bundle containing path to USDZ and camera selection.
        """
        self.ctx.clear_scenes()
        self._physx_debug_scene.reset()
        self._dynamic_scene = None

        clipgt_scene = load_ludus_scene(
            scene.scene_path,
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
                device=self._device,
            )
            self._sensor_to_rig[_BEV_CAMERA_NAME] = self._bev_sensor_to_rig
        else:
            self._bev_camera_id = None
            self._bev_sensor_to_rig = None

        self.ctx.upload_cameras(self._all_cameras)

        base_scene = clipgt_scene.timestamped_scene
        if self._bev is not None and self._bev.show_ego_car:
            assert self._ego_dimensions_lwh is not None
            placeholder_poses = torch.eye(
                4, dtype=torch.float32, device=self._device
            ).repeat(self._max_chunk_frames, 1, 1)
            placeholder_poses[:, :3, 3] = torch.tensor(
                [1_000_000.0, 1_000_000.0, -1_000_000.0],
                dtype=torch.float32,
                device=self._device,
            )
            ego_pool = _build_bev_ego_car_pool(
                rig_poses=placeholder_poses,
                timestamps_us=torch.arange(
                    self._max_chunk_frames, dtype=torch.int64, device=self._device
                ),
                dimensions_lwh=self._ego_dimensions_lwh,
                track_capacity=self._max_chunk_frames,
            )
            base_scene = TimestampedScene(
                polyline_pools=base_scene.polyline_pools,
                polygon_pools=base_scene.polygon_pools,
                cube_pools=[
                    *(
                        pool
                        for pool in (base_scene.cube_pools or [])
                        if pool.prim_type_id != PRIM_EGO_OBSTACLE
                    ),
                    ego_pool,
                ],
            )

        # Single scene upload shared by regular cameras and the BEV camera.
        # Ludus suppresses ``PRIM_EGO_OBSTACLE`` for regular-camera renders.
        self._scene_id = self.ctx.upload_scene(base_scene)
        self._dynamic_scene = MutableObjectSceneBuffer(
            self.ctx,
            self._scene_id,
            base_scene,
            device=self._device,
        )

    def render_chunk(
        self,
        rig_poses_world: npt.NDArray[np.float32],
        timestamps_us: npt.NDArray[np.int64],
        dynamic_actors: tuple[DynamicActorTrajectory, ...] = (),
        physics_debug_frames: tuple[PhysicsDebugFrame, ...] = (),
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
        if dynamic_actors:
            self._replace_dynamic_actor_scene(dynamic_actors)

        if (
            self._scene_data is None
            or self._scene_id is None
            or self._selected_camera_name is None
        ):
            raise RuntimeError("load_scene() must be called before render_chunk().")

        camera_name = self._selected_camera_name
        if camera_name not in self._all_camera_map:
            available = sorted(self._all_camera_map.keys())
            raise RuntimeError(
                f"Camera {camera_name!r} not found. Available: {available}"
            )

        rig_poses_torch = torch.from_numpy(
            np.ascontiguousarray(rig_poses_world, dtype=np.float32)
        ).to(device=self._device)
        rgb_frames = self._render_one_camera(
            rig_poses=rig_poses_torch,
            timestamps_us=timestamps_us,
            scene_id=self._scene_id,
            camera_id=self._all_camera_map[camera_name],
            sensor_to_rig=self._sensor_to_rig[camera_name],
            camera_type=CAMERA_TYPE_REGULAR,
            resolution=(self._raster.height, self._raster.width),
        )

        bev_frames = self.render_bev_frames(
            rig_poses_torch=rig_poses_torch,
            timestamps_us=timestamps_us,
        )
        physx_frames = self.render_physx_debug_frames(
            rig_poses_torch=rig_poses_torch,
            timestamps_us=timestamps_us,
            physics_debug_frames=physics_debug_frames,
        )
        return self.build_chunk(
            timestamps_us=timestamps_us,
            rgb_frames=rgb_frames,
            bev_frames=bev_frames,
            physics_debug_frames=physics_debug_frames,
            physx_frames=physx_frames,
        )

    def _replace_dynamic_actor_scene(
        self, actors: tuple[DynamicActorTrajectory, ...]
    ) -> None:
        """Replace recorded obstacles with authoritative rigid-body tracks."""
        dynamic_scene = getattr(self, "_dynamic_scene", None)
        if dynamic_scene is None and hasattr(self, "_base_timestamped_scene"):
            dynamic_scene = MutableObjectSceneBuffer(
                self.ctx,
                self._scene_id,
                self._base_timestamped_scene,
                device=self._device,
            )
            self._dynamic_scene = dynamic_scene
        if dynamic_scene is None:
            raise RuntimeError("load_scene() must be called before actor replacement.")
        dynamic_scene.update(actors)
        self._scene_id = dynamic_scene.scene_id

    def render_rgb_frames(
        self,
        rig_poses_world: npt.NDArray[np.float32],
        timestamps_us: npt.NDArray[np.int64],
        dynamic_actors: tuple[DynamicActorTrajectory, ...] = (),
    ) -> tuple[npt.NDArray[np.int64], Tensor, _RenderedCameraFrames]:
        """Render only the main camera frames needed for model conditioning."""
        if dynamic_actors:
            self._replace_dynamic_actor_scene(dynamic_actors)

        if (
            self._scene_data is None
            or self._scene_id is None
            or self._selected_camera_name is None
        ):
            raise RuntimeError("load_scene() must be called before render_chunk().")

        camera_name = self._selected_camera_name
        if camera_name not in self._all_camera_map:
            available = sorted(self._all_camera_map.keys())
            raise RuntimeError(
                f"Camera {camera_name!r} not found. Available: {available}"
            )

        rig_poses_torch = torch.from_numpy(
            np.ascontiguousarray(rig_poses_world, dtype=np.float32)
        ).to(device=self._device)
        rgb_frames = self._render_one_camera(
            rig_poses=rig_poses_torch,
            timestamps_us=timestamps_us,
            scene_id=self._scene_id,
            camera_id=self._all_camera_map[camera_name],
            sensor_to_rig=self._sensor_to_rig[camera_name],
            camera_type=CAMERA_TYPE_REGULAR,
            resolution=(self._raster.height, self._raster.width),
        )
        return (
            np.asarray(timestamps_us, dtype=np.int64),
            rig_poses_torch,
            rgb_frames,
        )

    def render_bev_frames(
        self,
        *,
        rig_poses_torch: Tensor,
        timestamps_us: npt.ArrayLike,
    ) -> _RenderedCameraFrames | None:
        """Render BEV frames for an already-prepared pose batch."""
        if not (
            self._scene_id is not None
            and self._bev is not None
            and self._bev.enabled
            and self._bev_camera_id is not None
            and self._bev_sensor_to_rig is not None
        ):
            return None
        level_rig_poses = _level_rig_poses_for_bev(rig_poses_torch)
        if self._bev.show_ego_car:
            assert self._ego_dimensions_lwh is not None
            ego_pool = _build_bev_ego_car_pool(
                rig_poses=level_rig_poses,
                timestamps_us=timestamps_us,
                dimensions_lwh=self._ego_dimensions_lwh,
                track_capacity=self._max_chunk_frames,
            )
            if not self.ctx.update_cube_pool(
                self._scene_id, PRIM_EGO_OBSTACLE, ego_pool
            ):
                raise RuntimeError("Could not update the BEV ego-car track")
        return self._render_one_camera(
            rig_poses=level_rig_poses,
            timestamps_us=timestamps_us,
            scene_id=self._scene_id,
            camera_id=self._bev_camera_id,
            sensor_to_rig=self._bev_sensor_to_rig,
            camera_type=CAMERA_TYPE_BEV,
            resolution=(self._bev.height, self._bev.width),
        )

    def render_physx_debug_frames(
        self,
        *,
        rig_poses_torch: Tensor,
        timestamps_us: npt.NDArray[np.int64],
        physics_debug_frames: tuple[PhysicsDebugFrame, ...],
    ) -> _RenderedCameraFrames | None:
        """Render exact PhysX collider snapshots with Ludus on CUDA."""
        if not physics_debug_frames:
            return None
        if len(physics_debug_frames) != len(timestamps_us):
            raise ValueError(
                "physics_debug_frames must match the rendered timestamp count"
            )
        if self._selected_camera_name is None:
            raise RuntimeError("load_scene() must be called before debug rendering.")
        camera_name = self._selected_camera_name
        scene_id = self._physx_debug_scene.update(
            physics_debug_frames, np.asarray(timestamps_us, dtype=np.int64)
        )
        return self._render_one_camera(
            rig_poses=rig_poses_torch,
            timestamps_us=timestamps_us,
            scene_id=scene_id,
            camera_id=self._all_camera_map[camera_name],
            sensor_to_rig=self._sensor_to_rig[camera_name],
            camera_type=CAMERA_TYPE_REGULAR,
            resolution=(self._raster.height, self._raster.width),
        )

    def render_physx_debug_lazy_frames(
        self,
        rig_poses_world: npt.NDArray[np.float32],
        timestamps_us: npt.NDArray[np.int64],
        physics_debug_frames: tuple[PhysicsDebugFrame, ...],
    ) -> tuple[_LazyRasterFrame, ...]:
        """Render only the debug view and expose each CUDA frame lazily."""
        rig_poses_torch = torch.from_numpy(
            np.ascontiguousarray(rig_poses_world, dtype=np.float32)
        ).to(device=self._device)
        rendered = self.render_physx_debug_frames(
            rig_poses_torch=rig_poses_torch,
            timestamps_us=timestamps_us,
            physics_debug_frames=physics_debug_frames,
        )
        if rendered is None:
            return ()
        return tuple(
            _LazyRasterFrame(
                rendered.frames_hwc_uint8,
                index,
                source_event=rendered.ready_event,
            )
            for index in range(len(timestamps_us))
        )

    def build_chunk(
        self,
        *,
        timestamps_us: npt.NDArray[np.int64],
        rgb_frames: _RenderedCameraFrames,
        bev_frames: _RenderedCameraFrames | None,
        physics_debug_frames: tuple[PhysicsDebugFrame, ...] = (),
        physx_frames: _RenderedCameraFrames | None = None,
    ) -> RasterChunk:
        """Wrap rendered camera tensors in lazy frame objects."""
        if physics_debug_frames and len(physics_debug_frames) != len(timestamps_us):
            raise ValueError(
                "physics_debug_frames must match the rendered timestamp count"
            )
        if bev_frames is not None and int(bev_frames.frames_hwc_uint8.shape[0]) == 0:
            bev_frames = None
        if physx_frames is not None and int(
            physx_frames.frames_hwc_uint8.shape[0]
        ) != len(timestamps_us):
            raise ValueError("PhysX debug render count must match the timestamp count")
        bev_frame_indices = _resampled_frame_indices(
            source_count=(
                int(bev_frames.frames_hwc_uint8.shape[0])
                if bev_frames is not None
                else 0
            ),
            target_count=len(timestamps_us),
        )
        if self._use_cuda_frames:
            frames = [
                PresentedFrame(
                    timestamp_us=int(timestamps_us[idx]),
                    rgb_host_uint8=_LazyRasterFrame(
                        rgb_frames.frames_hwc_uint8,
                        idx,
                        source_event=rgb_frames.ready_event,
                    ),
                    depth_host_f32=None,
                    bev_host_uint8=(
                        _LazyRasterFrame(
                            bev_frames.frames_hwc_uint8,
                            bev_frame_indices[idx],
                            source_event=bev_frames.ready_event,
                        )
                        if bev_frames is not None
                        else None
                    ),
                    physx_debug=(
                        physics_debug_frames[idx] if physics_debug_frames else None
                    ),
                    physx_rgb_host_uint8=(
                        _LazyRasterFrame(
                            physx_frames.frames_hwc_uint8,
                            idx,
                            source_event=physx_frames.ready_event,
                        )
                        if physx_frames is not None
                        else None
                    ),
                )
                for idx in range(len(timestamps_us))
            ]
            return RasterChunk(frames=tuple(frames))

        rgb_host_frames = _rendered_frames_to_numpy(rgb_frames)
        bev_host_frames = (
            _rendered_frames_to_numpy(bev_frames) if bev_frames is not None else None
        )
        frames = [
            PresentedFrame(
                timestamp_us=int(timestamps_us[idx]),
                rgb_host_uint8=rgb_host_frames[idx],
                depth_host_f32=None,
                bev_host_uint8=(
                    bev_host_frames[bev_frame_indices[idx]]
                    if bev_host_frames is not None
                    else None
                ),
                physx_debug=(
                    physics_debug_frames[idx] if physics_debug_frames else None
                ),
                physx_rgb_host_uint8=(
                    _LazyRasterFrame(
                        physx_frames.frames_hwc_uint8,
                        idx,
                        source_event=physx_frames.ready_event,
                    )
                    if physx_frames is not None
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
        timestamps_us: npt.ArrayLike,
        scene_id: int,
        camera_id: int,
        sensor_to_rig: Tensor,
        camera_type: int,
        resolution: tuple[int, int],
    ) -> _RenderedCameraFrames:
        """Single-camera rasterizer dispatch shared by the main view and BEV.

        Frames stay CUDA-backed so the world model consumes HDMap conditioning
        without a GPU->CPU->GPU round trip (presenters materialize NumPy lazily).
        """
        camera_poses_world = torch.einsum(
            "nij,jk->nik", rig_poses, sensor_to_rig.to(self._device)
        )
        camera_poses_ludus = self._to_ludus_camera_pose(camera_poses_world)

        height, width = resolution
        images = self.ctx.render_uniform(
            scene_id=scene_id,
            camera_id=camera_id,
            timestamps_us=timestamps_us,
            camera_type_id=camera_type,
            camera_poses=camera_poses_ludus,
            resolution=(height, width),
        )

        rgb = images[:, :, :, :3]
        if self.ctx.needs_vflip:
            rgb = rgb.flip(1)
        if rgb.dtype != torch.uint8:
            rgb = (rgb.clamp(0.0, 1.0) * 255.0 + 0.5).to(torch.uint8)
        rgb = rgb.detach().contiguous()
        ready_event = None
        if rgb.is_cuda:
            ready_event = torch.cuda.Event()
            ready_event.record(torch.cuda.current_stream(rgb.device))
        return _RenderedCameraFrames(frames_hwc_uint8=rgb, ready_event=ready_event)

    def cleanup(self) -> None:
        # getattr guard: __init__ can raise before _temp_dir is assigned (e.g.
        # the ludus extension build fails), and __del__ still calls cleanup --
        # without this the AttributeError masks the real __init__ error.
        temp_dir = getattr(self, "_temp_dir", None)
        if temp_dir is not None:
            temp_dir.cleanup()
            self._temp_dir = None

    def __del__(self) -> None:
        self.cleanup()


class LudusConditionRasterizer:
    """Thread-pinned facade over :class:`_LudusConditionRasterizerImpl`.

    NVIDIA EGL on the Blackwell + 595.58.03 driver can't migrate a headless
    surfaceless GL context across threads (``eglMakeCurrent`` fails off the
    init thread), so every public entry point runs synchronously on one
    dedicated worker that owns the GL context for its lifetime. Behaves
    exactly like the underlying implementation.
    """

    def __init__(
        self,
        raster: RasterConfig,
        bev: BevConfig | None = None,
        *,
        ego_dimensions_lwh: tuple[float, float, float] | None = None,
        max_chunk_frames: int = 1,
    ) -> None:
        self._exec = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ludus-render"
        )
        self._impl: _LudusConditionRasterizerImpl | None = self._exec.submit(
            _LudusConditionRasterizerImpl,
            raster,
            bev,
            ego_dimensions_lwh=ego_dimensions_lwh,
            max_chunk_frames=max_chunk_frames,
        ).result()
        self._bev_enabled = bool(bev is not None and bev.enabled)
        self._pending_bev: (
            concurrent.futures.Future[_RenderedCameraFrames | None] | None
        ) = None
        self._latest_bev: _RenderedCameraFrames | None = None

    def load_scene(self, scene: SceneBundle) -> None:
        exec_, impl = self._require_alive()
        self._clear_pending_bev()
        self._latest_bev = None
        return exec_.submit(impl.load_scene, scene).result()

    def render_chunk(
        self,
        rig_poses_world: npt.NDArray[np.float32],
        timestamps_us: npt.NDArray[np.int64],
        dynamic_actors: tuple[DynamicActorTrajectory, ...] = (),
        physics_debug_frames: tuple[PhysicsDebugFrame, ...] = (),
    ) -> "RasterChunk":
        exec_, impl = self._require_alive()
        actors_detached = any(actor.detached_from_track for actor in dynamic_actors)
        if actors_detached:
            self._clear_pending_bev()
            self._latest_bev = None
            return exec_.submit(
                impl.render_chunk,
                rig_poses_world,
                timestamps_us,
                dynamic_actors,
                physics_debug_frames,
            ).result()
        if not self._bev_enabled:
            return exec_.submit(
                impl.render_chunk,
                rig_poses_world,
                timestamps_us,
                dynamic_actors,
                physics_debug_frames,
            ).result()

        lagged_bev = self._poll_ready_bev()

        (
            chunk_timestamps_us,
            rig_poses_torch,
            rgb_frames,
        ) = exec_.submit(
            impl.render_rgb_frames,
            rig_poses_world,
            timestamps_us,
            dynamic_actors,
        ).result()

        # RGB rendering gives the previous BEV another chance to finish without
        # ever making it part of the critical path. Render PhysX before queuing
        # the next BEV: both use the same single-thread executor, so submitting
        # BEV first would put the debug view behind unrelated minimap work.
        lagged_bev = self._poll_ready_bev()
        physx_frames = exec_.submit(
            impl.render_physx_debug_frames,
            rig_poses_torch=rig_poses_torch,
            timestamps_us=chunk_timestamps_us,
            physics_debug_frames=physics_debug_frames,
        ).result()
        if self._pending_bev is None:
            self._pending_bev = exec_.submit(
                impl.render_bev_frames,
                rig_poses_torch=rig_poses_torch,
                timestamps_us=chunk_timestamps_us,
            )
        return impl.build_chunk(
            timestamps_us=chunk_timestamps_us,
            rgb_frames=rgb_frames,
            bev_frames=lagged_bev,
            physics_debug_frames=physics_debug_frames,
            physx_frames=physx_frames,
        )

    def render_physx_debug_lazy_frames(
        self,
        rig_poses_world: npt.NDArray[np.float32],
        timestamps_us: npt.NDArray[np.int64],
        physics_debug_frames: tuple[PhysicsDebugFrame, ...],
    ) -> tuple[_LazyRasterFrame, ...]:
        """Render a debug-only chunk on the rasterizer's pinned worker."""
        exec_, impl = self._require_alive()
        return exec_.submit(
            impl.render_physx_debug_lazy_frames,
            rig_poses_world,
            timestamps_us,
            physics_debug_frames,
        ).result()

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
        self._clear_pending_bev()
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

    def _clear_pending_bev(self) -> None:
        pending = getattr(self, "_pending_bev", None)
        if pending is None:
            return
        pending.cancel()
        with contextlib.suppress(Exception):
            pending.result(timeout=0)
        self._pending_bev = None

    def _poll_ready_bev(self) -> _RenderedCameraFrames | None:
        pending = self._pending_bev
        if pending is not None and pending.done():
            self._latest_bev = pending.result()
            self._pending_bev = None
        return self._latest_bev


def _rendered_frames_to_numpy(rendered: _RenderedCameraFrames) -> list[np.ndarray]:
    synchronize = getattr(rendered.ready_event, "synchronize", None)
    if callable(synchronize):
        synchronize()
    frames = rendered.frames_hwc_uint8.detach().cpu().numpy()
    frames = np.ascontiguousarray(frames, dtype=np.uint8)
    return [frames[idx] for idx in range(frames.shape[0])]


def _resampled_frame_indices(*, source_count: int, target_count: int) -> list[int]:
    """Map a lagged chunk across the current chunk without assuming equal sizes."""
    if source_count <= 0 or target_count <= 0:
        return []
    if source_count == 1 or target_count == 1:
        return [0] * target_count
    scale = (source_count - 1) / (target_count - 1)
    return [round(index * scale) for index in range(target_count)]


def _build_bev_ego_car_pool(
    *,
    rig_poses: Tensor,
    timestamps_us: npt.ArrayLike | Tensor,
    dimensions_lwh: tuple[float, float, float],
    track_capacity: int,
) -> CubePool:
    """Build a fixed-capacity green ego-car track for BEV rendering.

    Args:
        rig_poses: Level rig-to-world poses with shape ``[frames, 4, 4]``.
        timestamps_us: Timestamp corresponding to each rig pose.
        dimensions_lwh: Full ego-car box extents in metres.
        track_capacity: Fixed pose count allocated in the Ludus scene.

    Returns:
        Ego-obstacle cube pool padded to ``track_capacity`` poses.

    Raises:
        ValueError: The inputs are empty, mismatched, or exceed the track capacity.
    """
    frame_count = int(rig_poses.shape[0])
    if frame_count <= 0:
        raise ValueError("BEV ego-car rendering requires at least one pose")
    if frame_count > track_capacity:
        raise ValueError(
            f"Ego-car frame count {frame_count} exceeds capacity {track_capacity}"
        )
    if isinstance(timestamps_us, Tensor):
        timestamps = timestamps_us.to(device=rig_poses.device, dtype=torch.int64)
    else:
        timestamps = torch.as_tensor(
            np.asarray(timestamps_us, dtype=np.int64),
            dtype=torch.int64,
            device=rig_poses.device,
        )
    if int(timestamps.shape[0]) != frame_count:
        raise ValueError("BEV ego-car poses and timestamps must have equal length")

    poses = rig_poses
    padding = track_capacity - frame_count
    if padding:
        interval = (
            torch.clamp(timestamps[-1] - timestamps[-2], min=1)
            if frame_count > 1
            else timestamps.new_tensor(1)
        )
        padded_timestamps = timestamps[-1] + interval * torch.arange(
            1, padding + 1, dtype=torch.int64, device=rig_poses.device
        )
        timestamps = torch.cat((timestamps, padded_timestamps))
        poses = torch.cat((poses, poses[-1:].expand(padding, -1, -1)))

    height = float(dimensions_lwh[2])
    translations = poses[:, :3, 3].clone()
    translations[:, 2] += height * 0.5
    yaw = torch.atan2(poses[:, 1, 0], poses[:, 0, 0])
    quaternions = torch.zeros(
        (track_capacity, 4), dtype=torch.float32, device=rig_poses.device
    )
    quaternions[:, 2] = torch.sin(yaw * 0.5)
    quaternions[:, 3] = torch.cos(yaw * 0.5)
    return CubePool(
        timestamps_us=timestamps,
        cube_ts_prefix_sum=torch.tensor(
            [track_capacity], dtype=torch.int32, device=rig_poses.device
        ),
        track_timestamps_us=timestamps,
        translations=translations,
        quaternions=quaternions,
        scales=torch.tensor(
            [dimensions_lwh], dtype=torch.float32, device=rig_poses.device
        ),
        colors=torch.tensor(
            [[118.0 / 255.0, 185.0 / 255.0, 0.0, 0.08, 0.22, 0.0]],
            dtype=torch.float32,
            device=rig_poses.device,
        ),
        prim_type_id=PRIM_EGO_OBSTACLE,
        render_flags=CUBE_FLAG_WIREFRAME,
    )


def _build_bev_camera(bev: BevConfig, device: torch.device) -> OrthographicCamera:
    """Construct an orthographic camera with the configured ground footprint."""
    return create_bev_camera(
        width=bev.width,
        height=bev.height,
        device=device,
        bev_height=bev.height_m,
        fov_deg=bev.fov_deg,
        far=max(150.0, float(bev.height_m) * 4.0),
    )


def _level_rig_poses_for_bev(rig_poses: Tensor) -> Tensor:
    """Keep BEV centered on the rig without inheriting its pitch or roll.

    The minimap remains heading-up, so retain the rig's planar yaw. When the
    forward axis is nearly vertical (for example after a collision), derive
    yaw from the projected left axis instead. Translation is copied exactly.

    Args:
        rig_poses: Rig-to-world poses with shape ``[..., 4, 4]``.

    Returns:
        Poses with the same translation and planar heading but world-up rotation.
    """
    rotation = rig_poses[..., :3, :3]
    forward_xy = rotation[..., :2, 0]
    left_xy = rotation[..., :2, 1]
    forward_norm = torch.linalg.vector_norm(forward_xy, dim=-1)
    yaw_from_forward = torch.atan2(forward_xy[..., 1], forward_xy[..., 0])
    yaw_from_left = torch.atan2(-left_xy[..., 0], left_xy[..., 1])
    yaw = torch.where(forward_norm > 1e-4, yaw_from_forward, yaw_from_left)

    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    level_poses = torch.zeros_like(rig_poses)
    level_poses[..., 0, 0] = cos_yaw
    level_poses[..., 0, 1] = -sin_yaw
    level_poses[..., 1, 0] = sin_yaw
    level_poses[..., 1, 1] = cos_yaw
    level_poses[..., 2, 2] = 1.0
    level_poses[..., :3, 3] = rig_poses[..., :3, 3]
    level_poses[..., 3, 3] = 1.0
    return level_poses


def _bev_sensor_to_rig(*, height_m: float, device: torch.device) -> Tensor:
    """Create the sensor-to-rig transform for a straight-down BEV camera.

    Sensor (FLU): X=forward (optical axis), Y=left, Z=up
    Rig (FLU):    X=forward, Y=left, Z=up

    Sensor X (depth) maps to rig -Z (down), sensor Y (left) maps to rig +Y,
    and sensor Z (image up) maps to rig +X (forward).
    """
    return torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, float(height_m)],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )
