# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-threaded Ludus conditioning owned by the V2 model thread."""

from __future__ import annotations

import math

import numpy as np
import torch
from ludus_renderer import (
    PRIM_BEV_ROAD_SURFACE,
    FThetaCamera,
    LudusCudaTimestampedContext,
    TimestampedPolygonPool,
    TimestampedScene,
)
from ludus_renderer import load_scene as load_ludus_scene
from ludus_renderer._ops import _triangulate_polygon_ear_clipping
from ludus_renderer.render_utils import SceneAdapter
from ludus_renderer.torch.ops import CAMERA_TYPE_BEV, CAMERA_TYPE_REGULAR
from shapely.geometry import Polygon
from torch import Tensor

from omnidreams_game_engine.config import BevConfig, RasterConfig
from omnidreams_game_engine.contracts import ConditionRenderer
from omnidreams_game_engine.dynamic_scene import MutableObjectSceneBuffer
from omnidreams_game_engine.game_map.types import GameMapElement
from omnidreams_game_engine.types import (
    ConditionBatch,
    SceneDefinition,
    TrajectoryChunk,
)

_BEV_CAMERA_NAME = "game_engine_bev"
_BEV_ROAD_SIMPLIFY_M = 0.05
_BEV_ROAD_DEPTH_OFFSET_M = -0.01


class LudusConditionRenderer(ConditionRenderer):
    """Render semantic frames without creating an internal worker thread."""

    def __init__(
        self,
        raster: RasterConfig,
        bev: BevConfig = BevConfig(),
        *,
        device: torch.device | str = "cuda",
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Ludus conditioning")
        self._raster = raster
        self._bev = bev
        self._device = torch.device(device)
        if self._device.type != "cuda":
            raise ValueError("Ludus conditioning requires a CUDA device")
        self._context = LudusCudaTimestampedContext(device=self._device)
        self._context.set_depth_scaling(True)
        self._context.set_msaa_samples(4)
        self._context.set_max_tessellation_levels(cube=3)
        self._context.set_line_widths(
            polyline_regular=float(raster.line_width_px),
            polyline_bev=max(1.5, float(raster.line_width_px) * 0.2),
            ego_traj_regular=float(raster.pole_width_px),
            ego_traj_bev=max(1.5, float(raster.pole_width_px) * 0.4),
            wireframe=4.0,
        )
        self._scene_id: int | None = None
        self._dynamic_scene: MutableObjectSceneBuffer | None = None
        self._camera_ids: dict[str, int] = {}
        self._sensor_to_rig: dict[str, Tensor] = {}
        self._selected_camera: str | None = None
        self._bev_camera_id: int | None = None
        self._bev_sensor_pose: Tensor | None = None
        self._closed = False

    def load_scene(self, scene: SceneDefinition) -> None:
        """Upload scene geometry and main/BEV cameras on the current thread."""
        if self._closed:
            raise RuntimeError("Condition renderer is closed")
        self._context.clear_scenes()
        loaded = load_ludus_scene(
            scene.scene_path,
            device=self._device,
            target_resolution=self._raster.resolution_wh,
            include_ego_trajectory=False,
            include_ego_obstacle=False,
        )
        SceneAdapter(loaded)
        cameras = list(loaded.cameras)
        self._camera_ids = dict(loaded.camera_name_to_id)
        self._sensor_to_rig = dict(loaded.sensor_to_rig)
        self._selected_camera = scene.selected_camera.clipgt_name
        if self._bev.enabled:
            self._bev_camera_id = len(cameras)
            cameras.append(_build_bev_camera(self._bev, self._device))
            self._camera_ids[_BEV_CAMERA_NAME] = self._bev_camera_id
            self._bev_sensor_pose = _bev_sensor_to_rig(
                height_m=self._bev.height_m,
                tilt_deg=self._bev.tilt_deg,
                device=self._device,
            )
            self._sensor_to_rig[_BEV_CAMERA_NAME] = self._bev_sensor_pose
        self._context.upload_cameras(cameras)
        base_scene = loaded.timestamped_scene
        if self._bev.enabled and scene.game_map is not None:
            road_surface_pool = _build_bev_road_surface_pool(
                scene.game_map.elements,
                self._device,
            )
            if road_surface_pool is not None:
                base_scene = TimestampedScene(
                    polyline_pools=base_scene.polyline_pools,
                    polygon_pools=[road_surface_pool, *base_scene.polygon_pools],
                    cube_pools=base_scene.cube_pools,
                )
        self._scene_id = self._context.upload_scene(base_scene)
        self._dynamic_scene = MutableObjectSceneBuffer(
            self._context,
            self._scene_id,
            base_scene,
            device=self._device,
        )

    def render(self, trajectory: TrajectoryChunk) -> ConditionBatch:
        """Render a model-ready camera tensor and an optional HUD BEV tensor."""
        if self._closed:
            raise RuntimeError("Condition renderer is closed")
        if (
            self._scene_id is None
            or self._dynamic_scene is None
            or self._selected_camera is None
        ):
            raise RuntimeError("load_scene() must run before render()")
        self._dynamic_scene.update(trajectory.dynamic_actors)
        self._scene_id = self._dynamic_scene.scene_id
        poses = torch.from_numpy(
            np.ascontiguousarray(trajectory.rig_poses_world, dtype=np.float32)
        ).to(self._device)
        main = self._render_camera(
            poses=poses,
            timestamps_us=trajectory.timestamps_us,
            camera_id=self._camera_ids[self._selected_camera],
            sensor_to_rig=self._sensor_to_rig[self._selected_camera],
            camera_type=CAMERA_TYPE_REGULAR,
            resolution=(self._raster.height, self._raster.width),
        )
        hdmap = _normalize_hwc(main).unsqueeze(0).unsqueeze(0)
        bev = None
        if self._bev_camera_id is not None and self._bev_sensor_pose is not None:
            bev_hwc = self._render_camera(
                poses=_level_rig_poses_for_bev(poses),
                timestamps_us=trajectory.timestamps_us,
                camera_id=self._bev_camera_id,
                sensor_to_rig=self._bev_sensor_pose,
                camera_type=CAMERA_TYPE_BEV,
                resolution=(self._bev.height, self._bev.width),
                preserve_alpha=True,
            )
            # BEV is a UI-only channel. Preserve the renderer's native bytes
            # instead of normalizing to BF16 only for presentation to reverse
            # the conversion before uploading the ImGui texture.
            bev = _bev_presentation_frames(bev_hwc)
        return ConditionBatch(hdmap_bvtchw=hdmap, bev_tchw=bev)

    def close(self) -> None:
        """Release all scene references on the owning thread."""
        if self._closed:
            return
        self._closed = True
        self._dynamic_scene = None
        self._context.clear_scenes()

    def _render_camera(
        self,
        *,
        poses: Tensor,
        timestamps_us: np.ndarray,
        camera_id: int,
        sensor_to_rig: Tensor,
        camera_type: int,
        resolution: tuple[int, int],
        preserve_alpha: bool = False,
    ) -> Tensor:
        assert self._scene_id is not None
        camera_to_world = torch.einsum(
            "nij,jk->nik",
            poses,
            sensor_to_rig.to(self._device),
        )
        images = self._context.render_uniform(
            scene_id=self._scene_id,
            camera_id=camera_id,
            timestamps_us=timestamps_us.tolist(),
            camera_type_id=camera_type,
            camera_poses=torch.linalg.inv(camera_to_world),
            resolution=resolution,
        )
        output = images if preserve_alpha else images[..., :3]
        if self._context.needs_vflip:
            output = output.flip(1)
        if output.dtype != torch.uint8:
            output = (output.clamp(0.0, 1.0) * 255.0 + 0.5).to(torch.uint8)
        return output.detach().contiguous()


def _normalize_hwc(frames: Tensor) -> Tensor:
    return frames.permute(0, 3, 1, 2).to(torch.bfloat16) / 127.5 - 1.0


def _bev_presentation_frames(frames: Tensor) -> Tensor:
    """Keep renderer-native BEV RGBA bytes while changing to TCHW layout."""
    if frames.dtype != torch.uint8 or frames.ndim != 4 or frames.shape[-1] != 4:
        raise ValueError("BEV renderer output must be uint8 THWC RGBA")
    return frames.permute(0, 3, 1, 2).contiguous()


def _build_bev_road_surface_pool(
    elements: tuple[GameMapElement, ...],
    device: torch.device,
) -> TimestampedPolygonPool | None:
    """Build a lightweight black paved-surface layer for BEV alpha coverage."""
    vertices: list[Tensor] = []
    triangles: list[Tensor] = []
    vertex_counts: list[int] = []
    triangle_counts: list[int] = []
    for element in elements:
        surface = np.asarray(element.surface_world, dtype=np.float32)
        polygon = Polygon(surface[:, :2]).simplify(
            _BEV_ROAD_SIMPLIFY_M,
            preserve_topology=True,
        )
        if polygon.is_empty or polygon.geom_type != "Polygon":
            continue
        xy = np.asarray(polygon.exterior.coords[:-1], dtype=np.float32)
        if len(xy) < 3:
            continue
        z_m = float(np.median(surface[:, 2])) + _BEV_ROAD_DEPTH_OFFSET_M
        polygon_vertices = torch.from_numpy(
            np.column_stack((xy, np.full(len(xy), z_m, dtype=np.float32))).astype(
                np.float32
            )
        )
        polygon_triangles = _triangulate_polygon_ear_clipping(polygon_vertices)
        if not polygon_triangles:
            continue
        vertices.append(polygon_vertices)
        triangles.append(torch.tensor(polygon_triangles, dtype=torch.int32))
        vertex_counts.append(len(polygon_vertices))
        triangle_counts.append(len(polygon_triangles))

    if not vertices:
        return None
    return TimestampedPolygonPool(
        timestamps_us=torch.tensor([0], dtype=torch.int64, device=device),
        timestamped_varrays_prefix_sum=torch.tensor(
            [len(vertices)], dtype=torch.int32, device=device
        ),
        varrays_prefix_sum=torch.tensor(
            np.cumsum(vertex_counts), dtype=torch.int32, device=device
        ),
        triangle_prefix_sum=torch.tensor(
            np.cumsum(triangle_counts), dtype=torch.int32, device=device
        ),
        vertices=torch.cat(vertices).to(device),
        triangles=torch.cat(triangles).to(device),
        prim_type_id=PRIM_BEV_ROAD_SURFACE,
    )


def _build_bev_camera(bev: BevConfig, device: torch.device) -> FThetaCamera:
    cx = bev.width / 2.0
    cy = bev.height / 2.0
    focal = (bev.height / 2.0) / math.tan(math.radians(bev.fov_deg) / 2.0)
    diagonal = math.hypot(bev.width / 2.0, bev.height / 2.0)
    return FThetaCamera(
        principal_point=torch.tensor([cx, cy], device=device, dtype=torch.float32),
        image_size=torch.tensor(
            [float(bev.width), float(bev.height)],
            device=device,
            dtype=torch.float32,
        ),
        fw_poly=torch.tensor(
            [0.0, focal, 0.0, focal / 3.0, 0.0, 2.0 * focal / 15.0],
            device=device,
            dtype=torch.float32,
        ),
        max_ray_angle=math.atan(diagonal / focal),
        depth_max=max(150.0, bev.height_m * 4.0),
    )


def _level_rig_poses_for_bev(poses: Tensor) -> Tensor:
    rotation = poses[..., :3, :3]
    forward_xy = rotation[..., :2, 0]
    left_xy = rotation[..., :2, 1]
    yaw = torch.where(
        torch.linalg.vector_norm(forward_xy, dim=-1) > 1.0e-4,
        torch.atan2(forward_xy[..., 1], forward_xy[..., 0]),
        torch.atan2(-left_xy[..., 0], left_xy[..., 1]),
    )
    result = torch.zeros_like(poses)
    result[..., 0, 0] = torch.cos(yaw)
    result[..., 0, 1] = -torch.sin(yaw)
    result[..., 1, 0] = torch.sin(yaw)
    result[..., 1, 1] = torch.cos(yaw)
    result[..., 2, 2] = 1.0
    result[..., :3, 3] = poses[..., :3, 3]
    result[..., 3, 3] = 1.0
    return result


def _bev_sensor_to_rig(
    *, height_m: float, tilt_deg: float, device: torch.device
) -> Tensor:
    theta = math.radians(tilt_deg)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    return torch.tensor(
        [
            [sin_theta, 0.0, cos_theta, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-cos_theta, 0.0, sin_theta, height_m],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=torch.float32,
    )
