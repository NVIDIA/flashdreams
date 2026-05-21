# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ludus rendering context - ``LudusCudaTimestampedContext``."""

from typing import List, Optional, Tuple

import torch

from ._plugin import _get_plugin
from .primitives import (
    FThetaCamera,
    TimestampedScene,
    _pack_cameras,
)


def _compute_element_aabbs(
    vertices: torch.Tensor, prefix_sum: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Compute per-element AABBs from vertices and prefix sum.

    Returns a flat ``[n_elements * 6]`` float32 tensor with
    ``(min_x, min_y, min_z, max_x, max_y, max_z)`` per element.
    """
    n_elem = len(prefix_sum)
    if n_elem == 0 or len(vertices) == 0:
        return torch.zeros(0, dtype=torch.float32, device=device)

    verts = vertices.to(device, dtype=torch.float32)
    if verts.ndim == 2 and verts.shape[1] > 3:
        verts = verts[:, :3]
    ps = prefix_sum.to(device, dtype=torch.int64).contiguous()

    vid = torch.arange(len(verts), device=device, dtype=torch.int64)
    elem_id = torch.searchsorted(ps, vid, side="right")

    e_min = torch.full((n_elem, 3), float("inf"), device=device)
    e_max = torch.full((n_elem, 3), float("-inf"), device=device)
    idx_exp = elem_id.unsqueeze(-1).expand(-1, 3)
    e_min.scatter_reduce_(0, idx_exp, verts, reduce="amin")
    e_max.scatter_reduce_(0, idx_exp, verts, reduce="amax")

    return torch.cat([e_min, e_max], dim=-1).reshape(-1)


class LudusCudaTimestampedContext:
    """CUDA context for timestamped scene rendering.

    All timestamp search, element extraction, color/width lookup, and
    geometry generation happen on the GPU via CUDA kernels.
    """

    def __init__(self, device=None):
        if device is None:
            cuda_device_idx = torch.cuda.current_device()
        else:
            with torch.cuda.device(device):
                cuda_device_idx = torch.cuda.current_device()
        self.cuda_device_idx = cuda_device_idx
        self.cpp_wrapper = _get_plugin().LudusCudaStateWrapper(cuda_device_idx)
        self._tessellation_threshold = 1.0
        self._max_extrapolation_us = 500_000

        self._cameras: List[FThetaCamera] = []
        self._camera_intrinsics: Optional[torch.Tensor] = None
        self.needs_vflip = False  # CUDA renders top-down (standard image convention)

        # Per-scene flat buffers
        self._scenes: List[dict] = []

    @property
    def max_batch_size(self) -> int:
        """Maximum number of images that can be rendered in a single batch."""
        return 2048

    # ------------------------------------------------------------------
    def upload_cameras(self, cameras: List[FThetaCamera]) -> None:
        device = torch.device(f"cuda:{self.cuda_device_idx}")
        self._cameras = cameras
        self._camera_intrinsics = _pack_cameras(cameras, device)

    # ------------------------------------------------------------------
    def upload_scene(self, scene: TimestampedScene) -> int:
        device = torch.device(f"cuda:{self.cuda_device_idx}")

        timestamps_list: List[torch.Tensor] = []
        int32_list: List[torch.Tensor] = []
        vertices_list: List[torch.Tensor] = []
        triangles_list: List[torch.Tensor] = []
        float_list: List[torch.Tensor] = []

        polyline_pool_headers: List[torch.Tensor] = []
        polygon_pool_headers: List[torch.Tensor] = []
        cube_pool_headers: List[torch.Tensor] = []

        ts_offset = 0
        int32_offset = 0
        vert_offset = 0
        tri_offset = 0
        float_offset = 0

        max_varrays_per_ts_polyline = 0
        max_varrays_per_ts_polygon = 0

        for pool in scene.polyline_pools:
            n_ts = pool.timestamps_us.shape[0]
            n_varrays = pool.varrays_prefix_sum.shape[0]
            n_verts = pool.vertices.shape[0]

            # Compute per-timestamp max varrays from prefix sum
            ps = pool.timestamped_varrays_prefix_sum
            diffs = torch.diff(
                ps, prepend=torch.tensor([0], dtype=ps.dtype, device=ps.device)
            )
            mvpt = int(diffs.max().item()) if len(diffs) > 0 else 0
            if mvpt > max_varrays_per_ts_polyline:
                max_varrays_per_ts_polyline = mvpt

            header = torch.zeros(16, dtype=torch.int32, device=device)
            header[0] = n_ts
            header[1] = n_varrays
            header[2] = n_verts
            header[3] = pool.prim_type_id
            header[4] = ts_offset
            header[5] = int32_offset  # ts_varrays_ps
            header[6] = int32_offset + n_ts  # varrays_ps
            header[7] = vert_offset
            header[8] = float_offset  # aabb

            aabbs = _compute_element_aabbs(
                pool.vertices, pool.varrays_prefix_sum, device
            )
            polyline_pool_headers.append(header)

            timestamps_list.append(pool.timestamps_us.to(device))
            int32_list.append(
                pool.timestamped_varrays_prefix_sum.to(device, dtype=torch.int32)
            )
            int32_list.append(pool.varrays_prefix_sum.to(device, dtype=torch.int32))

            verts_padded = torch.zeros(n_verts, 4, dtype=torch.float32, device=device)
            verts_padded[:, :3] = pool.vertices.to(device)
            vertices_list.append(verts_padded)

            float_list.append(aabbs)

            ts_offset += n_ts
            int32_offset += n_ts + n_varrays
            vert_offset += n_verts
            float_offset += len(aabbs)

        for pool in scene.polygon_pools:
            n_ts = pool.timestamps_us.shape[0]
            n_varrays = pool.varrays_prefix_sum.shape[0]
            n_verts = pool.vertices.shape[0]
            n_tris = pool.triangles.shape[0]

            ps = pool.timestamped_varrays_prefix_sum
            diffs = torch.diff(
                ps, prepend=torch.tensor([0], dtype=ps.dtype, device=ps.device)
            )
            mvpt = int(diffs.max().item()) if len(diffs) > 0 else 0
            if mvpt > max_varrays_per_ts_polygon:
                max_varrays_per_ts_polygon = mvpt

            header = torch.zeros(16, dtype=torch.int32, device=device)
            header[0] = n_ts
            header[1] = n_varrays
            header[2] = n_verts
            header[3] = n_tris
            header[4] = pool.prim_type_id
            header[5] = ts_offset
            header[6] = int32_offset  # ts_varrays_ps
            header[7] = int32_offset + n_ts  # varrays_ps
            header[8] = int32_offset + n_ts + n_varrays  # tri_ps
            header[9] = vert_offset
            header[10] = tri_offset
            aabbs = _compute_element_aabbs(
                pool.vertices, pool.varrays_prefix_sum, device
            )
            header[11] = float_offset
            polygon_pool_headers.append(header)

            timestamps_list.append(pool.timestamps_us.to(device))
            int32_list.append(
                pool.timestamped_varrays_prefix_sum.to(device, dtype=torch.int32)
            )
            int32_list.append(pool.varrays_prefix_sum.to(device, dtype=torch.int32))
            int32_list.append(pool.triangle_prefix_sum.to(device, dtype=torch.int32))

            verts_padded = torch.zeros(n_verts, 4, dtype=torch.float32, device=device)
            verts_padded[:, :3] = pool.vertices.to(device)
            vertices_list.append(verts_padded)

            tris_padded = torch.zeros(n_tris, 4, dtype=torch.int32, device=device)
            tris_padded[:, :3] = pool.triangles.to(device, dtype=torch.int32)
            triangles_list.append(tris_padded)

            float_list.append(aabbs)

            ts_offset += n_ts
            int32_offset += n_ts + 2 * n_varrays
            vert_offset += n_verts
            tri_offset += n_tris
            float_offset += len(aabbs)

        for pool in scene.cube_pools or []:
            n_global_ts = pool.timestamps_us.shape[0]
            n_cubes = pool.scales.shape[0]
            n_track_poses = pool.translations.shape[0]

            header = torch.zeros(16, dtype=torch.int32, device=device)
            header[0] = n_cubes
            header[1] = n_global_ts
            header[2] = n_track_poses
            header[3] = pool.prim_type_id
            header[4] = ts_offset  # global timestamps
            header[5] = int32_offset  # cube_ts_ps
            header[6] = ts_offset + n_global_ts  # track timestamps
            header[7] = float_offset  # translations
            header[8] = float_offset + n_track_poses * 3  # quaternions
            header[9] = float_offset + n_track_poses * 7  # scales
            header[10] = float_offset + n_track_poses * 7 + n_cubes * 3  # colors
            header[11] = pool.render_flags
            cube_pool_headers.append(header)

            timestamps_list.append(pool.timestamps_us.to(device))
            timestamps_list.append(pool.track_timestamps_us.to(device))
            int32_list.append(pool.cube_ts_prefix_sum.to(device, dtype=torch.int32))
            float_list.append(pool.translations.to(device).reshape(-1))
            float_list.append(pool.quaternions.to(device).reshape(-1))
            float_list.append(pool.scales.to(device).reshape(-1))
            float_list.append(pool.colors.to(device).reshape(-1))

            ts_offset += n_global_ts + n_track_poses
            int32_offset += n_cubes
            float_offset += n_track_poses * 7 + n_cubes * 9

        all_timestamps = (
            torch.cat(timestamps_list).contiguous()
            if timestamps_list
            else torch.empty(0, dtype=torch.int64, device=device)
        )
        all_int32 = (
            torch.cat(int32_list).contiguous()
            if int32_list
            else torch.empty(0, dtype=torch.int32, device=device)
        )
        all_vertices = (
            torch.cat(vertices_list).contiguous()
            if vertices_list
            else torch.empty((0, 4), dtype=torch.float32, device=device)
        )
        all_triangles = (
            torch.cat(triangles_list).contiguous()
            if triangles_list
            else torch.empty((0, 4), dtype=torch.int32, device=device)
        )
        all_floats = (
            torch.cat(float_list).contiguous()
            if float_list
            else torch.empty(0, dtype=torch.float32, device=device)
        )

        all_pl_pools = (
            torch.stack(polyline_pool_headers).contiguous()
            if polyline_pool_headers
            else torch.empty((0, 16), dtype=torch.int32, device=device)
        )
        all_pg_pools = (
            torch.stack(polygon_pool_headers).contiguous()
            if polygon_pool_headers
            else torch.empty((0, 16), dtype=torch.int32, device=device)
        )
        all_cb_pools = (
            torch.stack(cube_pool_headers).contiguous()
            if cube_pool_headers
            else torch.empty((0, 16), dtype=torch.int32, device=device)
        )

        scene_id = len(self._scenes)
        self._scenes.append(
            {
                "timestamps": all_timestamps,
                "int32": all_int32,
                "vertices": all_vertices,
                "triangles": all_triangles,
                "floats": all_floats,
                "polyline_pools": all_pl_pools,
                "polygon_pools": all_pg_pools,
                "cube_pools": all_cb_pools,
                "max_varrays_per_ts_polyline": max_varrays_per_ts_polyline,
                "max_varrays_per_ts_polygon": max_varrays_per_ts_polygon,
            }
        )
        return scene_id

    # ------------------------------------------------------------------
    def set_tessellation_threshold(self, threshold: float) -> None:
        self._tessellation_threshold = threshold

    def set_depth_scaling(self, enabled: bool = True) -> None:
        self.cpp_wrapper.set_depth_scaling(1.0 if enabled else 0.0)

    def set_resolution_scale(
        self,
        width: int,
        height: int,
        reference_width: int = 1280,
        reference_height: int = 720,
    ) -> None:
        scale = min(width / reference_width, height / reference_height)
        self.cpp_wrapper.set_resolution_scale(scale)

    def set_cull_radius(self, scale: float = 1.5) -> None:
        self.cpp_wrapper.set_cull_radius(scale)

    def set_msaa_samples(self, samples: int) -> None:
        self.cpp_wrapper.set_msaa_samples(samples)

    def set_line_widths(
        self,
        polyline_regular: float = 0.0,
        polyline_bev: float = 0.0,
        ego_traj_regular: float = 0.0,
        ego_traj_bev: float = 0.0,
        wireframe: float = 0.0,
    ) -> None:
        self.cpp_wrapper.set_line_widths(
            polyline_regular,
            polyline_bev,
            ego_traj_regular,
            ego_traj_bev,
            wireframe,
        )

    def set_max_tessellation_levels(
        self,
        polyline: int = 4,
        polygon: int = 3,
        cube: int = 3,
    ) -> None:
        self.cpp_wrapper.set_max_tessellation_levels(polyline, polygon, cube)

    def upload_color_palette(self, colors: dict) -> None:
        """Upload a custom color palette.

        ``colors`` maps ``prim_type_id`` (int) to an RGBA tuple
        ``(r, g, b, a)`` where each component is in [0, 255].
        """
        max_prim = max(colors.keys()) + 1 if colors else 0
        palette = torch.zeros(max_prim, dtype=torch.int32)
        for prim_id, rgba in colors.items():
            r, g, b = int(rgba[0]), int(rgba[1]), int(rgba[2])
            a = int(rgba[3]) if len(rgba) > 3 else 255
            packed = r | (g << 8) | (b << 16) | (a << 24)
            palette[prim_id] = packed
        self.cpp_wrapper.upload_color_palette(palette)

    def clear_scenes(self) -> None:
        self._scenes.clear()

    # ------------------------------------------------------------------
    def render_batch(
        self,
        queries: List[Tuple[int, int, int, int]],
        camera_poses: torch.Tensor,
        resolution: Tuple[int, int],
    ) -> torch.Tensor:
        """Render a batch of queries (tuple-based API).

        Each query is ``(scene_id, camera_id, timestamp_us[, camera_type_id])``.
        Delegates to :meth:`render`.
        """
        device = camera_poses.device
        n = len(queries)
        scene_ids = torch.empty(n, dtype=torch.int32, device=device)
        camera_ids = torch.empty(n, dtype=torch.int32, device=device)
        timestamps_us = torch.empty(n, dtype=torch.int64, device=device)
        camera_type_ids = torch.empty(n, dtype=torch.int32, device=device)
        for i, q in enumerate(queries):
            scene_ids[i] = int(q[0])
            camera_ids[i] = int(q[1])
            ts = q[2]
            timestamps_us[i] = int(ts.item() if isinstance(ts, torch.Tensor) else ts)
            camera_type_ids[i] = int(q[3]) if len(q) > 3 else 0
        return self.render(
            scene_ids,
            camera_ids,
            timestamps_us,
            camera_type_ids,
            camera_poses,
            resolution,
        )

    # ------------------------------------------------------------------
    def render(
        self,
        scene_ids: torch.Tensor,
        camera_ids: torch.Tensor,
        timestamps_us: torch.Tensor,
        camera_type_ids: torch.Tensor,
        camera_poses: torch.Tensor,
        resolution: Tuple[int, int],
    ) -> torch.Tensor:
        """Render a batch of queries.

        Currently only single-query rendering is supported (one scene, one
        camera at a time); the batch is dispatched as a Python loop over
        the per-query CUDA kernel call.
        """
        assert self._camera_intrinsics is not None, "call upload_cameras first"

        n = scene_ids.shape[0]
        all_images = []

        plugin = _get_plugin()

        for i in range(n):
            sid = int(scene_ids[i].item())
            cid = int(camera_ids[i].item())
            ts_val = int(timestamps_us[i].item())
            cam_type = int(camera_type_ids[i].item())

            sd = self._scenes[sid]
            cam_intrinsics = self._camera_intrinsics[cid : cid + 1].contiguous()
            pose = camera_poses[i : i + 1].contiguous()

            img = plugin.ludus_render_fwd_cuda_timestamped(
                self.cpp_wrapper,
                sd["timestamps"],
                sd["int32"],
                sd["vertices"],
                sd["triangles"],
                sd["floats"],
                sd["polyline_pools"],
                sd["polygon_pools"],
                sd["cube_pools"],
                ts_val,
                self._max_extrapolation_us,
                sd["max_varrays_per_ts_polyline"],
                sd["max_varrays_per_ts_polygon"],
                cam_type,
                cam_intrinsics,
                pose,
                resolution,
                self._tessellation_threshold,
            )
            all_images.append(img)

        return torch.cat(all_images, dim=0)
