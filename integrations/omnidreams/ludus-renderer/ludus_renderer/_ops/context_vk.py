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

"""Ludus Vulkan timestamped rendering context.

:class:`LudusTimestampedContext` is the Vulkan-backed counterpart to
:class:`LudusCudaTimestampedContext`. They expose the same public method
surface so callers can swap them at runtime.

Compared to the CUDA backend:

- Scene data is uploaded once into GPU-resident Vulkan SSBOs (rather than
  cached in Python tensors and passed to the kernel on every render).
- Rendering of a query batch is a single Vulkan submission instead of one
  CUDA kernel launch per query, so larger batches scale much better.
- Geometry is generated procedurally in task/mesh shaders rather than in
  CUDA kernels.

The Vulkan extension is built on demand the first time you instantiate
this class. If the Vulkan loader or headers are missing, the constructor
raises :class:`ImportError` with an actionable hint.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from ._plugin_vk import _get_vk_plugin
from .context import _compute_element_aabbs
from .primitives import (
    FThetaCamera,
    TimestampedScene,
    _pack_cameras,
)


def _max_varrays_per_ts(timestamped_varrays_prefix_sum: torch.Tensor) -> int:
    """Max number of varrays present at any single timestamp in a pool.

    The prefix sum is cumulative across timestamps, so consecutive differences
    give the per-timestamp varray counts; the max drives the mesh-task dispatch
    stride (mirrors the CUDA backend's ``max_varrays_per_ts_*``).
    """
    ps = timestamped_varrays_prefix_sum
    if ps.numel() == 0:
        return 0
    diffs = torch.diff(ps, prepend=torch.zeros(1, dtype=ps.dtype, device=ps.device))
    return int(diffs.max().item())


class LudusTimestampedContext:
    """Vulkan context for timestamped scene rendering.

    All timestamp search, element extraction, color lookup, and procedural
    geometry generation happen on the GPU in task/mesh shaders.

    Example::

        ctx = LudusTimestampedContext(device="cuda")
        ctx.upload_cameras(cameras)
        scene_id = ctx.upload_scene(scene)

        queries = [(scene_id, cam_id, ts_us, CAMERA_TYPE_REGULAR)]
        poses   = scene.get_ego_poses_at_timestamps(...)
        images  = ctx.render_batch(queries, poses, resolution=(720, 1280))
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, device=None):
        if device is None:
            cuda_device_idx = torch.cuda.current_device()
        else:
            with torch.cuda.device(device):
                cuda_device_idx = torch.cuda.current_device()
        self.cuda_device_idx = cuda_device_idx
        self._device = torch.device(f"cuda:{cuda_device_idx}")

        try:
            plugin = _get_vk_plugin()
        except RuntimeError as exc:
            raise ImportError(str(exc)) from exc

        self._plugin = plugin
        self.cpp_wrapper = plugin.LudusTimestampedVkStateWrapper(cuda_device_idx)

        # Some defaults that the wrapper sets but Python should track.
        self._tessellation_threshold = 1.0

        self._cameras: List[FThetaCamera] = []
        self._camera_intrinsics: Optional[torch.Tensor] = None

        # Track the set of scene ids known to the C++ side so callers can
        # iterate / debug.
        self._scene_ids: List[int] = []

        # Running SSBO base offsets per scene (mirror the C++ *Used counters),
        # so scene N reads its own data, not scene 0's. Reset by clear_scenes.
        self._global_ts_off = 0
        self._global_i32_off = 0
        self._global_vert_off = 0
        self._global_tri_off = 0
        self._global_float_off = 0
        self._global_pl_pool_off = 0
        self._global_pg_pool_off = 0
        self._global_cb_pool_off = 0

        # GL/Vulkan render their layered framebuffer top-down by default;
        # Vulkan's viewport y-flip in the renderer matches the CUDA backend's
        # output convention so callers don't need to vflip.
        self.needs_vflip = False

    # ------------------------------------------------------------------
    @property
    def max_batch_size(self) -> int:
        """Maximum number of queries the backend can render in one call."""
        return int(self.cpp_wrapper.get_max_batch_size())

    # ------------------------------------------------------------------
    def upload_cameras(self, cameras: List[FThetaCamera]) -> None:
        """Upload camera intrinsics to the GPU."""
        self._cameras = list(cameras)
        self._camera_intrinsics = _pack_cameras(cameras, self._device)
        self.cpp_wrapper.upload_cameras(self._camera_intrinsics)

    # ------------------------------------------------------------------
    # Pool packers: each builds 16-uint32 row matching the C++ struct
    # layout (`ludus_types.h`) so the binding can `reinterpret_cast` rows
    # into structured pool descriptors.
    # ------------------------------------------------------------------
    @staticmethod
    def _pack_polyline_pool_row(num_timestamps: int, num_varrays: int,
                                num_vertices: int, prim_type_id: int,
                                ts_offset: int, ts_varrays_ps_offset: int,
                                varrays_ps_offset: int, vertices_offset: int,
                                aabb_offset: int,
                                ) -> torch.Tensor:
        row = torch.zeros(16, dtype=torch.uint32)
        row[0] = num_timestamps
        row[1] = num_varrays
        row[2] = num_vertices
        row[3] = prim_type_id
        row[4] = ts_offset
        row[5] = ts_varrays_ps_offset
        row[6] = varrays_ps_offset
        row[7] = vertices_offset
        row[8] = aabb_offset
        return row

    @staticmethod
    def _pack_polygon_pool_row(num_timestamps: int, num_varrays: int,
                               num_vertices: int, num_triangles: int,
                               prim_type_id: int,
                               ts_offset: int, ts_varrays_ps_offset: int,
                               varrays_ps_offset: int, tri_ps_offset: int,
                               vertices_offset: int, triangles_offset: int,
                               aabb_offset: int,
                               ) -> torch.Tensor:
        row = torch.zeros(16, dtype=torch.uint32)
        row[0] = num_timestamps
        row[1] = num_varrays
        row[2] = num_vertices
        row[3] = num_triangles
        row[4] = prim_type_id
        row[5] = ts_offset
        row[6] = ts_varrays_ps_offset
        row[7] = varrays_ps_offset
        row[8] = tri_ps_offset
        row[9] = vertices_offset
        row[10] = triangles_offset
        row[11] = aabb_offset
        return row

    @staticmethod
    def _pack_cube_pool_row(num_cubes: int, num_timestamps: int,
                            num_track_poses: int, prim_type_id: int,
                            ts_offset: int, cube_ts_ps_offset: int,
                            track_ts_offset: int, translations_offset: int,
                            quaternions_offset: int, scales_offset: int,
                            colors_offset: int, render_flags: int,
                            ) -> torch.Tensor:
        row = torch.zeros(16, dtype=torch.uint32)
        row[0] = num_cubes
        row[1] = num_timestamps
        row[2] = num_track_poses
        row[3] = prim_type_id
        row[4] = ts_offset
        row[5] = cube_ts_ps_offset
        row[6] = track_ts_offset
        row[7] = translations_offset
        row[8] = quaternions_offset
        row[9] = scales_offset
        row[10] = colors_offset
        row[11] = render_flags
        return row

    @staticmethod
    def _pack_scene_desc(num_pl_pools: int, pl_pools_offset: int,
                        num_pg_pools: int, pg_pools_offset: int,
                        num_cb_pools: int, cb_pools_offset: int,
                        ts_buf_offset: int, int32_buf_offset: int,
                        vert_buf_offset: int, tri_buf_offset: int,
                        pose_buf_offset: int, float_buf_offset: int,
                        ) -> torch.Tensor:
        # 128-byte scene descriptor = 32 uint32.
        row = torch.zeros(32, dtype=torch.uint32)
        row[0] = num_pl_pools
        row[1] = pl_pools_offset
        row[2] = num_pg_pools
        row[3] = pg_pools_offset
        row[4] = num_cb_pools
        row[5] = cb_pools_offset
        row[6] = ts_buf_offset
        row[7] = int32_buf_offset
        row[8] = vert_buf_offset
        row[9] = tri_buf_offset
        row[10] = pose_buf_offset
        row[11] = float_buf_offset
        return row

    # ------------------------------------------------------------------
    def upload_scene(self, scene: TimestampedScene) -> int:
        """Pack the scene and upload it to GPU SSBOs in a single call.

        Returns the scene id assigned by the backend.
        """
        device = self._device

        timestamps_list: List[torch.Tensor] = []
        int32_list:      List[torch.Tensor] = []
        vertices_list:   List[torch.Tensor] = []
        triangles_list:  List[torch.Tensor] = []
        float_list:      List[torch.Tensor] = []

        polyline_pool_rows: List[torch.Tensor] = []
        polygon_pool_rows:  List[torch.Tensor] = []
        cube_pool_rows:     List[torch.Tensor] = []

        # Per-scene local offsets. The C++ side appends each scene's data
        # to the global SSBOs and patches absolute offsets later via the
        # scene descriptor's *_buffer_offset fields.
        ts_off = 0
        i32_off = 0
        vert_off = 0
        tri_off = 0
        # Start float_off at 1 (and prepend a dummy float below) so that the
        # first pool never gets aabb_offset == 0. The task shaders use 0 as
        # a sentinel meaning "no AABB available, skip culling", so a real
        # aabb located at offset 0 silently disables culling for that pool,
        # which lets behind-camera polygons through and produces giant
        # garbage triangles when they straddle the camera plane.
        float_off = 1
        float_list.append(torch.zeros(1, dtype=torch.float32, device=device))

        # Max varrays present at any single timestamp, per family. Drives the
        # per-pool mesh-task dispatch stride (u_max_varrays_per_pool) so pools
        # larger than this are not silently truncated.
        max_varrays_per_ts_polyline = 0
        max_varrays_per_ts_polygon = 0

        # ---------- polyline pools ----------
        for pool in scene.polyline_pools:
            n_ts = int(pool.timestamps_us.shape[0])
            n_var = int(pool.varrays_prefix_sum.shape[0])
            n_v = int(pool.vertices.shape[0])
            max_varrays_per_ts_polyline = max(
                max_varrays_per_ts_polyline,
                _max_varrays_per_ts(pool.timestamped_varrays_prefix_sum),
            )

            aabbs = _compute_element_aabbs(pool.vertices, pool.varrays_prefix_sum, device)

            row = self._pack_polyline_pool_row(
                num_timestamps=n_ts, num_varrays=n_var, num_vertices=n_v,
                prim_type_id=pool.prim_type_id,
                ts_offset=ts_off,
                ts_varrays_ps_offset=i32_off,
                varrays_ps_offset=i32_off + n_ts,
                vertices_offset=vert_off,
                aabb_offset=float_off,
            )
            polyline_pool_rows.append(row)

            timestamps_list.append(pool.timestamps_us.to(device, dtype=torch.int64))
            int32_list.append(pool.timestamped_varrays_prefix_sum.to(device, dtype=torch.int32))
            int32_list.append(pool.varrays_prefix_sum.to(device, dtype=torch.int32))

            # Vertices are stored as Vertex (vec3 + pad) in the SSBO.
            v_padded = torch.zeros(n_v, 4, dtype=torch.float32, device=device)
            v_padded[:, :3] = pool.vertices.to(device, dtype=torch.float32)
            vertices_list.append(v_padded)

            float_list.append(aabbs)

            ts_off += n_ts
            i32_off += n_ts + n_var
            vert_off += n_v
            float_off += int(aabbs.numel())

        # ---------- polygon pools ----------
        for pool in scene.polygon_pools:
            n_ts = int(pool.timestamps_us.shape[0])
            n_var = int(pool.varrays_prefix_sum.shape[0])
            n_v = int(pool.vertices.shape[0])
            n_t = int(pool.triangles.shape[0])
            max_varrays_per_ts_polygon = max(
                max_varrays_per_ts_polygon,
                _max_varrays_per_ts(pool.timestamped_varrays_prefix_sum),
            )

            aabbs = _compute_element_aabbs(pool.vertices, pool.varrays_prefix_sum, device)

            row = self._pack_polygon_pool_row(
                num_timestamps=n_ts, num_varrays=n_var, num_vertices=n_v,
                num_triangles=n_t, prim_type_id=pool.prim_type_id,
                ts_offset=ts_off,
                ts_varrays_ps_offset=i32_off,
                varrays_ps_offset=i32_off + n_ts,
                tri_ps_offset=i32_off + n_ts + n_var,
                vertices_offset=vert_off,
                triangles_offset=tri_off,
                aabb_offset=float_off,
            )
            polygon_pool_rows.append(row)

            timestamps_list.append(pool.timestamps_us.to(device, dtype=torch.int64))
            int32_list.append(pool.timestamped_varrays_prefix_sum.to(device, dtype=torch.int32))
            int32_list.append(pool.varrays_prefix_sum.to(device, dtype=torch.int32))
            int32_list.append(pool.triangle_prefix_sum.to(device, dtype=torch.int32))

            v_padded = torch.zeros(n_v, 4, dtype=torch.float32, device=device)
            v_padded[:, :3] = pool.vertices.to(device, dtype=torch.float32)
            vertices_list.append(v_padded)

            t_padded = torch.zeros(n_t, 4, dtype=torch.int32, device=device)
            t_padded[:, :3] = pool.triangles.to(device, dtype=torch.int32)
            triangles_list.append(t_padded)

            float_list.append(aabbs)

            ts_off += n_ts
            i32_off += n_ts + 2 * n_var
            vert_off += n_v
            tri_off += n_t
            float_off += int(aabbs.numel())

        # ---------- cube pools ----------
        for pool in (scene.cube_pools or []):
            n_global_ts = int(pool.timestamps_us.shape[0])
            n_cubes = int(pool.scales.shape[0])
            n_track = int(pool.translations.shape[0])

            row = self._pack_cube_pool_row(
                num_cubes=n_cubes, num_timestamps=n_global_ts,
                num_track_poses=n_track, prim_type_id=pool.prim_type_id,
                ts_offset=ts_off,
                cube_ts_ps_offset=i32_off,
                track_ts_offset=ts_off + n_global_ts,
                translations_offset=float_off,
                quaternions_offset=float_off + n_track * 3,
                scales_offset=float_off + n_track * 7,
                colors_offset=float_off + n_track * 7 + n_cubes * 3,
                render_flags=int(pool.render_flags),
            )
            cube_pool_rows.append(row)

            timestamps_list.append(pool.timestamps_us.to(device, dtype=torch.int64))
            timestamps_list.append(pool.track_timestamps_us.to(device, dtype=torch.int64))
            int32_list.append(pool.cube_ts_prefix_sum.to(device, dtype=torch.int32))
            float_list.append(pool.translations.to(device, dtype=torch.float32).reshape(-1))
            float_list.append(pool.quaternions.to(device, dtype=torch.float32).reshape(-1))
            float_list.append(pool.scales.to(device, dtype=torch.float32).reshape(-1))
            float_list.append(pool.colors.to(device, dtype=torch.float32).reshape(-1))

            ts_off += n_global_ts + n_track
            i32_off += n_cubes
            float_off += n_track * 7 + n_cubes * 9

        # ---------- concatenate flat buffers ----------
        all_timestamps = (torch.cat(timestamps_list).contiguous()
                          if timestamps_list else torch.empty(0, dtype=torch.int64, device=device))
        all_int32 = (torch.cat(int32_list).contiguous()
                     if int32_list else torch.empty(0, dtype=torch.int32, device=device))
        all_vertices = (torch.cat(vertices_list).contiguous()
                        if vertices_list else torch.empty((0, 4), dtype=torch.float32, device=device))
        all_triangles = (torch.cat(triangles_list).contiguous()
                         if triangles_list else torch.empty((0, 4), dtype=torch.int32, device=device))
        all_floats = (torch.cat(float_list).contiguous()
                      if float_list else torch.empty(0, dtype=torch.float32, device=device))

        # Pool descriptor tensors (16 uint32 per row, viewed as raw bytes
        # so the C++ side can reinterpret_cast<TimestampedPool*>).
        def _pools_to_bytes(rows: List[torch.Tensor], stride_u32: int = 16) -> torch.Tensor:
            if not rows:
                return torch.empty((0, stride_u32 * 4), dtype=torch.uint8, device=device)
            stacked = torch.stack(rows).to(device).contiguous()  # [N, 16] uint32
            return stacked.view(torch.uint8).reshape(-1, stride_u32 * 4).contiguous()

        polyline_pool_bytes = _pools_to_bytes(polyline_pool_rows, stride_u32=16)
        polygon_pool_bytes  = _pools_to_bytes(polygon_pool_rows,  stride_u32=16)
        cube_pool_bytes     = _pools_to_bytes(cube_pool_rows,     stride_u32=16)

        # Scene descriptor (128 bytes).
        # This scene's base offset in each shared SSBO (0 for the first scene).
        # Shaders index as scene.<buf>_offset + pool.<buf>_offset.
        scene_desc_row = self._pack_scene_desc(
            num_pl_pools=len(polyline_pool_rows),
            pl_pools_offset=self._global_pl_pool_off,
            num_pg_pools=len(polygon_pool_rows),
            pg_pools_offset=self._global_pg_pool_off,
            num_cb_pools=len(cube_pool_rows),
            cb_pools_offset=self._global_cb_pool_off,
            ts_buf_offset=self._global_ts_off,
            int32_buf_offset=self._global_i32_off,
            vert_buf_offset=self._global_vert_off,
            tri_buf_offset=self._global_tri_off,
            pose_buf_offset=0,
            float_buf_offset=self._global_float_off,
        )
        scene_desc = scene_desc_row.to(device).contiguous().view(torch.uint8)

        # Empty CameraPose tensor -- this backend uses per-query camera
        # poses uploaded at render time, not per-scene poses.
        empty_poses = torch.empty((0, 16), dtype=torch.float32, device=device)

        # Per-pool max obstacle count drives one dispatch per pool.
        max_obstacles = max((int(p.scales.shape[0]) for p in (scene.cube_pools or [])),
                            default=0)

        scene_id = int(self.cpp_wrapper.upload_scene(
            scene_desc,
            polyline_pool_bytes,
            polygon_pool_bytes,
            cube_pool_bytes,
            max_obstacles,
            max_varrays_per_ts_polyline,
            max_varrays_per_ts_polygon,
            all_timestamps,
            all_int32,
            all_vertices,
            all_triangles,
            empty_poses,
            all_floats,
        ))

        # Advance the cursors past this scene for the next upload_scene.
        self._global_ts_off += ts_off
        self._global_i32_off += i32_off
        self._global_vert_off += vert_off
        self._global_tri_off += tri_off
        self._global_float_off += float_off
        self._global_pl_pool_off += len(polyline_pool_rows)
        self._global_pg_pool_off += len(polygon_pool_rows)
        self._global_cb_pool_off += len(cube_pool_rows)

        self._scene_ids.append(scene_id)
        return scene_id

    # ------------------------------------------------------------------
    def remove_scene(self, scene_id: int) -> None:
        self.cpp_wrapper.remove_scene(int(scene_id))
        if scene_id in self._scene_ids:
            self._scene_ids.remove(scene_id)

    def clear_scenes(self) -> None:
        self.cpp_wrapper.clear_scenes()
        self._scene_ids.clear()
        self._global_ts_off = 0
        self._global_i32_off = 0
        self._global_vert_off = 0
        self._global_tri_off = 0
        self._global_float_off = 0
        self._global_pl_pool_off = 0
        self._global_pg_pool_off = 0
        self._global_cb_pool_off = 0

    # ------------------------------------------------------------------
    # Render-tuning settings (mirror LudusCudaTimestampedContext)
    # ------------------------------------------------------------------
    def set_tessellation_threshold(self, threshold: float) -> None:
        self._tessellation_threshold = float(threshold)
        self.cpp_wrapper.set_tessellation_threshold(float(threshold))

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
        self.cpp_wrapper.set_resolution_scale(float(scale))

    def set_cull_radius(self, scale: float = 1.5) -> None:
        self.cpp_wrapper.set_cull_radius(float(scale))

    def set_msaa_samples(self, samples: int) -> None:
        self.cpp_wrapper.set_msaa_samples(int(samples))

    def set_line_widths(
        self,
        polyline_regular: float = 0.0,
        polyline_bev: float = 0.0,
        ego_traj_regular: float = 0.0,
        ego_traj_bev: float = 0.0,
        wireframe: float = 0.0,
    ) -> None:
        self.cpp_wrapper.set_line_widths(
            float(polyline_regular), float(polyline_bev),
            float(ego_traj_regular), float(ego_traj_bev),
            float(wireframe),
        )

    def set_max_tessellation_levels(
        self,
        polyline: int = 4,
        polygon: int = 3,
        cube: int = 3,
    ) -> None:
        self.cpp_wrapper.set_max_tessellation_levels(int(polyline), int(polygon), int(cube))

    def upload_color_palette(self, colors: dict) -> None:
        """Upload a custom color palette.

        ``colors`` maps ``prim_type_id`` (int) to an RGBA tuple
        ``(r, g, b, a)`` with each component in [0, 255]. Internally the
        Vulkan binding accepts either int32 packed RGBA8 or float [N,4]
        in [0,1]; we send float so the shader can use it as ``vec4``.
        """
        max_prim = max(colors.keys()) + 1 if colors else 0
        palette = torch.zeros((max_prim, 4), dtype=torch.float32)
        for prim_id, rgba in colors.items():
            r, g, b = int(rgba[0]), int(rgba[1]), int(rgba[2])
            a = int(rgba[3]) if len(rgba) > 3 else 255
            palette[prim_id, 0] = r / 255.0
            palette[prim_id, 1] = g / 255.0
            palette[prim_id, 2] = b / 255.0
            palette[prim_id, 3] = a / 255.0
        self.cpp_wrapper.upload_color_palette(palette.to(self._device))

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------
    def _pack_queries(self, scene_ids: torch.Tensor, camera_ids: torch.Tensor,
                      timestamps_us: torch.Tensor, camera_type_ids: torch.Tensor,
                      ) -> torch.Tensor:
        """Pack into ``RenderQuery[]`` (32 bytes each, returned as uint8)."""
        n = int(scene_ids.shape[0])
        # Layout (32 bytes per query):
        #   u32 scene_id        offset 0..4
        #   u32 camera_id       offset 4..8
        #   i64 timestamp_us    offset 8..16
        #   u32 camera_type_id  offset 16..20
        #   u32 pad[3]          offset 20..32
        #
        # We assemble the 32-byte record from two CPU int32/int64 columns to
        # avoid relying on `torch.uint32` / view-dtype machinery (which has
        # historically been finicky across torch versions). The final
        # tensor is then copied to the target device in one go.
        ints = torch.zeros((n, 5), dtype=torch.int32)
        ints[:, 0] = scene_ids.to("cpu", dtype=torch.int32)
        ints[:, 1] = camera_ids.to("cpu", dtype=torch.int32)
        # Skip slots 2..3 (int64 timestamp - filled separately).
        ints[:, 4] = camera_type_ids.to("cpu", dtype=torch.int32)
        ts = timestamps_us.to("cpu", dtype=torch.int64)

        buf = torch.zeros(n, 32, dtype=torch.uint8)
        buf_i32_view = buf.view(torch.int32).reshape(n, 8)
        buf_i64_view = buf.view(torch.int64).reshape(n, 4)
        buf_i32_view[:, 0] = ints[:, 0]   # scene_id
        buf_i32_view[:, 1] = ints[:, 1]   # camera_id
        buf_i64_view[:, 1] = ts            # timestamp_us
        buf_i32_view[:, 4] = ints[:, 4]   # camera_type_id
        return buf.to(self._device).contiguous()

    def render(
        self,
        scene_ids: torch.Tensor,
        camera_ids: torch.Tensor,
        timestamps_us: torch.Tensor,
        camera_type_ids: torch.Tensor,
        camera_poses: torch.Tensor,
        resolution: Tuple[int, int],
    ) -> torch.Tensor:
        """Render a batch of queries with one Vulkan submission."""
        if self._camera_intrinsics is None:
            raise RuntimeError("call upload_cameras() before render()")
        n = int(scene_ids.shape[0])
        if n == 0:
            h, w = resolution
            return torch.empty((0, h, w, 4), dtype=torch.uint8, device=self._device)

        queries = self._pack_queries(
            scene_ids.to(self._device),
            camera_ids.to(self._device),
            timestamps_us.to(self._device),
            camera_type_ids.to(self._device),
        )
        poses = camera_poses.to(self._device, dtype=torch.float32).contiguous()
        return self._plugin.ludus_timestamped_render_batch(
            self.cpp_wrapper, queries, poses, resolution
        )

    def render_batch(
        self,
        queries: List[Tuple[int, int, int, int]],
        camera_poses: torch.Tensor,
        resolution: Tuple[int, int],
    ) -> torch.Tensor:
        """Render a batch of queries (tuple-based API). Mirrors
        :meth:`LudusCudaTimestampedContext.render_batch`."""
        device = camera_poses.device if camera_poses.is_cuda else self._device

        # Build each column on the host and copy once; per-element GPU writes
        # would force a host/device sync per query.
        def _ts(q):
            ts = q[2]
            return int(ts.item() if isinstance(ts, torch.Tensor) else ts)

        scene_ids = torch.tensor([int(q[0]) for q in queries], dtype=torch.int32, device=device)
        camera_ids = torch.tensor([int(q[1]) for q in queries], dtype=torch.int32, device=device)
        timestamps_us = torch.tensor([_ts(q) for q in queries], dtype=torch.int64, device=device)
        camera_type_ids = torch.tensor(
            [int(q[3]) if len(q) > 3 else 0 for q in queries], dtype=torch.int32, device=device)
        return self.render(
            scene_ids, camera_ids, timestamps_us, camera_type_ids,
            camera_poses, resolution,
        )
