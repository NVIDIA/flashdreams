# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Ground-snap physics for interactive_drive.

After kinematic integration produces ``(x, y, yaw)``, this module re-aligns
``z + pitch + roll`` so the ego sits on top of the ground mesh shipped in the
USDZ as ``mesh_ground.ply``. Mirrors the alpasim physics service
(:mod:`alpasim_physics.backend.PhysicsBackend.update_pose`) using a numpy-only
CPU vertical raycaster so interactive_drive keeps a small dependency surface and
works on the ``--backend raster`` (no GPU) path too.
"""

import logging
import math

import numpy as np
import numpy.typing as npt
from omnidreams.interactive_drive.config import VehicleConfig
from omnidreams.interactive_drive.types import VehicleState

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int32]


class GroundSnapper:
    def __init__(
        self,
        vertices_xyz: FloatArray,
        faces_ijk: IntArray,
        *,
        grid_resolution_m: float = 2.0,
        max_translation_m: float = 1.5,
        max_rotation_deg: float = 10.0,
        num_sample_points: int = 16,
        min_intersections: int = 6,
    ) -> None:
        if vertices_xyz.ndim != 2 or vertices_xyz.shape[1] != 3:
            raise ValueError(f"vertices must be (N, 3), got {vertices_xyz.shape}")
        if faces_ijk.ndim != 2 or faces_ijk.shape[1] != 3:
            raise ValueError(f"faces must be (M, 3), got {faces_ijk.shape}")
        if vertices_xyz.shape[0] == 0 or faces_ijk.shape[0] == 0:
            raise ValueError("vertices and faces must be non-empty")

        self._max_translation_m = float(max_translation_m)
        self._max_rotation_rad = math.radians(max_rotation_deg)
        self._num_sample_points = int(num_sample_points)
        self._min_intersections = int(min_intersections)
        self._anchor_offset_m: float | None = None
        # Updated as a side effect of every ``snap()`` call. ``snap_hit_ratio``
        # is the fraction of body-grid sample rays that found ground in
        # the most recent call; the runtime loop uses it as a *quality*
        # signal for the snap itself (does it apply or bail out), NOT
        # as the OOB metric. OOB is computed separately by
        # :meth:`oob_proximity_at` from the ego's XY against the mesh
        # AABB -- mirroring alpasim's ``is_ego_off_map`` algorithm,
        # where the body-grid hit count would false-positive on every
        # sidewalk or curb the ego brushes against.
        self._last_snap_hit_ratio: float = 1.0

        vertices_d = np.asarray(vertices_xyz, dtype=np.float64)
        faces_i = np.asarray(faces_ijk, dtype=np.int32)
        if int(faces_i.max()) >= vertices_d.shape[0] or int(faces_i.min()) < 0:
            raise ValueError("face indices out of range for given vertex array")
        tri_vertices = vertices_d[faces_i]

        a = tri_vertices[:, 0, :]
        b = tri_vertices[:, 1, :]
        c = tri_vertices[:, 2, :]
        self._a_x = a[:, 0]
        self._a_y = a[:, 1]
        self._a_z = a[:, 2]
        self._b_z = b[:, 2]
        self._c_z = c[:, 2]
        self._v0x = b[:, 0] - a[:, 0]
        self._v0y = b[:, 1] - a[:, 1]
        self._v1x = c[:, 0] - a[:, 0]
        self._v1y = c[:, 1] - a[:, 1]
        denom = self._v0x * self._v1y - self._v1x * self._v0y
        self._inv_denom = np.where(
            np.abs(denom) >= 1e-12, 1.0 / np.where(denom != 0, denom, 1.0), 0.0
        )
        self._denom_valid = np.abs(denom) >= 1e-12

        self._grid_resolution_m = float(grid_resolution_m)
        tri_xy = tri_vertices[:, :, :2]
        self._tri_xy_min = tri_xy.min(axis=1)
        self._tri_xy_max = tri_xy.max(axis=1)
        self._grid_origin = self._tri_xy_min.min(axis=0)
        grid_extent = self._tri_xy_max.max(axis=0) - self._grid_origin
        self._grid_shape = (
            max(1, int(np.ceil(grid_extent[0] / self._grid_resolution_m))),
            max(1, int(np.ceil(grid_extent[1] / self._grid_resolution_m))),
        )
        cell_buckets: dict[tuple[int, int], list[int]] = {}
        for tri_idx in range(faces_i.shape[0]):
            i_min = self._cell_x(self._tri_xy_min[tri_idx, 0])
            i_max = self._cell_x(self._tri_xy_max[tri_idx, 0])
            j_min = self._cell_y(self._tri_xy_min[tri_idx, 1])
            j_max = self._cell_y(self._tri_xy_max[tri_idx, 1])
            for i in range(i_min, i_max + 1):
                for j in range(j_min, j_max + 1):
                    cell_buckets.setdefault((i, j), []).append(tri_idx)
        self._cell_candidates: dict[tuple[int, int], np.ndarray] = {
            cell: np.asarray(idxs, dtype=np.int32)
            for cell, idxs in cell_buckets.items()
        }

    @property
    def anchor_offset_m(self) -> float | None:
        return self._anchor_offset_m

    @property
    def last_snap_hit_ratio(self) -> float:
        """Fraction of body-grid sample rays that hit ground in the latest ``snap()``.

        ``1.0`` means every ray landed on a ground triangle (clean snap);
        ``0.0`` means none did. Useful for diagnosing snap quality on
        sparse meshes; **not** the OOB signal -- see
        :meth:`oob_proximity_at` for that.
        """
        return self._last_snap_hit_ratio

    def _cell_x(self, x: float) -> int:
        i = int((x - self._grid_origin[0]) / self._grid_resolution_m)
        return max(0, min(self._grid_shape[0] - 1, i))

    def _cell_y(self, y: float) -> int:
        j = int((y - self._grid_origin[1]) / self._grid_resolution_m)
        return max(0, min(self._grid_shape[1] - 1, j))

    def _ground_z_at(self, x: float, y: float, z_ref: float) -> float | None:
        if (
            x < self._grid_origin[0]
            or y < self._grid_origin[1]
            or x > self._grid_origin[0] + self._grid_shape[0] * self._grid_resolution_m
            or y > self._grid_origin[1] + self._grid_shape[1] * self._grid_resolution_m
        ):
            return None
        candidates = self._cell_candidates.get((self._cell_x(x), self._cell_y(y)))
        if candidates is None or candidates.size == 0:
            return None
        v2x = x - self._a_x[candidates]
        v2y = y - self._a_y[candidates]
        v0y_c = self._v0y[candidates]
        v1x_c = self._v1x[candidates]
        v0x_c = self._v0x[candidates]
        v1y_c = self._v1y[candidates]
        inv = self._inv_denom[candidates]
        v = (v2x * v1y_c - v1x_c * v2y) * inv
        w = (v0x_c * v2y - v2x * v0y_c) * inv
        u = 1.0 - v - w
        eps = 1e-6
        inside = self._denom_valid[candidates] & (u >= -eps) & (v >= -eps) & (w >= -eps)
        if not bool(inside.any()):
            return None
        z = (
            u * self._a_z[candidates]
            + v * self._b_z[candidates]
            + w * self._c_z[candidates]
        )
        z_inside = z[inside]
        best = int(np.argmin(np.abs(z_inside - z_ref)))
        return float(z_inside[best])

    def _sample_body_grid(self, vehicle: VehicleConfig) -> np.ndarray:
        n = max(2, int(np.ceil(self._num_sample_points**0.5)))
        p = np.linspace(0.0, 1.0, n)
        u, v = np.meshgrid(p, p)
        return np.column_stack(
            [
                u.ravel() * vehicle.aabb_length_m - vehicle.aabb_length_m * 0.5,
                v.ravel() * vehicle.aabb_width_m - vehicle.aabb_width_m * 0.5,
                np.full(u.size, -vehicle.aabb_height_m * 0.5),
            ]
        )

    def snap(self, state: VehicleState, vehicle: VehicleConfig) -> VehicleState:
        body_pts = self._sample_body_grid(vehicle)
        rot = _euler_to_rotation(state.yaw_rad, state.pitch_rad, state.roll_rad)
        world_pts = body_pts @ rot.T + np.array(
            [state.x_m, state.y_m, state.z_m], dtype=np.float64
        )
        ground_zs = np.array(
            [self._raycast(float(p[0]), float(p[1]), float(p[2])) for p in world_pts],
            dtype=np.float64,
        )
        mask = ~np.isnan(ground_zs)
        n_hits = int(mask.sum())
        n_total = int(len(world_pts))
        # Track snap quality (hit ratio) for diagnostics; OOB proximity
        # is computed separately by ``oob_proximity_at`` from the ego XY
        # against the mesh AABB.
        self._last_snap_hit_ratio = (
            float(n_hits) / float(n_total) if n_total > 0 else 1.0
        )
        if n_hits < self._min_intersections:
            logger.debug(
                "ground snap: %d/%d sample rays hit, below min=%d; passing through",
                n_hits,
                n_total,
                self._min_intersections,
            )
            return state
        ground_pts = np.column_stack(
            [world_pts[mask, 0], world_pts[mask, 1], ground_zs[mask]]
        )
        try:
            centroid_g, normal_g = _fit_plane(ground_pts.T)
        except _InsufficientPoints:
            return state
        if normal_g[2] < 0.0:
            normal_g = -normal_g
        local_ground_z = float(
            centroid_g[2]
            - (
                normal_g[0] * (state.x_m - centroid_g[0])
                + normal_g[1] * (state.y_m - centroid_g[1])
            )
            / normal_g[2]
        )
        if self._anchor_offset_m is None:
            self._anchor_offset_m = float(state.z_m) - local_ground_z
        new_z = local_ground_z + self._anchor_offset_m
        cy = math.cos(state.yaw_rad)
        sy = math.sin(state.yaw_rad)
        target_x = float(cy * normal_g[0] + sy * normal_g[1])
        target_y = float(-sy * normal_g[0] + cy * normal_g[1])
        target_z = float(normal_g[2])
        new_roll = -math.asin(max(-1.0, min(1.0, target_y)))
        new_pitch = math.atan2(target_x, target_z)
        delta_z = abs(new_z - state.z_m)
        if delta_z > self._max_translation_m:
            return state
        delta_rot = max(
            abs(new_pitch - state.pitch_rad), abs(new_roll - state.roll_rad)
        )
        if delta_rot > self._max_rotation_rad:
            return state
        return VehicleState(
            x_m=state.x_m,
            y_m=state.y_m,
            z_m=float(new_z),
            yaw_rad=state.yaw_rad,
            speed_mps=state.speed_mps,
            steer_rad=state.steer_rad,
            pitch_rad=float(new_pitch),
            roll_rad=float(new_roll),
        )

    def _raycast(self, x: float, y: float, z_ref: float) -> float:
        z = self._ground_z_at(x, y, z_ref)
        return float("nan") if z is None else z


class _InsufficientPoints(Exception):
    pass


def _euler_to_rotation(yaw_rad: float, pitch_rad: float, roll_rad: float) -> np.ndarray:
    cr, sr = math.cos(roll_rad), math.sin(roll_rad)
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _fit_plane(points_3xn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if points_3xn.shape[0] != 3:
        raise ValueError(f"points must have shape (3, N), got {points_3xn.shape}")
    if points_3xn.shape[1] < 3:
        raise _InsufficientPoints(
            f"Need >=3 points to fit a plane, got {points_3xn.shape[1]}"
        )
    centroid = points_3xn.mean(axis=1)
    centered = points_3xn - centroid[:, np.newaxis]
    M = centered @ centered.T
    return centroid, np.linalg.svd(M)[0][:, -1]
