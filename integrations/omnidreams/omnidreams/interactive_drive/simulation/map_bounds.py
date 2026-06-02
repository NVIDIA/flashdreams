# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Out-of-bounds detection bounds for interactive_drive.

Stands in for alpasim's GT-trajectory bounds in the interactive case
where the user drives freely instead of replaying a recorded path. The
AABB is the union of *every* spatial layer in the
:class:`SceneBundle` -- ground mesh, lane markers, drivable triangles,
vehicle bbox tracks, polygons -- so it covers the full extent of the
scene's content rather than only the road surface.

The proximity calculation itself is a verbatim port of
:meth:`alpasim_runtime.events.state.is_ego_off_map`: distance from the
AABB+margin edge, ramped over a 100 m warning zone, with a hard ``2.0``
sentinel when the ego has actually crossed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from omnidreams.interactive_drive.types import SceneBundle


@dataclass(frozen=True)
class MapBounds:
    """XY axis-aligned bounding box of the scene's navigable area.

    Built once at scene load time; the runtime loop reads :meth:`proximity`
    once per chunk to drive the OOB warning overlay and auto-respawn.
    Values are plain ``float`` so the per-chunk read path doesn't pay any
    array indexing cost.
    """

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width_m(self) -> float:
        return self.x_max - self.x_min

    @property
    def height_m(self) -> float:
        return self.y_max - self.y_min

    @classmethod
    def from_scene(cls, scene: SceneBundle) -> "MapBounds | None":
        """Compute the union AABB of every spatial layer in ``scene``.

        Returns ``None`` when the scene has no usable spatial content
        (empty fixtures, etc.) -- callers then default to "always
        in-bounds" so the OOB respawn path is a no-op for that scene.
        """
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []

        # Ground mesh vertices. Authoritative for the drivable surface
        # but typically the smallest piece of geometry in the scene.
        if scene.ground_mesh_vertices is not None:
            verts = np.asarray(scene.ground_mesh_vertices, dtype=np.float32)
            if verts.size:
                xs.append(verts[:, 0])
                ys.append(verts[:, 1])

        # Line segments (lane lines, polylines). ``segments_world`` is
        # shaped ``(N, 2, 3)`` (pair of endpoints) per the scene loader,
        # so flatten the first two axes to harvest every endpoint.
        for layer in scene.line_layers:
            seg = np.asarray(layer.segments_world, dtype=np.float32)
            if seg.size:
                flat = seg.reshape(-1, seg.shape[-1])
                xs.append(flat[:, 0])
                ys.append(flat[:, 1])

        # Triangles (drivable surface, intersection plates). ``(N, 3, 3)``;
        # flatten to vertices.
        for layer in scene.triangle_layers:
            tri = np.asarray(layer.triangles_world, dtype=np.float32)
            if tri.size:
                flat = tri.reshape(-1, tri.shape[-1])
                xs.append(flat[:, 0])
                ys.append(flat[:, 1])

        # Polygons -- each entry is its own ``(N, 3)`` ring.
        for layer in scene.polygon_layers:
            for ring in layer.polygons_world:
                pts = np.asarray(ring, dtype=np.float32)
                if pts.size:
                    xs.append(pts[:, 0])
                    ys.append(pts[:, 1])

        # Vehicle track centers (other actors driving through the scene).
        # The scene loader stores them as ``(N, 3)`` per track.
        for track in scene.vehicle_bbox_tracks:
            centers = np.asarray(track.centers_world, dtype=np.float32)
            if centers.size:
                xs.append(centers[:, 0])
                ys.append(centers[:, 1])

        if not xs or not ys:
            return None

        all_x = np.concatenate(xs)
        all_y = np.concatenate(ys)
        return cls(
            x_min=float(all_x.min()),
            y_min=float(all_y.min()),
            x_max=float(all_x.max()),
            y_max=float(all_y.max()),
        )

    def proximity(
        self,
        ego_xy: tuple[float, float],
        *,
        margin_m: float = 50.0,
        warning_zone_m: float = 100.0,
    ) -> float:
        """Distance-from-AABB proximity, matching alpasim's ``is_ego_off_map``.

        Returns:
        - ``0.0`` when the ego is more than ``warning_zone_m`` inside
          the AABB expanded by ``margin_m`` -- solidly in-bounds.
        - ``(0.0, 1.0]`` linearly ramping over the ``warning_zone_m``
          band as the ego approaches the edge.
        - ``2.0`` (alpasim's sentinel) when the ego has actually
          crossed the AABB+margin boundary.
        """
        bx_min = self.x_min - margin_m
        by_min = self.y_min - margin_m
        bx_max = self.x_max + margin_m
        by_max = self.y_max + margin_m

        dist_to_edge = min(
            ego_xy[0] - bx_min,
            bx_max - ego_xy[0],
            ego_xy[1] - by_min,
            by_max - ego_xy[1],
        )
        if dist_to_edge < 0.0:
            return 2.0
        if warning_zone_m > 0.0 and dist_to_edge < warning_zone_m:
            return float(np.clip(1.0 - dist_to_edge / warning_zone_m, 0.0, 1.0))
        return 0.0
