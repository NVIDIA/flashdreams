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

"""Mirror-stitch augmentation for static :class:`SceneBundle` instances.

This is a simplified port of ``ludus_renderer.augmentation.mirror_augment_scene``.
The original handled time-stamped polyline / polygon / cube pools and rebuilt
ego-trajectory and ego-obstacle pools across each segment. The flashdreams
gRPC server only ever exercises the static HD-map path
(``include_ego_trajectory=False``, ``include_ego_obstacle=False``), so the
relevant behaviour reduces to:

1. Pick a mirror plane near the end of the ego trajectory (``lookahead_m``
   metres beyond the last ego position along its forward heading).
2. Reflect every line segment, triangle vertex, and polygon vertex of the
   source scene across that plane.
3. Concatenate the reflected geometry with the original layers.

We repeat steps 2-3 for ``n_mirrors`` iterations, alternating which plane each
new copy is reflected across (so the result is `[orig | mirror | orig | ...]`
just like ludus). For ``n_mirrors == 0`` or no ego pose data, the bundle is
returned unchanged.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import numpy.typing as npt
from loguru import logger
from scipy.spatial.transform import Rotation

from alpadreams.conditioning.slang_renderer.types import (
    SceneBundle,
    WorldLineSegments,
    WorldPolygonList,
    WorldTriangleList,
)
from alpadreams.conditioning.world_scenario.data_types import EgoPose


def mirror_augment_bundle(
    bundle: SceneBundle,
    ego_poses: list[EgoPose],
    *,
    n_mirrors: int = 1,
    lookahead_m: float = 50.0,
) -> SceneBundle:
    """Extend ``bundle`` by mirror-stitching it ``n_mirrors`` times.

    Args:
        bundle: The static SceneBundle to augment.
        ego_poses: Ego trajectory used to derive the first mirror plane. Must
            contain at least two poses.
        n_mirrors: Number of mirror copies to add (total tiles = ``n_mirrors + 1``).
        lookahead_m: Distance past the last ego pose to place the first mirror plane.

    Returns:
        A new SceneBundle containing the original layers plus reflected copies.
        If ``n_mirrors <= 0`` or insufficient ego data is available the original
        bundle is returned.
    """
    if n_mirrors <= 0:
        return bundle
    if len(ego_poses) < 2:
        logger.warning(
            "mirror_augment_bundle: at least 2 ego poses are required to compute a mirror plane; "
            "returning the original bundle unchanged."
        )
        return bundle

    last_position = np.asarray(ego_poses[-1].position, dtype=np.float32)
    forward = _ego_forward_direction(ego_poses)
    if forward is None:
        logger.warning(
            "mirror_augment_bundle: could not determine ego forward direction; "
            "returning the original bundle unchanged."
        )
        return bundle

    line_layers = list(bundle.line_layers)
    triangle_layers = list(bundle.triangle_layers)
    polygon_layers = list(bundle.polygon_layers)

    plane_point = (last_position + forward * float(lookahead_m)).astype(np.float32)
    plane_normal = forward.astype(np.float32)

    for tile_index in range(1, n_mirrors + 1):
        plane_point_iter = (last_position + forward * float(lookahead_m * tile_index)).astype(np.float32)
        line_layers.extend(_reflect_lines(bundle.line_layers, plane_point_iter, plane_normal))
        triangle_layers.extend(_reflect_triangles(bundle.triangle_layers, plane_point_iter, plane_normal))
        polygon_layers.extend(_reflect_polygons(bundle.polygon_layers, plane_point_iter, plane_normal))

    _ = plane_point  # tile_index=1 already covers the initial plane

    return SceneBundle(
        line_layers=tuple(line_layers),
        triangle_layers=tuple(triangle_layers),
        polygon_layers=tuple(polygon_layers),
    )


def _ego_forward_direction(ego_poses: list[EgoPose]) -> npt.NDArray[np.float32] | None:
    last_position = np.asarray(ego_poses[-1].position, dtype=np.float32)
    look_back = min(10, len(ego_poses) - 1)
    if look_back >= 1:
        ref_position = np.asarray(ego_poses[-1 - look_back].position, dtype=np.float32)
        delta = last_position - ref_position
        norm = float(np.linalg.norm(delta))
        if norm > 1e-1:
            return (delta / norm).astype(np.float32)

    quat = np.asarray(ego_poses[-1].orientation, dtype=np.float32)
    if quat.shape != (4,):
        return None
    rotation = Rotation.from_quat(quat).as_matrix().astype(np.float32)
    forward = rotation[:, 0].astype(np.float32)
    norm = float(np.linalg.norm(forward))
    if not math.isfinite(norm) or norm < 1e-6:
        return None
    return (forward / norm).astype(np.float32)


def _reflect_points(
    points_xyz: npt.NDArray[np.float32],
    plane_point: npt.NDArray[np.float32],
    plane_normal: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    if points_xyz.size == 0:
        return points_xyz
    diffs = points_xyz - plane_point
    distances = np.einsum("...i,i->...", diffs, plane_normal).astype(np.float32)
    return (points_xyz - 2.0 * distances[..., None] * plane_normal[None, :]).astype(np.float32)


def _reflect_lines(
    layers: Iterable[WorldLineSegments],
    plane_point: npt.NDArray[np.float32],
    plane_normal: npt.NDArray[np.float32],
) -> list[WorldLineSegments]:
    reflected: list[WorldLineSegments] = []
    for layer in layers:
        segments = np.asarray(layer.segments_world, dtype=np.float32)
        if len(segments) == 0:
            continue
        flat = segments.reshape(-1, 3)
        flat = _reflect_points(flat, plane_point, plane_normal)
        reflected.append(
            WorldLineSegments(
                segments_world=flat.reshape(segments.shape).astype(np.float32),
                color_rgba=layer.color_rgba,
                width_px=layer.width_px,
                layer_name=f"{layer.layer_name}_mirror",
            )
        )
    return reflected


def _reflect_triangles(
    layers: Iterable[WorldTriangleList],
    plane_point: npt.NDArray[np.float32],
    plane_normal: npt.NDArray[np.float32],
) -> list[WorldTriangleList]:
    reflected: list[WorldTriangleList] = []
    for layer in layers:
        triangles = np.asarray(layer.triangles_world, dtype=np.float32)
        if len(triangles) == 0:
            continue
        flat = triangles.reshape(-1, 3)
        flat = _reflect_points(flat, plane_point, plane_normal)
        # Reverse winding so the reflection's triangle normals stay correct.
        new_triangles = flat.reshape(triangles.shape)[:, ::-1, :].copy().astype(np.float32)
        reflected.append(
            WorldTriangleList(
                triangles_world=new_triangles,
                color_rgba=layer.color_rgba,
                layer_name=f"{layer.layer_name}_mirror",
            )
        )
    return reflected


def _reflect_polygons(
    layers: Iterable[WorldPolygonList],
    plane_point: npt.NDArray[np.float32],
    plane_normal: npt.NDArray[np.float32],
) -> list[WorldPolygonList]:
    reflected: list[WorldPolygonList] = []
    for layer in layers:
        new_polygons: list[npt.NDArray[np.float32]] = []
        for polygon in layer.polygons_world:
            polygon_arr = np.asarray(polygon, dtype=np.float32)
            if polygon_arr.shape[0] < 3:
                continue
            mirrored = _reflect_points(polygon_arr, plane_point, plane_normal)
            new_polygons.append(mirrored[::-1].copy().astype(np.float32))
        if not new_polygons:
            continue
        reflected.append(
            WorldPolygonList(
                polygons_world=tuple(new_polygons),
                color_rgba=layer.color_rgba,
                layer_name=f"{layer.layer_name}_mirror",
            )
        )
    return reflected
