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

"""Convert flashdreams :class:`SceneData` into a :class:`SceneBundle`.

flashdreams already loads CLIPGT parquet into a typed :class:`SceneData` (see
:mod:`alpadreams.conditioning.world_scenario.clipgt_loader`). The Slang
rasterizer consumes a flat, world-space layer representation
(:class:`SceneBundle`), so this module is responsible for that one-time
conversion.

The mapping mirrors what ``ludus_renderer.load_clipgt_scene`` produced and
what ``roaddreams.scene_loader`` builds when reading USDZ archives directly:

* Lane lines are grouped by ``(color, width_px)`` with dash/dual patterning
  applied, then emitted as line layers.
* Road boundaries / wait lines / poles become solid line layers.
* Traffic signs become triangle layers (cuboid plate faces).
* Traffic lights become line layers (cuboid edges).
* Crosswalks / road markings / intersection areas / road islands become
  polygon layers (the rasterizer triangulates them on the CPU at upload time).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import numpy.typing as npt

from alpadreams.conditioning.slang_renderer.colors import (
    HDMAP_V3_COLORS,
    LANE_LINE_STYLE_CONFIG,
)
from alpadreams.conditioning.slang_renderer.config import RasterConfig
from alpadreams.conditioning.slang_renderer.math3d import quaternion_to_matrix_xyzw
from alpadreams.conditioning.slang_renderer.patterns import (
    apply_pattern,
    concatenate_segments,
    resample_polyline,
    segments_from_polyline,
    split_segment_runs,
    subdivide_polyline,
    triangulate_polygon_fan,
)
from alpadreams.conditioning.slang_renderer.types import (
    SceneBundle,
    WorldLineSegments,
    WorldPolygonList,
    WorldTriangleList,
)
from alpadreams.conditioning.world_scenario.data_types import (
    PolygonElement,
    PolylineElement,
    SceneData,
)


def scene_data_to_bundle(scene_data: SceneData, raster: RasterConfig) -> SceneBundle:
    """Convert a flashdreams :class:`SceneData` into a :class:`SceneBundle`."""
    line_layers: list[WorldLineSegments] = []
    triangle_layers: list[WorldTriangleList] = []
    polygon_layers: list[WorldPolygonList] = []

    line_layers.extend(_build_lane_segments(scene_data.lane_lines, raster))

    if scene_data.road_boundaries:
        line_layers.append(
            _build_polyline_layer(
                scene_data.road_boundaries,
                layer_name="road_boundaries",
                raster=raster,
                width_px=raster.line_width_px,
            )
        )
    if scene_data.wait_lines:
        line_layers.append(
            _build_polyline_layer(
                scene_data.wait_lines,
                layer_name="wait_lines",
                raster=raster,
                width_px=raster.line_width_px,
            )
        )
    if scene_data.poles:
        line_layers.append(
            _build_polyline_layer(
                scene_data.poles,
                layer_name="poles",
                raster=raster,
                width_px=raster.pole_width_px,
            )
        )
    if scene_data.traffic_signs:
        triangle_layers.append(
            _build_oriented_box_face_layer(
                scene_data.traffic_signs, layer_name="traffic_signs"
            )
        )
    if scene_data.traffic_lights:
        line_layers.append(
            _build_oriented_box_edge_layer(
                scene_data.traffic_lights,
                layer_name="traffic_lights",
                width_px=raster.line_width_px,
            )
        )
    if scene_data.crosswalks:
        polygon_layers.append(
            _build_polygon_layer(scene_data.crosswalks, layer_name="crosswalks")
        )
    if scene_data.road_markings:
        polygon_layers.append(
            _build_polygon_layer(scene_data.road_markings, layer_name="road_markings")
        )
    if scene_data.intersection_areas:
        polygon_layers.append(
            _build_polygon_layer(
                scene_data.intersection_areas, layer_name="intersection_areas"
            )
        )
    if scene_data.road_islands:
        polygon_layers.append(
            _build_polygon_layer(scene_data.road_islands, layer_name="road_islands")
        )

    return SceneBundle(
        line_layers=tuple(line_layers),
        triangle_layers=tuple(triangle_layers),
        polygon_layers=tuple(polygon_layers),
    )


def _coarsen_segment_group(
    segments_world: npt.NDArray[np.float32], interval_m: float
) -> npt.NDArray[np.float32]:
    coarsened_runs: list[npt.NDArray[np.float32]] = []
    for run in split_segment_runs(segments_world):
        if len(run) <= 2:
            coarsened_runs.append(segments_from_polyline(run))
            continue
        sampled = resample_polyline(run, interval_m=interval_m)
        coarsened_runs.append(segments_from_polyline(sampled))
    return concatenate_segments(coarsened_runs)


def _is_finite_polyline(points: npt.NDArray[np.float32], min_count: int) -> bool:
    """True if ``points`` has at least ``min_count`` rows and is NaN/Inf-free.

    A SINGLE non-finite value in a GPU vertex buffer poisons the whole raster
    pass (NaN-vs-finite comparisons in the shader fail in ways that paint
    unbounded regions, see commit
    https://gitlab-master.nvidia.com/sil/omni-dreams/-/commit/92091220).
    flashdreams' upstream :class:`ClipGTLoader` already guards most paths,
    but we re-check here so a single stray null in a parquet row can't
    destroy the entire conditioning frame.
    """
    if points.ndim != 2 or points.shape[0] < min_count:
        return False
    return bool(np.all(np.isfinite(points)))


def _build_lane_segments(lane_lines: list[Any], raster: RasterConfig) -> list[WorldLineSegments]:
    grouped_segments: dict[tuple[tuple[float, float, float, float], float], list[npt.NDArray[np.float32]]] = (
        defaultdict(list)
    )
    grouped_names: dict[tuple[tuple[float, float, float, float], float], set[str]] = defaultdict(set)
    skipped = 0

    for lane_line in lane_lines:
        polyline = np.asarray(lane_line.points, dtype=np.float32)
        if not _is_finite_polyline(polyline, min_count=2):
            skipped += 1
            continue
        type_hint = lane_line.lane_type.canonical_name
        config = LANE_LINE_STYLE_CONFIG.get(type_hint, LANE_LINE_STYLE_CONFIG["OTHER"])
        subdivided = subdivide_polyline(polyline, raster.lane_segment_interval_m)
        base_segments = segments_from_polyline(subdivided)
        patterned_segments = apply_pattern(
            base_segments,
            pattern=str(config.get("pattern", "solid")),
            dual_pattern=config.get("dual_pattern"),  # type: ignore[arg-type]
            dual_offset_m=raster.dual_line_offset_m,
        )
        color_rgba = tuple(float(value) for value in config["color"])  # type: ignore[arg-type]
        width_px = raster.line_width_px * float(config.get("width_scale", 1.0))
        group_key = (color_rgba, width_px)
        for group in patterned_segments:
            if len(group) == 0 or not bool(np.all(np.isfinite(group))):
                continue
            grouped_segments[group_key].append(
                _coarsen_segment_group(group, interval_m=raster.polyline_segment_interval_m)
            )
            grouped_names[group_key].add(type_hint)

    if skipped:
        print(
            f"[scene_loader] lanelines: skipped {skipped} of {len(lane_lines)} "
            "polyline(s) with NaN/Inf vertices.",
            flush=True,
        )

    layers: list[WorldLineSegments] = []
    for key, segment_groups in grouped_segments.items():
        color_rgba, width_px = key
        style_names = "+".join(sorted(name.lower().replace(" ", "_") for name in grouped_names[key]))
        merged = concatenate_segments(segment_groups)
        if len(merged) == 0 or not bool(np.all(np.isfinite(merged))):
            continue
        layers.append(
            WorldLineSegments(
                segments_world=merged,
                color_rgba=color_rgba,
                width_px=width_px,
                layer_name=f"lanelines_{style_names}",
            )
        )
    return layers


def _build_polyline_layer(
    polylines: list[PolylineElement],
    *,
    layer_name: str,
    raster: RasterConfig,
    width_px: float,
) -> WorldLineSegments:
    segment_groups: list[npt.NDArray[np.float32]] = []
    skipped = 0
    for polyline_obj in polylines:
        polyline = np.asarray(polyline_obj.points, dtype=np.float32)
        if not _is_finite_polyline(polyline, min_count=2):
            skipped += 1
            continue
        subdivided = subdivide_polyline(polyline, raster.polyline_segment_interval_m)
        line_segments = segments_from_polyline(subdivided)
        if len(line_segments) > 0 and bool(np.all(np.isfinite(line_segments))):
            segment_groups.append(line_segments)
    if skipped:
        print(
            f"[scene_loader] {layer_name}: skipped {skipped} of {len(polylines)} "
            "polyline(s) with NaN/Inf vertices.",
            flush=True,
        )
    return WorldLineSegments(
        segments_world=concatenate_segments(segment_groups),
        color_rgba=HDMAP_V3_COLORS[layer_name],
        width_px=width_px,
        layer_name=layer_name,
    )


def _build_polygon_layer(
    polygons: list[PolygonElement],
    *,
    layer_name: str,
) -> WorldPolygonList:
    polygons_world: list[npt.NDArray[np.float32]] = []
    skipped = 0
    for polygon_obj in polygons:
        vertices = np.asarray(polygon_obj.vertices, dtype=np.float32)
        if not _is_finite_polyline(vertices, min_count=3):
            skipped += 1
            continue
        if np.linalg.norm(vertices[0] - vertices[-1]) <= 1e-4:
            vertices = vertices[:-1]
        if vertices.shape[0] < 3:
            continue
        polygons_world.append(vertices.astype(np.float32))
    if skipped:
        print(
            f"[scene_loader] {layer_name}: skipped {skipped} of {len(polygons)} "
            "polygon(s) with NaN/Inf vertices.",
            flush=True,
        )
    return WorldPolygonList(
        polygons_world=tuple(polygons_world),
        color_rgba=HDMAP_V3_COLORS[layer_name],
        layer_name=layer_name,
    )


def _cuboid_corners(box: Any) -> npt.NDArray[np.float32]:
    center = np.asarray(box.center, dtype=np.float32)
    dimensions = np.asarray(box.dimensions, dtype=np.float32)
    quat = (
        float(box.orientation[0]),
        float(box.orientation[1]),
        float(box.orientation[2]),
        float(box.orientation[3]),
    )
    rotation = quaternion_to_matrix_xyzw(quat)
    half = dimensions * 0.5
    corners = np.array(
        [
            [-half[0], -half[1], -half[2]],
            [half[0], -half[1], -half[2]],
            [half[0], half[1], -half[2]],
            [-half[0], half[1], -half[2]],
            [-half[0], -half[1], half[2]],
            [half[0], -half[1], half[2]],
            [half[0], half[1], half[2]],
            [-half[0], half[1], half[2]],
        ],
        dtype=np.float32,
    )
    return ((corners @ rotation.T) + center).astype(np.float32)


def _box_has_null_geometry(box: Any) -> bool:
    """True if any field of an OrientedBoxElement is NaN/None.

    Some CLIPGT scenes ship rows whose center / dimensions / orientation
    are entirely null because upstream pose-fitting failed to localise
    the feature. NumPy silently casts those nulls to NaN, and a single
    NaN cuboid in a GPU primitive buffer corrupts the entire raster pass
    (NaN-vs-finite comparisons fail, lane lines and road boundaries
    collapse to noise specs). Skip such rows here.

    See https://gitlab-master.nvidia.com/sil/omni-dreams/-/commit/92091220
    for the original report on clipgt-065dcac9-...
    """
    for arr in (box.center, box.dimensions, box.orientation):
        a = np.asarray(arr, dtype=np.float32)
        if a.size == 0 or not np.all(np.isfinite(a)):
            return True
    return False


def _build_oriented_box_face_layer(
    boxes: list[Any], *, layer_name: str
) -> WorldTriangleList:
    triangles: list[npt.NDArray[np.float32]] = []
    skipped = 0
    for box in boxes:
        if _box_has_null_geometry(box):
            skipped += 1
            continue
        corners = _cuboid_corners(box)
        thinnest_axis = int(np.argmin(np.asarray(box.dimensions, dtype=np.float32)))
        face_indices_by_axis = {
            0: ((0, 3, 7, 4), (1, 2, 6, 5)),
            1: ((0, 1, 5, 4), (3, 2, 6, 7)),
            2: ((0, 1, 2, 3), (4, 5, 6, 7)),
        }
        quads = [
            corners[np.array(indices, dtype=np.int32)]
            for indices in face_indices_by_axis[thinnest_axis]
        ]
        triangles.append(
            np.concatenate([triangulate_polygon_fan(quad) for quad in quads], axis=0).astype(np.float32)
        )
    if skipped:
        print(
            f"[scene_loader] {layer_name}: skipped {skipped} of {len(boxes)} "
            "row(s) with null/NaN center/dimensions/orientation. "
            "Upstream dataset bug (missing pose-fit for some features).",
            flush=True,
        )
    triangles_world = (
        np.concatenate(triangles, axis=0).astype(np.float32)
        if triangles
        else np.empty((0, 3, 3), dtype=np.float32)
    )
    return WorldTriangleList(
        triangles_world=triangles_world,
        color_rgba=HDMAP_V3_COLORS[layer_name],
        layer_name=layer_name,
    )


def _build_oriented_box_edge_layer(
    boxes: list[Any], *, layer_name: str, width_px: float
) -> WorldLineSegments:
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    groups: list[npt.NDArray[np.float32]] = []
    skipped = 0
    for box in boxes:
        if _box_has_null_geometry(box):
            skipped += 1
            continue
        corners = _cuboid_corners(box)
        groups.append(
            np.array([[corners[a], corners[b]] for a, b in edges], dtype=np.float32)
        )
    if skipped:
        print(
            f"[scene_loader] {layer_name}: skipped {skipped} of {len(boxes)} "
            "cuboid(s) with null/NaN center/dimensions/orientation. "
            "Upstream dataset bug (missing pose-fit for some features).",
            flush=True,
        )
    return WorldLineSegments(
        segments_world=concatenate_segments(groups),
        color_rgba=HDMAP_V3_COLORS[layer_name],
        width_px=width_px,
        layer_name=layer_name,
    )
