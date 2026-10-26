# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic first-person fallback renders for semantic-map spawns."""

from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from omnidreams_game_engine.camera_defaults import (
    DEFAULT_FIRST_FRAME_RESOLUTION_WH,
    default_front_camera_calibration,
)
from omnidreams_game_engine.game_map.types import (
    GameMapSpawn,
    ResolvedGameMap,
)
from omnidreams_game_engine.math3d import rig_pose_from_state

SPAWN_RENDERER_VERSION = "2"
"""Version included in compiled-map cache keys for fallback rendering."""

_MAX_GROUND_DISTANCE_M = 600.0
_PAINT_WIDTH_M = 0.12
_BOUNDARY_WIDTH_M = 0.10
_CURB_WIDTH_M = 0.28

_SKY_TOP_RGB = np.asarray([104, 154, 202], dtype=np.float32)
_SKY_HORIZON_RGB = np.asarray([208, 222, 226], dtype=np.float32)
_TERRAIN_RGB = np.asarray([116, 125, 83], dtype=np.float32)
_ROAD_RGB = np.asarray([64, 67, 69], dtype=np.uint8)
_PARKING_RGB = np.asarray([72, 75, 76], dtype=np.uint8)
_BOUNDARY_RGB = np.asarray([52, 54, 55], dtype=np.uint8)
_CURB_RGB = np.asarray([151, 151, 145], dtype=np.uint8)
_WHITE_PAINT_RGB = np.asarray([226, 225, 213], dtype=np.uint8)
_YELLOW_PAINT_RGB = np.asarray([222, 177, 47], dtype=np.uint8)


def _camera_mesh_distances(
    directions: np.ndarray, vertices: np.ndarray, faces: np.ndarray
) -> np.ndarray:
    """Return nearest mesh-hit distances for camera-local FLU rays.

    Clip triangle bounds at the camera's forward plane before indexing their
    perspective projections. Exact ray/triangle tests resolve visibility inside
    those bounds, including ground rising above the camera's horizon.

    Args:
        directions: Unit ray directions shaped ``[N, 3]``.
        vertices: Camera-local mesh vertices shaped ``[V, 3]``.
        faces: Triangle vertex indices shaped ``[F, 3]``.

    Returns:
        Distances along each ray, or infinity for rays missing the mesh.
    """
    triangles = np.asarray(vertices, dtype=np.float64)[faces]
    near = 1e-4
    front = triangles[:, :, 0] >= near
    visible = front.any(axis=1)
    triangles = triangles[visible]
    front = front[visible]
    distances = np.full(len(directions), np.inf)
    if not len(triangles):
        return distances

    bounds = np.empty((len(triangles), 4))
    whole = front.all(axis=1)
    projected = triangles[whole, :, 1:] / triangles[whole, :, :1]
    bounds[whole, :2] = projected.min(axis=1)
    bounds[whole, 2:] = projected.max(axis=1)
    for index in np.flatnonzero(~whole):
        triangle = triangles[index]
        clipped = list(triangle[front[index]])
        for start, end in zip(triangle, np.roll(triangle, -1, axis=0), strict=True):
            if (start[0] >= near) != (end[0] >= near):
                clipped.append(
                    start + (end - start) * ((near - start[0]) / (end[0] - start[0]))
                )
        polygon = np.asarray(clipped)
        projected = polygon[:, 1:] / polygon[:, :1]
        bounds[index, :2] = projected.min(axis=0)
        bounds[index, 2:] = projected.max(axis=0)
    index = shapely.STRtree(shapely.box(*bounds.T))
    starts = triangles[:, 0]
    edge1 = triangles[:, 1] - starts
    edge2 = triangles[:, 2] - starts
    q = np.cross(-starts, edge1)
    numerators = np.einsum("ij,ij->i", edge2, q)
    forward = np.flatnonzero(directions[:, 0] > 0)
    # ponytail: Batch candidate pairs to bound memory at full seed resolution;
    # a native BVH would be appropriate for substantially denser scene meshes.
    for begin in range(0, len(forward), 16384):
        rays = forward[begin : begin + 16384]
        points = directions[rays, 1:] / directions[rays, :1]
        ray_ids, triangle_ids = index.query(shapely.points(points))
        if not len(ray_ids):
            continue
        ray_ids = rays[ray_ids]
        p = np.cross(directions[ray_ids], edge2[triangle_ids])
        determinant = np.einsum("ij,ij->i", edge1[triangle_ids], p)
        inverse = np.divide(
            1.0,
            determinant,
            out=np.zeros_like(determinant),
            where=np.abs(determinant) > 1e-12,
        )
        u = np.einsum("ij,ij->i", -starts[triangle_ids], p) * inverse
        v = np.einsum("ij,ij->i", directions[ray_ids], q[triangle_ids]) * inverse
        distance = numerators[triangle_ids] * inverse
        hit = (
            (np.abs(determinant) > 1e-12)
            & (u >= -1e-7)
            & (v >= -1e-7)
            & (u + v <= 1 + 1e-7)
            & (distance > 0)
            & (distance <= _MAX_GROUND_DISTANCE_M)
        )
        np.minimum.at(distances, ray_ids[hit], distance[hit])
    return distances


def _camera_ground_intersections(
    spawn: GameMapSpawn,
    width: int,
    height: int,
    game_map: ResolvedGameMap | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    calibration = default_front_camera_calibration()
    u, v = np.meshgrid(
        np.arange(width, dtype=np.float32) + np.float32(0.5),
        np.arange(height, dtype=np.float32) + np.float32(0.5),
    )
    pixels = np.column_stack((u.reshape(-1), v.reshape(-1)))

    scale = np.asarray(
        [width / calibration.width, height / calibration.height], dtype=np.float32
    )
    native_pixels = pixels / scale
    relative = native_pixels - np.asarray(
        [calibration.cx, calibration.cy], dtype=np.float32
    )
    relative = (
        relative
        @ np.linalg.inv(
            np.asarray(
                [
                    [calibration.linear_cde[0], calibration.linear_cde[1]],
                    [calibration.linear_cde[2], 1.0],
                ],
                dtype=np.float32,
            )
        ).T
    )
    radius = np.linalg.norm(relative, axis=1).astype(np.float32)
    angle = np.zeros_like(radius)
    for power, coefficient in enumerate(calibration.polynomial):
        angle += np.float32(coefficient) * np.power(radius, power, dtype=np.float32)
    sin_angle = np.sin(angle).astype(np.float32)
    radial_scale = np.divide(
        sin_angle,
        np.maximum(radius, np.float32(1.0e-6)),
        out=np.zeros_like(radius),
        where=radius > np.float32(1.0e-6),
    )
    directions_rdf = np.column_stack(
        (
            relative[:, 0] * radial_scale,
            relative[:, 1] * radial_scale,
            np.cos(angle),
        )
    ).astype(np.float32)
    directions_sensor_flu = np.column_stack(
        (
            directions_rdf[:, 2],
            -directions_rdf[:, 0],
            -directions_rdf[:, 1],
        )
    ).astype(np.float32)

    rig_to_world = rig_pose_from_state(
        float(spawn.position_world[0]),
        float(spawn.position_world[1]),
        float(spawn.position_world[2]),
        spawn.yaw_rad,
        pitch_rad=spawn.pitch_rad,
    )
    sensor_to_world = rig_to_world @ calibration.sensor_to_rig_flu
    directions_world = directions_sensor_flu @ sensor_to_world[:3, :3].T
    origin_world = sensor_to_world[:3, 3]
    ground_z = float(spawn.position_world[2])
    distance_along_ray = np.divide(
        np.float32(ground_z) - origin_world[2],
        directions_world[:, 2],
        out=np.full(len(directions_world), np.float32(-1.0)),
        where=np.abs(directions_world[:, 2]) > np.float32(1.0e-6),
    )
    if game_map is not None and np.ptp(game_map.ground_vertices[:, 2]) > 0:
        vertices_camera = (game_map.ground_vertices - origin_world) @ sensor_to_world[
            :3, :3
        ]
        distance_along_ray = _camera_mesh_distances(
            directions_sensor_flu, vertices_camera, game_map.ground_faces
        )
        distance_along_ray = np.where(
            np.isfinite(distance_along_ray), distance_along_ray, -1.0
        )
    valid = (
        (directions_sensor_flu[:, 0] > 0.0)
        & (distance_along_ray > 0.0)
        & (distance_along_ray <= _MAX_GROUND_DISTANCE_M)
    )
    points_world = origin_world[None, :] + (
        directions_world * distance_along_ray[:, None]
    )
    planar_distance = np.linalg.norm(points_world[:, :2] - origin_world[:2], axis=1)
    return points_world[:, 0], points_world[:, 1], planar_distance, valid


def _polygon_geometry(polygons: list[np.ndarray]) -> BaseGeometry:
    geometries = [
        Polygon(np.asarray(points, dtype=np.float64)[:, :2])
        for points in polygons
        if len(points) >= 3
    ]
    return unary_union(geometries) if geometries else Polygon()


def _line_geometry(polylines: list[np.ndarray], width_m: float) -> BaseGeometry:
    geometries = [
        LineString(np.asarray(points, dtype=np.float64)[:, :2]).buffer(
            width_m * 0.5, cap_style="flat", join_style="round"
        )
        for points in polylines
        if len(points) >= 2
    ]
    return unary_union(geometries) if geometries else Polygon()


def _paint(
    image_flat: np.ndarray,
    ground_indices: np.ndarray,
    ground_x: np.ndarray,
    ground_y: np.ndarray,
    geometry: BaseGeometry,
    color: np.ndarray,
) -> None:
    if geometry.is_empty:
        return
    covered = shapely.intersects_xy(geometry, ground_x, ground_y)
    image_flat[ground_indices[covered]] = color


def render_spawn_first_frame(
    game_map: ResolvedGameMap,
    spawn: GameMapSpawn,
    *,
    resolution_wh: tuple[int, int] = DEFAULT_FIRST_FRAME_RESOLUTION_WH,
) -> np.ndarray:
    """Render a deterministic synthetic first frame aligned to ``spawn``.

    Args:
        game_map: Resolved semantic map containing drawable surfaces and lines.
        spawn: Spawn supplying the camera position and heading.
        resolution_wh: Output resolution as ``(width, height)``.

    Returns:
        RGB image with shape ``[height, width, 3]`` and dtype ``uint8``.
    """
    width, height = (int(resolution_wh[0]), int(resolution_wh[1]))
    if width <= 0 or height <= 0:
        raise ValueError(f"resolution_wh must be positive, got {resolution_wh!r}")

    vertical = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    sky = (
        _SKY_TOP_RGB[None, None, :] * (1.0 - vertical)
        + _SKY_HORIZON_RGB[None, None, :] * vertical
    )
    image = np.broadcast_to(sky, (height, width, 3)).copy()

    ground_x_all, ground_y_all, distance_all, valid_ground = (
        _camera_ground_intersections(spawn, width, height, game_map)
    )
    flat = image.reshape(-1, 3)
    ground_indices = np.flatnonzero(valid_ground)
    ground_x = ground_x_all[valid_ground]
    ground_y = ground_y_all[valid_ground]
    distance = distance_all[valid_ground]
    terrain_shade = np.clip(1.0 - distance / 1400.0, 0.72, 1.0)[:, None]
    texture = 3.0 * np.sin(ground_x[:, None] * 0.37) * np.cos(ground_y[:, None] * 0.29)
    flat[ground_indices] = np.clip(
        _TERRAIN_RGB[None, :] * terrain_shade + texture, 0.0, 255.0
    )

    parking_surfaces = [
        element.surface_world
        for element in game_map.elements
        if element.element_type == "parking_lot"
    ]
    road_surfaces = [
        element.surface_world
        for element in game_map.elements
        if element.element_type != "parking_lot"
    ]
    _paint(
        flat,
        ground_indices,
        ground_x,
        ground_y,
        _polygon_geometry(road_surfaces),
        _ROAD_RGB,
    )
    _paint(
        flat,
        ground_indices,
        ground_x,
        ground_y,
        _polygon_geometry(parking_surfaces),
        _PARKING_RGB,
    )

    boundaries = [
        boundary.polyline_world
        for element in game_map.elements
        for boundary in element.road_boundaries
    ]
    curbs = [
        curb.polyline_world for element in game_map.elements for curb in element.curbs
    ]
    _paint(
        flat,
        ground_indices,
        ground_x,
        ground_y,
        _line_geometry(boundaries, _BOUNDARY_WIDTH_M),
        _BOUNDARY_RGB,
    )
    _paint(
        flat,
        ground_indices,
        ground_x,
        ground_y,
        _line_geometry(curbs, _CURB_WIDTH_M),
        _CURB_RGB,
    )

    _paint(
        flat,
        ground_indices,
        ground_x,
        ground_y,
        _polygon_geometry(list(game_map.road_marking_polygons_world)),
        _WHITE_PAINT_RGB,
    )
    for color_name, color in (
        ("WHITE", _WHITE_PAINT_RGB),
        ("YELLOW", _YELLOW_PAINT_RGB),
    ):
        polylines = [
            divider.polyline_world
            for divider in game_map.lane_dividers
            if divider.color == color_name
        ] + [
            marking.polyline_world
            for marking in game_map.line_markings
            if marking.color == color_name
        ]
        _paint(
            flat,
            ground_indices,
            ground_x,
            ground_y,
            _line_geometry(polylines, _PAINT_WIDTH_M),
            color,
        )
    return np.clip(image, 0.0, 255.0).astype(np.uint8)
