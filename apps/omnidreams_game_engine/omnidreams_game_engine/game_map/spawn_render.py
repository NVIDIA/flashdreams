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

"""Deterministic first-person fallback renders for semantic-map spawns."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import shapely
from PIL import Image
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

SPAWN_RENDERER_VERSION = "1"
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


def _camera_ground_intersections(
    spawn: GameMapSpawn, width: int, height: int
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
        _camera_ground_intersections(spawn, width, height)
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


def write_spawn_first_frame_preview(
    source: Path,
    destination: Path,
    *,
    spawn_id: str | None = None,
) -> Path:
    """Write the deterministic fallback render for one authored spawn.

    Args:
        source: Semantic map YAML path.
        destination: PNG path to create.
        spawn_id: Spawn identifier; ``None`` selects the first spawn.

    Returns:
        Resolved output path.

    Raises:
        GameMapError: ``spawn_id`` does not identify a map spawn.
    """
    from omnidreams_game_engine.game_map._schema import GameMapError
    from omnidreams_game_engine.game_map.loader import load_game_map

    game_map = load_game_map(source)
    if spawn_id is None:
        spawn = game_map.default_spawn
    else:
        spawn = next(
            (
                candidate
                for candidate in game_map.spawns
                if candidate.spawn_id == spawn_id
            ),
            None,
        )
        if spawn is None:
            available = ", ".join(item.spawn_id for item in game_map.spawns)
            raise GameMapError(
                f"Unknown spawn {spawn_id!r}; available spawns: {available}"
            )
    output = Path(destination).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(render_spawn_first_frame(game_map, spawn)).save(output)
    return output
