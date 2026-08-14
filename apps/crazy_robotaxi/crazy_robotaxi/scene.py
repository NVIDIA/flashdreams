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

"""Crazy Robotaxi navigation geometry loading."""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq
from omnidreams_game_engine.types import SceneBundle
from shapely.geometry import Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from crazy_robotaxi.navigation import NavigationLane

_PHYSICAL_ROAD_EDGE_STYLES = frozenset({"TALL_CURB", "ROAD_BOUNDARY", "WALL", "FENCE"})
"""ClipGT edge styles that unambiguously bound drivable pavement."""

_PAINTED_ROAD_EDGE_STYLES = frozenset({"SOLID_SINGLE", "SOLID_GROUP"})
"""Solid white edge styles usable when a physical curb is unavailable."""


@dataclass(frozen=True)
class CrazyRobotaxiSceneData:
    """Navigation geometry loaded only when Crazy Robotaxi is selected."""

    reference_route_world: np.ndarray
    """Recorded ego route used when mapped lanes are unavailable."""

    navigation_lanes: tuple[NavigationLane, ...]
    """Directed car-lane centerlines used for target routing."""

    perimeter_segments_world: npt.NDArray[np.float32]
    """Taxi-only walls enclosing the player's lane-network component."""

    @property
    def navigation_routes_world(self) -> tuple[np.ndarray, ...]:
        """Return centerline arrays for compatibility with route consumers."""
        return tuple(lane.centerline_world for lane in self.navigation_lanes)

    @property
    def enclosure_segments_world(self) -> npt.NDArray[np.float32]:
        """Return every Taxi-only enclosure wall."""
        return self.perimeter_segments_world


_PERIMETER_MARGIN_M = 20.0
"""Distance between boundary-only legacy geometry and its outer wall."""

_LANE_JOIN_TOLERANCE_M = 0.25
"""Morphological closing distance used to join adjacent lane polygons."""

_LANE_PERIMETER_CLEARANCE_M = 3.0
"""Distance between mapped lane rails and the Taxi-only enclosure."""

_LANE_PERIMETER_SIMPLIFY_M = 0.5
"""Maximum geometric deviation when simplifying the enclosure ring."""


def _empty_segments() -> npt.NDArray[np.float32]:
    return np.empty((0, 2, 3), dtype=np.float32)


def load_scene_data(scene: SceneBundle) -> CrazyRobotaxiSceneData:
    """Load recorded and mapped routes only for a Crazy Robotaxi session."""
    if scene.game_map is not None:
        lanes = tuple(
            NavigationLane(
                centerline_world=lane.centerline_world,
                road_edge_world=(
                    lane.right_edge_world if lane.allows_taxi_stops else None
                ),
                allows_taxi_stops=lane.allows_taxi_stops,
                lane_id=lane.lane_id,
                successor_ids=lane.successor_ids,
            )
            for lane in scene.game_map.lanes
        )
        spawn_lane = next(
            lane
            for lane in scene.game_map.lanes
            if lane.lane_id == scene.game_map.default_spawn.lane_id
        )
        return CrazyRobotaxiSceneData(
            reference_route_world=spawn_lane.centerline_world,
            navigation_lanes=lanes,
            perimeter_segments_world=scene.game_map.collision_segments_world,
        )
    with zipfile.ZipFile(scene.scene_path, "r") as archive:
        trajectory_doc = json.loads(archive.read("rig_trajectories.json"))
        poses = np.asarray(
            trajectory_doc["rig_trajectories"][0]["T_rig_worlds"],
            dtype=np.float32,
        )
        reference_route_world = poses[:, :3, 3].astype(np.float32)
        lane_member = "clipgt/lane.parquet"
        if lane_member not in archive.namelist():
            lane_rows: list[dict[str, Any]] = []
            navigation_lanes = ()
        else:
            with archive.open(lane_member) as handle:
                lane_rows = pq.read_table(handle).to_pylist()
            mapped_lanes = _build_navigation_lanes(lane_rows)
            navigation_lanes = (
                mapped_lanes
                if any(lane.allows_taxi_stops for lane in mapped_lanes)
                else ()
            )
        boundary_member = "clipgt/road_boundary.parquet"
        if boundary_member in archive.namelist():
            with archive.open(boundary_member) as handle:
                boundary_rows = pq.read_table(handle).to_pylist()
        else:
            boundary_rows = []

    perimeter = _build_lane_network_perimeter(
        lane_rows,
        reference_route_world[0, :2],
    )
    if not len(perimeter):
        perimeter = _build_fallback_perimeter(lane_rows, boundary_rows)

    return CrazyRobotaxiSceneData(
        reference_route_world=reference_route_world,
        navigation_lanes=navigation_lanes,
        perimeter_segments_world=perimeter,
    )


def _points_from_records(points: list[dict[str, float]]) -> np.ndarray:
    return np.array(
        [[point["x"], point["y"], point["z"]] for point in points],
        dtype=np.float32,
    )


def _sample_polyline_fractions(
    points_xyz: np.ndarray, fractions: np.ndarray
) -> np.ndarray:
    segment_lengths = np.linalg.norm(np.diff(points_xyz[:, :2], axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = float(cumulative[-1])
    if total_length <= 1.0e-4:
        return np.repeat(points_xyz[:1], len(fractions), axis=0)
    distances = fractions * total_length
    return np.stack(
        [np.interp(distances, cumulative, points_xyz[:, axis]) for axis in range(3)],
        axis=1,
    ).astype(np.float32)


def _aligned_lane_rails(
    payload: dict[str, Any],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]] | None:
    left_rail = _points_from_records(payload.get("left_rail", []))
    right_rail = _points_from_records(payload.get("right_rail", []))
    if len(left_rail) < 2 or len(right_rail) < 2:
        return None
    aligned_cost = float(
        np.linalg.norm(left_rail[0, :2] - right_rail[0, :2])
        + np.linalg.norm(left_rail[-1, :2] - right_rail[-1, :2])
    )
    reversed_cost = float(
        np.linalg.norm(left_rail[0, :2] - right_rail[-1, :2])
        + np.linalg.norm(left_rail[-1, :2] - right_rail[0, :2])
    )
    if reversed_cost < aligned_cost:
        right_rail = right_rail[::-1]
    return left_rail, right_rail


def _car_lane(payload: dict[str, Any]) -> bool:
    vehicle_types = {
        str(vehicle_type).upper()
        for vehicle_type in payload.get("vehicle_types", [])
        if vehicle_type
    }
    return not vehicle_types or "CAR" in vehicle_types


def _polygon_components(geometry: BaseGeometry) -> tuple[Polygon, ...]:
    """Return every nonempty polygon contained in a Shapely geometry."""
    if isinstance(geometry, Polygon):
        return (geometry,) if geometry.area > 1.0e-2 else ()
    if hasattr(geometry, "geoms"):
        return tuple(
            polygon
            for child in geometry.geoms
            for polygon in _polygon_components(child)
        )
    return ()


def _build_lane_network_perimeter(
    lane_rows: list[dict[str, Any]],
    spawn_xy_m: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Build closed walls around the spawn-connected drivable lane surface.

    Args:
        lane_rows: ClipGT lane records.
        spawn_xy_m: Initial player position in world XY coordinates.

    Returns:
        World-space wall segments with shape ``[N, 2, 3]``. Segments are
        consecutive within each closed boundary ring.
    """
    lane_surfaces: list[Polygon] = []
    lane_heights: list[npt.NDArray[np.float32]] = []
    for row in lane_rows:
        payload = row.get("lane", {})
        if not _car_lane(payload):
            continue
        rails = _aligned_lane_rails(payload)
        if rails is None:
            continue
        left_rail, right_rail = rails
        surface = Polygon(
            np.concatenate((left_rail[:, :2], right_rail[::-1, :2]), axis=0)
        )
        if not surface.is_valid:
            surface = surface.buffer(0)
        lane_surfaces.extend(_polygon_components(surface))
        lane_heights.extend((left_rail[:, 2], right_rail[:, 2]))
    if not lane_surfaces:
        return _empty_segments()

    joined_surface = unary_union(lane_surfaces)
    joined_surface = joined_surface.buffer(
        _LANE_JOIN_TOLERANCE_M,
        join_style="mitre",
    ).buffer(-_LANE_JOIN_TOLERANCE_M, join_style="mitre")
    components = _polygon_components(joined_surface)
    if not components:
        return _empty_segments()

    spawn_point = Point(float(spawn_xy_m[0]), float(spawn_xy_m[1]))
    playable_surface = min(
        components, key=lambda component: component.distance(spawn_point)
    )
    enclosure_geometry = playable_surface.buffer(
        _LANE_PERIMETER_CLEARANCE_M,
        join_style="mitre",
    ).simplify(_LANE_PERIMETER_SIMPLIFY_M, preserve_topology=True)
    enclosure_components = _polygon_components(enclosure_geometry)
    if not enclosure_components:
        return _empty_segments()
    enclosure = min(
        enclosure_components,
        key=lambda component: component.distance(spawn_point),
    )

    z_m = float(np.median(np.concatenate(lane_heights)))
    ring_segments: list[npt.NDArray[np.float32]] = []
    for ring in (enclosure.exterior, *enclosure.interiors):
        ring_xy = np.asarray(ring.coords, dtype=np.float32)
        if len(ring_xy) < 4:
            continue
        ring_xyz = np.column_stack(
            (ring_xy, np.full(len(ring_xy), z_m, dtype=np.float32))
        ).astype(np.float32)
        ring_segments.append(np.stack((ring_xyz[:-1], ring_xyz[1:]), axis=1))
    if not ring_segments:
        return _empty_segments()
    return np.concatenate(ring_segments, axis=0).astype(np.float32)


def _boundary_polylines(
    rows: list[dict[str, Any]],
) -> tuple[npt.NDArray[np.float32], ...]:
    polylines: list[npt.NDArray[np.float32]] = []
    for row in rows:
        points = _points_from_records(row.get("road_boundary", {}).get("location", []))
        if len(points) >= 2:
            polylines.append(points)
    return tuple(polylines)


def _build_fallback_perimeter(
    lane_rows: list[dict[str, Any]],
    boundary_rows: list[dict[str, Any]],
) -> npt.NDArray[np.float32]:
    points: list[npt.NDArray[np.float32]] = []
    for row in lane_rows:
        payload = row.get("lane", {})
        if not _car_lane(payload):
            continue
        rails = _aligned_lane_rails(payload)
        if rails is not None:
            points.extend(rails)
    points.extend(_boundary_polylines(boundary_rows))
    if not points:
        return _empty_segments()
    all_points = np.concatenate(points, axis=0)
    x_min, y_min = np.min(all_points[:, :2], axis=0) - _PERIMETER_MARGIN_M
    x_max, y_max = np.max(all_points[:, :2], axis=0) + _PERIMETER_MARGIN_M
    z_m = float(np.median(all_points[:, 2]))
    corners = np.asarray(
        [
            [x_min, y_min, z_m],
            [x_max, y_min, z_m],
            [x_max, y_max, z_m],
            [x_min, y_max, z_m],
        ],
        dtype=np.float32,
    )
    return np.stack(
        [np.stack((corners[index - 1], corners[index])) for index in range(4)]
    ).astype(np.float32)


def _build_lane_centerlines(rows: list[dict[str, Any]]) -> tuple[np.ndarray, ...]:
    """Return directed car-lane centerlines from ClipGT records."""
    return tuple(lane.centerline_world for lane in _build_navigation_lanes(rows))


def _build_navigation_lanes(
    rows: list[dict[str, Any]],
) -> tuple[NavigationLane, ...]:
    """Return directed car lanes and their mapped roadside stopping edges."""
    lanes: list[NavigationLane] = []
    for row in rows:
        payload = row["lane"]
        if not _car_lane(payload):
            continue
        rails = _aligned_lane_rails(payload)
        if rails is None:
            continue
        left_rail, right_rail = rails
        sample_count = max(2, len(left_rail), len(right_rail))
        fractions = np.linspace(0.0, 1.0, sample_count, dtype=np.float32)
        left_rail = _sample_polyline_fractions(left_rail, fractions)
        right_rail = _sample_polyline_fractions(right_rail, fractions)
        centerline = 0.5 * (left_rail + right_rail)
        if float(np.linalg.norm(centerline[-1, :2] - centerline[0, :2])) > 1.0e-4:
            road_edge = _roadside_edge(payload, left_rail, right_rail)
            lanes.append(
                NavigationLane(
                    centerline.astype(np.float32),
                    road_edge,
                    allows_taxi_stops=road_edge is not None,
                )
            )

    return tuple(lanes)


def _roadside_edge(
    payload: dict[str, Any],
    left_rail: np.ndarray,
    right_rail: np.ndarray,
) -> np.ndarray | None:
    left_score = _road_edge_score(
        payload.get("left_edge_styles", []),
        payload.get("left_edge_colors", []),
    )
    right_score = _road_edge_score(
        payload.get("right_edge_styles", []),
        payload.get("right_edge_colors", []),
    )
    if left_score == right_score == 0:
        return None
    return right_rail if right_score >= left_score else left_rail


def _road_edge_score(styles: list[str] | None, colors: list[str] | None) -> int:
    point_scores = []
    for style, color in zip(styles or (), colors or (), strict=True):
        normalized_style = str(style).upper()
        normalized_color = str(color).upper()
        if normalized_style in _PHYSICAL_ROAD_EDGE_STYLES:
            point_scores.append(2)
        elif (
            normalized_style in _PAINTED_ROAD_EDGE_STYLES
            and normalized_color == "WHITE"
        ):
            point_scores.append(1)
        else:
            point_scores.append(0)
    return min(point_scores, default=0)
