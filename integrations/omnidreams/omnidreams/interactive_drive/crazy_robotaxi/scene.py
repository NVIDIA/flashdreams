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
import math
import zipfile
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq
from omnidreams.interactive_drive.crazy_robotaxi.navigation import NavigationLane
from omnidreams.interactive_drive.types import SceneBundle


@dataclass(frozen=True)
class CrazyRobotaxiSceneData:
    """Navigation geometry loaded only when Crazy Robotaxi is selected."""

    reference_route_world: np.ndarray
    """Recorded ego route used when mapped lanes are unavailable."""

    navigation_lanes: tuple[NavigationLane, ...]
    """Directed car-lane centerlines used for target routing."""

    exit_cap_segments_world: npt.NDArray[np.float32]
    """Taxi-only walls closing mapped road exits."""

    perimeter_segments_world: npt.NDArray[np.float32]
    """Taxi-only outer wall guaranteeing a closed play area."""

    @property
    def navigation_routes_world(self) -> tuple[np.ndarray, ...]:
        """Return centerline arrays for compatibility with route consumers."""
        return tuple(lane.centerline_world for lane in self.navigation_lanes)

    @property
    def enclosure_segments_world(self) -> npt.NDArray[np.float32]:
        """Return every Taxi-only enclosure wall."""
        if len(self.exit_cap_segments_world) == 0:
            return self.perimeter_segments_world
        if len(self.perimeter_segments_world) == 0:
            return self.exit_cap_segments_world
        return np.concatenate(
            (self.exit_cap_segments_world, self.perimeter_segments_world), axis=0
        )


_PERIMETER_MARGIN_M = 20.0
_MIN_EXIT_WIDTH_M = 1.5
_MAX_EXIT_WIDTH_M = 30.0
_LEGACY_EDGE_BAND_M = 8.0


def _empty_segments() -> npt.NDArray[np.float32]:
    return np.empty((0, 2, 3), dtype=np.float32)


def load_scene_data(scene: SceneBundle) -> CrazyRobotaxiSceneData:
    """Load recorded and mapped routes only for a Crazy Robotaxi session."""
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
            navigation_lanes = _build_navigation_lanes(lane_rows)
        boundary_member = "clipgt/road_boundary.parquet"
        if boundary_member in archive.namelist():
            with archive.open(boundary_member) as handle:
                boundary_rows = pq.read_table(handle).to_pylist()
        else:
            boundary_rows = []

    exit_caps = _build_lane_exit_caps(lane_rows)
    if len(exit_caps) == 0:
        exit_caps = _build_legacy_exit_caps(boundary_rows)
    perimeter = _build_fallback_perimeter(lane_rows, boundary_rows)

    return CrazyRobotaxiSceneData(
        reference_route_world=reference_route_world,
        navigation_lanes=navigation_lanes,
        exit_cap_segments_world=exit_caps,
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


def _deduplicate_segments(
    segments: list[npt.NDArray[np.float32]],
) -> npt.NDArray[np.float32]:
    unique: list[npt.NDArray[np.float32]] = []
    keys: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for segment in segments:
        if segment.shape != (2, 3):
            continue
        length = float(np.linalg.norm(segment[1, :2] - segment[0, :2]))
        if length < _MIN_EXIT_WIDTH_M:
            continue
        points = sorted(
            (
                (round(float(point[0]) * 4.0), round(float(point[1]) * 4.0))
                for point in segment
            )
        )
        key = (points[0], points[1])
        if key in keys:
            continue
        keys.add(key)
        unique.append(segment.astype(np.float32))
    if not unique:
        return _empty_segments()
    return np.stack(unique).astype(np.float32)


def _build_lane_exit_caps(
    rows: list[dict[str, Any]],
) -> npt.NDArray[np.float32]:
    """Close lane ends explicitly labelled as map boundaries by ClipGT."""
    caps: list[npt.NDArray[np.float32]] = []
    for row in rows:
        payload = row.get("lane", {})
        if not _car_lane(payload):
            continue
        map_end = str(payload.get("map_end", "NONE")).upper()
        if map_end not in {"FRONT", "BACK"}:
            continue
        rails = _aligned_lane_rails(payload)
        if rails is None:
            continue
        left_rail, right_rail = rails
        endpoint_index = -1 if map_end == "FRONT" else 0
        cap = np.stack((left_rail[endpoint_index], right_rail[endpoint_index])).astype(
            np.float32
        )
        if float(np.linalg.norm(cap[1, :2] - cap[0, :2])) <= _MAX_EXIT_WIDTH_M:
            caps.append(cap)
    return _deduplicate_segments(caps)


def _boundary_polylines(
    rows: list[dict[str, Any]],
) -> tuple[npt.NDArray[np.float32], ...]:
    polylines: list[npt.NDArray[np.float32]] = []
    for row in rows:
        points = _points_from_records(row.get("road_boundary", {}).get("location", []))
        if len(points) >= 2:
            polylines.append(points)
    return tuple(polylines)


def _build_legacy_exit_caps(
    boundary_rows: list[dict[str, Any]],
) -> npt.NDArray[np.float32]:
    """Conservatively pair compatible boundary ends at a legacy map's AABB."""
    polylines = _boundary_polylines(boundary_rows)
    if not polylines:
        return _empty_segments()
    all_points = np.concatenate(polylines, axis=0)
    xy_min = np.min(all_points[:, :2], axis=0)
    xy_max = np.max(all_points[:, :2], axis=0)
    endpoints: list[tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]] = []
    for points in polylines:
        for index, neighbor_index in ((0, 1), (-1, -2)):
            point = points[index]
            distance_to_edge = min(
                float(point[0] - xy_min[0]),
                float(xy_max[0] - point[0]),
                float(point[1] - xy_min[1]),
                float(xy_max[1] - point[1]),
            )
            tangent = points[neighbor_index, :2] - point[:2]
            tangent_length = float(np.linalg.norm(tangent))
            if distance_to_edge <= _LEGACY_EDGE_BAND_M and tangent_length > 1.0e-4:
                endpoints.append((point, tangent / tangent_length))

    candidates: list[tuple[float, int, int]] = []
    for left_index, (left_point, left_tangent) in enumerate(endpoints):
        for right_index in range(left_index + 1, len(endpoints)):
            right_point, right_tangent = endpoints[right_index]
            connection = right_point[:2] - left_point[:2]
            distance = float(np.linalg.norm(connection))
            if not _MIN_EXIT_WIDTH_M <= distance <= _MAX_EXIT_WIDTH_M:
                continue
            connection /= distance
            parallel = abs(float(np.dot(left_tangent, right_tangent)))
            left_crossing = abs(float(np.dot(left_tangent, connection)))
            right_crossing = abs(float(np.dot(right_tangent, connection)))
            if parallel >= math.cos(math.radians(30.0)) and max(
                left_crossing, right_crossing
            ) <= math.sin(math.radians(35.0)):
                candidates.append((distance, left_index, right_index))

    used: set[int] = set()
    caps: list[npt.NDArray[np.float32]] = []
    for _distance, left_index, right_index in sorted(candidates):
        if left_index in used or right_index in used:
            continue
        used.update((left_index, right_index))
        caps.append(np.stack((endpoints[left_index][0], endpoints[right_index][0])))
    return _deduplicate_segments(caps)


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
    """Return directed car-lane centerlines from ClipGT records."""
    centerlines: list[np.ndarray] = []
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
        centerline = 0.5 * (
            _sample_polyline_fractions(left_rail, fractions)
            + _sample_polyline_fractions(right_rail, fractions)
        )
        if float(np.linalg.norm(centerline[-1, :2] - centerline[0, :2])) > 1.0e-4:
            centerlines.append(centerline.astype(np.float32))
    return tuple(NavigationLane(centerline) for centerline in centerlines)
