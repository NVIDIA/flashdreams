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

    @property
    def navigation_routes_world(self) -> tuple[np.ndarray, ...]:
        """Return centerline arrays for compatibility with route consumers."""
        return tuple(lane.centerline_world for lane in self.navigation_lanes)


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
            navigation_lanes = ()
        else:
            with archive.open(lane_member) as handle:
                rows = pq.read_table(handle).to_pylist()
            navigation_lanes = _build_navigation_lanes(rows)

    return CrazyRobotaxiSceneData(
        reference_route_world=reference_route_world,
        navigation_lanes=navigation_lanes,
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
        vehicle_types = {
            str(vehicle_type).upper()
            for vehicle_type in payload.get("vehicle_types", [])
            if vehicle_type
        }
        if vehicle_types and "CAR" not in vehicle_types:
            continue
        left_rail = _points_from_records(payload.get("left_rail", []))
        right_rail = _points_from_records(payload.get("right_rail", []))
        if len(left_rail) < 2 or len(right_rail) < 2:
            continue
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
        sample_count = max(2, len(left_rail), len(right_rail))
        fractions = np.linspace(0.0, 1.0, sample_count, dtype=np.float32)
        centerline = 0.5 * (
            _sample_polyline_fractions(left_rail, fractions)
            + _sample_polyline_fractions(right_rail, fractions)
        )
        if float(np.linalg.norm(centerline[-1, :2] - centerline[0, :2])) > 1.0e-4:
            centerlines.append(centerline.astype(np.float32))
    return tuple(NavigationLane(centerline) for centerline in centerlines)
