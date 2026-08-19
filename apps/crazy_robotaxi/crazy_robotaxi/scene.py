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

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from omnidreams_game_engine.types import SceneBundle

from crazy_robotaxi.navigation import NavigationLane


@dataclass(frozen=True)
class CrazyRobotaxiSceneData:
    """Navigation geometry loaded only when Crazy Robotaxi is selected."""

    reference_route_world: np.ndarray
    """Route used to initialize navigation."""

    navigation_lanes: tuple[NavigationLane, ...]
    """Directed car-lane centerlines used for target routing."""

    curb_segments_world: npt.NDArray[np.float32]
    """Physical curb segments compiled from map-element boundaries."""

    @property
    def navigation_routes_world(self) -> tuple[np.ndarray, ...]:
        """Return centerline arrays for compatibility with route consumers."""
        return tuple(lane.centerline_world for lane in self.navigation_lanes)


def load_scene_data(scene: SceneBundle) -> CrazyRobotaxiSceneData:
    """Load Crazy Robotaxi navigation geometry from the compiled game map."""
    game_map = scene.game_map
    assert game_map is not None, "compiled game-map metadata is required"
    lanes = tuple(
        NavigationLane(
            centerline_world=lane.centerline_world,
            road_edge_world=(
                lane.roadside_edge_world if lane.allows_taxi_stops else None
            ),
            allows_taxi_stops=lane.allows_taxi_stops,
            lane_id=lane.lane_id,
            successor_ids=lane.successor_ids,
        )
        for lane in game_map.lanes
    )
    spawn_lane = next(
        lane
        for lane in game_map.lanes
        if lane.lane_id == game_map.default_spawn.lane_id
    )
    curb_segments = [
        np.stack((start, end))
        for element in game_map.elements
        for curb in element.curbs
        for start, end in zip(
            curb.polyline_world[:-1], curb.polyline_world[1:], strict=True
        )
    ]
    return CrazyRobotaxiSceneData(
        reference_route_world=spawn_lane.centerline_world,
        navigation_lanes=lanes,
        curb_segments_world=(
            np.asarray(curb_segments, dtype=np.float32)
            if curb_segments
            else np.empty((0, 2, 3), dtype=np.float32)
        ),
    )
