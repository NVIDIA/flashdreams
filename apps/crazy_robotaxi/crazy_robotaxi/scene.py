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
from omnidreams_game_engine.types import SceneDefinition

from crazy_robotaxi.navigation import NavigationFareRegion, NavigationLane


@dataclass(frozen=True)
class CrazyRobotaxiSceneData:
    """Navigation geometry loaded only when Crazy Robotaxi is selected."""

    reference_route_world: np.ndarray
    """Route used to initialize navigation."""

    navigation_lanes: tuple[NavigationLane, ...]
    """Directed car-lane centerlines used for target routing."""

    fare_regions: tuple[NavigationFareRegion, ...]
    """Parking areas and exposed node edges used for fare placement."""

    curb_segments_world: npt.NDArray[np.float32]
    """Physical curb segments compiled from map-element boundaries."""

    @property
    def navigation_routes_world(self) -> tuple[np.ndarray, ...]:
        """Return centerline arrays for compatibility with route consumers."""
        return tuple(lane.centerline_world for lane in self.navigation_lanes)


def load_scene_data(scene: SceneDefinition) -> CrazyRobotaxiSceneData:
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
            element_id=lane.element_id,
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
    runtime_lanes = {lane.lane_id: lane for lane in game_map.lanes}
    roads_by_node: dict[str, list[str]] = {
        node.node_id: [] for node in game_map.topology.nodes
    }
    for road in game_map.topology.roads:
        roads_by_node[road.from_node_id].append(road.road_id)
        roads_by_node[road.to_node_id].append(road.road_id)

    def node_anchors(node_id: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        node = next(item for item in game_map.topology.nodes if item.node_id == node_id)
        center = np.asarray([node.x_m, node.y_m], dtype=np.float32)
        arrivals: list[str] = []
        departures: list[str] = []
        for road_id in roads_by_node[node_id]:
            for lane_id, lane in runtime_lanes.items():
                if lane.element_id != road_id:
                    continue
                start_distance = float(
                    np.linalg.norm(lane.centerline_world[0, :2] - center)
                )
                end_distance = float(
                    np.linalg.norm(lane.centerline_world[-1, :2] - center)
                )
                if end_distance <= start_distance:
                    arrivals.append(lane_id)
                if start_distance <= end_distance:
                    departures.append(lane_id)
        return tuple(dict.fromkeys(arrivals)), tuple(dict.fromkeys(departures))

    node_anchor_cache = {
        node.node_id: node_anchors(node.node_id)
        for node in game_map.topology.nodes
        if node.node_type != "parking_lot"
    }
    accesses_by_lot: dict[str, list[str]] = {}
    for access in game_map.topology.parking_accesses:
        accesses_by_lot.setdefault(access.parking_lot_node_id, []).append(
            access.source_node_id
        )
    elements = {element.element_id: element for element in game_map.elements}
    fare_regions: list[NavigationFareRegion] = []
    for node in game_map.topology.nodes:
        element = elements[node.node_id]
        if node.node_type == "parking_lot":
            source_ids = accesses_by_lot[node.node_id]
            arrivals = tuple(
                lane_id
                for source_id in source_ids
                for lane_id in node_anchor_cache[source_id][0]
            )
            departures = tuple(
                lane_id
                for source_id in source_ids
                for lane_id in node_anchor_cache[source_id][1]
            )
            fare_regions.append(
                NavigationFareRegion(
                    node.node_id,
                    "area",
                    (element.surface_world,),
                    tuple(dict.fromkeys(arrivals)),
                    tuple(dict.fromkeys(departures)),
                )
            )
            continue
        arrivals, departures = node_anchor_cache[node.node_id]
        boundaries = tuple(
            boundary.polyline_world for boundary in element.road_boundaries
        )
        if boundaries and arrivals and departures:
            fare_regions.append(
                NavigationFareRegion(
                    node.node_id,
                    "boundary",
                    boundaries,
                    arrivals,
                    departures,
                )
            )
    return CrazyRobotaxiSceneData(
        reference_route_world=spawn_lane.centerline_world,
        navigation_lanes=lanes,
        fare_regions=tuple(fare_regions),
        curb_segments_world=(
            np.asarray(curb_segments, dtype=np.float32)
            if curb_segments
            else np.empty((0, 2, 3), dtype=np.float32)
        ),
    )
