# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Runtime types for resolved semantic game maps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]


@dataclass(frozen=True)
class GameMapVisualVariant:
    """Seed image and prompt for one visual variant."""

    name: str
    """Variant slug exposed by the scene selector."""

    image: str
    """Map-relative path or ``package://package/resource`` asset reference."""

    prompt: str
    """World-model text prompt paired with the seed image."""


@dataclass(frozen=True)
class GameMapSpawn:
    """Vehicle spawn resolved onto a directed lane."""

    spawn_id: str
    """Stable author-defined spawn identifier."""

    lane_id: str
    """Directed lane containing the spawn."""

    distance_m: float
    """Distance from the directed lane start."""

    position_world: FloatArray
    """World position with shape ``[3]``."""

    yaw_rad: float
    """World heading following the directed lane."""

    variants: tuple[GameMapVisualVariant, ...]
    """Available visual seed variants; ``default`` is always present."""


@dataclass(frozen=True)
class GameMapLane:
    """Explicit directed lane and its legal successors."""

    lane_id: str
    """Stable compiler-generated lane identifier."""

    element_id: str
    """Owning routable map-element identifier."""

    centerline_world: FloatArray
    """Directed centerline with shape ``[N, 3]``."""

    left_edge_world: FloatArray
    """Left rail in travel direction with shape ``[N, 3]``."""

    right_edge_world: FloatArray
    """Right rail in travel direction with shape ``[N, 3]``."""

    roadside_edge_world: FloatArray
    """Physical roadside edge to the right of travel with shape ``[N, 3]``."""

    speed_limit_mps: float
    """Authored speed limit for this lane."""

    marking_style: str
    """ClipGT-compatible lane-marking style."""

    marking_color: str
    """ClipGT-compatible lane-marking color."""

    left_marking_style: str
    """ClipGT-compatible marking style for the directed left rail."""

    left_marking_color: str
    """ClipGT-compatible marking color for the directed left rail."""

    right_marking_style: str
    """ClipGT-compatible marking style for the directed right rail."""

    right_marking_color: str
    """ClipGT-compatible marking color for the directed right rail."""

    successor_ids: tuple[str, ...]
    """Legal successor lane identifiers."""

    allows_taxi_stops: bool = True
    """Whether taxi targets may be sampled from this lane."""


@dataclass(frozen=True)
class GameMapElement:
    """Resolved map-element geometry used by previews and diagnostics."""

    element_id: str
    """Stable author-defined identifier."""

    element_type: str
    """Schema discriminator such as ``road_segment`` or ``intersection``."""

    profile_id: str
    """Primary road or access profile used by this element."""

    surface_world: FloatArray
    """Closed surface polygon with shape ``[N, 3]``."""

    ports: tuple[tuple[str, float, float, float, bool], ...]
    """Port name, world position, heading, and connected state tuples."""


@dataclass(frozen=True)
class GameMapLineMarking:
    """Resolved line marking emitted into model conditioning."""

    marking_id: str
    """Stable compiler-generated marking identifier."""

    polyline_world: FloatArray
    """World-space marking centerline with shape ``[N, 3]``."""

    style: str
    """ClipGT-compatible lane-line style."""

    color: str
    """ClipGT-compatible lane-line color."""


@dataclass(frozen=True)
class ResolvedGameMap:
    """Validated semantic map with generated runtime geometry."""

    schema_version: int
    """Authoring schema version."""

    map_id: str
    """Stable map identifier."""

    name: str
    """Human-readable map name."""

    source_path: Path
    """Canonical YAML source path."""

    lanes: tuple[GameMapLane, ...]
    """Directed road and intersection lanes."""

    elements: tuple[GameMapElement, ...]
    """Resolved element surfaces and ports."""

    collision_segments_world: FloatArray
    """Explicit curb colliders with shape ``[N, 2, 3]``."""

    road_marking_polygons_world: tuple[FloatArray, ...]
    """Closed road-marking polygons used by conditioning and previews."""

    line_markings: tuple[GameMapLineMarking, ...]
    """Standalone painted lines used by conditioning and previews."""

    ground_vertices: FloatArray
    """Flat ground-mesh vertices."""

    ground_faces: npt.NDArray[np.int32]
    """Ground-mesh triangle indices."""

    spawns: tuple[GameMapSpawn, ...]
    """Playable vehicle spawns."""

    @property
    def default_spawn(self) -> GameMapSpawn:
        """Return the first declared spawn."""
        return self.spawns[0]

    @property
    def variants(self) -> tuple[str, ...]:
        """Return variants available at the default spawn."""
        names = [variant.name for variant in self.default_spawn.variants]
        return tuple(names)


def game_map_to_dict(game_map: ResolvedGameMap) -> dict[str, Any]:
    """Serialize a resolved map into JSON-compatible values."""
    return {
        "schema_version": game_map.schema_version,
        "map_id": game_map.map_id,
        "name": game_map.name,
        "source_path": str(game_map.source_path),
        "lanes": [
            {
                "lane_id": lane.lane_id,
                "element_id": lane.element_id,
                "centerline_world": lane.centerline_world.tolist(),
                "left_edge_world": lane.left_edge_world.tolist(),
                "right_edge_world": lane.right_edge_world.tolist(),
                "roadside_edge_world": lane.roadside_edge_world.tolist(),
                "speed_limit_mps": lane.speed_limit_mps,
                "marking_style": lane.marking_style,
                "marking_color": lane.marking_color,
                "left_marking_style": lane.left_marking_style,
                "left_marking_color": lane.left_marking_color,
                "right_marking_style": lane.right_marking_style,
                "right_marking_color": lane.right_marking_color,
                "successor_ids": list(lane.successor_ids),
                "allows_taxi_stops": lane.allows_taxi_stops,
            }
            for lane in game_map.lanes
        ],
        "elements": [
            {
                "element_id": element.element_id,
                "element_type": element.element_type,
                "profile_id": element.profile_id,
                "surface_world": element.surface_world.tolist(),
                "ports": [list(port) for port in element.ports],
            }
            for element in game_map.elements
        ],
        "collision_segments_world": game_map.collision_segments_world.tolist(),
        "road_marking_polygons_world": [
            polygon.tolist() for polygon in game_map.road_marking_polygons_world
        ],
        "line_markings": [
            {
                "marking_id": marking.marking_id,
                "polyline_world": marking.polyline_world.tolist(),
                "style": marking.style,
                "color": marking.color,
            }
            for marking in game_map.line_markings
        ],
        "ground_vertices": game_map.ground_vertices.tolist(),
        "ground_faces": game_map.ground_faces.tolist(),
        "spawns": [
            {
                "spawn_id": spawn.spawn_id,
                "lane_id": spawn.lane_id,
                "distance_m": spawn.distance_m,
                "position_world": spawn.position_world.tolist(),
                "yaw_rad": spawn.yaw_rad,
                "variants": [
                    {
                        "name": variant.name,
                        "image": variant.image,
                        "prompt": variant.prompt,
                    }
                    for variant in spawn.variants
                ],
            }
            for spawn in game_map.spawns
        ],
    }


def game_map_from_dict(value: dict[str, Any]) -> ResolvedGameMap:
    """Deserialize embedded semantic-map metadata."""
    lanes = tuple(
        GameMapLane(
            lane_id=str(raw["lane_id"]),
            element_id=str(raw["element_id"]),
            centerline_world=np.asarray(raw["centerline_world"], dtype=np.float32),
            left_edge_world=np.asarray(raw["left_edge_world"], dtype=np.float32),
            right_edge_world=np.asarray(raw["right_edge_world"], dtype=np.float32),
            roadside_edge_world=np.asarray(
                raw.get("roadside_edge_world", raw["right_edge_world"]),
                dtype=np.float32,
            ),
            speed_limit_mps=float(raw["speed_limit_mps"]),
            marking_style=str(raw["marking_style"]),
            marking_color=str(raw["marking_color"]),
            left_marking_style=str(raw.get("left_marking_style", raw["marking_style"])),
            left_marking_color=str(raw.get("left_marking_color", raw["marking_color"])),
            right_marking_style=str(
                raw.get("right_marking_style", raw["marking_style"])
            ),
            right_marking_color=str(
                raw.get("right_marking_color", raw["marking_color"])
            ),
            successor_ids=tuple(str(item) for item in raw["successor_ids"]),
            allows_taxi_stops=bool(raw["allows_taxi_stops"]),
        )
        for raw in value["lanes"]
    )
    elements = tuple(
        GameMapElement(
            element_id=str(raw["element_id"]),
            element_type=str(raw["element_type"]),
            profile_id=str(raw["profile_id"]),
            surface_world=np.asarray(raw["surface_world"], dtype=np.float32),
            ports=tuple(
                (
                    str(port[0]),
                    float(port[1]),
                    float(port[2]),
                    float(port[3]),
                    bool(port[4]),
                )
                for port in raw["ports"]
            ),
        )
        for raw in value["elements"]
    )
    spawns = tuple(
        GameMapSpawn(
            spawn_id=str(raw["spawn_id"]),
            lane_id=str(raw["lane_id"]),
            distance_m=float(raw["distance_m"]),
            position_world=np.asarray(raw["position_world"], dtype=np.float32),
            yaw_rad=float(raw["yaw_rad"]),
            variants=tuple(
                GameMapVisualVariant(
                    name=str(variant["name"]),
                    image=str(variant["image"]),
                    prompt=str(variant["prompt"]),
                )
                for variant in raw["variants"]
            ),
        )
        for raw in value["spawns"]
    )
    return ResolvedGameMap(
        schema_version=int(value["schema_version"]),
        map_id=str(value["map_id"]),
        name=str(value["name"]),
        source_path=Path(str(value["source_path"])),
        lanes=lanes,
        elements=elements,
        collision_segments_world=np.asarray(
            value["collision_segments_world"], dtype=np.float32
        ).reshape(-1, 2, 3),
        road_marking_polygons_world=tuple(
            np.asarray(polygon, dtype=np.float32)
            for polygon in value.get("road_marking_polygons_world", [])
        ),
        line_markings=tuple(
            GameMapLineMarking(
                marking_id=str(raw["marking_id"]),
                polyline_world=np.asarray(raw["polyline_world"], dtype=np.float32),
                style=str(raw["style"]),
                color=str(raw["color"]),
            )
            for raw in value.get("line_markings", [])
        ),
        ground_vertices=np.asarray(value["ground_vertices"], dtype=np.float32),
        ground_faces=np.asarray(value["ground_faces"], dtype=np.int32),
        spawns=spawns,
    )
