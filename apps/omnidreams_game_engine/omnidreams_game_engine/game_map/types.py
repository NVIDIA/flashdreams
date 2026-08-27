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
class GameMapBoundaryAttributes:
    """Resolved attributes for a structural map element."""

    curb: bool
    """Whether the element emits physical curb boundaries."""


@dataclass(frozen=True)
class GameMapLinearAttributes(GameMapBoundaryAttributes):
    """Resolved lane, surface, and marking attributes."""

    lane_width_m: float
    """Width of one directed lane in metres."""

    curb_offset_m: float
    """Paved offset between the outer lane edge and curb."""

    directions: tuple[str, ...]
    """Ordered lane directions across the element."""

    speed_limit_mps: float
    """Lane speed limit in metres per second."""

    marking_style: str
    """ClipGT-compatible outer lane-marking style."""

    marking_color: str
    """ClipGT-compatible outer lane-marking color."""

    divider_markings: tuple[tuple[str, str], ...]
    """Style and color for each adjacent lane pair."""

    @property
    def lane_width_total_m(self) -> float:
        """Return the total width occupied by lanes."""
        return self.lane_width_m * len(self.directions)

    @property
    def surface_width_m(self) -> float:
        """Return the curb-to-curb paved width."""
        return self.lane_width_total_m + 2.0 * self.curb_offset_m


@dataclass(frozen=True)
class GameMapNode:
    """One explicitly posed node in the authored road network."""

    node_id: str
    """Stable author-defined node identifier."""

    node_type: str
    """Node discriminator such as intersection, road joint, or parking lot."""

    x_m: float
    """Map-space x coordinate of the node origin."""

    y_m: float
    """Map-space y coordinate of the node origin."""

    profile_id: str | None
    """Optional source profile used to resolve node attributes."""

    attributes: GameMapBoundaryAttributes | GameMapLinearAttributes
    """Effective node attributes after applying profile defaults."""

    geometry: dict[str, float]
    """Validated node-type-specific geometry parameters."""

    polygon_vertices_xy: tuple[tuple[float, float], ...] = ()
    """Authored map-space polygon vertices for a parking-lot node."""


@dataclass(frozen=True, eq=False)
class GameMapRoad:
    """One topological road edge between two structural nodes."""

    road_id: str
    """Stable author-defined road identifier."""

    from_node_id: str
    """Node at the beginning of the authored road geometry."""

    to_node_id: str
    """Node at the end of the authored road geometry."""

    profile_id: str | None
    """Optional source profile used to resolve road attributes."""

    attributes: GameMapLinearAttributes
    """Effective road attributes after applying profile defaults."""

    bezier_spans_world: tuple[FloatArray, ...]
    """Compiler-generated map-space cubic spans shaped ``[4, 3]``; empty is straight."""

    def __eq__(self, other: object) -> bool:
        """Compare road metadata and cubic span values."""
        if not isinstance(other, GameMapRoad):
            return NotImplemented
        return (
            self.road_id == other.road_id
            and self.from_node_id == other.from_node_id
            and self.to_node_id == other.to_node_id
            and self.profile_id == other.profile_id
            and self.attributes == other.attributes
            and len(self.bezier_spans_world) == len(other.bezier_spans_world)
            and all(
                np.array_equal(first, second)
                for first, second in zip(
                    self.bezier_spans_world,
                    other.bezier_spans_world,
                    strict=True,
                )
            )
        )


@dataclass(frozen=True)
class GameMapParkingAccess:
    """A parking-lot node's inferred access corridor."""

    access_id: str
    """Stable identifier derived from the parking-lot node identifier."""

    source_node_id: str
    """Intersection or driveway node at the road end of the access."""

    parking_lot_node_id: str
    """Parking-lot node reached by the access."""

    opening_vertex_index: int
    """Zero-based runtime index of the first vertex in the opening edge."""


@dataclass(frozen=True)
class GameMapTopology:
    """Persisted node graph and its derived adjacency."""

    nodes: tuple[GameMapNode, ...]
    """Typed, explicitly posed graph nodes."""

    roads: tuple[GameMapRoad, ...]
    """Authored topological road edges."""

    parking_accesses: tuple[GameMapParkingAccess, ...]
    """Access corridors derived from parking-lot node connections."""

    adjacency: tuple[tuple[str, tuple[str, ...]], ...]
    """Node identifiers paired with stable incident edge/link references."""


@dataclass(frozen=True)
class GameMapVisualVariant:
    """Optional seed image and prompt for one visual variant."""

    name: str
    """Variant slug used to select this visual conditioning."""

    image: str | None
    """Optional map-relative or ``package://`` seed-image reference."""

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
class GameMapTrafficVehicle:
    """One map-authored vehicle and its compiled cyclic route."""

    vehicle_id: str
    """Stable author-defined traffic identifier."""

    node_ids: tuple[str, ...]
    """Ordered author-defined node waypoints."""

    end_behavior: str
    """Whether the waypoint list wraps or is traversed in reverse."""

    vehicle_type: str
    """Motor-vehicle category used by conditioning and physics."""

    dimensions_lwh_m: tuple[float, float, float]
    """Full vehicle length, width, and height in metres."""

    speed_mps: float | None
    """Optional maximum speed; ``None`` follows lane speed limits."""

    start_distance_m: float
    """Initial arc distance along the resolved cyclic route."""

    centerline_world: FloatArray
    """Closed, directed route centerline with shape ``[N, 3]``."""

    speed_limits_mps: FloatArray
    """Per-route-sample target speeds with shape ``[N]``."""

    route_element_ids: tuple[str, ...]
    """Owning road or node identifier for each route segment."""

    def __post_init__(self) -> None:
        if len(self.route_element_ids) != len(self.centerline_world) - 1:
            raise ValueError(
                "route_element_ids must contain one identifier per route segment"
            )


@dataclass(frozen=True)
class GameMapRaceCourse:
    """One ordered race course authored from map nodes and roads."""

    course_id: str
    """Stable course identifier scoped to the containing map."""

    start_element_id: str
    """Node or road surface that starts the timer and closes each lap."""

    checkpoint_element_ids: tuple[str, ...]
    """Ordered node or road surfaces that must be reached."""

    lap_count: int
    """Required laps, or zero for a point-to-point course."""

    checkpoint_markers: bool = True
    """Whether presenters display camera-view start and checkpoint markers."""


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

    conditioning_visible: bool = True
    """Whether the lane is emitted into world-model map conditioning."""


@dataclass(frozen=True)
class GameMapElement:
    """Resolved map-element geometry used by previews and diagnostics."""

    element_id: str
    """Stable author-defined identifier."""

    element_type: str
    """Schema discriminator such as ``road`` or ``intersection``."""

    profile_id: str | None
    """Optional source profile used to resolve element attributes."""

    attributes: GameMapBoundaryAttributes | GameMapLinearAttributes
    """Effective attributes controlling this element."""

    surface_world: FloatArray
    """Closed surface polygon with shape ``[N, 3]``."""

    road_boundaries: tuple[GameMapRoadBoundary, ...]
    """Element-owned semantic boundary polylines excluding declared openings."""

    curbs: tuple[GameMapCurb, ...]
    """Physical curb polylines used as collision barriers."""


@dataclass(frozen=True)
class GameMapRoadBoundary:
    """One semantic road-boundary polyline owned by a resolved map element."""

    boundary_id: str
    """Stable compiler-generated identifier scoped to the owning element."""

    polyline_world: FloatArray
    """World-space boundary points with shape ``[N, 3]``."""


@dataclass(frozen=True)
class GameMapCurb:
    """One stable curb polyline owned by a resolved map element."""

    curb_id: str
    """Stable compiler-generated identifier scoped to the owning element."""

    polyline_world: FloatArray
    """World-space curb points with shape ``[N, 3]``."""


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
class GameMapLaneDivider:
    """One resolved divider shared by two adjacent authored lanes."""

    divider_id: str
    """Stable compiler-generated divider identifier."""

    lane_edges: tuple[tuple[str, str], tuple[str, str]]
    """Adjacent ``(lane_id, side)`` pairs represented by the divider."""

    polyline_world: FloatArray
    """World-space divider centerline with shape ``[N, 3]``."""

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

    compiler_settings: dict[str, object]
    """Resolved authoring settings that affect generated map geometry."""

    topology: GameMapTopology
    """First-class authored topology retained alongside derived lane geometry."""

    lanes: tuple[GameMapLane, ...]
    """Directed road and intersection lanes."""

    elements: tuple[GameMapElement, ...]
    """Resolved element surfaces used by conditioning and previews."""

    road_marking_polygons_world: tuple[FloatArray, ...]
    """Closed road-marking polygons used by conditioning and previews."""

    lane_dividers: tuple[GameMapLaneDivider, ...]
    """Resolved non-virtual dividers between adjacent authored lanes."""

    line_markings: tuple[GameMapLineMarking, ...]
    """Standalone painted lines used by conditioning and previews."""

    ground_vertices: FloatArray
    """Flat ground-mesh vertices."""

    ground_faces: npt.NDArray[np.int32]
    """Ground-mesh triangle indices."""

    spawns: tuple[GameMapSpawn, ...]
    """Playable vehicle spawns."""

    race_courses: tuple[GameMapRaceCourse, ...] = ()
    """Optional ordered race courses authored for this map."""

    traffic: tuple[GameMapTrafficVehicle, ...] = ()
    """Map-authored vehicles with compiled cyclic public-road routes."""

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
        "compiler_settings": game_map.compiler_settings,
        "topology": {
            "nodes": [
                {
                    "node_id": node.node_id,
                    "node_type": node.node_type,
                    "x_m": node.x_m,
                    "y_m": node.y_m,
                    "profile_id": node.profile_id,
                    "attributes": _attributes_to_dict(node.attributes),
                    "geometry": node.geometry,
                    "polygon_vertices_xy": [
                        list(point) for point in node.polygon_vertices_xy
                    ],
                }
                for node in game_map.topology.nodes
            ],
            "roads": [
                {
                    "road_id": road.road_id,
                    "from_node_id": road.from_node_id,
                    "to_node_id": road.to_node_id,
                    "profile_id": road.profile_id,
                    "attributes": _attributes_to_dict(road.attributes),
                    "bezier_spans_world": [
                        span.tolist() for span in road.bezier_spans_world
                    ],
                }
                for road in game_map.topology.roads
            ],
            "parking_accesses": [
                {
                    "access_id": access.access_id,
                    "source_node_id": access.source_node_id,
                    "parking_lot_node_id": access.parking_lot_node_id,
                    "opening_vertex_index": access.opening_vertex_index,
                }
                for access in game_map.topology.parking_accesses
            ],
            "adjacency": [
                [node_id, list(references)]
                for node_id, references in game_map.topology.adjacency
            ],
        },
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
                "conditioning_visible": lane.conditioning_visible,
            }
            for lane in game_map.lanes
        ],
        "elements": [
            {
                "element_id": element.element_id,
                "element_type": element.element_type,
                "profile_id": element.profile_id,
                "attributes": _attributes_to_dict(element.attributes),
                "surface_world": element.surface_world.tolist(),
                "road_boundaries": [
                    {
                        "boundary_id": boundary.boundary_id,
                        "polyline_world": boundary.polyline_world.tolist(),
                    }
                    for boundary in element.road_boundaries
                ],
                "curbs": [
                    {
                        "curb_id": curb.curb_id,
                        "polyline_world": curb.polyline_world.tolist(),
                    }
                    for curb in element.curbs
                ],
            }
            for element in game_map.elements
        ],
        "road_marking_polygons_world": [
            polygon.tolist() for polygon in game_map.road_marking_polygons_world
        ],
        "lane_dividers": [
            {
                "divider_id": divider.divider_id,
                "lane_edges": [list(edge) for edge in divider.lane_edges],
                "polyline_world": divider.polyline_world.tolist(),
                "style": divider.style,
                "color": divider.color,
            }
            for divider in game_map.lane_dividers
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
        "race_courses": [
            {
                "course_id": course.course_id,
                "start_element_id": course.start_element_id,
                "checkpoint_element_ids": list(course.checkpoint_element_ids),
                "lap_count": course.lap_count,
                "checkpoint_markers": course.checkpoint_markers,
            }
            for course in game_map.race_courses
        ],
        "traffic": [
            {
                "vehicle_id": vehicle.vehicle_id,
                "node_ids": list(vehicle.node_ids),
                "end_behavior": vehicle.end_behavior,
                "vehicle_type": vehicle.vehicle_type,
                "dimensions_lwh_m": list(vehicle.dimensions_lwh_m),
                "speed_mps": vehicle.speed_mps,
                "start_distance_m": vehicle.start_distance_m,
                "centerline_world": vehicle.centerline_world.tolist(),
                "speed_limits_mps": vehicle.speed_limits_mps.tolist(),
                "route_element_ids": list(vehicle.route_element_ids),
            }
            for vehicle in game_map.traffic
        ],
    }


def _attributes_to_dict(
    attributes: GameMapBoundaryAttributes | GameMapLinearAttributes,
) -> dict[str, Any]:
    """Serialize resolved element attributes."""
    result: dict[str, Any] = {"curb": attributes.curb}
    if isinstance(attributes, GameMapLinearAttributes):
        result.update(
            {
                "lane_width_m": attributes.lane_width_m,
                "curb_offset_m": attributes.curb_offset_m,
                "directions": list(attributes.directions),
                "speed_limit_mps": attributes.speed_limit_mps,
                "marking_style": attributes.marking_style,
                "marking_color": attributes.marking_color,
                "divider_markings": [
                    list(marking) for marking in attributes.divider_markings
                ],
            }
        )
    return result


def _attributes_from_dict(
    raw: dict[str, Any], *, linear: bool
) -> GameMapBoundaryAttributes | GameMapLinearAttributes:
    """Deserialize resolved attributes for one map element."""
    if not linear:
        return GameMapBoundaryAttributes(curb=bool(raw["curb"]))
    return GameMapLinearAttributes(
        curb=bool(raw["curb"]),
        lane_width_m=float(raw["lane_width_m"]),
        curb_offset_m=float(raw["curb_offset_m"]),
        directions=tuple(str(value) for value in raw["directions"]),
        speed_limit_mps=float(raw["speed_limit_mps"]),
        marking_style=str(raw["marking_style"]),
        marking_color=str(raw["marking_color"]),
        divider_markings=tuple(
            (str(value[0]), str(value[1])) for value in raw["divider_markings"]
        ),
    )


def _lane_divider_from_dict(raw: dict[str, Any]) -> GameMapLaneDivider:
    edges = list(raw["lane_edges"])
    if len(edges) != 2 or any(len(edge) != 2 for edge in edges):
        raise ValueError("lane_dividers[].lane_edges must contain exactly two pairs")
    return GameMapLaneDivider(
        divider_id=str(raw["divider_id"]),
        lane_edges=(
            (str(edges[0][0]), str(edges[0][1])),
            (str(edges[1][0]), str(edges[1][1])),
        ),
        polyline_world=np.asarray(raw["polyline_world"], dtype=np.float32),
        style=str(raw["style"]),
        color=str(raw["color"]),
    )


def game_map_from_dict(value: dict[str, Any]) -> ResolvedGameMap:
    """Deserialize embedded semantic-map metadata."""
    raw_topology = dict(value["topology"])
    topology = GameMapTopology(
        nodes=tuple(
            GameMapNode(
                node_id=str(raw["node_id"]),
                node_type=str(raw["node_type"]),
                x_m=float(raw["x_m"]),
                y_m=float(raw["y_m"]),
                profile_id=(
                    None if raw.get("profile_id") is None else str(raw["profile_id"])
                ),
                attributes=_attributes_from_dict(
                    dict(raw["attributes"]),
                    linear=str(raw["node_type"]) in {"driveway", "road_joint"},
                ),
                geometry={
                    str(key): float(item) for key, item in raw["geometry"].items()
                },
                polygon_vertices_xy=tuple(
                    (float(point[0]), float(point[1]))
                    for point in raw.get("polygon_vertices_xy", ())
                ),
            )
            for raw in raw_topology["nodes"]
        ),
        roads=tuple(
            GameMapRoad(
                road_id=str(raw["road_id"]),
                from_node_id=str(raw["from_node_id"]),
                to_node_id=str(raw["to_node_id"]),
                profile_id=(
                    None if raw.get("profile_id") is None else str(raw["profile_id"])
                ),
                attributes=_attributes_from_dict(dict(raw["attributes"]), linear=True),
                bezier_spans_world=tuple(
                    np.asarray(span, dtype=np.float32)
                    for span in raw["bezier_spans_world"]
                ),
            )
            for raw in raw_topology["roads"]
        ),
        parking_accesses=tuple(
            GameMapParkingAccess(
                access_id=str(raw["access_id"]),
                source_node_id=str(raw["source_node_id"]),
                parking_lot_node_id=str(raw["parking_lot_node_id"]),
                opening_vertex_index=int(raw["opening_vertex_index"]),
            )
            for raw in raw_topology["parking_accesses"]
        ),
        adjacency=tuple(
            (str(raw[0]), tuple(str(reference) for reference in raw[1]))
            for raw in raw_topology["adjacency"]
        ),
    )
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
            conditioning_visible=bool(raw.get("conditioning_visible", True)),
        )
        for raw in value["lanes"]
    )
    elements = tuple(
        GameMapElement(
            element_id=str(raw["element_id"]),
            element_type=str(raw["element_type"]),
            profile_id=(
                None if raw.get("profile_id") is None else str(raw["profile_id"])
            ),
            attributes=_attributes_from_dict(
                dict(raw["attributes"]),
                linear=str(raw["element_type"])
                in {
                    "road",
                    "road_joint",
                    "driveway",
                    "parking_access",
                },
            ),
            surface_world=np.asarray(raw["surface_world"], dtype=np.float32),
            road_boundaries=tuple(
                GameMapRoadBoundary(
                    boundary_id=str(
                        boundary.get("boundary_id", boundary.get("curb_id"))
                    ),
                    polyline_world=np.asarray(
                        boundary["polyline_world"], dtype=np.float32
                    ),
                )
                for boundary in raw.get("road_boundaries", raw.get("curbs", []))
            ),
            curbs=tuple(
                GameMapCurb(
                    curb_id=str(curb["curb_id"]),
                    polyline_world=np.asarray(curb["polyline_world"], dtype=np.float32),
                )
                for curb in raw.get("curbs", [])
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
                    image=(
                        None if variant.get("image") is None else str(variant["image"])
                    ),
                    prompt=str(variant["prompt"]),
                )
                for variant in raw["variants"]
            ),
        )
        for raw in value["spawns"]
    )
    traffic = tuple(
        GameMapTrafficVehicle(
            vehicle_id=str(raw["vehicle_id"]),
            node_ids=tuple(str(item) for item in raw["node_ids"]),
            end_behavior=str(raw["end_behavior"]),
            vehicle_type=str(raw["vehicle_type"]),
            dimensions_lwh_m=tuple(float(item) for item in raw["dimensions_lwh_m"]),
            speed_mps=(
                None if raw.get("speed_mps") is None else float(raw["speed_mps"])
            ),
            start_distance_m=float(raw["start_distance_m"]),
            centerline_world=np.asarray(raw["centerline_world"], dtype=np.float32),
            speed_limits_mps=np.asarray(raw["speed_limits_mps"], dtype=np.float32),
            route_element_ids=tuple(str(item) for item in raw["route_element_ids"]),
        )
        for raw in value.get("traffic", [])
    )
    race_courses = tuple(
        GameMapRaceCourse(
            course_id=str(raw["course_id"]),
            start_element_id=str(raw["start_element_id"]),
            checkpoint_element_ids=tuple(
                str(item) for item in raw["checkpoint_element_ids"]
            ),
            lap_count=int(raw["lap_count"]),
            checkpoint_markers=bool(raw.get("checkpoint_markers", True)),
        )
        for raw in value.get("race_courses", [])
    )
    return ResolvedGameMap(
        schema_version=int(value["schema_version"]),
        map_id=str(value["map_id"]),
        name=str(value["name"]),
        source_path=Path(str(value["source_path"])),
        compiler_settings=dict(value.get("compiler_settings", {})),
        topology=topology,
        lanes=lanes,
        elements=elements,
        road_marking_polygons_world=tuple(
            np.asarray(polygon, dtype=np.float32)
            for polygon in value.get("road_marking_polygons_world", [])
        ),
        lane_dividers=tuple(
            _lane_divider_from_dict(raw) for raw in value.get("lane_dividers", [])
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
        race_courses=race_courses,
        traffic=traffic,
    )
