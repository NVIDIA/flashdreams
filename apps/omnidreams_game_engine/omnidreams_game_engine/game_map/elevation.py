# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Elevation of node-graph road geometry and its continuous ground mesh."""

from dataclasses import replace
from collections.abc import Iterator

import numpy as np
import shapely
from shapely.geometry import LineString, MultiPoint, Polygon
from shapely.ops import triangulate

from omnidreams_game_engine.game_map._schema import _LaneBuild
from omnidreams_game_engine.game_map.types import (
    GameMapElement,
    GameMapLaneDivider,
    GameMapTopology,
)
from omnidreams_game_engine.simulation.ground_snap import GroundSnapper


def _pavement_samples(region: Polygon, spacing: float) -> Iterator[np.ndarray]:
    """Yield covered points on a shared XY lattice in bounded tiles."""
    x0, y0, x1, y1 = region.bounds
    left, bottom = int(np.floor(x0 / spacing)), int(np.floor(y0 / spacing))
    right, top = int(np.ceil(x1 / spacing)) + 1, int(np.ceil(y1 / spacing)) + 1
    for row in range(bottom, top, 256):
        for column in range(left, right, 256):
            xs = np.arange(column, min(column + 256, right)) * spacing
            ys = np.arange(row, min(row + 256, top)) * spacing
            if not region.intersects(
                shapely.box(xs[0], ys[0], xs[-1] + spacing, ys[-1] + spacing)
            ):
                continue
            x, y = np.meshgrid(xs, ys)
            xy = np.column_stack((x.ravel(), y.ravel()))
            inside = shapely.intersects_xy(region, xy[:, 0], xy[:, 1])
            if inside.any():
                yield xy[inside]


def elevate_geometry(
    topology: GameMapTopology,
    elements: list[GameMapElement],
    lanes: list[_LaneBuild],
    dividers: list[GameMapLaneDivider],
    reference_lines: dict[str, np.ndarray],
    *,
    sample_spacing_m: float,
    ground_margin_m: float,
) -> tuple[list[GameMapElement], list[GameMapLaneDivider], np.ndarray, np.ndarray]:
    """Lift geometry in place and sample one ground height field.

    Roads use quintic smoothstep along their trimmed reference line, with zero
    grade and curvature at the level node surfaces. Parking accesses share
    their source node's height. Ground between pavement and the bounding
    rectangle is linearly interpolated; the outer rectangle is at height zero.
    This is a single surface, so overlapping roads and overpasses remain unsupported.

    Args:
        topology: Nodes, road endpoints, and parking accesses.
        elements: Flat road and node footprints with boundaries and curbs.
        lanes: Mutable compiler lanes, lifted before spawns and traffic resolve.
        dividers: Paint paths associated with the compiler lanes.
        reference_lines: Trimmed road and parking-access centerlines in XY.
        sample_spacing_m: Maximum spacing of pavement samples, capped at 1 m.
        ground_margin_m: Ground extent beyond the outermost pavement.

    Returns:
        Lifted elements and dividers, followed by ground vertices and faces.
    """
    nodes = {node.node_id: node for node in topology.nodes}
    polygons = {
        element.element_id: Polygon(element.surface_world[:, :2])
        for element in elements
    }
    node_footprints = {key: polygons[key].buffer(1e-4) for key in nodes}
    endpoints = {
        road.road_id: (road.from_node_id, road.to_node_id) for road in topology.roads
    }
    endpoints.update(
        {
            access.access_id: (access.source_node_id, access.parking_lot_node_id)
            for access in topology.parking_accesses
        }
    )
    lines = {key: LineString(points[:, :2]) for key, points in reference_lines.items()}

    def lift(element_id: str, points: np.ndarray) -> np.ndarray:
        xy = np.asarray(points[:, :2], dtype=np.float64)
        if element_id in nodes:
            z = np.full(len(xy), nodes[element_id].z_m)
        else:
            first, last = endpoints[element_id]
            begin, end = nodes[first].z_m, nodes[last].z_m
            if begin == end:
                z = np.full(len(xy), begin)
            else:
                line = lines[element_id]
                t = np.clip(
                    shapely.line_locate_point(line, shapely.points(xy)) / line.length,
                    0,
                    1,
                )
                z = begin + (end - begin) * t**3 * (10 - 15 * t + 6 * t**2)
                for node_id in (first, last):
                    at_node = shapely.intersects_xy(
                        node_footprints[node_id], xy[:, 0], xy[:, 1]
                    )
                    z[at_node] = nodes[node_id].z_m
        return np.column_stack((xy, z)).astype(np.float32)

    clouds = []
    lifted = []
    spacing = min(sample_spacing_m, 1.0)
    distances = []
    for element in elements:
        key = element.element_id
        surface = lift(key, element.surface_world)
        lifted.append(
            replace(
                element,
                surface_world=surface,
                road_boundaries=tuple(
                    replace(boundary, polyline_world=lift(key, boundary.polyline_world))
                    for boundary in element.road_boundaries
                ),
                curbs=tuple(
                    replace(curb, polyline_world=lift(key, curb.polyline_world))
                    for curb in element.curbs
                ),
            )
        )
        region = polygons[key].buffer(3 * spacing)
        for xy in _pavement_samples(region, spacing):
            clouds.append(lift(key, xy))
            distances.append(shapely.distance(shapely.points(xy), polygons[key]))
    lane_by_id = {lane.lane_id: lane for lane in lanes}
    for lane in lanes:
        for name in ("centerline", "left_edge", "right_edge", "roadside_edge"):
            points = lift(lane.element_id, getattr(lane, name))
            setattr(lane, name, points)
    dividers = [
        replace(
            divider,
            polyline_world=lift(
                lane_by_id[divider.lane_edges[0][0]].element_id,
                divider.polyline_world,
            ),
        )
        for divider in dividers
    ]
    # A shared XY lattice avoids skinny triangles between independently sampled
    # lane rails and road edges, which otherwise create spurious surface normals.
    points = np.concatenate(clouds)
    scores = np.concatenate(distances)
    order = np.lexsort((scores, points[:, 1], points[:, 0]))
    points = points[order]
    _, indices = np.unique(points[:, :2], axis=0, return_index=True)
    vertices = points[indices]
    pavement = np.concatenate([element.surface_world for element in lifted])
    minimum = pavement[:, :2].min(axis=0) - ground_margin_m
    maximum = pavement[:, :2].max(axis=0) + ground_margin_m
    x, y = np.meshgrid(
        np.arange(minimum[0], maximum[0], 10),
        np.arange(minimum[1], maximum[1], 10),
    )
    xy = np.column_stack((x.ravel(), y.ravel()))
    footprint = shapely.union_all(list(polygons.values()))
    fine_region = footprint.buffer(3 * spacing)
    xy = xy[~shapely.intersects_xy(fine_region, xy[:, 0], xy[:, 1])]
    nearest = shapely.STRtree(shapely.points(vertices[:, :2])).nearest(
        shapely.points(xy)
    )
    distance = shapely.distance(shapely.points(xy), footprint)
    height = vertices[nearest, 2] * np.clip(
        1 - distance / max(ground_margin_m, 1), 0, 1
    )
    corners = np.asarray(
        [
            [minimum[0], minimum[1], 0],
            [maximum[0], minimum[1], 0],
            [maximum[0], maximum[1], 0],
            [minimum[0], maximum[1], 0],
        ],
        dtype=np.float32,
    )
    points = np.concatenate((vertices, np.column_stack((xy, height)), corners)).astype(
        np.float32
    )
    _, indices = np.unique(points[:, :2], axis=0, return_index=True)
    vertices = points[indices]
    lookup = {(float(x), float(y)): index for index, (x, y, _) in enumerate(vertices)}
    faces = []
    for triangle in triangulate(MultiPoint(vertices[:, :2])):
        face = [lookup[(x, y)] for x, y in triangle.exterior.coords[:3]]
        a, b, c = vertices[face, :2]
        area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(area) <= 1e-8:
            continue
        if area < 0:
            face.reverse()
        faces.append(face)
    faces = np.asarray(faces, dtype=np.int32)
    ground = GroundSnapper(vertices, faces)

    def drape(points: np.ndarray) -> np.ndarray:
        result = points.copy()
        result[:, 2] = ground._raycast(points)
        return result

    # Paint, curb bases and gameplay paths share the sampled physical surface.
    for lane in lanes:
        for name in ("centerline", "left_edge", "right_edge", "roadside_edge"):
            setattr(lane, name, drape(getattr(lane, name)))
    lifted = [
        replace(
            element,
            surface_world=drape(element.surface_world),
            road_boundaries=tuple(
                replace(item, polyline_world=drape(item.polyline_world))
                for item in element.road_boundaries
            ),
            curbs=tuple(
                replace(item, polyline_world=drape(item.polyline_world))
                for item in element.curbs
            ),
        )
        for element in lifted
    ]
    dividers = [
        replace(item, polyline_world=drape(item.polyline_world)) for item in dividers
    ]
    return lifted, dividers, vertices, faces
