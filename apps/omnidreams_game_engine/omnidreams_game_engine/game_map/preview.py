# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""SVG previews for semantic game maps."""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np

from omnidreams_game_engine.game_map.loader import load_game_map


def _points(points: np.ndarray, transform: object) -> str:
    convert = transform
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in (convert(point) for point in points))


def _label(text: str, point: np.ndarray, transform: object, color: str) -> str:
    x, y = transform(point)
    return (
        f'<text x="{x + 3.0:.2f}" y="{y - 3.0:.2f}" fill="{color}" '
        'font-family="monospace" font-size="4" font-weight="600" '
        'stroke="#f8f4ea" stroke-width="0.25" paint-order="stroke" '
        f'stroke-linejoin="round">{html.escape(text)}</text>'
    )


def _point_at_distance(
    points: np.ndarray, distance_m: float
) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    cumulative = np.concatenate((np.zeros(1), np.cumsum(lengths)))
    index = min(
        int(np.searchsorted(cumulative, distance_m, side="right") - 1),
        len(lengths) - 1,
    )
    index = max(0, index)
    alpha = (distance_m - cumulative[index]) / max(float(lengths[index]), 1.0e-9)
    point = points[index] + alpha * (points[index + 1] - points[index])
    tangent = points[index + 1, :2] - points[index, :2]
    tangent /= max(float(np.linalg.norm(tangent)), 1.0e-9)
    return point, tangent


def write_game_map_preview(
    source: Path, destination: Path, *, include_annotations: bool = True
) -> Path:
    """Render a top-down semantic-map preview as SVG."""
    game_map = load_game_map(source)
    points = np.concatenate(
        [element.surface_world[:, :2] for element in game_map.elements], axis=0
    )
    x_min, y_min = np.min(points, axis=0) - 8.0
    x_max, y_max = np.max(points, axis=0) + 8.0
    width = max(1.0, float(x_max - x_min))
    height = max(1.0, float(y_max - y_min))
    scale = min(1000.0 / width, 800.0 / height)

    def convert(point: np.ndarray) -> tuple[float, float]:
        return (
            (float(point[0]) - float(x_min)) * scale,
            (float(y_max) - float(point[1])) * scale,
        )

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width * scale:.2f} {height * scale:.2f}">',
        '<rect width="100%" height="100%" fill="#e9e2d2"/>',
    ]
    for element in game_map.elements:
        fill = "#14a878" if element.element_type == "parking_lot" else "#4b4f55"
        lines.append(
            f'<polygon points="{_points(element.surface_world[:, :2], convert)}" '
            f'fill="{fill}" stroke="none"/>'
        )
    for polygon in game_map.road_marking_polygons_world:
        lines.append(
            f'<polygon points="{_points(polygon[:, :2], convert)}" '
            'fill="#f4f4f4" stroke="none"/>'
        )
    for marking in game_map.line_markings:
        color = "#ffd60a" if marking.color == "YELLOW" else "#f4f4f4"
        lines.append(
            f'<polyline points="{_points(marking.polyline_world[:, :2], convert)}" '
            f'fill="none" stroke="{color}" stroke-width="1.5"/>'
        )
    for divider in game_map.lane_dividers:
        color = "#ffd60a" if divider.color == "YELLOW" else "#f4f4f4"
        lines.append(
            f'<polyline points="{_points(divider.polyline_world[:, :2], convert)}" '
            f'fill="none" stroke="{color}" stroke-width="1.5"/>'
        )
    for element in game_map.elements:
        for boundary in element.road_boundaries:
            lines.append(
                f'<polyline points="{_points(boundary.polyline_world[:, :2], convert)}" '
                'fill="none" stroke="#8b929c" stroke-width="1.5"/>'
            )
        for curb in element.curbs:
            lines.append(
                f'<polyline points="{_points(curb.polyline_world[:, :2], convert)}" '
                'fill="none" stroke="#5f6673" stroke-width="2.5"/>'
            )
    for traffic in game_map.traffic:
        lines.append(
            f'<polyline points="{_points(traffic.centerline_world[:, :2], convert)}" '
            'fill="none" stroke="#ef476f" stroke-width="1.2" '
            'stroke-dasharray="3 2" opacity="0.8"/>'
        )
        point, forward = _point_at_distance(
            traffic.centerline_world, traffic.start_distance_m
        )
        left = np.asarray([-forward[1], forward[0]])
        half_length = traffic.dimensions_lwh_m[0] * 0.5
        half_width = traffic.dimensions_lwh_m[1] * 0.5
        corners = np.asarray(
            [
                point[:2] + forward * half_length + left * half_width,
                point[:2] + forward * half_length - left * half_width,
                point[:2] - forward * half_length - left * half_width,
                point[:2] - forward * half_length + left * half_width,
            ]
        )
        lines.append(
            f'<polygon points="{_points(corners, convert)}" fill="#ef476f" '
            'stroke="#ffffff" stroke-width="0.8"/>'
        )
        if include_annotations:
            lines.append(_label(traffic.vehicle_id, point, convert, "#9f1239"))
    if include_annotations:
        lane_by_element = {
            lane.element_id: lane
            for lane in game_map.lanes
            if lane.conditioning_visible and ":connector:" not in lane.lane_id
        }
        for road in game_map.topology.roads:
            lane = lane_by_element[road.road_id]
            point = lane.centerline_world[len(lane.centerline_world) // 2, :2]
            lines.append(
                _label(
                    f"{road.road_id} [road:{road.profile_id}; "
                    f"{road.from_node_id}→{road.to_node_id}]",
                    point,
                    convert,
                    "#17233d",
                )
            )
        for access in game_map.topology.parking_accesses:
            lane = lane_by_element[access.access_id]
            point = lane.centerline_world[len(lane.centerline_world) // 2, :2]
            lines.append(
                _label(
                    f"{access.access_id} [parking access; "
                    f"{access.source_node_id}→{access.parking_lot_node_id}]",
                    point,
                    convert,
                    "#064e3b",
                )
            )
        node_colors = {
            "intersection": "#2d6cdf",
            "road_joint": "#0891b2",
            "cul_de_sac": "#8b5cf6",
            "driveway": "#f59e0b",
            "parking_lot": "#059669",
        }
        for node in game_map.topology.nodes:
            point = np.asarray([node.x_m, node.y_m])
            x, y = convert(point)
            lines.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2" '
                f'fill="{node_colors[node.node_type]}" stroke="#ffffff" '
                'stroke-width="1"/>'
            )
            lines.append(
                _label(
                    f"{node.node_id} [node:{node.node_type}]",
                    point,
                    convert,
                    "#111827",
                )
            )
    lines.append("</svg>")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination
