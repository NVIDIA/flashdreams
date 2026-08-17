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


def write_game_map_preview(source: Path, destination: Path) -> Path:
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
            f'fill="{fill}" stroke="#222" stroke-width="1"/>'
        )
        center = np.mean(element.surface_world[:-1, :2], axis=0)
        x, y = convert(center)
        lines.append(
            f'<text x="{x:.2f}" y="{y:.2f}" fill="white" font-size="11" '
            f'text-anchor="middle">{html.escape(element.element_id)}</text>'
        )
        for name, port_x, port_y, _heading, connected in element.ports:
            x, y = convert(np.asarray([port_x, port_y]))
            color = "#25d366" if connected else "#ff3b30"
            lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>')
            lines.append(
                f'<text x="{x + 5:.2f}" y="{y - 5:.2f}" fill="#111" '
                f'font-size="9">{html.escape(name)}</text>'
            )
    for polygon in game_map.road_marking_polygons_world:
        lines.append(
            f'<polygon points="{_points(polygon[:, :2], convert)}" '
            'fill="#f4f4f4" stroke="none"/>'
        )
    for lane in game_map.lanes:
        color = "#ffd60a" if lane.allows_taxi_stops else "#64d2ff"
        lines.append(
            f'<polyline points="{_points(lane.centerline_world[:, :2], convert)}" '
            f'fill="none" stroke="{color}" stroke-width="1.5"/>'
        )
    for segment in game_map.collision_segments_world:
        lines.append(
            f'<polyline points="{_points(segment[:, :2], convert)}" '
            'fill="none" stroke="#ff453a" stroke-width="2.5"/>'
        )
    lines.append("</svg>")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination
