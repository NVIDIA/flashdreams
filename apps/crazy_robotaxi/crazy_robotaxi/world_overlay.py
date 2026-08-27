# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ImGui draw-list geometry for world-anchored Crazy Robotaxi markers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from omnidreams_game_engine.camera import FThetaCameraModel
from omnidreams_game_engine.types import CameraCalibration

from crazy_robotaxi.rules import (
    TaxiCameraMarkerProjection,
    TaxiGameSnapshot,
    project_taxi_markers_to_camera,
)

_PICKUP_RGB = (118.0 / 255.0, 185.0 / 255.0, 0.0)
"""NVIDIA green used for pickup waypoint geometry."""

_DROPOFF_RGB = (200.0 / 255.0, 150.0 / 255.0, 50.0 / 255.0)
"""Amber used for drop-off waypoint geometry."""

_WHITE = (1.0, 1.0, 1.0, 1.0)
_LABEL_BACKGROUND = (8.0 / 255.0, 8.0 / 255.0, 12.0 / 255.0, 225.0 / 255.0)


def project_waypoints(
    snapshot: TaxiGameSnapshot,
    rig_to_world: npt.NDArray[np.float32],
    calibration: CameraCalibration,
    *,
    width: int,
    height: int,
) -> tuple[TaxiCameraMarkerProjection, ...]:
    """Project the current world waypoints into presentation pixels."""
    if width <= 0 or height <= 0:
        raise ValueError("Waypoint projection dimensions must be positive")
    pose = np.asarray(rig_to_world, dtype=np.float32)
    if pose.shape != (4, 4):
        raise ValueError("Waypoint projection requires one [4,4] rig pose")
    if snapshot.session_state != "playing":
        return ()
    camera = FThetaCameraModel(
        calibration,
        output_width=width,
        output_height=height,
    )
    return project_taxi_markers_to_camera(
        snapshot,
        pose,
        camera,
        image_width=width,
        image_height=height,
    )


def draw_waypoints(
    imgui: Any,
    projections: Sequence[TaxiCameraMarkerProjection],
    *,
    phase: Literal["seeking_pickup", "to_dropoff"],
    width: int,
    height: int,
) -> None:
    """Draw projected world markers beneath all ImGui HUD windows."""
    if not projections:
        return
    draw_list = imgui.get_background_draw_list()
    rgb = _PICKUP_RGB if phase == "seeking_pickup" else _DROPOFF_RGB
    ring_color = _imgui_color(imgui, (*rgb, 245.0 / 255.0))
    solid_color = _imgui_color(imgui, (*rgb, 1.0))
    black_ring = _imgui_color(imgui, (0.0, 0.0, 0.0, 220.0 / 255.0))
    black_beacon = _imgui_color(imgui, (0.0, 0.0, 0.0, 235.0 / 255.0))
    white = _imgui_color(imgui, _WHITE)
    panel = _imgui_color(imgui, _LABEL_BACKGROUND)
    label = "PICKUP" if phase == "seeking_pickup" else "DROPOFF"

    for projection in projections:
        for start, end in projection.ring_edges_uv:
            draw_list.add_line(
                _point(imgui, start),
                _point(imgui, end),
                black_ring,
                7.0,
            )
    for projection in projections:
        for start, end in projection.ring_edges_uv:
            draw_list.add_line(
                _point(imgui, start),
                _point(imgui, end),
                ring_color,
                4.0,
            )

    beacon_tops = tuple(_beacon_top(projection) for projection in projections)
    for projection, top in zip(projections, beacon_tops, strict=True):
        draw_list.add_line(
            _point(imgui, projection.anchor_uv),
            _point(imgui, top),
            black_beacon,
            9.0,
        )
    for projection, top in zip(projections, beacon_tops, strict=True):
        draw_list.add_line(
            _point(imgui, projection.anchor_uv),
            _point(imgui, top),
            solid_color,
            5.0,
        )
    for projection, top in zip(projections, beacon_tops, strict=True):
        anchor = _point(imgui, projection.anchor_uv)
        draw_list.add_circle_filled(anchor, 9.0, solid_color)
        draw_list.add_circle(anchor, 7.5, white, 0, 3.0)
        _draw_label(
            imgui,
            draw_list,
            top,
            label,
            color=solid_color,
            panel=panel,
            scale=max(1, min(width, height) // 360),
        )


def _draw_label(
    imgui: Any,
    draw_list: Any,
    top: tuple[float, float],
    label: str,
    *,
    color: int,
    panel: int,
    scale: int,
) -> None:
    text_size = imgui.calc_text_size(label)
    text_width = float(text_size.x)
    text_height = float(text_size.y)
    text_left = float(top[0]) - text_width / 2.0
    text_top = float(top[1]) - text_height - 10.0 * scale
    padding_x = 4.0 * scale
    padding_y = 3.0 * scale
    panel_min = imgui.ImVec2(text_left - padding_x, text_top - padding_y)
    panel_max = imgui.ImVec2(
        text_left + text_width + padding_x,
        text_top + text_height + padding_y,
    )
    draw_list.add_rect_filled(panel_min, panel_max, panel)
    draw_list.add_rect(panel_min, panel_max, color, 0.0, float(max(1, scale)))
    draw_list.add_text(imgui.ImVec2(text_left, text_top), color, label)


def _beacon_top(projection: TaxiCameraMarkerProjection) -> tuple[float, float]:
    anchor_x, anchor_y = projection.anchor_uv
    if projection.beacon_top_uv is None:
        return anchor_x, anchor_y - 64.0
    vector_x = float(projection.beacon_top_uv[0] - anchor_x)
    vector_y = float(projection.beacon_top_uv[1] - anchor_y)
    length = max(1.0, math.hypot(vector_x, vector_y))
    display_length = min(170.0, max(52.0, length))
    return (
        anchor_x + vector_x * display_length / length,
        anchor_y + vector_y * display_length / length,
    )


def _point(imgui: Any, value: tuple[float, float]) -> Any:
    return imgui.ImVec2(float(value[0]), float(value[1]))


def _imgui_color(imgui: Any, rgba: tuple[float, float, float, float]) -> int:
    return int(imgui.color_convert_float4_to_u32(imgui.ImVec4(*rgba)))


__all__ = ["draw_waypoints", "project_waypoints"]
