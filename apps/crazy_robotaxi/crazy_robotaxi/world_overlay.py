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

"""Frame-aligned Crazy Robotaxi presentation layers."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as functional
from omnidreams_game_engine.camera import FThetaCameraModel
from omnidreams_game_engine.types import CameraCalibration
from torch import Tensor

from crazy_robotaxi.rules import (
    TaxiCameraMarkerProjection,
    TaxiGameSnapshot,
    project_taxi_markers_to_camera,
)

_PICKUP_COLOR = (-0.0745098, 0.4509804, -1.0)
"""Normalized RGB used for pickup waypoint geometry."""

_DROPOFF_COLOR = (0.5686275, 0.1764706, -0.60784316)
"""Normalized RGB used for drop-off waypoint geometry."""

_BLACK = (-1.0, -1.0, -1.0)
_WHITE = (1.0, 1.0, 1.0)
_LABEL_BACKGROUND = (-0.9372549, -0.9372549, -0.90588236)

_GLYPHS = {
    "C": ("1111", "1000", "1000", "1000", "1000", "1000", "1111"),
    "D": ("1110", "1001", "1001", "1001", "1001", "1001", "1110"),
    "F": ("1111", "1000", "1000", "1110", "1000", "1000", "1000"),
    "I": ("111", "010", "010", "010", "010", "010", "111"),
    "K": ("1001", "1010", "1100", "1100", "1010", "1001", "1001"),
    "O": ("0110", "1001", "1001", "1001", "1001", "1001", "0110"),
    "P": ("1110", "1001", "1001", "1110", "1000", "1000", "1000"),
    "R": ("1110", "1001", "1001", "1110", "1010", "1001", "1001"),
    "U": ("1001", "1001", "1001", "1001", "1001", "1001", "0110"),
}
"""Tiny dependency-free glyphs used for world-anchored marker labels."""


def render_bev_overlay(
    bev_tchw: Tensor,
    *,
    width: int,
    height: int,
) -> Tensor:
    """Render raw BEV frames into frame-aligned transparent RGBA layers.

    Args:
        bev_tchw: Floating-point BEV frames in ``[T,3,H,W]``.
        width: Output width in pixels.
        height: Output height in pixels.

    Returns:
        Floating-point layers in ``[T,4,H,W]`` with the BEV panel in the
        bottom-right corner and alpha in ``[0,1]``.

    Raises:
        ValueError: The BEV tensor or output dimensions are invalid.
    """
    if bev_tchw.ndim != 4 or bev_tchw.shape[1] != 3:
        raise ValueError("BEV frames must use [T,3,H,W]")
    if not bev_tchw.is_floating_point():
        raise ValueError("BEV overlays require floating-point frames")
    if width <= 0 or height <= 0:
        raise ValueError("BEV overlay dimensions must be positive")

    overlay = torch.zeros(
        (int(bev_tchw.shape[0]), 4, height, width),
        device=bev_tchw.device,
        dtype=bev_tchw.dtype,
    )
    size = min(width // 4, height // 3)
    if size <= 4:
        return overlay

    panel = functional.interpolate(
        bev_tchw,
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )
    margin = max(8, min(width, height) // 80)
    top = height - size - margin
    left = width - size - margin
    target = overlay[:, :, top : top + size, left : left + size]
    target[:, :3].copy_(panel)
    target[:, 3].fill_(0.82)
    target[:, :, :2, :].fill_(1.0)
    target[:, :, -2:, :].fill_(1.0)
    target[:, :, :, :2].fill_(1.0)
    target[:, :, :, -2:].fill_(1.0)
    return overlay


def render_waypoint_layers(
    snapshots: Sequence[object],
    rig_poses_world: np.ndarray,
    calibration: CameraCalibration,
    *,
    width: int,
    height: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Render projected waypoint geometry as transparent RGBA layers.

    Args:
        snapshots: Per-frame taxi game snapshots.
        rig_poses_world: Camera-rig poses in ``[T,4,4]``.
        calibration: Camera calibration used for world projection.
        width: Output width in pixels.
        height: Output height in pixels.
        device: Device receiving the resulting tensor.
        dtype: Floating-point output dtype.

    Returns:
        Floating-point layers in ``[T,4,H,W]`` with normalized RGB and
        alpha in ``[0,1]``.

    Raises:
        ValueError: Inputs are not frame-aligned or output dimensions are invalid.
        TypeError: A frame does not contain a taxi game snapshot.
    """
    frame_count = len(snapshots)
    poses = np.asarray(rig_poses_world, dtype=np.float32)
    if poses.shape != (frame_count, 4, 4):
        raise ValueError("Waypoint snapshots and camera poses must align")
    if width <= 0 or height <= 0:
        raise ValueError("Waypoint layer dimensions must be positive")
    if not dtype.is_floating_point:
        raise ValueError("Waypoint layers require a floating-point dtype")
    camera = FThetaCameraModel(
        calibration,
        output_width=width,
        output_height=height,
    )
    layers = torch.zeros(
        (frame_count, 4, height, width),
        device=device,
        dtype=dtype,
    )
    for frame_index, snapshot in enumerate(snapshots):
        if not isinstance(snapshot, TaxiGameSnapshot):
            raise TypeError("Waypoint layer received a non-taxi game snapshot")
        if snapshot.session_state != "playing":
            continue
        projections = project_taxi_markers_to_camera(
            snapshot,
            poses[frame_index],
            camera,
            image_width=width,
            image_height=height,
        )
        color = _PICKUP_COLOR if snapshot.phase == "seeking_pickup" else _DROPOFF_COLOR
        label = "PICKUP" if snapshot.phase == "seeking_pickup" else "DROPOFF"
        ring_edges = []
        beacons = []
        for projection in projections:
            ring_edges.extend(projection.ring_edges_uv)
            beacons.append((projection.anchor_uv, _beacon_top(projection)))

        layer = layers[frame_index]
        _paint_lines(layer, ring_edges, width, height, 7, _BLACK, 220.0 / 255.0)
        _paint_lines(layer, ring_edges, width, height, 4, color, 245.0 / 255.0)
        _paint_lines(layer, beacons, width, height, 9, _BLACK, 235.0 / 255.0)
        _paint_lines(layer, beacons, width, height, 5, color, 1.0)
        for projection, (_, top) in zip(projections, beacons, strict=True):
            _paint_marker_anchor(layer, projection.anchor_uv, width, height, color)
            _paint_marker_label(layer, top, label, width, height, color)
    return layers


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


def _paint_lines(
    layer: Tensor,
    lines: Sequence[tuple[tuple[float, float], tuple[float, float]]],
    width: int,
    height: int,
    width_px: int,
    color: tuple[float, float, float],
    alpha: float,
) -> None:
    if not lines:
        return
    pixels = [
        _line_pixels(
            start,
            end,
            width=width,
            height=height,
            width_px=width_px,
        )
        for start, end in lines
    ]
    _paint_pixels(
        layer,
        np.concatenate([points[0] for points in pixels]),
        np.concatenate([points[1] for points in pixels]),
        color,
        alpha,
    )


def _paint_marker_anchor(
    layer: Tensor,
    anchor: tuple[float, float],
    width: int,
    height: int,
    color: tuple[float, float, float],
) -> None:
    center_x, center_y = int(round(anchor[0])), int(round(anchor[1]))
    x, y = _disk_pixels(center_x, center_y, 9, width, height)
    _paint_pixels(layer, x, y, color, 1.0)
    x, y = _ring_pixels(center_x, center_y, 6, 9, width, height)
    _paint_pixels(layer, x, y, _WHITE, 1.0)


def _paint_marker_label(
    layer: Tensor,
    top: tuple[float, float],
    label: str,
    width: int,
    height: int,
    color: tuple[float, float, float],
) -> None:
    scale = max(1, min(width, height) // 360)
    text_x, text_y = _text_pixels(label, scale)
    text_width = int(text_x.max()) + 1
    text_height = int(text_y.max()) + 1
    center_x = int(round(top[0]))
    text_left = center_x - text_width // 2
    text_top = int(round(top[1])) - text_height - 10 * scale
    left = text_left - 4 * scale
    top_y = text_top - 3 * scale
    right = text_left + text_width + 4 * scale
    bottom = text_top + text_height + 3 * scale
    x, y = _rectangle_pixels(left, top_y, right, bottom, width, height)
    _paint_pixels(layer, x, y, _LABEL_BACKGROUND, 225.0 / 255.0)
    border = max(1, scale)
    border_lines = (
        ((left, top_y), (right, top_y)),
        ((right, top_y), (right, bottom)),
        ((right, bottom), (left, bottom)),
        ((left, bottom), (left, top_y)),
    )
    _paint_lines(layer, border_lines, width, height, border, color, 1.0)
    _paint_pixels(layer, text_x + text_left, text_y + text_top, color, 1.0)


def _paint_pixels(
    layer: Tensor,
    x: np.ndarray,
    y: np.ndarray,
    color: tuple[float, float, float],
    alpha: float,
) -> None:
    height, width = int(layer.shape[-2]), int(layer.shape[-1])
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x = x[valid]
    y = y[valid]
    if x.size == 0:
        return
    x_tensor = torch.from_numpy(x).to(device=layer.device)
    y_tensor = torch.from_numpy(y).to(device=layer.device)
    layer[:3, y_tensor, x_tensor] = torch.tensor(
        color,
        device=layer.device,
        dtype=layer.dtype,
    ).view(3, 1)
    layer[3, y_tensor, x_tensor] = alpha


def _line_pixels(
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    width: int,
    height: int,
    width_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    delta_x = float(end[0] - start[0])
    delta_y = float(end[1] - start[1])
    sample_count = max(2, int(math.ceil(max(abs(delta_x), abs(delta_y)))) + 1)
    center_x = np.rint(np.linspace(start[0], end[0], sample_count)).astype(np.int32)
    center_y = np.rint(np.linspace(start[1], end[1], sample_count)).astype(np.int32)
    radius = max(0.0, (float(width_px) - 1.0) / 2.0)
    line_x = []
    line_y = []
    integer_radius = int(math.ceil(radius))
    for offset_y in range(-integer_radius, integer_radius + 1):
        for offset_x in range(-integer_radius, integer_radius + 1):
            if offset_x * offset_x + offset_y * offset_y > radius * radius:
                continue
            x = center_x + offset_x
            y = center_y + offset_y
            valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
            line_x.append(x[valid])
            line_y.append(y[valid])
    return np.concatenate(line_x), np.concatenate(line_y)


def _disk_pixels(
    center_x: int,
    center_y: int,
    radius: int,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.arange(-radius, radius + 1, dtype=np.int32)
    offset_x, offset_y = np.meshgrid(offsets, offsets)
    selected = offset_x * offset_x + offset_y * offset_y <= radius * radius
    x = center_x + offset_x[selected]
    y = center_y + offset_y[selected]
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    return x[valid], y[valid]


def _ring_pixels(
    center_x: int,
    center_y: int,
    inner_radius: int,
    outer_radius: int,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.arange(-outer_radius, outer_radius + 1, dtype=np.int32)
    offset_x, offset_y = np.meshgrid(offsets, offsets)
    distance_squared = offset_x * offset_x + offset_y * offset_y
    selected = (distance_squared >= inner_radius * inner_radius) & (
        distance_squared <= outer_radius * outer_radius
    )
    x = center_x + offset_x[selected]
    y = center_y + offset_y[selected]
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    return x[valid], y[valid]


def _rectangle_pixels(
    left: int,
    top: int,
    right: int,
    bottom: int,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    clipped_left = max(0, left)
    clipped_top = max(0, top)
    clipped_right = min(width - 1, right)
    clipped_bottom = min(height - 1, bottom)
    if clipped_left > clipped_right or clipped_top > clipped_bottom:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    x, y = np.meshgrid(
        np.arange(clipped_left, clipped_right + 1, dtype=np.int32),
        np.arange(clipped_top, clipped_bottom + 1, dtype=np.int32),
    )
    return x.ravel(), y.ravel()


def _text_pixels(text: str, scale: int) -> tuple[np.ndarray, np.ndarray]:
    x_values = []
    y_values = []
    cursor = 0
    for character in text:
        glyph = _GLYPHS[character]
        for row, bits in enumerate(glyph):
            for column, bit in enumerate(bits):
                if bit == "0":
                    continue
                for offset_y in range(scale):
                    for offset_x in range(scale):
                        x_values.append(cursor + column * scale + offset_x)
                        y_values.append(row * scale + offset_y)
        cursor += (len(glyph[0]) + 1) * scale
    return np.asarray(x_values, dtype=np.int32), np.asarray(y_values, dtype=np.int32)


__all__ = ["render_bev_overlay", "render_waypoint_layers"]
