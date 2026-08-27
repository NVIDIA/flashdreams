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
from functools import lru_cache

import numpy as np
import torch
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

_BLACK_RING = 0
_PICKUP_RING = 1
_DROPOFF_RING = 2
_BLACK_BEACON = 3
_PICKUP_SOLID = 4
_DROPOFF_SOLID = 5
_WHITE_SOLID = 6
_LABEL_PANEL = 7

_WAYPOINT_PALETTE = np.asarray(
    (
        (*_BLACK, 220.0 / 255.0),
        (*_PICKUP_COLOR, 245.0 / 255.0),
        (*_DROPOFF_COLOR, 245.0 / 255.0),
        (*_BLACK, 235.0 / 255.0),
        (*_PICKUP_COLOR, 1.0),
        (*_DROPOFF_COLOR, 1.0),
        (*_WHITE, 1.0),
        (*_LABEL_BACKGROUND, 225.0 / 255.0),
    ),
    dtype=np.float32,
)
"""Painter styles uploaded once for each frame-aligned waypoint batch."""

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
    styles = np.full((frame_count, height, width), -1, dtype=np.int8)
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
        if snapshot.phase == "seeking_pickup":
            ring_style = _PICKUP_RING
            solid_style = _PICKUP_SOLID
        else:
            ring_style = _DROPOFF_RING
            solid_style = _DROPOFF_SOLID
        label = "PICKUP" if snapshot.phase == "seeking_pickup" else "DROPOFF"
        ring_edges = []
        beacons = []
        for projection in projections:
            ring_edges.extend(projection.ring_edges_uv)
            beacons.append((projection.anchor_uv, _beacon_top(projection)))

        layer = styles[frame_index]
        _paint_lines(layer, ring_edges, width, height, 7, _BLACK_RING)
        _paint_lines(layer, ring_edges, width, height, 4, ring_style)
        _paint_lines(layer, beacons, width, height, 9, _BLACK_BEACON)
        _paint_lines(layer, beacons, width, height, 5, solid_style)
        for projection, (_, top) in zip(projections, beacons, strict=True):
            _paint_marker_anchor(
                layer,
                projection.anchor_uv,
                width,
                height,
                solid_style,
            )
            _paint_marker_label(
                layer,
                top,
                label,
                width,
                height,
                solid_style,
            )
    return _materialize_waypoint_layers(styles, device=device, dtype=dtype)


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
    layer: np.ndarray,
    lines: Sequence[tuple[tuple[float, float], tuple[float, float]]],
    width: int,
    height: int,
    width_px: int,
    style: int,
) -> None:
    if not lines:
        return
    x, y = _line_pixels(lines, width=width, height=height, width_px=width_px)
    _paint_pixels(
        layer,
        x,
        y,
        style,
    )


def _paint_marker_anchor(
    layer: np.ndarray,
    anchor: tuple[float, float],
    width: int,
    height: int,
    style: int,
) -> None:
    center_x, center_y = int(round(anchor[0])), int(round(anchor[1]))
    x, y = _disk_pixels(center_x, center_y, 9, width, height)
    _paint_pixels(layer, x, y, style)
    x, y = _ring_pixels(center_x, center_y, 6, 9, width, height)
    _paint_pixels(layer, x, y, _WHITE_SOLID)


def _paint_marker_label(
    layer: np.ndarray,
    top: tuple[float, float],
    label: str,
    width: int,
    height: int,
    style: int,
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
    _paint_pixels(layer, x, y, _LABEL_PANEL)
    border = max(1, scale)
    border_lines = (
        ((left, top_y), (right, top_y)),
        ((right, top_y), (right, bottom)),
        ((right, bottom), (left, bottom)),
        ((left, bottom), (left, top_y)),
    )
    _paint_lines(layer, border_lines, width, height, border, style)
    _paint_pixels(
        layer,
        text_x + text_left,
        text_y + text_top,
        style,
        clip=True,
    )


def _paint_pixels(
    layer: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    style: int,
    *,
    clip: bool = False,
) -> None:
    if clip:
        height, width = layer.shape
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        x = x[valid]
        y = y[valid]
    if x.size == 0:
        return
    layer[y, x] = style


def _line_pixels(
    lines: Sequence[tuple[tuple[float, float], tuple[float, float]]],
    *,
    width: int,
    height: int,
    width_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    endpoints = np.asarray(lines, dtype=np.float64)
    starts = endpoints[:, 0]
    deltas = endpoints[:, 1] - starts
    sample_counts = np.maximum(
        2,
        np.ceil(np.max(np.abs(deltas), axis=1)).astype(np.int32) + 1,
    )
    sample_index = np.arange(int(sample_counts.max()), dtype=np.float64)[None]
    fraction = sample_index / (sample_counts[:, None] - 1)
    selected_samples = sample_index < sample_counts[:, None]
    center_x = np.rint(starts[:, 0, None] + deltas[:, 0, None] * fraction).astype(
        np.int32
    )
    center_y = np.rint(starts[:, 1, None] + deltas[:, 1, None] * fraction).astype(
        np.int32
    )
    offset_x, offset_y = _line_offsets(width_px)
    x = center_x[:, :, None] + offset_x
    y = center_y[:, :, None] + offset_y
    valid = (
        selected_samples[:, :, None] & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    )
    return x[valid], y[valid]


@lru_cache(maxsize=None)
def _line_offsets(width_px: int) -> tuple[np.ndarray, np.ndarray]:
    radius = max(0.0, (float(width_px) - 1.0) / 2.0)
    integer_radius = int(math.ceil(radius))
    offsets = np.arange(-integer_radius, integer_radius + 1, dtype=np.int32)
    offset_x, offset_y = np.meshgrid(offsets, offsets)
    selected = offset_x * offset_x + offset_y * offset_y <= radius * radius
    return offset_x[selected], offset_y[selected]


def _disk_pixels(
    center_x: int,
    center_y: int,
    radius: int,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    offset_x, offset_y = _disk_offsets(radius)
    x = center_x + offset_x
    y = center_y + offset_y
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    return x[valid], y[valid]


@lru_cache(maxsize=None)
def _disk_offsets(radius: int) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.arange(-radius, radius + 1, dtype=np.int32)
    offset_x, offset_y = np.meshgrid(offsets, offsets)
    selected = offset_x * offset_x + offset_y * offset_y <= radius * radius
    return offset_x[selected], offset_y[selected]


def _ring_pixels(
    center_x: int,
    center_y: int,
    inner_radius: int,
    outer_radius: int,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    offset_x, offset_y = _ring_offsets(inner_radius, outer_radius)
    x = center_x + offset_x
    y = center_y + offset_y
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    return x[valid], y[valid]


@lru_cache(maxsize=None)
def _ring_offsets(
    inner_radius: int,
    outer_radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.arange(-outer_radius, outer_radius + 1, dtype=np.int32)
    offset_x, offset_y = np.meshgrid(offsets, offsets)
    distance_squared = offset_x * offset_x + offset_y * offset_y
    selected = (distance_squared >= inner_radius * inner_radius) & (
        distance_squared <= outer_radius * outer_radius
    )
    return offset_x[selected], offset_y[selected]


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


@lru_cache(maxsize=None)
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


def _materialize_waypoint_layers(
    styles: np.ndarray,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> Tensor:
    style_tensor = torch.from_numpy(styles).to(device=device)
    transparent = torch.zeros((1, 4), device=device, dtype=dtype)
    palette = torch.cat(
        (
            transparent,
            torch.as_tensor(_WAYPOINT_PALETTE, device=device, dtype=dtype),
        )
    )
    # A GPU nonzero() has a data-dependent output size and therefore waits for
    # all earlier work on the stream before returning to Python. Dense palette
    # lookup keeps materialization asynchronous; style -1 selects transparent.
    return palette[style_tensor.to(torch.int64) + 1].permute(0, 3, 1, 2).contiguous()


__all__ = ["render_waypoint_layers"]
