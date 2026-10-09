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

"""Angled overhead point-cloud viewport with height colors and an ego vehicle."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor


class LidarView(str, Enum):
    """Available views of one calibrated LiDAR sweep."""

    POINT_CLOUD = "point-cloud"
    RANGE = "range"
    INTENSITY = "intensity"


@dataclass(frozen=True)
class LidarViewport:
    """Orthographic ego-frame view: x forward, y left, z up, in metres."""

    width: int = 832
    height: int = 480
    extent_m: float = 60.0
    elevation_degrees: float = 55.0
    azimuth_degrees: float = 25.0
    ego_length_m: float = 4.8
    ego_width_m: float = 2.0

    def __post_init__(self) -> None:
        if min(self.width, self.height) < 1 or not all(
            math.isfinite(v) and v > 0
            for v in (self.extent_m, self.ego_length_m, self.ego_width_m)
        ):
            raise ValueError("Viewport dimensions and extents must be positive")
        if (
            not math.isfinite(self.azimuth_degrees)
            or not 0 < self.elevation_degrees <= 90
        ):
            raise ValueError("Elevation must lie in (0, 90] and azimuth must be finite")

    def render_sweep(
        self,
        ranges: Tensor,
        ray_directions: Tensor,
        *,
        sensor_to_ego: Tensor | None = None,
        view: LidarView = LidarView.POINT_CLOUD,
    ) -> Tensor:
        """Render metric range/intensity/validity to RGB using a fixed color scale.

        Range uses blue through red over 0–100 metres; intensity uses black
        through white over 0–1. Both panoramas preserve their aspect ratio,
        with black padding and invalid returns. Point clouds use height colors.
        """
        view = LidarView(view)
        if view is LidarView.POINT_CLOUD:
            return self.render(
                range_image_points(ranges, ray_directions, sensor_to_ego=sensor_to_ego)
            )
        if ranges.ndim != 3 or ranges.shape[0] != 3 or min(ranges.shape[1:]) < 1:
            raise ValueError("Expected [3,H,W] range/intensity/validity")
        valid = torch.isfinite(ranges).all(0) & (ranges[2] >= 0.5) & (ranges[0] > 0)
        value = torch.nan_to_num(
            ranges[0] / 100 if view is LidarView.RANGE else ranges[1]
        )
        value = value.float().clamp(0, 1)
        if view is LidarView.RANGE:
            # Fixed false colors preserve distance comparisons between sweeps.
            colors = torch.stack((value, 1 - (2 * value - 1).abs(), 1 - value))
        else:
            colors = value.expand(3, -1, -1)
        colors = colors * valid[None]
        scale = min(self.width / ranges.shape[2], self.height / ranges.shape[1])
        width = max(1, min(self.width, round(ranges.shape[2] * scale)))
        height = max(1, min(self.height, round(ranges.shape[1] * scale)))
        resized = torch.nn.functional.interpolate(
            colors[None], size=(height, width), mode="nearest"
        )[0]
        image = colors.new_zeros(3, self.height, self.width)
        top, left = (self.height - height) // 2, (self.width - width) // 2
        image[:, top : top + height, left : left + width] = resized
        return image

    def render(self, points: Tensor) -> Tensor:
        """Render ``[N,3]`` ego-frame points to float RGB ``[3,H,W]`` in [0,1]."""
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Expected [N,3] points in ego-frame metres")
        device = points.device
        xyz = points.to(torch.float32)
        xyz = xyz[
            torch.isfinite(xyz).all(1) & (xyz[:, :2].abs().amax(1) <= self.extent_m)
        ]
        # A solid, bright ego box remains distinguishable from the height palette.
        ex = torch.linspace(
            -self.ego_length_m / 2, self.ego_length_m / 2, 45, device=device
        )
        ey = torch.linspace(
            -self.ego_width_m / 2, self.ego_width_m / 2, 21, device=device
        )
        x, y = torch.meshgrid(ex, ey, indexing="ij")
        ego = torch.stack(
            (x.flatten(), y.flatten(), torch.full_like(x.flatten(), 1.5)), 1
        )
        t = ((xyz[:, 2] + 2) / 6).clamp(0, 1)
        anchors = torch.tensor(
            [[0.1, 0.25, 0.9], [0.05, 0.9, 0.85], [1.0, 0.85, 0.1], [1.0, 0.2, 0.15]],
            device=device,
        )
        segment = (t * 3).long().clamp(max=2)
        colors = torch.lerp(
            anchors[segment], anchors[segment + 1], (t * 3 - segment)[:, None]
        )
        ego_colors = (
            torch.tensor([1.0, 0.15, 0.75], device=device).expand(len(ego), 3).clone()
        )
        ego_colors[ego[:, 0] > self.ego_length_m * 0.32] = 1.0
        xyz = torch.cat((xyz, ego))
        colors = torch.cat((colors, ego_colors))
        az, el = (
            math.radians(self.azimuth_degrees),
            math.radians(self.elevation_degrees),
        )
        forward = xyz[:, 0] * math.cos(az) + xyz[:, 1] * math.sin(az)
        side = -xyz[:, 0] * math.sin(az) + xyz[:, 1] * math.cos(az)
        scale = min(self.width, self.height) / (2 * self.extent_m)
        u = (self.width / 2 - side * scale).round().long()
        v = (
            (
                self.height * 0.58
                - (forward * math.sin(el) + xyz[:, 2] * math.cos(el)) * scale
            )
            .round()
            .long()
        )
        depth = forward * math.cos(el) - xyz[:, 2] * math.sin(el)
        valid = (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        u, v, depth, colors = u[valid], v[valid], depth[valid], colors[valid]
        index = v * self.width + u
        z = torch.full((self.width * self.height,), torch.inf, device=device)
        z.scatter_reduce_(0, index, depth, reduce="amin", include_self=True)
        visible = depth == z[index]
        # Break equal-depth ties deterministically, including coincident returns.
        winner = torch.full_like(z, -1, dtype=torch.long)
        order = torch.arange(len(index), device=device)
        winner.scatter_reduce_(
            0, index[visible], order[visible], reduce="amax", include_self=True
        )
        image = (
            torch.tensor([0.025, 0.035, 0.065], device=device)
            .expand(self.width * self.height, 3)
            .clone()
        )
        occupied = winner >= 0
        image[occupied] = colors[winner[occupied]]
        return image.reshape(self.height, self.width, 3).permute(2, 0, 1).contiguous()


def range_image_points(
    ranges: Tensor, ray_directions: Tensor, *, sensor_to_ego: Tensor | None = None
) -> Tensor:
    """Unproject metric ``[3,H,W]`` range/intensity/validity with calibrated rays."""
    if (
        ranges.ndim != 3
        or ranges.shape[0] != 3
        or ray_directions.shape != (*ranges.shape[1:], 3)
    ):
        raise ValueError("Expected [3,H,W] range image and matching [H,W,3] rays")
    rays = ray_directions.to(device=ranges.device, dtype=torch.float32)
    valid = (
        (ranges[2] >= 0.5)
        & (ranges[0] > 0)
        & torch.isfinite(ranges).all(0)
        & torch.isfinite(rays).all(-1)
    )
    points = rays[valid] * ranges[0][valid, None]
    if sensor_to_ego is not None:
        if sensor_to_ego.shape != (4, 4):
            raise ValueError("sensor_to_ego must be a 4x4 transform")
        pose = sensor_to_ego.to(points)
        points = points @ pose[:3, :3].T + pose[:3, 3]
    return points
