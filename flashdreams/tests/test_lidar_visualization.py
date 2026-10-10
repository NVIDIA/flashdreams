# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for calibrated LiDAR projection and viewport rendering."""

from __future__ import annotations

import pytest
import torch

from flashdreams.core.visualization.lidar import (
    LidarView,
    LidarViewport,
    range_image_points,
)

pytestmark = pytest.mark.ci_cpu


def test_unprojection_and_viewport_ego_and_invalid_points():
    ranges = torch.tensor([[[2.0, 3.0]], [[0.1, 0.2]], [[1.0, 0.0]]])
    rays = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    pose = torch.eye(4)
    pose[2, 3] = 1.5
    assert torch.equal(
        range_image_points(ranges, rays, sensor_to_ego=pose),
        torch.tensor([[2.0, 0.0, 1.5]]),
    )
    viewport = LidarViewport(width=128, height=96, extent_m=12)
    image = viewport.render(
        torch.tensor([[float("nan"), 0, 0], [1000, 0, 0], [3, 0, 2]])
    )
    assert image.shape == (3, 96, 128) and torch.isfinite(image).all()
    assert image.min() >= 0 and image.max() <= 1
    assert ((image[0] > 0.9) & (image[2] > 0.7)).any()  # magenta ego / white nose
    assert not torch.equal(image, viewport.render(torch.empty(0, 3)))


@pytest.mark.parametrize(
    "azimuth,pixels",
    [(0, ((38, 30), (58, 40))), (90, ((38, 70), (48, 50)))],
)
def test_rotated_sensor_pose_places_points_at_known_pixels(azimuth, pixels):
    ranges = torch.tensor([[[2.0, 4.0]], [[0.5, 0.5]], [[1.0, 1.0]]])
    rays = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    # A quarter turn about z followed by translation into the ego frame.
    pose = torch.tensor(
        [
            [0.0, -1.0, 0.0, 4.0],
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 2.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    torch.testing.assert_close(
        range_image_points(ranges, rays, sensor_to_ego=pose),
        torch.tensor([[4.0, 4.0, 2.0], [0.0, 2.0, 2.0]]),
    )
    viewport = LidarViewport(
        width=100,
        height=100,
        extent_m=10,
        elevation_degrees=90,
        azimuth_degrees=azimuth,
    )
    # The overhead view has five pixels per metre and origin (column 50, row 58).
    expected = viewport.render(torch.empty(0, 3))
    for row, column in pixels:
        expected[:, row, column] = torch.tensor([1.0, 0.85, 0.1])
    torch.testing.assert_close(
        viewport.render_sweep(ranges, rays, sensor_to_ego=pose), expected
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_overlapping_points_show_the_nearest_return(reverse):
    viewport = LidarViewport(
        width=100,
        height=100,
        extent_m=10,
        elevation_degrees=45,
        azimuth_degrees=0,
    )
    # Both points project to (column 30, row 44); the higher point is nearer.
    points = torch.tensor([[4.0, 4.0, 0.0], [2.0, 4.0, 2.0]])
    if reverse:
        points = points.flip(0)
    expected = viewport.render(torch.empty(0, 3))
    expected[:, 44, 30] = torch.tensor([1.0, 0.85, 0.1])
    torch.testing.assert_close(viewport.render(points), expected)


@pytest.mark.parametrize("reverse", [False, True])
def test_equal_depth_returns_use_the_last_point_color(reverse):
    viewport = LidarViewport(
        width=100,
        height=100,
        extent_m=10,
        elevation_degrees=45,
        azimuth_degrees=0,
    )
    # Equal x-z gives equal depth, while the small height change stays in one pixel.
    points = torch.tensor([[4.0, 4.0, 0.0], [4.03125, 4.0, 0.03125]])
    if reverse:
        points = points.flip(0)
    expected = viewport.render(torch.empty(0, 3))
    expected[:, 44, 30] = torch.tensor(
        [0.05, 0.9, 0.85] if reverse else [0.06484375, 0.89921875, 0.83828125]
    )
    torch.testing.assert_close(viewport.render(points), expected)


@pytest.mark.parametrize("view", ["range", "intensity"])
def test_range_views_mask_invalid_returns_and_keep_fixed_scales(view):
    viewport = LidarViewport(width=4, height=4)
    ranges = torch.tensor(
        [
            [[25.0, 75.0, 50.0, float("nan")], [25.0, 75.0, 50.0, 100.0]],
            [[0.25, 0.75, 0.5, 1.0], [0.25, 0.75, 0.5, 1.0]],
            [[1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
        ]
    )
    image = viewport.render_sweep(ranges, torch.empty(0), view=LidarView(view))
    assert image.shape == (3, 4, 4) and torch.isfinite(image).all()
    assert (image[:, [0, 3]] == 0).all()  # panorama stays 2:1, with padding
    assert (image[:, 1, 2:] == 0).all()  # invalid flag and NaN
    expected = [0.25, 0.5, 0.75] if view == "range" else [0.25] * 3
    torch.testing.assert_close(image[:, 1, 0], torch.tensor(expected))
    changed = ranges.clone()
    changed[0, 1, -1] = 10  # no per-frame autoscale
    torch.testing.assert_close(
        viewport.render_sweep(changed, torch.empty(0), view=LidarView(view))[:, 1, 0],
        image[:, 1, 0],
    )
