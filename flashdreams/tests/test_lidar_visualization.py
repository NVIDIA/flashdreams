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
