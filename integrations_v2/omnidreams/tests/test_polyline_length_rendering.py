# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete long-polyline rendering on independent synthetic geometry."""

import pytest
import torch
from ludus_renderer import (
    CAMERA_TYPE_REGULAR,
    FThetaCamera,
    LudusCudaTimestampedContext,
    PRIM_LANE_LINE_WHITE_SOLID,
    TimestampedPolylinePool,
    TimestampedScene,
)

pytestmark = pytest.mark.ci_gpu


@pytest.fixture(scope="module")
def renderer():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for semantic rendering")
    context = LudusCudaTimestampedContext(device="cuda")
    context.set_depth_scaling(False)
    # Keep the initial geometry estimate small so the longest line needs growth.
    context.set_max_tessellation_levels(polyline=0, polygon=0, cube=0)
    context.upload_cameras(
        [
            FThetaCamera(
                principal_point=torch.tensor([64, 64], device="cuda"),
                image_size=torch.tensor([128, 128], device="cuda"),
                fw_poly=torch.tensor([0, 80, 0, 0, 0, 0], device="cuda"),
                max_ray_angle=1.2,
                depth_max=100,
            )
        ]
    )
    yield context
    context.clear_scenes()


def _render(context, x):
    vertices = torch.stack((x, torch.zeros_like(x), torch.zeros_like(x)), dim=1)
    pool = TimestampedPolylinePool(
        timestamps_us=torch.tensor([0], dtype=torch.int64, device="cuda"),
        timestamped_varrays_prefix_sum=torch.tensor(
            [1], dtype=torch.int32, device="cuda"
        ),
        varrays_prefix_sum=torch.tensor([len(x)], dtype=torch.int32, device="cuda"),
        vertices=vertices,
        prim_type_id=PRIM_LANE_LINE_WHITE_SOLID,
        surface_planes=torch.tensor(
            [0, 0, 1, 0], dtype=torch.float32, device="cuda"
        ).repeat(len(x), 1),
    )
    # Forward/left/up camera axes: look straight down at the horizontal line.
    camera_to_world = torch.tensor(
        [[0, -1, 0, 0], [0, 0, 1, 0], [-1, 0, 0, 5], [0, 0, 0, 1]],
        dtype=torch.float32,
        device="cuda",
    )
    context.clear_scenes()
    scene_id = context.upload_scene(TimestampedScene([pool], []))
    return context.render_uniform(
        scene_id=scene_id,
        camera_id=0,
        timestamps_us=[0],
        camera_type_id=CAMERA_TYPE_REGULAR,
        camera_poses=torch.linalg.inv(camera_to_world)[None],
        resolution=(128, 128),
    )[..., :3].cpu()


@pytest.mark.parametrize("point_count", [255, 256, 257, 1025, 8193])
@pytest.mark.parametrize("reverse", [False, True])
def test_polyline_keeps_both_ends_beyond_256_points(renderer, point_count, reverse):
    # Place a substantial visible tail beyond the old silent truncation boundary.
    if point_count > 256:
        x = torch.cat(
            (torch.linspace(-4, 0, 256), torch.linspace(0, 4, point_count - 255)[1:])
        )
    else:
        x = torch.linspace(-4, 4, point_count)
    if reverse:
        x = x.flip(0)
    actual = _render(renderer, x.to("cuda"))
    expected = _render(
        renderer, torch.tensor([-4, 0, 4], dtype=torch.float32, device="cuda")
    )
    expected_mask = expected.any(dim=-1)
    actual_mask = actual.any(dim=-1)
    assert expected_mask.sum() > 100
    # Resampling can shift subpixel edge coverage, but cannot delete an endpoint
    # or create a cap where the renderer formerly stopped at vertex 255.
    assert (expected_mask & actual_mask).sum() >= 0.98 * expected_mask.sum()
    assert (expected_mask & actual_mask).sum() >= 0.98 * actual_mask.sum()
