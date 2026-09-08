# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU contracts for Ludus camera intrinsics."""

import pytest
import torch
from ludus_renderer import OrthographicCamera
from ludus_renderer._ops.primitives import _pack_cameras
from ludus_renderer.render_utils import create_bev_camera

pytestmark = pytest.mark.ci_cpu


def test_bev_camera_uses_constant_orthographic_scale() -> None:
    camera = create_bev_camera(
        width=800,
        height=400,
        device=torch.device("cpu"),
        bev_height=50.0,
        fov_deg=90.0,
    )

    assert isinstance(camera, OrthographicCamera)
    assert camera.principal_point.tolist() == [400.0, 200.0]
    assert camera.pixels_per_meter.tolist() == pytest.approx([4.0, 4.0])

    camera_points = torch.tensor([[10.0, -5.0, 50.0], [10.0, -5.0, 25.0]])
    pixels = camera.principal_point + camera_points[:, :2] * camera.pixels_per_meter

    assert camera_points[0, 2] != camera_points[1, 2]
    torch.testing.assert_close(pixels[0], pixels[1])


def test_pack_cameras_marks_orthographic_projection() -> None:
    camera = create_bev_camera(
        width=800,
        height=400,
        device=torch.device("cpu"),
        bev_height=50.0,
        fov_deg=90.0,
    )

    packed = _pack_cameras([camera], torch.device("cpu"))

    assert tuple(packed.shape) == (1, 18)
    assert packed[0, 4:6].tolist() == pytest.approx([4.0, 4.0])
    assert packed[0, 10].item() < 0.0


@pytest.mark.parametrize(
    ("width", "height", "bev_height", "fov_deg"),
    [
        (0, 400, 50.0, 90.0),
        (800, 0, 50.0, 90.0),
        (800, 400, 0.0, 90.0),
        (800, 400, 50.0, 0.0),
        (800, 400, 50.0, 180.0),
    ],
)
def test_bev_camera_rejects_invalid_footprint(
    width: int, height: int, bev_height: float, fov_deg: float
) -> None:
    with pytest.raises(ValueError):
        create_bev_camera(
            width=width,
            height=height,
            device=torch.device("cpu"),
            bev_height=bev_height,
            fov_deg=fov_deg,
        )
