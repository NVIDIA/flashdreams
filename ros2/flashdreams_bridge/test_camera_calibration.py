# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 FlashDreams contributors

import math
import unittest

import numpy as np
import pytest

try:
    import cv2
    from std_msgs.msg import Header
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("Run with ROS 2 Humble and OpenCV") from exc

from camera_calibration import camera_info

pytestmark = pytest.mark.ci_cpu


def test_equidistant_projection_follows_raster_and_resize_at_centre_edges_and_corners():
    cal = dict(
        cx=954.0,
        cy=535.0,
        width=1924,
        height=1084,
        ludus_draw_size=[1024, 577],
        is_backward_polynomial=True,
        polynomial=[0.0, 0.001, 0.0, 0.0, 0.0, 0.0],
        linear_cde=[1, 0, 0],
    )
    header = Header(frame_id="front_optical_frame")
    width, height = 832, 480
    info = camera_info(header, width, height, cal, opening=False)
    assert (info.header, info.width, info.height, info.distortion_model) == (
        header,
        width,
        height,
        "equidistant",
    )
    draw_size = np.array(cal["ludus_draw_size"])
    native_size = np.array([cal["width"], cal["height"]])
    pp = np.array([cal["cx"], cal["cy"]]) * draw_size / native_size
    draw_focal = np.mean(draw_size / native_size) / cal["polynomial"][1]
    pixels = np.array(
        [
            (x, y)
            for x in (0, (width - 1) / 2, width - 1)
            for y in (0, (height - 1) / 2, height - 1)
        ]
    )
    # Undo the renderer's viewport in edge coordinates, turn those directions
    # into optical-frame XYZ, then project using OpenCV's ROS fisheye model.
    local = (pixels + 0.5) * draw_size / [width, height] - pp
    rays = []
    for x, y in local:
        radius = math.hypot(x, y)
        theta = radius / draw_focal
        scale = math.sin(theta) / radius if radius else 0.0
        rays.append((scale * x, scale * y, math.cos(theta)))
    projected, _ = cv2.fisheye.projectPoints(
        np.array([rays]),
        np.zeros(3),
        np.zeros(3),
        info.k.reshape(3, 3),
        np.array(info.d),
    )
    np.testing.assert_allclose(projected[0], pixels, atol=1e-8)
    for changes in (
        dict(ludus_draw_size=None),
        dict(linear_cde=[1, 0.02, 0.01]),
        dict(polynomial=[0.0, 0.001, 0.0001, 0, 0, 0]),
    ):
        assert (
            camera_info(header, width, height, cal | changes, opening=False).k[0] == 0
        )
    assert camera_info(header, width, height, cal, opening=True).k[0] == 0
