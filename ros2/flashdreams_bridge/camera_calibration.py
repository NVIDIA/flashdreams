# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 FlashDreams contributors

"""ROS pixel-centre calibration for a linear equidistant virtual camera."""

import math

from sensor_msgs.msg import CameraInfo


def camera_info(header, width, height, cal, *, opening):
    """Describe the emitted image at its actual size, on its optical frame."""
    info = CameraInfo(header=header, width=width, height=height)
    size = cal.get("ludus_draw_size")
    # Seed PNGs have their own export/resampling path. The prepared format
    # does not declare a pixel transform from the native sensor to the PNG.
    if opening or not size:
        return info

    poly = cal["polynomial"]
    if (
        len(poly) != 6
        or poly[0] != 0
        or any(poly[2:])
        or len(size) != 2
        or list(cal["linear_cde"]) != [1, 0, 0]
    ):
        return info
    draw_w, draw_h = size
    native_w, native_h = cal["width"], cal["height"]
    if min(draw_w, draw_h, native_w, native_h, poly[1]) <= 0 or not all(
        math.isfinite(x) for x in (poly[1], cal["cx"], cal["cy"])
    ):
        return info
    focal = 1 / poly[1] if cal["is_backward_polynomial"] else poly[1]
    sx, sy = width / native_w, height / native_h
    # Ludus uses the mean draw scale for its radial polynomial and edge-origin
    # NDC for projection. ROS counts the first texel centre as pixel zero.
    draw_focal = focal * (draw_w / native_w + draw_h / native_h) / 2
    fx, fy = draw_focal * width / draw_w, draw_focal * height / draw_h
    cx, cy = cal["cx"] * sx - 0.5, cal["cy"] * sy - 0.5
    info.distortion_model = "equidistant"
    info.d = [0.0] * 4
    info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return info
