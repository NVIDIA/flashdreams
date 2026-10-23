# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Camera calibration and rendering conventions for prepared scenes."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from ludus_renderer import FThetaCamera

OUTLINE_WIDTH = 4.0
"""Width in pixels of the edges around a box, at the camera's own size.

The reference draws them at this, narrower than the twelve it gives a lane
line, and in a plain light grey. A caller drawing smaller shrinks it to match,
as it does the line widths.
"""


VIEWPORT_LIMIT = 2048
"""The widest picture ludus's rasterizer can address, from its own constants.

Ludus smooths by drawing at twice the size, so a frame is held to half of this
a side. Asking for more is not an error: it says so and draws unsmoothed.
"""


def frame_rate(frames: int, start_us: int, end_us: int) -> float:
    """The rate a clip's frame numbering counts in, counted off its own span.

    Thirty is what these rigs are described by and not what the data keeps: a
    clip of 654 frames spanning exactly twenty seconds runs at 32.7. Assuming
    thirty does no harm while both halves of a render assume it, since
    stamping a box and asking for a camera at a frame cancel out, but it walks
    our frame away from the reference's frame of the same number -- by a
    second and more over a clip -- and a comparison then quietly stops
    comparing the same moment.
    """
    return frames * 1e6 / (end_us - start_us)


def smoothable(native: tuple[int, int]) -> tuple[int, int]:
    """The largest picture ludus will smooth as it draws, in a camera's shape.

    The reference drew at the camera's own size, 1924x1084, and multisampled it
    four times over, which grades a pixel by how much of a line covers it. Ludus
    asks its rasterizer for twice the size instead, and 3848 is past what that
    can address; tiles are no way out either, since a viewport is what clip
    coordinates are mapped onto and each tile would hold the whole picture
    squeezed into it. So a frame is drawn at 1024x577, still finer than the
    832x480 the model is fed, and smoothed rather than not: drawn without that,
    a line lying across the picture is a flight of steps, and shrinking the
    picture afterwards averages the steps rather than removing them.
    """
    scale = min(1.0, VIEWPORT_LIMIT / 2 / max(native))
    return max(round(native[0] * scale), 1), max(round(native[1] * scale), 1)


def ftheta_camera(
    intrinsics: Any,
    device: Any = "cuda",
    resolution: tuple[int, int] | None = None,
    depth_max: float = 200.0,
) -> FThetaCamera:
    """A ludus camera from the fourteen numbers the rig records per sensor.

    Those are ``cx, cy, width, height``, six polynomial coefficients, a flag,
    and three distortion terms. The flag matters: rigs almost always store the
    polynomial backwards, mapping radius to angle, and the renderer projects
    with the forward direction, so it has to be turned round. Sampling the
    backward polynomial across the image and fitting the samples the other way
    is what ludus does for its own scenes, and doing the same here keeps the
    two agreeing about where a point lands.

    The distortion terms are frequently absent, and eleven numbers rather than
    fourteen means an undistorted lens rather than a malformed rig.

    Args:
        intrinsics: the rig's numbers for one sensor.
        device: where the camera's tensors go.
        resolution: the ``(width, height)`` to draw at, if not the rig's own.
            The polynomial gives radii in the rig's pixels and the optical
            centre is in them too, so drawing at another size means scaling
            both. Skip it and the lens has the wrong focal length and its
            centre sits somewhere off the picture.
        depth_max: how far the renderer's depth buffer reaches, in metres.
    """
    intrinsics = np.asarray(intrinsics, dtype=np.float64).reshape(-1)
    cx, cy, width, height = intrinsics[:4]
    poly = intrinsics[4:10]
    backwards = intrinsics[10] > 0
    distortion = (
        intrinsics[11:14] if len(intrinsics) >= 14 else np.array([1.0, 0.0, 0.0])
    )

    corners = np.array(
        [[0, 0], [width, 0], [0, height], [width, height]], dtype=np.float64
    )
    radii = np.hypot(corners[:, 0] - cx, corners[:, 1] - cy)
    furthest = radii.max()

    if backwards and poly[1] != 0:
        sampled = np.linspace(0, 1, 200) * furthest
        angles = np.polynomial.polynomial.polyval(sampled, poly)
        usable = angles > 1e-6
        if usable.any():
            forward = np.polyfit(angles[usable], sampled[usable], deg=5)[::-1].copy()
            forward[0] = 0.0
        else:
            forward = np.array([0.0, 1.0 / poly[1], 0.0, 0.0, 0.0, 0.0])
        corner_angle = np.polynomial.polynomial.polyval(furthest, poly)
    else:
        forward = poly.copy()
        low, high = 0.0, np.pi
        for _ in range(50):
            corner_angle = (low + high) / 2
            if np.polynomial.polynomial.polyval(corner_angle, poly) < furthest:
                low = corner_angle
            else:
                high = corner_angle

    # How wide the lens sees is a fact about the lens, so it is measured above
    # from the rig's own frame and left alone by any rescaling below.
    if resolution is not None:
        across, down = resolution
        sideways, downwards = across / width, down / height
        forward = forward * (sideways + downwards) / 2
        cx, cy = cx * sideways, cy * downwards
        width, height = float(across), float(down)

    return FThetaCamera(
        principal_point=torch.tensor([cx, cy], dtype=torch.float32, device=device),
        image_size=torch.tensor([width, height], dtype=torch.float32, device=device),
        fw_poly=torch.tensor(forward[:6], dtype=torch.float32, device=device),
        max_ray_angle=float(abs(corner_angle) * 1.05),
        linear_distortion=torch.tensor(
            [[distortion[0], distortion[1]], [distortion[2], 1.0]],
            dtype=torch.float32,
            device=device,
        ),
        depth_max=depth_max,
    )


RDF_TO_FLU = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
"""Between the two ways of naming a camera's own axes.

The rig names them right, down and forward, which is the convention the
annotations were projected with. The renderer names them forward, left and up,
and converts to the other itself on the way in. Handing it the rig's directly
therefore turns the world a quarter turn: the road underneath becomes a wall
out to the left.
"""


def camera_pose(camera_to_world: np.ndarray) -> np.ndarray:
    """What the renderer wants, from a pose as the rig records it.

    Two changes of direction, both easy to leave out. The recorded pose says
    where the camera is in the world and the renderer wants the reverse, and
    the two disagree about which axis a camera looks along.

    Args:
        camera_to_world: a ``[4, 4]`` pose, or a ``[T, 4, 4]`` stack of them.
    """
    return RDF_TO_FLU @ np.linalg.inv(camera_to_world).astype(np.float32)
