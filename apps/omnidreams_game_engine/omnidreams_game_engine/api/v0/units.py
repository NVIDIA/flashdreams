# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unreal Engine to FLU conversions for engine state decoders.

FLU: x forward, y left, z up, meters, right-handed. Unreal: X forward, Y
right, Z up, centimeters, left-handed, rotators as pitch, yaw and roll in
degrees, rotated as Unreal's ``FRotationMatrix`` does.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

Float64Array = npt.NDArray[np.float64]

_S = np.diag([1.0, -1.0, 1.0])


def ue_pos_to_flu(p_cm: npt.ArrayLike) -> Float64Array:
    """Convert Unreal positions in cm to FLU meters (the last axis holds x, y, z)."""
    p = np.asarray(p_cm, dtype=np.float64)
    return np.stack([p[..., 0] / 100.0, -p[..., 1] / 100.0, p[..., 2] / 100.0], axis=-1)


def ue_rotator_matrix(
    pitch_deg: float, yaw_deg: float, roll_deg: float
) -> Float64Array:
    """Return the Unreal local X, Y, Z axes in Unreal world axes, as columns."""
    pitch, yaw, roll = (math.radians(v) for v in (pitch_deg, yaw_deg, roll_deg))
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cr, sr = math.cos(roll), math.sin(roll)
    x = [cp * cy, cp * sy, sp]
    y = [sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, -sr * cp]
    z = [-(cr * sp * cy + sr * sy), cy * sr - cr * sp * sy, cr * cp]
    return np.array([x, y, z], dtype=np.float64).T


def ue_rot_to_flu(rot_pyr_deg: Sequence[float]) -> Float64Array:
    """Convert an Unreal ``[pitch, yaw, roll]`` in degrees to an FLU 3x3 rotation."""
    pitch, yaw, roll = rot_pyr_deg
    return _S @ ue_rotator_matrix(pitch, yaw, roll) @ _S


def pose(rotation: npt.ArrayLike, translation: npt.ArrayLike) -> Float64Array:
    """Return the 4x4 homogeneous transform of a rotation and a translation."""
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def ue_transform_to_flu(
    position_cm: Sequence[float], rot_pyr_deg: Sequence[float]
) -> Float64Array:
    """Convert an Unreal location and rotator to the 4x4 FLU body-to-world transform."""
    return pose(ue_rot_to_flu(rot_pyr_deg), ue_pos_to_flu(position_cm))


def ue_yaw_to_flu_rad(yaw_deg: float) -> float:
    """Convert an Unreal yaw in degrees (clockwise from above) to FLU radians."""
    return -math.radians(yaw_deg)


def rotation_to_quat_xyzw(rotation: npt.ArrayLike) -> Float64Array:
    """Return the unit quaternion (x, y, z, w) of a 3x3 rotation, with w >= 0."""
    m = np.asarray(rotation, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = [
            (m[2, 1] - m[1, 2]) / s,
            (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s,
            0.25 * s,
        ]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = [
            0.25 * s,
            (m[0, 1] + m[1, 0]) / s,
            (m[0, 2] + m[2, 0]) / s,
            (m[2, 1] - m[1, 2]) / s,
        ]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = [
            (m[0, 1] + m[1, 0]) / s,
            0.25 * s,
            (m[1, 2] + m[2, 1]) / s,
            (m[0, 2] - m[2, 0]) / s,
        ]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = [
            (m[0, 2] + m[2, 0]) / s,
            (m[1, 2] + m[2, 1]) / s,
            0.25 * s,
            (m[1, 0] - m[0, 1]) / s,
        ]
    quat = np.asarray(q, dtype=np.float64)
    quat /= np.linalg.norm(quat)
    if quat[3] < 0.0:
        quat = -quat
    return quat


__all__ = [
    "pose",
    "rotation_to_quat_xyzw",
    "ue_pos_to_flu",
    "ue_rot_to_flu",
    "ue_rotator_matrix",
    "ue_transform_to_flu",
    "ue_yaw_to_flu_rad",
]
