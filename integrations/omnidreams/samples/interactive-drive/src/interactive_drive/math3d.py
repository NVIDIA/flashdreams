# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt


def normalize_camera_name(name: str) -> tuple[str, str]:
    if ":" in name:
        clipgt_name = name
        logical_name = name.replace(":", "_")
        return clipgt_name, logical_name

    logical_name = name
    clipgt_name = logical_name.replace("camera_", "camera:", 1).replace("_", ":")
    return clipgt_name, logical_name


def euler_xyz_degrees_to_matrix(
    rpy_deg: list[float] | tuple[float, float, float],
) -> npt.NDArray[np.float32]:
    roll, pitch, yaw = [math.radians(v) for v in rpy_deg]

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return (rz @ ry @ rx).astype(np.float32)


def quaternion_to_matrix_xyzw(
    quat: list[float] | tuple[float, float, float, float],
) -> npt.NDArray[np.float32]:
    x, y, z, w = quat
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def transform_from_rt(
    rotation: npt.NDArray[np.float32], translation_xyz: list[float] | tuple[float, float, float]
) -> npt.NDArray[np.float32]:
    result = np.eye(4, dtype=np.float32)
    result[:3, :3] = rotation
    result[:3, 3] = np.asarray(translation_xyz, dtype=np.float32)
    return result


def invert_transform(matrix: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    inv = np.eye(4, dtype=np.float32)
    inv[:3, :3] = rotation.T
    inv[:3, 3] = -(rotation.T @ translation)
    return inv


def transform_points(
    matrix: npt.NDArray[np.float32], points_xyz: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    ones = np.ones((points_xyz.shape[0], 1), dtype=np.float32)
    points_h = np.concatenate([points_xyz, ones], axis=1)
    return (points_h @ matrix.T)[:, :3].astype(np.float32)


def extract_yaw_from_transform(matrix: npt.NDArray[np.float32]) -> float:
    return float(math.atan2(matrix[1, 0], matrix[0, 0]))


def rig_pose_from_state(
    x_m: float,
    y_m: float,
    z_m: float,
    yaw_rad: float,
    pitch_rad: float = 0.0,
    roll_rad: float = 0.0,
) -> npt.NDArray[np.float32]:
    cr, sr = math.cos(roll_rad), math.sin(roll_rad)
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
    rotation = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float32,
    )
    return transform_from_rt(rotation, [x_m, y_m, z_m])
