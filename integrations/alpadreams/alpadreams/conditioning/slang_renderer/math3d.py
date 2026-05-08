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

"""3D transform helpers used by the slang rasterizer.

Lifted from ``roaddreams.math3d``.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt


def quaternion_to_matrix_xyzw(
    quat: list[float] | tuple[float, float, float, float],
) -> npt.NDArray[np.float32]:
    """Convert an ``(x, y, z, w)`` quaternion to a 3x3 rotation matrix."""
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
    rotation: npt.NDArray[np.float32],
    translation_xyz: list[float] | tuple[float, float, float],
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


def yaw_from_matrix(matrix: npt.NDArray[np.float32]) -> float:
    return float(math.atan2(matrix[1, 0], matrix[0, 0]))
