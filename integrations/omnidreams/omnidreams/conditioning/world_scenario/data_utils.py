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

"""World scenario utilities.

This module contains utilities for the world scenario.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

# =============================================================================
# Coordinate Frame Conversion (FLU <-> RDF)
# =============================================================================
# FLU: Forward-Left-Up (common robotics convention, used by clients)
# RDF: Right-Down-Forward (OpenCV convention, used internally by renderer)

# Transformation matrix from FLU to OpenCV RDF
FLU_TO_RDF_MATRIX: NDArray[np.float32] = np.array(
    [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)

# Transformation matrix from OpenCV RDF to FLU (inverse = transpose for orthonormal)
RDF_TO_FLU_MATRIX: NDArray[np.float32] = FLU_TO_RDF_MATRIX.T.astype(np.float32)


def convert_pose_flu_to_rdf(pose_matrix: np.ndarray) -> np.ndarray:
    """
    Convert a 4x4 pose matrix from FLU to RDF coordinates.

    This applies a basis change transformation:
    - Position: p_rdf = R * p_flu
    - Rotation: R_rdf = R * R_flu * R^T

    Args:
        pose_matrix: 4x4 transformation matrix in FLU coordinates.

    Returns:
        4x4 transformation matrix in RDF coordinates.
    """
    result = np.eye(4, dtype=np.float32)

    # Convert position
    result[:3, 3] = FLU_TO_RDF_MATRIX @ pose_matrix[:3, 3]

    # Convert rotation: R_rdf = S * R_flu * S^T (double-sided transformation)
    result[:3, :3] = FLU_TO_RDF_MATRIX @ pose_matrix[:3, :3] @ RDF_TO_FLU_MATRIX

    return result


def convert_position_flu_to_rdf(position: np.ndarray) -> np.ndarray:
    """
    Convert a 3D position from FLU to RDF coordinates.

    Args:
        position: Position vector [x, y, z] in FLU coordinates.

    Returns:
        Position vector [x, y, z] in RDF coordinates.
    """
    return (FLU_TO_RDF_MATRIX @ position.reshape(3)).astype(np.float32)


def convert_quaternion_flu_to_rdf(quat_xyzw: np.ndarray) -> np.ndarray:
    """
    Convert a quaternion from FLU to RDF coordinates.

    Args:
        quat_xyzw: Quaternion [x, y, z, w] in FLU coordinates (scipy convention).

    Returns:
        Quaternion [x, y, z, w] in RDF coordinates (scipy convention).
    """
    rot_flu = Rotation.from_quat(quat_xyzw)
    rot_matrix_flu = rot_flu.as_matrix()
    rot_matrix_rdf = FLU_TO_RDF_MATRIX @ rot_matrix_flu @ RDF_TO_FLU_MATRIX
    return Rotation.from_matrix(rot_matrix_rdf).as_quat().astype(np.float32)


def convert_points_flu_to_rdf(points: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert an array of 3D points from FLU to OpenCV RDF coordinates."""

    if points.size == 0:
        return points.astype(np.float32, copy=False)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Points must have shape (N, 3), got {points.shape}")
    return (FLU_TO_RDF_MATRIX @ points.T).T.astype(np.float32)


def convert_quaternions_flu_to_rdf(
    quaternions: NDArray[np.float32],
    *,
    double_sided: bool = False,
) -> NDArray[np.float32]:
    """Convert an array of quaternions from FLU to OpenCV RDF coordinates."""

    if quaternions.size == 0:
        return quaternions.astype(np.float32, copy=False)
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError(f"Quaternions must have shape (N, 4), got {quaternions.shape}")

    rotations_flu = Rotation.from_quat(quaternions)
    r_flu = rotations_flu.as_matrix()
    r_rdf = np.einsum("ij,njk->nik", FLU_TO_RDF_MATRIX, r_flu)
    if double_sided:
        r_rdf = np.matmul(r_rdf, FLU_TO_RDF_MATRIX.T)
    quaternions_rdf = Rotation.from_matrix(r_rdf).as_quat().astype(np.float32)
    return quaternions_rdf


def normalize_quaternions(
    quaternions: NDArray[np.float32], eps: float = 1e-6
) -> Tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Normalize quaternions and return mask of valid entries."""

    if quaternions.size == 0:
        mask = np.zeros((0,), dtype=bool)
        return quaternions.astype(np.float32, copy=False), mask

    norms = np.linalg.norm(quaternions, axis=1)
    valid_mask = norms > eps

    normalized = quaternions[valid_mask].copy()
    if normalized.size > 0:
        normalized /= norms[valid_mask][:, None]

    return normalized.astype(np.float32), valid_mask
