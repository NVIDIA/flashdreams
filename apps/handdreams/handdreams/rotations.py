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

"""Pure-NumPy rotation and rigid-pose helpers for HandDreams V2."""

from __future__ import annotations

from typing import Any

import numpy as np

HOMOGENEOUS_BOTTOM_ROW = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
"""Final row every 4x4 rigid transform must carry."""

# Orthonormality is checked with a loose tolerance because these rotations
# arrive from float32 WebXR quaternions and accumulated basis products.
_ROTATION_ATOL = 2e-3


def quat_xyzw_to_matrix(quaternion: Any) -> np.ndarray:
    """Convert ``[...,4]`` ``(x, y, z, w)`` quaternions to ``[...,3,3]`` rotations.

    Args:
        quaternion: Quaternion values in ``(x, y, z, w)`` order. Normalized
            internally, so unnormalized input is accepted.

    Returns:
        The equivalent rotation matrices as ``float32``.

    Raises:
        ValueError: If any quaternion has effectively zero norm.
    """
    values = np.asarray(quaternion, dtype=np.float32)
    if values.shape[-1:] != (4,):
        raise ValueError(f"quaternion must end in 4 values, got {values.shape}")
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("quaternion has zero norm")
    x, y, z, w = np.moveaxis(values / norm, -1, 0)
    matrix = np.empty((*values.shape[:-1], 3, 3), dtype=np.float32)
    matrix[..., 0, 0] = 1 - 2 * (y * y + z * z)
    matrix[..., 0, 1] = 2 * (x * y - z * w)
    matrix[..., 0, 2] = 2 * (x * z + y * w)
    matrix[..., 1, 0] = 2 * (x * y + z * w)
    matrix[..., 1, 1] = 1 - 2 * (x * x + z * z)
    matrix[..., 1, 2] = 2 * (y * z - x * w)
    matrix[..., 2, 0] = 2 * (x * z - y * w)
    matrix[..., 2, 1] = 2 * (y * z + x * w)
    matrix[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return matrix


def matrix_to_rot6d(matrix: Any) -> np.ndarray:
    """Encode ``[...,3,3]`` rotations as the first two columns, ``[...,6]``.

    This is the rot6d convention the AgiBot action contract expects; the
    identity rotation maps to ``[1, 0, 0, 0, 1, 0]``.
    """
    values = np.asarray(matrix, dtype=np.float32)
    if values.shape[-2:] != (3, 3):
        raise ValueError(f"rotation must end in [3,3], got {values.shape}")
    return np.concatenate((values[..., :, 0], values[..., :, 1]), axis=-1).astype(
        np.float32
    )


def validate_rotation(matrix: Any, name: str) -> np.ndarray:
    """Return ``matrix`` as ``float32`` after checking it is a proper rotation.

    Args:
        matrix: Candidate rotation of shape ``[...,3,3]``.
        name: Subject used in error messages.

    Raises:
        ValueError: If the value is not finite, orthonormal, and right-handed.
    """
    values = np.asarray(matrix, dtype=np.float32)
    if values.shape[-2:] != (3, 3) or not np.isfinite(values).all():
        raise ValueError(f"{name} rotation must be finite [...,3,3]")
    flat = values.reshape(-1, 3, 3)
    gram = np.swapaxes(flat, -1, -2) @ flat
    if not np.allclose(gram, np.eye(3), atol=_ROTATION_ATOL, rtol=_ROTATION_ATOL):
        raise ValueError(f"{name} rotation must be orthonormal")
    if not np.allclose(
        np.linalg.det(flat), 1.0, atol=_ROTATION_ATOL, rtol=_ROTATION_ATOL
    ):
        raise ValueError(f"{name} rotation must be right-handed")
    return values


def validate_pose_track(
    poses: Any,
    name: str = "pose",
    *,
    minimum_frames: int = 2,
) -> np.ndarray:
    """Return an ``[F,4,4]`` track as ``float32`` after checking rigidity.

    Args:
        poses: Homogeneous transforms of shape ``[F,4,4]``.
        name: Subject used in error messages.
        minimum_frames: Smallest acceptable ``F``.

    Raises:
        ValueError: If the shape, finiteness, bottom row, or rotation block is
            invalid.
    """
    values = np.asarray(poses, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (4, 4):
        raise ValueError(f"{name} track must have shape [F,4,4], got {values.shape}")
    if values.shape[0] < minimum_frames:
        raise ValueError(f"{name} track needs at least {minimum_frames} frames")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} track contains non-finite values")
    if not np.allclose(values[:, 3, :], HOMOGENEOUS_BOTTOM_ROW, atol=1e-5, rtol=0.0):
        raise ValueError(f"{name} track must contain homogeneous transforms")
    validate_rotation(values[:, :3, :3], name)
    return values


def unit_vector(vector: Any, *, name: str = "vector") -> np.ndarray:
    """Normalize a 3D vector, rejecting degenerate input."""
    values = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError(f"cannot normalize a degenerate {name}")
    return (values / norm).astype(np.float32)
