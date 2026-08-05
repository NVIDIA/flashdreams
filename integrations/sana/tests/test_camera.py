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

"""CPU-safe tests for SANA-WM camera and intrinsics helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from sana_wm.camera import (
    action_string_to_c2w,
    default_intrinsics_vec4,
    fit_camera_trajectory,
    fit_intrinsics_sequence,
    load_intrinsics,
    prepare_camera,
    resize_center_crop_geometry,
    snap_num_frames,
    transform_intrinsics_for_crop,
)

pytestmark = pytest.mark.ci_cpu


def test_default_intrinsics_vec4_centers_and_matches_hfov() -> None:
    vec = default_intrinsics_vec4((1000, 500), num_frames=3, hfov_deg=90.0)
    assert vec.shape == (3, 4)
    fx, fy, cx, cy = vec[0]
    # 90 deg hfov with square pixels -> fx = 0.5 * W / tan(45) = 0.5 * W.
    assert fx == pytest.approx(500.0)
    assert fy == pytest.approx(500.0)
    # Principal point at the image center.
    assert cx == pytest.approx(500.0)
    assert cy == pytest.approx(250.0)
    # Same intrinsics for every frame.
    assert np.allclose(vec, vec[0])


def test_default_intrinsics_vec4_rejects_bad_fov() -> None:
    for bad in (0.0, 180.0, 200.0):
        with pytest.raises(ValueError):
            default_intrinsics_vec4((640, 360), num_frames=1, hfov_deg=bad)


def test_action_string_rolls_out_identity_plus_motion() -> None:
    """Expand ``w-3`` to an identity frame plus three motion frames."""
    c2w = action_string_to_c2w("w-3", smooth=False)

    assert c2w.shape == (4, 4, 4)
    np.testing.assert_allclose(c2w[0], np.eye(4), atol=1e-6)
    assert np.all(np.diff(c2w[:, 2, 3]) > 0)


def test_action_string_repeats_to_requested_frame_count() -> None:
    """Use action prompts to derive the requested number of camera poses."""
    c2w = action_string_to_c2w("w-1", smooth=False, num_frames=5)

    assert c2w.shape == (5, 4, 4)
    np.testing.assert_allclose(c2w[0], np.eye(4), atol=1e-6)
    np.testing.assert_allclose(
        c2w[:, 2, 3],
        [0.0, 0.025, 0.05, 0.075, 0.1],
        atol=1e-6,
    )


def test_action_string_bounds_oversized_duration_to_requested_frames() -> None:
    """Avoid materializing action frames beyond the requested output length."""
    c2w = action_string_to_c2w("w-500000000", smooth=False, num_frames=5)

    assert c2w.shape == (5, 4, 4)
    np.testing.assert_allclose(
        c2w[:, 2, 3],
        [0.0, 0.025, 0.05, 0.075, 0.1],
        atol=1e-6,
    )


def test_action_string_validates_segments_after_requested_frame_limit() -> None:
    """Do not skip validation for segments beyond ``num_frames`` truncation."""
    with pytest.raises(ValueError, match="unknown keys"):
        action_string_to_c2w("w-500000000,q-1", smooth=False, num_frames=5)


def test_action_string_rejects_unknown_keys() -> None:
    """Reject invalid action DSL tokens."""
    with pytest.raises(ValueError, match="unknown keys"):
        action_string_to_c2w("q-3")


def test_snap_num_frames_matches_ltx2_stride() -> None:
    """Snap requested frames to nearest ``8k + 1`` value."""
    assert snap_num_frames(321) == 321
    assert snap_num_frames(322) == 321
    assert snap_num_frames(325) == 329
    assert snap_num_frames(325, upper_bound=326) == 321


def test_fit_intrinsics_sequence_interpolates_short_sequences() -> None:
    """Interpolate per-frame intrinsics when the source sequence is shorter."""
    source = np.array([[10.0, 20.0, 1.0, 2.0], [30.0, 40.0, 3.0, 4.0]])

    fitted = fit_intrinsics_sequence(source, 3)

    np.testing.assert_allclose(
        fitted,
        np.array(
            [
                [10.0, 20.0, 1.0, 2.0],
                [20.0, 30.0, 2.0, 3.0],
                [30.0, 40.0, 3.0, 4.0],
            ],
            dtype=np.float32,
        ),
    )


def test_fit_camera_trajectory_interpolates_short_sequences() -> None:
    """Fit explicit camera paths to the requested rollout length."""
    source = np.broadcast_to(np.eye(4, dtype=np.float32), (2, 4, 4)).copy()
    source[1, 2, 3] = 2.0

    fitted = fit_camera_trajectory(source, 5)

    assert fitted.shape == (5, 4, 4)
    np.testing.assert_allclose(fitted[:, 2, 3], [0.0, 0.5, 1.0, 1.5, 2.0])
    np.testing.assert_allclose(
        fitted[:, 3],
        np.broadcast_to(np.array([0.0, 0.0, 0.0, 1.0]), (5, 4)),
    )
    for rotation in fitted[:, :3, :3]:
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize(
    ("array", "expected"),
    [
        (
            np.array([100.0, 110.0, 50.0, 55.0], dtype=np.float32),
            np.array([[100.0, 110.0, 50.0, 55.0]] * 3, dtype=np.float32),
        ),
        (
            np.array(
                [[100.0, 0.0, 50.0], [0.0, 110.0, 55.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
            np.array([[100.0, 110.0, 50.0, 55.0]] * 3, dtype=np.float32),
        ),
    ],
)
def test_load_intrinsics_accepts_static_shapes(
    tmp_path: Path, array: np.ndarray, expected: np.ndarray
) -> None:
    """Load static vector and matrix intrinsics as per-frame vectors."""
    path = tmp_path / "intrinsics.npy"
    np.save(path, array)

    loaded = load_intrinsics(path, num_frames=3)

    np.testing.assert_allclose(loaded, expected)


def test_resize_crop_geometry_and_intrinsics_transform() -> None:
    """Map source intrinsics through SANA-WM resize and center-crop geometry."""
    src_size = (640, 480)
    resized_size, crop_offset = resize_center_crop_geometry(src_size)
    intrinsics = np.array([[400.0, 420.0, 320.0, 240.0]], dtype=np.float32)

    transformed = transform_intrinsics_for_crop(
        intrinsics, src_size, resized_size, crop_offset
    )

    assert resized_size == (1280, 960)
    assert crop_offset == (0, 128)
    np.testing.assert_allclose(
        transformed,
        np.array([[800.0, 840.0, 640.0, 352.0]], dtype=np.float32),
    )


def test_prepare_camera_shapes_for_sana_wm_resolution() -> None:
    """Build raymap and chunk-Plucker tensors at 704x1280 SANA-WM shape."""
    num_frames = 17
    poses = np.broadcast_to(np.eye(4, dtype=np.float32), (num_frames, 4, 4)).copy()
    poses[:, 2, 3] = np.linspace(0.0, 1.0, num_frames)
    intrinsics = np.broadcast_to(
        np.array([900.0, 900.0, 640.0, 352.0], dtype=np.float32),
        (num_frames, 4),
    ).copy()

    camera = prepare_camera(poses, intrinsics)

    assert camera["raymap"].shape == (3, 20)
    assert camera["chunk_plucker"].shape == (48, 3, 22, 40)
    assert camera["raymap"].dtype == torch.float32
    assert camera["chunk_plucker"].dtype == torch.float32
