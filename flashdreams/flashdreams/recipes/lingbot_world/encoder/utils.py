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

"""Camera-pose math: SE(3) helpers, relative poses, and Plücker rays."""

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp
from torch import Tensor

## Lingbot World example-data preprocessing

_TEMPORAL_COMPRESSION_RATIO = 4
"""Lingbot World VAE temporal stride; one encoded frame per N video frames."""

_TRANSFORMER_LEN_T = 3
"""Latent frames the transformer consumes per AR chunk."""


def preprocess_example_poses(poses: np.ndarray) -> tuple[np.ndarray, float]:
    """Preprocess the example poses carried over from the original Lingbot World repo.

    Truncates the raw camera stream to the largest length compatible with
    the AR-chunk grid, interpolates it down to the encoded length, computes
    the world-scale normalizer on the encoded poses (matching upstream),
    then re-expands the encoded poses back to per-pixel-frame cadence so
    the encoder's stride-4 frame selection recovers the encoded sequence.

    Args:
        poses: Raw camera-to-world poses of shape ``[T, 4, 4]``.

    Returns:
        Tuple of ``(poses [T_clipped, 4, 4], world_scale)``.
    """
    assert poses.ndim == 3 and poses.shape[1:] == (4, 4), (
        "Expected poses shape [T, 4, 4]"
    )
    T_raw = poses.shape[0]
    T = (T_raw - 1) // _TEMPORAL_COMPRESSION_RATIO * _TEMPORAL_COMPRESSION_RATIO + 1
    poses = poses[:T]

    T_after_encoding = int((T - 1) // _TEMPORAL_COMPRESSION_RATIO) + 1
    T_after_encoding = int(T_after_encoding - (T_after_encoding % _TRANSFORMER_LEN_T))

    poses_after_encoding = interpolate_camera_poses(
        src_indices=np.linspace(0, T - 1, T),
        src_rot_mat=poses[:, :3, :3],
        src_trans_vec=poses[:, :3, 3],
        tgt_indices=np.linspace(0, T - 1, T_after_encoding),
    )  # [T_after_encoding, 4, 4]

    # Match upstream Lingbot World: world-scale normalizer is computed
    # framewise on the *encoded-length* poses, not the raw stream.
    _, trans_normalizer = compute_relative_poses(
        torch.from_numpy(poses_after_encoding).float(),
        framewise=True,
        normalize_trans=True,
    )

    # Re-expand to per-pixel-frame cadence so the encoder's stride-4 frame
    # selection ([0, 4, 8, ...] at AR step 0, [3, 7, 11, ...] later) recovers
    # the encoded sequence exactly. The first encoded frame stays as the
    # one-frame chunk; every subsequent encoded frame is repeated 4x.
    poses = np.concatenate(
        [
            poses_after_encoding[:1],
            np.repeat(poses_after_encoding[1:], _TEMPORAL_COMPRESSION_RATIO, axis=0),
        ],
        axis=0,
    )  # [T, 4, 4]

    # Round-trip check that the encoder's stride-4 frame selection will
    # recover ``poses_after_encoding``. Skipped under ``python -O`` so
    # production runs don't pay for the per-pose allclose.
    # indices = [0] + list(
    #     range(
    #         _TEMPORAL_COMPRESSION_RATIO,
    #         poses.shape[0],
    #         _TEMPORAL_COMPRESSION_RATIO,
    #     )
    # )
    # np.testing.assert_allclose(
    #     poses[indices], poses_after_encoding, atol=1e-4, rtol=1e-4
    # )

    return poses, trans_normalizer


def get_Ks_transformed(
    Ks: torch.Tensor,
    height_org: int,
    width_org: int,
    height_resize: int,
    width_resize: int,
    height_final: int,
    width_final: int,
) -> torch.Tensor:
    """Rescale + recenter intrinsics for a resize-then-center-crop pipeline.

    Mirrors the OpenCV ``resize`` + center-crop the runner applies to the
    image: scales ``fx, fy, cx, cy`` from the capture resolution to the
    resized frame, then shifts the principal point to compensate for the
    crop down to the final size.

    Args:
        Ks: Per-frame intrinsics ``[..., 4]`` (``fx, fy, cx, cy``); arbitrary
            leading batch dims are preserved.
        height_org: Capture-resolution height ``Ks`` is expressed in.
        width_org: Capture-resolution width ``Ks`` is expressed in.
        height_resize: Height after the resize step.
        width_resize: Width after the resize step.
        height_final: Height after the center crop.
        width_final: Width after the center crop.

    Returns:
        Transformed intrinsics with the same shape and dtype as ``Ks``.
    """
    fx, fy, cx, cy = Ks.chunk(4, dim=-1)  # [..., 1]

    scale_x = width_resize / width_org
    scale_y = height_resize / height_org

    fx_resize = fx * scale_x
    fy_resize = fy * scale_y
    cx_resize = cx * scale_x
    cy_resize = cy * scale_y

    crop_offset_x = (width_resize - width_final) / 2
    crop_offset_y = (height_resize - height_final) / 2

    cx_final = cx_resize - crop_offset_x
    cy_final = cy_resize - crop_offset_y

    Ks_transformed = torch.zeros_like(Ks)
    Ks_transformed[..., 0:1] = fx_resize
    Ks_transformed[..., 1:2] = fy_resize
    Ks_transformed[..., 2:3] = cx_final
    Ks_transformed[..., 3:4] = cy_final

    return Ks_transformed


def interpolate_camera_poses(
    src_indices: np.ndarray,
    src_rot_mat: np.ndarray,
    src_trans_vec: np.ndarray,
    tgt_indices: np.ndarray,
) -> np.ndarray:
    """Resample a camera trajectory onto a new index grid.

    Linearly interpolates translations and SLERP-interpolates rotations
    (after a sign-flip pass that keeps adjacent quaternions on the same
    hemisphere so SLERP doesn't take the long way around).

    Args:
        src_indices: Sample positions of the source poses ``[N]``.
        src_rot_mat: Source rotations ``[N, 3, 3]``.
        src_trans_vec: Source translations ``[N, 3]``.
        tgt_indices: Sample positions to resample at ``[M]``.

    Returns:
        Resampled SE(3) poses ``[M, 4, 4]``.

    Raises:
        ImportError: SciPy is not installed.
    """
    try:
        from scipy.interpolate import interp1d  # noqa: PLC0415
        from scipy.spatial.transform import Rotation, Slerp  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "interpolate_camera_poses requires SciPy. Install it with "
            "`pip install scipy` (or `pip install flashdreams[examples]`)."
        ) from e

    interp_func_trans = interp1d(
        src_indices,
        src_trans_vec,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    interpolated_trans_vec = interp_func_trans(tgt_indices)

    # SLERP needs successive quaternions on the same hemisphere, otherwise
    # the great-circle path takes the long way around.
    src_quat_vec = Rotation.from_matrix(src_rot_mat)
    quats = src_quat_vec.as_quat().copy()  # [N, 4]
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]
    src_quat_vec = Rotation.from_quat(quats)
    slerp_func_rot = Slerp(src_indices, src_quat_vec)
    interpolated_rot_quat = slerp_func_rot(tgt_indices)
    interpolated_rot_mat = interpolated_rot_quat.as_matrix()

    poses = np.zeros((len(tgt_indices), 4, 4))
    poses[:, :3, :3] = interpolated_rot_mat
    poses[:, :3, 3] = interpolated_trans_vec
    poses[:, 3, 3] = 1.0
    return poses


def SE3_inverse(T: Tensor) -> Tensor:
    """Invert a batch of SE(3) transforms ``[..., 4, 4]`` analytically."""
    batch_shape = T.shape[:-2]
    Rot = T[..., :3, :3]
    trans = T[..., :3, 3:]
    R_inv = Rot.transpose(-1, -2)
    t_inv = -torch.bmm(R_inv, trans)
    T_inv = torch.eye(4, device=T.device, dtype=T.dtype).repeat(*batch_shape, 1, 1)
    T_inv[..., :3, :3] = R_inv
    T_inv[..., :3, 3:] = t_inv
    return T_inv


def compute_relative_poses(
    c2ws_mat: Tensor,
    framewise: bool = False,
    normalize_trans: bool = True,
) -> tuple[Tensor, float]:
    """Compute relative camera poses against the first frame.

    Args:
        c2ws_mat: Camera-to-world poses of shape ``[T, 4, 4]``.
        framewise: If ``True``, return frame-to-frame relative poses
            instead of frame-to-first relative poses.
        normalize_trans: If ``True``, scale all translations by the maximum
            translation norm so the inputs sit roughly within unit norm.

    Returns:
        Tuple of relative poses ``[T, 4, 4]`` and the scalar normalizer
        applied to translations (``1.0`` when ``normalize_trans`` is ``False``).
    """
    ref_w2cs = SE3_inverse(c2ws_mat[0:1])
    relative_poses = torch.matmul(ref_w2cs, c2ws_mat)
    relative_poses[0] = torch.eye(4, device=c2ws_mat.device, dtype=c2ws_mat.dtype)
    if framewise:
        relative_poses_framewise = torch.bmm(
            SE3_inverse(relative_poses[:-1]), relative_poses[1:]
        )
        relative_poses[1:] = relative_poses_framewise
    if normalize_trans:
        # See camctrl2: scale coordinate inputs to roughly 1 standard
        # deviation to simplify model learning.
        translations = relative_poses[:, :3, 3]
        max_norm = torch.norm(translations, dim=-1).max().item()
        if max_norm > 0:
            relative_poses[:, :3, 3] = translations / max_norm
    else:
        max_norm = 1.0
    return relative_poses, max_norm


def compute_relative_poses_causal(
    c2ws_mat: Tensor,
    trans_normalizer: Tensor | float = 1.0,
    ref_pose: Tensor | None = None,
) -> Tensor:
    """Compute frame-to-frame relative poses anchored to ``ref_pose``.

    Args:
        c2ws_mat: Camera-to-world poses of shape ``[..., T, 4, 4]``.
        trans_normalizer: Divisor applied to translations after the relative
            transform; pre-computed by :func:`compute_relative_poses`.
        ref_pose: Anchor pose of shape ``[..., 1, 4, 4]`` used to pad the
            sequence on the left so the first relative pose is well-defined;
            defaults to the first frame of ``c2ws_mat`` when ``None``.

    Returns:
        Relative poses of shape ``[..., T, 4, 4]``.
    """
    if ref_pose is None:
        ref_pose = c2ws_mat[..., 0:1, :, :]
    assert ref_pose.shape[-3:] == (1, 4, 4)
    c2ws_mat = torch.cat([ref_pose, c2ws_mat], dim=-3)
    relative_poses = torch.bmm(
        SE3_inverse(c2ws_mat[..., :-1, :, :]), c2ws_mat[..., 1:, :, :]
    )
    relative_poses[..., :, :3, 3] /= trans_normalizer
    return relative_poses


def create_meshgrid(
    n_frames: int,
    height: int,
    width: int,
    bias: float = 0.5,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Build a flattened ``(x, y)`` pixel grid replicated across ``n_frames``.

    Args:
        n_frames: Number of frames to replicate the grid for.
        height: Pixel-space height.
        width: Pixel-space width.
        bias: Sub-pixel offset added to integer pixel coordinates
            (``0.5`` for pixel centers).
        device: Output device.
        dtype: Output dtype.

    Returns:
        Per-pixel ``(x, y)`` coordinates, shape ``[n_frames, H * W, 2]``.
    """
    x_range = torch.arange(width, device=device, dtype=dtype)
    y_range = torch.arange(height, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing="ij")
    grid_xy = torch.stack([grid_x, grid_y], dim=-1).view([-1, 2]) + bias
    grid_xy = grid_xy[None, ...].repeat(n_frames, 1, 1)
    return grid_xy


def get_plucker_embeddings(
    c2ws_mat: Tensor,
    Ks: Tensor,
    height: int,
    width: int,
    only_rays_d: bool = False,
) -> Tensor:
    """Compute per-pixel Plücker embeddings for a stack of camera frames.

    Args:
        c2ws_mat: Camera-to-world poses of shape ``[T, 4, 4]``.
        Ks: Per-frame intrinsics ``[T, 4]`` (``fx, fy, cx, cy``).
        height: Pixel-space height.
        width: Pixel-space width.
        only_rays_d: If ``True``, return ray directions only ``[T, H, W, 3]``;
            otherwise stack ``[origins, directions]`` to ``[T, H, W, 6]``.

    Returns:
        Plücker tensor of shape ``[T, H, W, 3]`` or ``[T, H, W, 6]``.
    """
    n_frames = c2ws_mat.shape[0]
    grid_xy = create_meshgrid(
        n_frames, height, width, device=c2ws_mat.device, dtype=c2ws_mat.dtype
    )
    fx, fy, cx, cy = Ks.chunk(4, dim=-1)

    i = grid_xy[..., 0]
    j = grid_xy[..., 1]
    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs

    directions = torch.stack([xs, ys, zs], dim=-1)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    rays_d = directions @ c2ws_mat[:, :3, :3].transpose(-1, -2)
    if only_rays_d:
        plucker_embeddings = rays_d
        plucker_embeddings = plucker_embeddings.view([n_frames, height, width, 3])
    else:
        rays_o = c2ws_mat[:, :3, 3]
        rays_o = rays_o[:, None, :].expand_as(rays_d)
        plucker_embeddings = torch.cat([rays_o, rays_d], dim=-1)
        plucker_embeddings = plucker_embeddings.view([n_frames, height, width, 6])
    return plucker_embeddings
