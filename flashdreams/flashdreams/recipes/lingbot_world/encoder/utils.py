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

import torch
from torch import Tensor


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
) -> tuple[Tensor, Tensor | float]:
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
        max_norm = torch.norm(translations, dim=-1).max()
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
