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

"""Patchification of opacity / Plucker camera-ray conditioning tensors.

Mirrors ``ArtifixerTransformer.forward`` L309-L341 in dreamfix:

  * opacity (B, T, H, W) -> opacity_extra_patches (B, L, opacity_embedding_dim)
  * camera_rays (B, T, H, W, 6) -> camera_extra_patches (B, L, camera_embedding_dim)

where ``L`` is the post-patch token count for the chunk.

The Wan VAE has a ``1 + 4 * (n - 1)`` temporal layout: the first decoded
frame corresponds to a single latent frame, every subsequent latent frame
covers 4 input frames. When the first patchify call covers ``frame_offset == 0``
(i.e. the first AR chunk), we left-pad the input frame stack by 3 copies of
the first frame so the rearrange treats every latent frame uniformly. Later
chunks already have the temporal downsampling baked in.
"""

from __future__ import annotations

import torch
from einops import rearrange
from torch import Tensor


def patchify_opacity(
    opacity: Tensor,
    *,
    vae_scale_factor_temporal: int,
    vae_scale_factor_spatial: int,
    patch_size: tuple[int, int, int],
    frame_offset: int,
) -> Tensor:
    """Rearrange per-pixel opacity into per-token features.

    Args:
        opacity: ``(B, T, H, W)`` -- raw per-frame alpha at input resolution.
        vae_scale_factor_temporal: e.g. ``4`` for Wan 2.1.
        vae_scale_factor_spatial: e.g. ``8`` for Wan 2.1.
        patch_size: Network ``patch_size``, ``(kt, kh, kw)``.
        frame_offset: Logical start of this chunk in latent-time tokens.
            When ``0`` we pad the first frame by 3 copies to account for the
            Wan VAE's ``1 + 4`` temporal layout (mirrors dreamfix L309-L311).

    Returns:
        ``(B, L, opacity_embedding_dim)`` where
        ``L = (T / vae_t / kt) * (H / vae_s / kh) * (W / vae_s / kw)``.
    """
    _, ph, pw = patch_size

    if frame_offset == 0:
        opacity = torch.cat(
            [opacity[:, :1].repeat_interleave(3, dim=1), opacity], dim=1
        )
    extra = rearrange(
        opacity,
        "b (t t4) (h h8) (w w8) -> b (h8 w8 t4) t h w",
        h8=vae_scale_factor_spatial * ph,
        w8=vae_scale_factor_spatial * pw,
        t4=vae_scale_factor_temporal,
    )
    return extra.flatten(2).transpose(1, 2)


def patchify_camera_rays(
    camera_rays: Tensor,
    *,
    hidden_post_patch_t: int,
    vae_scale_factor_temporal: int,
    vae_scale_factor_spatial: int,
    patch_size: tuple[int, int, int],
    frame_offset: int,
) -> Tensor:
    """Rearrange per-pixel Plucker rays into per-token features.

    Handles dreamfix's branch on whether ``camera_rays``' temporal axis is
    already at the latent rate (``camera_rays.shape[1] == hidden_post_patch_t``)
    or at the input rate (needs the VAE ``1 + 4`` left-pad on the first AR
    chunk).

    Args:
        camera_rays: ``(B, T, H, W, 6)`` Plucker rays.
        hidden_post_patch_t: Number of post-patch latent frames in the chunk
            (i.e. ``hidden_states.shape[2]`` after :meth:`patch_embedding`).
        vae_scale_factor_temporal: e.g. ``4`` for Wan 2.1.
        vae_scale_factor_spatial: e.g. ``8`` for Wan 2.1.
        patch_size: Network ``patch_size``, ``(kt, kh, kw)``.
        frame_offset: Logical start of this chunk in latent-time tokens.

    Returns:
        ``(B, L, camera_embedding_dim)``.
    """
    _, ph, pw = patch_size

    if camera_rays.shape[1] == hidden_post_patch_t:
        extra = rearrange(
            camera_rays,
            "b t (h h8) (w w8) c -> b (c h8 w8) t h w",
            h8=vae_scale_factor_spatial * ph,
            w8=vae_scale_factor_spatial * pw,
        )
    else:
        if frame_offset == 0:
            camera_rays = torch.cat(
                [camera_rays[:, :1].repeat_interleave(3, dim=1), camera_rays], dim=1
            )
        extra = rearrange(
            camera_rays,
            "b (t t4) (h h8) (w w8) c -> b (c h8 w8 t4) t h w",
            h8=vae_scale_factor_spatial * ph,
            w8=vae_scale_factor_spatial * pw,
            t4=vae_scale_factor_temporal,
        )
    return extra.flatten(2).transpose(1, 2)


__all__ = ["patchify_opacity", "patchify_camera_rays"]
