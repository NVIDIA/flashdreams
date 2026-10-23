# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Latent <-> token patchify for the vae2llm / llm2vae boundary.

The MoT consumes vision as tokens: each token is a ``patch_spatial x
patch_spatial`` spatial patch of one latent frame, flattened to
``patch_latent_dim = patch_spatial**2 * latent_channels`` — the ``vae2llm``
input width. Tokens are laid out T-major then H then W, matching the mRoPE-3D
vision position order (``_rope.vae_mrope_ids``), so token ``i`` and mRoPE id
``i`` refer to the same patch.

``patchify_latent`` turns a VAE latent ``[C, T, H, W]`` into ``[T*(H/p)*(W/p),
p*p*C]`` tokens (the noisy input the rollout denoises); ``unpatchify_latent``
inverts it to feed the VAE/TAEHV decoder. The exact channel/patch ordering
within a token is preserved by the inverse operation and matches the mRoPE layout.

Some embodiments encode a canvas larger than the checkpoint's logical latent
grid; the pad/crop helpers below reconcile the two without disturbing source
pixels. Padding goes on the bottom and right so content stays anchored at the
top-left origin the model was trained on.
"""

from __future__ import annotations

import torch
from einops import rearrange
from torch import Tensor


def pad_latent_bottom(latent: Tensor, rows: int) -> Tensor:
    """Zero-pad ``rows`` onto latent height, leaving source content untouched."""
    if latent.ndim != 4:
        raise ValueError(f"expected [C,T,H,W] latent, got {tuple(latent.shape)}")
    rows = int(rows)
    if rows < 0:
        raise ValueError(f"bottom padding must be non-negative, got {rows}")
    if rows == 0:
        return latent
    padding = torch.zeros(
        (*latent.shape[:-2], rows, latent.shape[-1]),
        dtype=latent.dtype,
        device=latent.device,
    )
    return torch.cat((latent, padding), dim=-2)


def pad_latent_right(latent: Tensor, columns: int) -> Tensor:
    """Zero-pad ``columns`` onto latent width, leaving source content untouched."""
    if latent.ndim != 4:
        raise ValueError(f"expected [C,T,H,W] latent, got {tuple(latent.shape)}")
    columns = int(columns)
    if columns < 0:
        raise ValueError(f"right padding must be non-negative, got {columns}")
    if columns == 0:
        return latent
    padding = torch.zeros(
        (*latent.shape[:-1], columns), dtype=latent.dtype, device=latent.device
    )
    return torch.cat((latent, padding), dim=-1)


def crop_latent_height(latent: Tensor, height: int) -> Tensor:
    """Crop latent height back to ``height`` before VAE decoding."""
    if latent.ndim != 4:
        raise ValueError(f"expected [C,T,H,W] latent, got {tuple(latent.shape)}")
    height = int(height)
    if height <= 0 or height > latent.shape[-2]:
        raise ValueError(
            f"decode height must be in [1, {latent.shape[-2]}], got {height}"
        )
    return latent[..., :height, :]


def crop_latent_width(latent: Tensor, width: int) -> Tensor:
    """Crop latent width back to ``width`` before VAE decoding."""
    if latent.ndim != 4:
        raise ValueError(f"expected [C,T,H,W] latent, got {tuple(latent.shape)}")
    width = int(width)
    if width <= 0 or width > latent.shape[-1]:
        raise ValueError(
            f"decode width must be in [1, {latent.shape[-1]}], got {width}"
        )
    return latent[..., :width]


def patchify_latent(latent: Tensor, patch_spatial: int) -> Tensor:
    """``[C, T, H, W]`` latent -> ``[T*(H/p)*(W/p), p*p*C]`` tokens (T-major, h, w)."""
    _, _, h, w = latent.shape
    assert h % patch_spatial == 0 and w % patch_spatial == 0, (
        f"H,W ({h},{w}) must be divisible by patch_spatial={patch_spatial}"
    )
    return rearrange(
        latent,
        "c t (h ph) (w pw) -> (t h w) (ph pw c)",
        ph=patch_spatial,
        pw=patch_spatial,
    )


def unpatchify_latent(
    tokens: Tensor,
    *,
    channels: int,
    grid_t: int,
    grid_h: int,
    grid_w: int,
    patch_spatial: int,
) -> Tensor:
    """Inverse of :func:`patchify_latent` -> ``[C, T, H, W]`` (grid_* are post-patch)."""
    return rearrange(
        tokens,
        "(t h w) (ph pw c) -> c t (h ph) (w pw)",
        t=grid_t,
        h=grid_h,
        w=grid_w,
        ph=patch_spatial,
        pw=patch_spatial,
        c=channels,
    )
