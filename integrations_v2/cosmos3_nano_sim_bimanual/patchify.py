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

"""Latent-to-token patching for the VAE/LLM boundary."""

from __future__ import annotations

from einops import rearrange
from torch import Tensor
from torch.nn import functional as F


def pad_latent(latent: Tensor, *, bottom: int, right: int) -> Tensor:
    """Zero-pad a ``[C,T,H,W]`` latent on its bottom and right edges."""
    if latent.ndim != 4:
        raise ValueError(f"expected [C,T,H,W] latent, got {tuple(latent.shape)}")
    bottom, right = int(bottom), int(right)
    if bottom < 0 or right < 0:
        raise ValueError(
            f"bottom/right padding must be non-negative, got {bottom}/{right}"
        )
    if bottom == right == 0:
        return latent
    return F.pad(latent, (0, right, 0, bottom))


def crop_latent(latent: Tensor, *, height: int, width: int) -> Tensor:
    """Crop a ``[C,T,H,W]`` latent from its bottom and right edges."""
    if latent.ndim != 4:
        raise ValueError(f"expected [C,T,H,W] latent, got {tuple(latent.shape)}")
    height, width = int(height), int(width)
    if height <= 0 or height > latent.shape[-2]:
        raise ValueError(
            f"decode height must be in [1, {latent.shape[-2]}], got {height}"
        )
    if width <= 0 or width > latent.shape[-1]:
        raise ValueError(
            f"decode width must be in [1, {latent.shape[-1]}], got {width}"
        )
    return latent[..., :height, :width]


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
