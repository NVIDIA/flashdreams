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

r"""PRoPE-style projective positional encoding for multi-view attention.

Applies per-camera projective transforms to Q/K/V before attention. The
transform is a block-diagonal matrix multiply on the per-head feature
axis: each camera's tokens get multiplied by a 4×4 matrix derived from
that camera's world-to-camera extrinsic and (optional) intrinsic. The
matrices are

* ``P = lift(K) @ viewmats``    (image ← world; passed through query)
* ``P_inv = inv(viewmats) @ lift(inv(K))``  (world ← image; passed through K, V)
* ``P_T = P.transpose(-1, -2)`` (applied to query)

so the attention score :math:`Q P_q \cdot K^T P_{k}^{-T}` evaluates to the
pair-wise projective positional encoding. The post-attention output is
multiplied by ``P`` of the query's camera to undo the input basis change.
``head_dim`` must be divisible by 4 because the projection matrices are
4×4.
"""

from __future__ import annotations

from functools import partial
from typing import Callable, Tuple

import torch
from torch import Tensor

__all__ = [
    "prope_qkv",
    "build_prope_apply_fns",
]


def prope_qkv(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    viewmats: Tensor,
    Ks: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, Callable[[Tensor], Tensor]]:
    """Apply PRoPE projective positional encoding to Q/K/V.

    Args:
        q: Query tensor, shape ``[batch, num_heads, seqlen, head_dim]``.
        k: Key tensor with the same shape as ``q``.
        v: Value tensor with the same shape as ``q``.
        viewmats: Per-camera world-to-camera ``SE(3)`` matrices, shape
            ``[batch, cameras, 4, 4]``. The token axis must satisfy
            ``seqlen % cameras == 0`` so each camera owns a contiguous
            block of ``seqlen // cameras`` tokens. ``cameras`` is the
            number of latent frames covered by ``q``; cached K/V from
            earlier AR steps must already be PRoPE-transformed and is
            concatenated by the caller *after* this function returns.
        Ks: Optional camera intrinsics ``[batch, cameras, 3, 3]``. When
            ``None`` the formula degenerates to the no-intrinsic variant.

    Returns:
        ``(q_prope, k_prope, v_prope, apply_fn_o)`` where the first three
        tensors carry the projective transform on Q, K, V respectively and
        ``apply_fn_o`` applies the matching projective transform to the
        attention output. ``head_dim`` must be divisible by 4.
    """
    batch, num_heads, seqlen, head_dim = q.shape
    cameras = viewmats.shape[1]
    assert q.shape == k.shape == v.shape, (
        f"PRoPE requires Q/K/V to share the same shape; got Q={tuple(q.shape)}, "
        f"K={tuple(k.shape)}, V={tuple(v.shape)}."
    )
    assert viewmats.shape == (batch, cameras, 4, 4), (
        f"viewmats must have shape [batch={batch}, cameras, 4, 4]; "
        f"got {tuple(viewmats.shape)}."
    )
    assert Ks is None or Ks.shape == (batch, cameras, 3, 3), (
        f"Ks (when provided) must have shape [batch={batch}, cameras={cameras}, "
        f"3, 3]; got {tuple(Ks.shape) if Ks is not None else None}."
    )
    assert head_dim % 4 == 0, (
        f"PRoPE applies a tiled 4x4 projmat on head_dim, which must be a "
        f"multiple of 4; got head_dim={head_dim}."
    )

    apply_fn_q, apply_fn_kv, apply_fn_o = build_prope_apply_fns(
        head_dim=head_dim, viewmats=viewmats, Ks=Ks
    )
    return apply_fn_q(q), apply_fn_kv(k), apply_fn_kv(v), apply_fn_o


def build_prope_apply_fns(
    *,
    head_dim: int,
    viewmats: Tensor,
    Ks: Tensor | None,
) -> tuple[
    Callable[[Tensor], Tensor],
    Callable[[Tensor], Tensor],
    Callable[[Tensor], Tensor],
]:
    """Precompute the per-camera projective transforms used by :func:`prope_qkv`.

    Returns three callables (``apply_fn_q``, ``apply_fn_kv``,
    ``apply_fn_o``) that each take a ``[batch, num_heads, seqlen, head_dim]``
    tensor and apply the corresponding tiled 4×4 matrix per camera. Use
    this directly to pay the matrix-prep cost once when reusing transforms
    across multiple attention layers.
    """
    batch, cameras, _, _ = viewmats.shape
    assert head_dim % 4 == 0, (
        f"PRoPE applies a tiled 4x4 projmat on head_dim, which must be a "
        f"multiple of 4; got head_dim={head_dim}."
    )

    if Ks is not None:
        # Keep only the focal lengths so PRoPE encodes focal / rotation /
        # translation; the principal point is dropped (callers pass K with
        # cx/cy already renormalised).
        Ks_norm = torch.zeros_like(Ks)
        Ks_norm[..., 0, 0] = Ks[..., 0, 0]
        Ks_norm[..., 1, 1] = Ks[..., 1, 1]
        Ks_norm[..., 2, 2] = 1.0
        Ks_norm = Ks_norm.to(dtype=Ks.dtype)

        # P = lift(K) @ viewmats is the image<-world transform; keep both
        # P (for the output projection / query) and P_inv (for K/V).
        P = torch.einsum("...ij,...jk->...ik", _lift_K(Ks_norm), viewmats)
        P_T = P.transpose(-1, -2).to(dtype=viewmats.dtype)
        P_inv = torch.einsum(
            "...ij,...jk->...ik",
            _invert_SE3(viewmats),
            _lift_K(_invert_K(Ks_norm)),
        ).to(dtype=viewmats.dtype)
    else:
        # Intrinsic-free variant.
        P = viewmats
        P_T = P.transpose(-1, -2)
        P_inv = _invert_SE3(viewmats)

    assert P.shape == P_inv.shape == (batch, cameras, 4, 4)
    apply_fn_q = partial(_apply_tiled_projmat, matrix=P_T)
    apply_fn_kv = partial(_apply_tiled_projmat, matrix=P_inv)
    apply_fn_o = partial(_apply_tiled_projmat, matrix=P)
    return apply_fn_q, apply_fn_kv, apply_fn_o


## Internal helpers


def _apply_tiled_projmat(
    feats: Tensor,
    *,
    matrix: Tensor,
) -> Tensor:
    """Apply a per-camera 4x4 matrix block-diagonally on the head_dim axis.

    ``feats`` is reshaped from ``[B, H, seqlen, feat_dim]`` to
    ``[B, H, cameras, patches_per_cam, feat_dim // 4, 4]`` so each
    consecutive 4-vector on the trailing axis gets multiplied by that
    camera's 4x4 matrix. Token ordering must place each camera's tokens
    contiguously on the seqlen axis (``seqlen == cameras * patches_per_cam``).
    """
    batch, num_heads, seqlen, feat_dim = feats.shape
    cameras = matrix.shape[1]
    assert seqlen >= cameras and seqlen % cameras == 0, (
        f"PRoPE token layout requires seqlen ({seqlen}) divisible by "
        f"cameras ({cameras})."
    )
    D = matrix.shape[-1]
    assert matrix.shape == (batch, cameras, D, D), (
        f"matrix must be [batch={batch}, cameras={cameras}, {D}, {D}]; "
        f"got {tuple(matrix.shape)}."
    )
    assert feat_dim % D == 0, (
        f"head_dim ({feat_dim}) must be divisible by {D} for the block-diagonal tiling."
    )
    return torch.einsum(
        "bcij,bncpkj->bncpki",
        matrix,
        feats.reshape((batch, num_heads, cameras, -1, feat_dim // D, D)),
    ).reshape(feats.shape)


def _invert_SE3(transforms: Tensor) -> Tensor:
    """Invert a stack of 4x4 ``SE(3)`` matrices using the rigid-body shortcut."""
    assert transforms.shape[-2:] == (4, 4)
    Rinv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out.to(dtype=transforms.dtype)


def _lift_K(Ks: Tensor) -> Tensor:
    """Embed 3x3 camera intrinsics into 4x4 homogeneous form.

    Allocates in ``Ks.dtype`` so a float64 input is not silently
    downcast.
    """
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros(Ks.shape[:-2] + (4, 4), device=Ks.device, dtype=Ks.dtype)
    out[..., :3, :3] = Ks
    out[..., 3, 3] = 1.0
    return out


def _invert_K(Ks: Tensor) -> Tensor:
    """Invert 3x3 intrinsics (no-skew, closed form)."""
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros_like(Ks)
    out[..., 0, 0] = 1.0 / Ks[..., 0, 0]
    out[..., 1, 1] = 1.0 / Ks[..., 1, 1]
    out[..., 0, 2] = -Ks[..., 0, 2] / Ks[..., 0, 0]
    out[..., 1, 2] = -Ks[..., 1, 2] / Ks[..., 1, 1]
    out[..., 2, 2] = 1.0
    return out.to(dtype=Ks.dtype)


_Tuple3Tensor = Tuple[Tensor, Tensor, Tensor]
