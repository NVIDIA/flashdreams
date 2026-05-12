# MIT License
#
# Copyright (c) Authors of
# "Cameras as Relative Positional Encoding" https://arxiv.org/pdf/2507.10496
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""PRoPE attention with precomputed RoPE coefficients.

Verbatim port of dreamfix ``model_training/net/prope.py`` (matching paper
"Cameras as Relative Positional Encoding", arXiv 2507.10496). The math is
unchanged; the only difference is that ``invert_SE3`` is inlined here so
this module is self-contained.

Usage for cross-attention follows the upstream docstring::

    attn_src = PropeDotProductAttention(...)
    attn_tgt = PropeDotProductAttention(...)
    attn_src._precompute_and_cache_apply_fns(viewmats_src, Ks_src)
    attn_tgt._precompute_and_cache_apply_fns(viewmats_tgt, Ks_tgt)
    q_src = attn_src._apply_to_q(q_src)
    k_tgt = attn_tgt._apply_to_kv(k_tgt)
    v_tgt = attn_tgt._apply_to_kv(v_tgt)
    o_src = F.scaled_dot_product_attention(q_src, k_tgt, v_tgt, **kwargs)
    o_src = attn_src._apply_to_o(o_src)

A numerical-parity unit test against the dreamfix reference lives at
``integrations/artifixer/tests/test_prope_parity.py``.
"""

from __future__ import annotations

from functools import partial
from typing import Callable

import torch
from torch import nn


class PropeDotProductAttention(nn.Module):
    """PRoPE attention with precomputed RoPE coefficients."""

    coeffs_x_0: torch.Tensor
    coeffs_x_1: torch.Tensor
    coeffs_y_0: torch.Tensor
    coeffs_y_1: torch.Tensor

    def __init__(
        self,
        head_dim: int,
        patches_x: int = 0,
        patches_y: int = 0,
        freq_base: float = 100.0,
        freq_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.patches_x = patches_x
        self.patches_y = patches_y
        self.freq_base = freq_base
        self.freq_scale = freq_scale

    def update_coeffs(
        self, patches_x: int, patches_y: int, device: str | torch.device
    ) -> None:
        self.patches_x = patches_x
        self.patches_y = patches_y
        coeffs_x = _rope_precompute_coeffs(
            torch.tile(torch.arange(patches_x, device=device), (patches_y,)),
            freq_base=self.freq_base,
            freq_scale=self.freq_scale,
            feat_dim=self.head_dim // 4,
        )
        coeffs_y = _rope_precompute_coeffs(
            torch.repeat_interleave(torch.arange(patches_y, device=device), patches_x),
            freq_base=self.freq_base,
            freq_scale=self.freq_scale,
            feat_dim=self.head_dim // 4,
        )
        # Non-persistent buffers: camera count may change between train/eval.
        self.coeffs_x_0 = nn.Buffer(coeffs_x[0], persistent=False)
        self.coeffs_x_1 = nn.Buffer(coeffs_x[1], persistent=False)
        self.coeffs_y_0 = nn.Buffer(coeffs_y[0], persistent=False)
        self.coeffs_y_1 = nn.Buffer(coeffs_y[1], persistent=False)

    def _precompute_and_cache_apply_fns(
        self, viewmats: torch.Tensor, Ks_norm: torch.Tensor
    ) -> None:
        batch, cameras, _, _ = viewmats.shape
        assert viewmats.shape == (batch, cameras, 4, 4)
        assert Ks_norm.shape == (batch, cameras, 3, 3)
        self.cameras = cameras

        self.apply_fn_q, self.apply_fn_kv, self.apply_fn_o = _prepare_apply_fns(
            head_dim=self.head_dim,
            viewmats=viewmats,
            Ks_norm=Ks_norm,
            coeffs_x=(self.coeffs_x_0, self.coeffs_x_1),
            coeffs_y=(self.coeffs_y_0, self.coeffs_y_1),
        )

    def _apply_to_q(self, q: torch.Tensor) -> torch.Tensor:
        batch, num_heads, seqlen, head_dim = q.shape
        assert seqlen == self.cameras * self.patches_x * self.patches_y
        assert head_dim == self.head_dim
        return self.apply_fn_q(q)

    def _apply_to_kv(self, kv: torch.Tensor) -> torch.Tensor:
        batch, num_heads, seqlen, head_dim = kv.shape
        assert seqlen == self.cameras * self.patches_x * self.patches_y
        assert head_dim == self.head_dim
        return self.apply_fn_kv(kv)

    def _apply_to_o(self, o: torch.Tensor) -> torch.Tensor:
        batch, num_heads, seqlen, head_dim = o.shape
        assert seqlen == self.cameras * self.patches_x * self.patches_y
        assert head_dim == self.head_dim
        return self.apply_fn_o(o)


def _prepare_apply_fns(
    head_dim: int,
    viewmats: torch.Tensor,
    Ks_norm: torch.Tensor,
    coeffs_x: tuple[torch.Tensor, torch.Tensor],
    coeffs_y: tuple[torch.Tensor, torch.Tensor],
) -> tuple[
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
]:
    """Prepare transforms for PRoPE-style positional encoding.

    - ``K`` is an ``image<-camera`` transform.
    - ``viewmats`` is a ``camera<-world`` transform.
    - ``P = lift(K) @ viewmats`` is an ``image<-world`` transform.
    """
    batch, cameras, _, _ = viewmats.shape

    P = torch.einsum("...ij,...jk->...ik", _lift_K(Ks_norm), viewmats)
    P_T = P.transpose(-1, -2)
    P_inv = torch.einsum(
        "...ij,...jk->...ik",
        _invert_SE3(viewmats),
        _lift_K(_invert_K(Ks_norm)),
    )
    assert P.shape == P_inv.shape == (batch, cameras, 4, 4)
    assert head_dim % 4 == 0

    transforms_q = [
        (partial(_apply_tiled_projmat, matrix=P_T), head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=coeffs_x), head_dim // 4),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y), head_dim // 4),
    ]
    transforms_kv = [
        (partial(_apply_tiled_projmat, matrix=P_inv), head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=coeffs_x), head_dim // 4),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y), head_dim // 4),
    ]
    transforms_o = [
        (partial(_apply_tiled_projmat, matrix=P), head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=coeffs_x, inverse=True), head_dim // 4),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y, inverse=True), head_dim // 4),
    ]

    apply_fn_q = partial(_apply_block_diagonal, func_size_pairs=transforms_q)
    apply_fn_kv = partial(_apply_block_diagonal, func_size_pairs=transforms_kv)
    apply_fn_o = partial(_apply_block_diagonal, func_size_pairs=transforms_o)
    return apply_fn_q, apply_fn_kv, apply_fn_o


def _apply_tiled_projmat(
    feats: torch.Tensor,
    matrix: torch.Tensor,
) -> torch.Tensor:
    """Apply projection matrix to features.

    ``seqlen => (cameras, patches_x * patches_y)`` and
    ``feat_dim => (feat_dim // 4, 4)``.
    """
    batch, num_heads, seqlen, feat_dim = feats.shape
    cameras = matrix.shape[1]
    assert seqlen > cameras and seqlen % cameras == 0
    D = matrix.shape[-1]
    assert matrix.shape == (batch, cameras, D, D)
    assert feat_dim % D == 0
    return torch.einsum(
        "bcij,bncpkj->bncpki",
        matrix,
        feats.reshape((batch, num_heads, cameras, -1, feat_dim // D, D)),
    ).reshape(feats.shape)


def _rope_precompute_coeffs(
    positions: torch.Tensor,
    freq_base: float,
    freq_scale: float,
    feat_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE (cos, sin) coefficients for given positions."""
    assert positions.ndim == 1
    assert feat_dim % 2 == 0
    num_freqs = feat_dim // 2
    freqs = freq_scale * (
        freq_base
        ** (
            -torch.arange(num_freqs, device=positions.device)[None, None, None, :]
            / num_freqs
        )
    )
    angles = positions[None, None, :, None] * freqs
    assert angles.shape == (1, 1, positions.shape[0], num_freqs)
    return torch.cos(angles), torch.sin(angles)


def _rope_apply_coeffs(
    feats: torch.Tensor,
    coeffs: tuple[torch.Tensor, torch.Tensor],
    inverse: bool = False,
) -> torch.Tensor:
    """Apply RoPE coefficients to features with the *split* ordering convention."""
    cos, sin = coeffs
    if cos.shape[2] != feats.shape[2]:
        n_repeats = feats.shape[2] // cos.shape[2]
        cos = cos.repeat(1, 1, n_repeats, 1)
        sin = sin.repeat(1, 1, n_repeats, 1)
    assert feats.ndim == cos.ndim == sin.ndim == 4
    assert cos.shape[-1] == sin.shape[-1] == feats.shape[-1] // 2
    x_in = feats[..., : feats.shape[-1] // 2]
    y_in = feats[..., feats.shape[-1] // 2 :]
    return torch.cat(
        (
            [cos * x_in + sin * y_in, -sin * x_in + cos * y_in]
            if not inverse
            else [cos * x_in - sin * y_in, sin * x_in + cos * y_in]
        ),
        dim=-1,
    )


def _apply_block_diagonal(
    feats: torch.Tensor,
    func_size_pairs: list[tuple[Callable[[torch.Tensor], torch.Tensor], int]],
) -> torch.Tensor:
    """Apply a block-diagonal function to an input array.

    Each function is specified as a tuple ``((Tensor) -> Tensor, int)`` where
    the integer is the size of the input slice consumed by the function.
    """
    funcs, block_sizes = zip(*func_size_pairs)
    assert feats.shape[-1] == sum(block_sizes)
    x_blocks = torch.split(feats, block_sizes, dim=-1)
    out = torch.cat([f(x) for f, x in zip(funcs, x_blocks)], dim=-1)
    assert out.shape == feats.shape
    return out


def _lift_K(Ks: torch.Tensor) -> torch.Tensor:
    """Lift 3x3 matrices to homogeneous 4x4 matrices."""
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros(Ks.shape[:-2] + (4, 4), device=Ks.device, dtype=Ks.dtype)
    out[..., :3, :3] = Ks
    out[..., 3, 3] = 1.0
    return out


def _invert_K(Ks: torch.Tensor) -> torch.Tensor:
    """Invert 3x3 intrinsics matrices. Assumes no skew."""
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros_like(Ks)
    out[..., 0, 0] = 1.0 / Ks[..., 0, 0]
    out[..., 1, 1] = 1.0 / Ks[..., 1, 1]
    out[..., 0, 2] = -Ks[..., 0, 2] / Ks[..., 0, 0]
    out[..., 1, 2] = -Ks[..., 1, 2] / Ks[..., 1, 1]
    out[..., 2, 2] = 1.0
    return out


def _invert_SE3(transforms: torch.Tensor) -> torch.Tensor:
    """Invert a 4x4 SE(3) matrix.

    Mirrors ``dreamfix/model_training/utils/pose_utils.invert_SE3`` so this
    module has no cross-repo dependencies.
    """
    assert transforms.shape[-2:] == (4, 4)
    Rinv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


__all__ = ["PropeDotProductAttention"]
