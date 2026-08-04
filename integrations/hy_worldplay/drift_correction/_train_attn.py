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

"""Grad-friendly functional dual-branch attention for corrector training.

The production ``forward_dual_branch`` routes the current chunk's K / V
through the rolling ``BlockKVCache`` (in-place write into a no-grad buffer,
then a buffer read), which (a) severs gradients into the ``k`` / ``v``
projections and (b) plants mutable storage inside the autograd tape,
breaking gradient-checkpoint recomputation.

Training probes run exactly one forward per chunk on a freshly reset cache,
where ``cached_k()`` would return precisely the current chunk -- so the
functional equivalent below (use the projected K / V directly, prepend the
prefilled memory K / V) is mathematically identical while keeping the whole
path differentiable and side-effect free. Deployment keeps the stock path;
the corrector's weights are shared, so val metrics transfer.

Must mirror :meth:`HyWorldPlayPRoPESelfAttention.forward_dual_branch`
(``hy_worldplay/_camera.py``); revisit on upstream changes.
"""

from __future__ import annotations

import math

import torch
from hy_worldplay._camera import (
    HyWorldPlayMemoryKVCache,
    HyWorldPlayPRoPESelfAttention,
)
from hy_worldplay._prope import prope_qkv
from torch import Tensor


def _rope_interleaved(x: Tensor, freqs: Tensor) -> Tensor:
    """Differentiable equivalent of the fused interleaved-RoPE Triton kernel.

    The production kernel (``rope_kernel.apply_rotary_pos_emb``) is
    inference-only: raw ``tl.store`` writes with no backward, so gradients
    through it are silently dropped. Mirrors its arithmetic exactly: pair
    ``(2k, 2k+1)`` rotated by angle ``freqs[..., k]``, cos/sin in fp32 then
    cast to ``x.dtype`` before the multiply.

    Args:
        x: Activations ``[B, S, H, D]``.
        freqs: Angles in the full-width ``[S, 1, 1, D]`` ``shift_t`` layout,
            where each pair's angle is duplicated; pair ``k`` reads lane
            ``2k`` (the kernel's ``stride_fd * 2`` skip).

    Returns:
        Rotated tensor, same shape/dtype, fresh storage.
    """
    ang = freqs[:, 0, 0, 0::2].float()
    cos = ang.cos().to(x.dtype)[None, :, None, :]
    sin = ang.sin().to(x.dtype)[None, :, None, :]
    pairs = x.unflatten(-1, (x.shape[-1] // 2, 2))
    a, b = pairs[..., 0], pairs[..., 1]
    return torch.stack((a * cos - b * sin, b * cos + a * sin), dim=-1).flatten(-2)


def _functional_dual_branch(
    self: HyWorldPlayPRoPESelfAttention,
    x: Tensor,
    kv_cache,
    prope_kv_cache,
    rope_freqs: Tensor,
    viewmats: Tensor,
    Ks: Tensor | None,
    memory_kv_cache: HyWorldPlayMemoryKVCache | None = None,
) -> Tensor:
    """Cache-free re-implementation of ``forward_dual_branch``.

    Valid only for the single-forward-per-chunk training regime: asserts the
    rolling cache is empty so the functional K / V equal what the stock path
    would have read back.
    """
    assert kv_cache._n_cached == 0, (
        "functional attention expects a freshly reset rolling cache "
        "(one forward per start/finalize bracket)."
    )
    assert self.apply_rope_before_kvcache, (
        "cache-relative RoPE mode would need the cached-window freqs; the "
        "TI2V-5B recipe applies RoPE before the cache."
    )
    rope_freqs_q, rope_freqs_k = self._slice_rope_freqs(rope_freqs, kv_cache)

    batch_shape = x.shape[:-2]
    batch_size = math.prod(batch_shape)
    L, _ = x.shape[-2:]
    n, d = self.n_heads, self.head_dim

    q_raw = self.norm_q(self.q(x)).reshape(batch_size, L, n, d)
    k_raw = self.norm_k(self.k(x)).reshape(batch_size, L, n, d)
    v_raw = self.v(x).reshape(batch_size, L, n, d)

    # The production kernel rotates in place, aliasing ``k_raw`` -- the
    # stock PRoPE branch therefore consumes the *roped* K (and the un-roped
    # Q, since the Q rotation happens after ``prope_qkv``). Replicate that
    # dataflow explicitly, out of place and differentiably.
    k_cur = k_raw
    if rope_freqs_k is not None:
        k_cur = _rope_interleaved(k_raw, rope_freqs_k)

    # PRoPE math runs in bhsd.
    q_prope, k_prope_bhsd, v_prope_bhsd, apply_fn_o = prope_qkv(
        q_raw.transpose(1, 2),
        k_cur.transpose(1, 2),
        v_raw.transpose(1, 2),
        viewmats=viewmats,
        Ks=Ks,
    )

    q_rope = q_raw
    if rope_freqs_q is not None:
        q_rope = _rope_interleaved(q_raw, rope_freqs_q)

    # Standard RoPE branch: [memory K/V, current K/V].
    keys, vals = k_cur, v_raw
    if memory_kv_cache is not None and memory_kv_cache.has_rope_kv:
        assert memory_kv_cache.k_rope is not None
        assert memory_kv_cache.v_rope is not None
        keys = torch.cat([memory_kv_cache.k_rope, keys], dim=-3)
        vals = torch.cat([memory_kv_cache.v_rope, vals], dim=-3)
    out_rope = self.attn_op(q_rope, keys, vals)
    out_rope = self.o(out_rope.reshape(batch_shape + (L, n * d)))

    # PRoPE branch; same memory prepend on the camera side.
    k_p = k_prope_bhsd.transpose(1, 2)
    v_p = v_prope_bhsd.transpose(1, 2)
    if memory_kv_cache is not None and memory_kv_cache.has_prope_kv:
        assert memory_kv_cache.k_prope is not None
        assert memory_kv_cache.v_prope is not None
        k_p = torch.cat([memory_kv_cache.k_prope, k_p], dim=-3)
        v_p = torch.cat([memory_kv_cache.v_prope, v_p], dim=-3)
    out_prope = self.attn_op_prope(q_prope.transpose(1, 2), k_p, v_p)
    out_prope = apply_fn_o(out_prope.transpose(1, 2)).transpose(1, 2)
    out_prope = self.o_prope(out_prope.reshape(batch_shape + (L, n * d)))

    return out_rope + out_prope


def patch_functional_attention() -> None:
    """Swap ``forward_dual_branch`` for the functional variant, process-wide."""
    HyWorldPlayPRoPESelfAttention.forward_dual_branch = _functional_dual_branch  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
