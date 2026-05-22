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

"""HY-WorldPlay camera-trajectory conditioner (phase 2b.4).

Adds dual-branch self-attention -- standard RoPE branch plus PRoPE
camera-projective branch -- on top of the action-aware DiT from 2b.3.
The two branches share the input tokens and Q/K/V projections but
otherwise run independently:

* The RoPE branch is exactly the stock :class:`SelfAttention` path:
  RoPE applied to Q/K, attention against the standard KV cache, output
  projection through ``self.o``.
* The PRoPE branch applies the camera-projective transforms from
  :func:`flashdreams.core.attention.prope.prope_qkv` to Q/K/V, writes
  the transformed K/V into a *second* :class:`BlockKVCache`, runs
  attention against that cache, applies the matching output transform
  (``apply_fn_o``), then projects through ``self.o_prope``. The two
  branch outputs are summed before exiting the self-attn module.

This mirrors upstream's ``arwan_w_action_w_mem_relative_rope.py``
attention forward bit-for-bit (modulo the ``sageattention`` → native
SDPA substitution and the per-camera precision improvements in
:mod:`flashdreams.core.attention.prope`). The ``o_prope`` linear is
zero-initialised so the PRoPE branch is a strict identity at random /
zero init -- the camera path stays parity-safe until HY-WorldPlay's
distilled checkpoint loads non-zero weights for it.

CP is intentionally restricted to size 1 here; multi-rank PRoPE
expansion is left for a follow-up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributed import ProcessGroup

from flashdreams.core.attention import BlockKVCache, RingAttention, prope_qkv
from flashdreams.core.attention.rope import apply_rope_freqs
from flashdreams.recipes.wan.transformer.impl.modules import (
    Block,
    BlockCache,
    SelfAttention,
)

__all__ = [
    "HyWorldPlayPRoPEBlock",
    "HyWorldPlayPRoPEBlockCache",
    "HyWorldPlayPRoPESelfAttention",
]


## ---------------------------------------------------------------------------
## Dual-branch self-attention
## ---------------------------------------------------------------------------


class HyWorldPlayPRoPESelfAttention(SelfAttention):
    """Self-attention with a parallel PRoPE-projected branch.

    Owns a second output projection :attr:`o_prope` (zero-initialised so
    the PRoPE branch is a strict no-op at random init) on top of the
    inherited Q/K/V/o projections from :class:`MultiHeadAttention`. The
    PRoPE branch reuses the same Q/K/V tensors (so HY-WorldPlay weights
    are loaded once into ``q`` / ``k`` / ``v`` / ``norm_q`` / ``norm_k``)
    and only differs on how those tensors are pre-/post-processed for
    attention.

    A second :class:`BlockKVCache` (created alongside the stock cache by
    :meth:`HyWorldPlayPRoPEBlock.initialize_cache`) stores the
    *already-PRoPE-transformed* K / V from previous AR steps so each AR
    step only transforms its current chunk's K / V. The query side is
    transformed fresh every call because it uses the current chunk's
    extrinsics + intrinsics.
    """

    def __init__(
        self,
        query_dim: int,
        n_heads: int = 8,
        head_dim: int = 64,
        eps: float = 1e-6,
        apply_rope_before_kvcache: bool = True,
    ) -> None:
        super().__init__(
            query_dim=query_dim,
            n_heads=n_heads,
            head_dim=head_dim,
            eps=eps,
            apply_rope_before_kvcache=apply_rope_before_kvcache,
        )
        # Second output projection used only by the PRoPE branch.
        # Zero-init mirrors upstream's
        # ``nn.init.zeros_(block.attn1.to_out_prope[0].weight)`` so the
        # camera path adds zero residual until HY-WorldPlay's distilled
        # checkpoint loads non-zero weights for it.
        self.o_prope = nn.Linear(self.inner_dim, self.query_dim)
        nn.init.zeros_(self.o_prope.weight)
        if self.o_prope.bias is not None:
            nn.init.zeros_(self.o_prope.bias)

        # Independent attention op for the PRoPE branch so CP setup
        # (when it lands) can route the two branches identically without
        # cross-branch state.
        self.attn_op_prope = RingAttention(qkv_format="bshd", backend="cudnn")

    def set_context_parallel_group(self, cp_group: ProcessGroup | None) -> None:
        """Route CP to both attention ops; the PRoPE branch follows the standard one."""
        super().set_context_parallel_group(cp_group)
        self.attn_op_prope.set_context_parallel_group(cp_group=cp_group)

    def initialize_prope_cache(
        self,
        batch_size: int,
        chunk_size: int,
        window_size: int,
        sink_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> BlockKVCache:
        """Build a second :class:`BlockKVCache` matching :attr:`attn_op`'s layout."""
        total_size = sink_size + window_size
        return BlockKVCache(
            k_shape=(batch_size, total_size, self.n_heads, self.head_dim),
            v_shape=(batch_size, total_size, self.n_heads, self.head_dim),
            seq_dim=-3,
            chunk_size=chunk_size,
            window_size=window_size,
            sink_size=sink_size,
            device=device,
            dtype=dtype,
        )

    def forward_dual_branch(
        self,
        x: Tensor,
        kv_cache: BlockKVCache,
        prope_kv_cache: BlockKVCache,
        rope_freqs: Tensor,
        viewmats: Tensor,
        Ks: Tensor | None,
    ) -> Tensor:
        """Run the dual-branch self-attention.

        Args:
            x: Input token tensor with shape ``[..., L, query_dim]``.
            kv_cache: Standard RoPE-branch KV cache.
            prope_kv_cache: Second KV cache that stores the PRoPE-
                transformed K / V.
            rope_freqs: Standard-mode RoPE frequencies with shape
                ``[L, 1, 1, head_dim]``.
            viewmats: Per-frame W2C matrices for the *current* chunk,
                shape ``[batch, cameras, 4, 4]`` where ``cameras`` is the
                per-AR-step latent-frame count (``len_t``).
            Ks: Optional per-frame intrinsics ``[batch, cameras, 3, 3]``.

        Returns:
            Sum of the two branches' projected outputs, shape
            ``[..., L, query_dim]``.
        """
        if self.is_context_parallel_enabled():
            raise NotImplementedError(
                "HyWorldPlayPRoPESelfAttention does not yet support "
                "context-parallel (cp_size > 1); CP wiring lands together "
                "with multi-rank action expansion in a follow-up."
            )

        rope_freqs_q, rope_freqs_k = self._slice_rope_freqs(rope_freqs, kv_cache)

        batch_shape = x.shape[:-2]
        batch_size = math.prod(batch_shape)
        L, _ = x.shape[-2:]
        n, d = self.n_heads, self.head_dim

        q_raw = self.norm_q(self.q(x)).reshape(batch_size, L, n, d)
        k_raw = self.norm_k(self.k(x)).reshape(batch_size, L, n, d)
        v_raw = self.v(x).reshape(batch_size, L, n, d)

        # --- RoPE branch K cache write --------------------------------
        k_for_rope_cache = k_raw
        if rope_freqs_k is not None and self.apply_rope_before_kvcache:
            k_for_rope_cache = apply_rope_freqs(
                k_for_rope_cache, rope_freqs_k, interleaved=True
            )
        kv_cache.update(k_for_rope_cache, v_raw)

        # --- PRoPE branch K / V cache write ---------------------------
        # PRoPE expects ``[batch, num_heads, seqlen, head_dim]``; the
        # cache stores ``[batch, seqlen, num_heads, head_dim]``. Transpose
        # in for the PRoPE math and back out before the cache write so the
        # cache layout matches the standard branch.
        q_prope, k_prope_bhsd, v_prope_bhsd, apply_fn_o = prope_qkv(
            q_raw.transpose(1, 2),
            k_raw.transpose(1, 2),
            v_raw.transpose(1, 2),
            viewmats=viewmats,
            Ks=Ks,
        )
        prope_kv_cache.update(
            k_prope_bhsd.transpose(1, 2), v_prope_bhsd.transpose(1, 2)
        )

        # --- Standard RoPE-branch attention ---------------------------
        q_rope = q_raw
        if rope_freqs_q is not None:
            q_rope = apply_rope_freqs(q_rope, rope_freqs_q, interleaved=True)
        if not self.apply_rope_before_kvcache:
            assert rope_freqs_k is not None, (
                "KV-cache-relative RoPE requires rope_freqs_k for cached K"
            )
            cached_k = kv_cache.cached_k().clone()
            cached_k = apply_rope_freqs(cached_k, rope_freqs_k, interleaved=True)
        else:
            cached_k = kv_cache.cached_k()
        out_rope = self.attn_op(q_rope, cached_k, kv_cache.cached_v())
        out_rope = out_rope.reshape(batch_shape + (L, n * d))
        out_rope = self.o(out_rope)

        # --- PRoPE-branch attention -----------------------------------
        out_prope = self.attn_op_prope(
            q_prope.transpose(1, 2),
            prope_kv_cache.cached_k(),
            prope_kv_cache.cached_v(),
        )
        # ``apply_fn_o`` expects ``[batch, num_heads, seqlen, head_dim]``;
        # we currently have ``[batch, seqlen, num_heads, head_dim]`` from
        # the cudnn-backed attn op output, so transpose for the matmul
        # and back for the final flatten + projection.
        out_prope = apply_fn_o(out_prope.transpose(1, 2)).transpose(1, 2)
        out_prope = out_prope.reshape(batch_shape + (L, n * d))
        out_prope = self.o_prope(out_prope)

        return out_rope + out_prope


## ---------------------------------------------------------------------------
## Block + cache subclasses
## ---------------------------------------------------------------------------


@dataclass
class HyWorldPlayPRoPEBlockCache(BlockCache):
    """:class:`BlockCache` plus a second cache for the PRoPE branch."""

    # ``prope_self_attn`` mirrors the layout of ``self_attn`` but stores
    # the *already-PRoPE-transformed* K / V so each AR step only pays
    # the per-frame projection cost once.
    prope_self_attn: BlockKVCache = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.prope_self_attn is None:
            raise ValueError(
                "HyWorldPlayPRoPEBlockCache requires prope_self_attn; "
                "use HyWorldPlayPRoPEBlock.initialize_cache to build one."
            )

    def before_update(self, chunk_idx: int) -> None:
        super().before_update(chunk_idx)
        self.prope_self_attn.before_update(chunk_idx)

    def after_update(self, chunk_idx: int) -> None:
        super().after_update(chunk_idx)
        self.prope_self_attn.after_update(chunk_idx)


class HyWorldPlayPRoPEBlock(Block):
    """Transformer block whose self-attn runs the dual-branch RoPE+PRoPE path.

    Replaces the stock :class:`SelfAttention` with
    :class:`HyWorldPlayPRoPESelfAttention` and overrides
    :meth:`initialize_cache` / :meth:`forward` so the PRoPE branch's
    independent KV cache is created and threaded alongside the standard
    one. Cross-attention + FFN are inherited unchanged.

    The block accepts ``viewmats`` and ``Ks`` as forward kwargs (passed
    via :attr:`WanDiTNetwork.forward`'s ``block_extra_kwargs``); when
    both are ``None`` the call still runs the PRoPE math against the
    cached zeros and the zero-init ``o_prope`` projects to zero, so the
    block is observationally a no-op vs the standard one until HY-
    WorldPlay weights are loaded.
    """

    self_attn: HyWorldPlayPRoPESelfAttention

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        i2v: bool = False,
        apply_rope_before_kvcache: bool = True,
    ) -> None:
        super().__init__(
            dim=dim,
            ffn_dim=ffn_dim,
            num_heads=num_heads,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
            i2v=i2v,
            apply_rope_before_kvcache=apply_rope_before_kvcache,
        )
        # Replace the stock self-attn with the dual-branch variant; we
        # do this after super().__init__ so the standard module's
        # ``q`` / ``k`` / ``v`` / ``o`` weights are picked up by any
        # checkpoint loader that addresses them by name (e.g.
        # ``blocks.{i}.self_attn.q.weight``). The dual-branch subclass
        # adds ``o_prope`` and ``attn_op_prope`` on top.
        self.self_attn = HyWorldPlayPRoPESelfAttention(
            query_dim=dim,
            n_heads=num_heads,
            head_dim=dim // num_heads,
            eps=eps,
            apply_rope_before_kvcache=apply_rope_before_kvcache,
        )

    def initialize_cache(
        self,
        chunk_size: int,
        window_size: int,
        sink_size: int,
        context_text: Tensor,
        context_img: Tensor | None = None,
    ) -> HyWorldPlayPRoPEBlockCache:
        base = super().initialize_cache(
            chunk_size=chunk_size,
            window_size=window_size,
            sink_size=sink_size,
            context_text=context_text,
            context_img=context_img,
        )
        prope_cache = self.self_attn.initialize_prope_cache(
            batch_size=base.self_attn.k_shape[0],
            chunk_size=chunk_size,
            window_size=window_size,
            sink_size=sink_size,
            device=context_text.device,
            dtype=context_text.dtype,
        )
        return HyWorldPlayPRoPEBlockCache(
            self_attn=base.self_attn,
            cross_attn=base.cross_attn,
            prope_self_attn=prope_cache,
        )

    def forward(
        self,
        x: Tensor,
        e: Tensor,
        cache: BlockCache,
        rope_freqs: Tensor,
        viewmats: Tensor | None = None,
        Ks: Tensor | None = None,
    ) -> Tensor:
        """Dual-branch variant of :meth:`Block.forward`.

        Args:
            x: Input tensor with shape ``[..., L, D]``.
            e: AdaLN modulation tensor (same shape contract as
                :class:`Block`).
            cache: Per-block cache. Must be a
                :class:`HyWorldPlayPRoPEBlockCache` so the PRoPE-branch
                cache is accessible.
            rope_freqs: Standard-mode RoPE frequencies.
            viewmats: Per-frame W2C matrices for the current chunk.
                Required for the PRoPE branch to have any non-trivial
                contribution.
            Ks: Optional per-frame intrinsics ``[batch, cameras, 3, 3]``.
        """
        assert self._parameters_updated_after_loading_checkpoint, (
            "We expect to have called update_parameters_after_loading_checkpoint() "
            "before running the forward pass"
        )
        # User-facing errors first: missing camera data is the most likely
        # misconfiguration here, so report it before the internal cache-type
        # assertion fires (which would otherwise mask the real problem
        # behind a confusing ``cache is object`` message in tests).
        if viewmats is None:
            raise ValueError(
                "HyWorldPlayPRoPEBlock.forward requires viewmats. "
                "Did the encoder bind camera data and the network thread "
                "it via block_extra_kwargs?"
            )
        assert isinstance(cache, HyWorldPlayPRoPEBlockCache), (
            "HyWorldPlayPRoPEBlock.forward requires a "
            f"HyWorldPlayPRoPEBlockCache; got {type(cache).__name__}. "
            "Did HyWorldPlayWanDiTNetwork.initialize_cache run?"
        )

        e_chunks = [c.squeeze(-2) for c in (self.modulation + e).chunk(6, dim=-2)]

        y = self.norm1(x) * (1 + e_chunks[1]) + e_chunks[0]
        y = self.self_attn.forward_dual_branch(
            y,
            kv_cache=cache.self_attn,
            prope_kv_cache=cache.prope_self_attn,
            rope_freqs=rope_freqs,
            viewmats=viewmats,
            Ks=Ks,
        )
        x = x + (y * e_chunks[2])

        x = x + self.cross_attn(
            self.norm3(x),
            kv_cache=cache.cross_attn,
        )
        y = self.norm2(x) * (1 + e_chunks[4]) + e_chunks[3]
        y = self.ffn(y)
        x = x + (y * e_chunks[5])
        return x

