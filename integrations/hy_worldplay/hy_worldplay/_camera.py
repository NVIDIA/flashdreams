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
from dataclasses import dataclass, field
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
    "HyWorldPlayMemoryKVCache",
    "HyWorldPlayPRoPEBlock",
    "HyWorldPlayPRoPEBlockCache",
    "HyWorldPlayPRoPESelfAttention",
]


## ---------------------------------------------------------------------------
## Memory KV cache (phase 2b.5b-part2)
## ---------------------------------------------------------------------------


@dataclass
class HyWorldPlayMemoryKVCache:
    """Per-block flat KV cache for HY-WorldPlay's reconstituted-context memory.

    Distinct from :class:`flashdreams.core.attention.kvcache.BlockKVCache`:
    this cache stores K / V at upstream's RoPE-collapsed positions
    ``[0, len(selected) * tokens_per_frame)``, not at original absolute
    frame positions. It has no rolling window and no chunk indexing --
    the prefill executor wipes and repopulates it at the start of every
    chunk past the first; within a chunk's denoising loop the contents
    are frozen.

    Mirrors upstream's ``self._kv_cache[idx]`` in
    ``wan/inference/pipeline_wan_w_mem_relative_rope.py``: a per-block
    ``{"k": Tensor | None, "v": Tensor | None}`` payload, but split
    into two branches (RoPE + PRoPE) so the dual-branch attention can
    address each independently. Upstream packs both branches into one
    tensor via ``torch.cat([key_rope, key_prope], dim=-1)`` and
    chunks at read time; we keep them separate for clarity (the
    underlying memory is identical).

    Tensor layout: ``[batch, S, n_heads, head_dim]`` where ``S`` is the
    number of prefilled tokens (``len(selected_frame_indices) *
    tokens_per_frame``). Matches the layout of
    :meth:`BlockKVCache.cached_k` so both caches can be concatenated
    along ``seq_dim=-3`` without a reshape.
    """

    k_rope: Tensor | None = None
    """Standard-RoPE-branch keys for the prefilled tokens."""

    v_rope: Tensor | None = None
    """Standard-RoPE-branch values for the prefilled tokens."""

    k_prope: Tensor | None = None
    """PRoPE-branch keys (camera-projected) for the prefilled tokens."""

    v_prope: Tensor | None = None
    """PRoPE-branch values (camera-projected) for the prefilled tokens."""

    def reset(self) -> None:
        """Clear the cache. Called at the start of each new chunk's prefill."""
        self.k_rope = None
        self.v_rope = None
        self.k_prope = None
        self.v_prope = None

    def write_rope(self, k: Tensor, v: Tensor) -> None:
        """Store the standard-branch K / V from a prefill pass."""
        self.k_rope = k
        self.v_rope = v

    def write_prope(self, k: Tensor, v: Tensor) -> None:
        """Store the PRoPE-branch K / V from a prefill pass."""
        self.k_prope = k
        self.v_prope = v

    @property
    def has_rope_kv(self) -> bool:
        """``True`` once the standard branch has been prefilled this chunk."""
        return self.k_rope is not None and self.v_rope is not None

    @property
    def has_prope_kv(self) -> bool:
        """``True`` once the PRoPE branch has been prefilled this chunk."""
        return self.k_prope is not None and self.v_prope is not None

    @property
    def is_empty(self) -> bool:
        """``True`` when neither branch is populated (chunk 0 baseline)."""
        return not self.has_rope_kv and not self.has_prope_kv


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
        memory_kv_cache: "HyWorldPlayMemoryKVCache | None" = None,
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
            memory_kv_cache: Optional reconstituted-context memory cache
                (phase 2b.5b-part2). When non-empty, the prefilled K / V
                are prepended to ``kv_cache`` / ``prope_kv_cache`` along
                ``seq_dim=-3`` before the attention call, mirroring
                upstream's ``cat([cache, current], dim=-2)`` in
                ``arwan_w_action_w_mem_relative_rope.py`` line 169-173.
                ``None`` (or empty) keeps the dual-branch path bit-
                identical to the 2b.4 baseline -- the cache stays a
                strict no-op until the prefill executor populates it
                at the start of chunk > 0.

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

        from hy_worldplay import _debug_dump

        if _debug_dump.enabled():
            _debug_dump.dump("attn.x_in", x)
            _debug_dump.dump("attn.q_raw", q_raw)
            _debug_dump.dump("attn.k_raw", k_raw)
            _debug_dump.dump("attn.v_raw", v_raw)
            if rope_freqs is not None:
                _debug_dump.dump("attn.rope_freqs_full", rope_freqs)
            if rope_freqs_q is not None:
                _debug_dump.dump("attn.rope_freqs_q", rope_freqs_q)
            if rope_freqs_k is not None:
                _debug_dump.dump("attn.rope_freqs_k", rope_freqs_k)

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
        cached_v = kv_cache.cached_v()
        # Phase 2b.5b-part2: prepend the prefilled memory K / V (if any)
        # so the attention sees ``[memory_K, current_K]`` along the
        # sequence dim. Mirrors upstream's prefill prepend at
        # ``arwan_w_action_w_mem_relative_rope.py`` line 169-170.
        if _debug_dump.enabled():
            _debug_dump.dump("attn.q_rope_post", q_rope)
            _debug_dump.dump("attn.cached_k_pre_mem_concat", cached_k)
            _debug_dump.dump("attn.cached_v_pre_mem_concat", cached_v)
        if memory_kv_cache is not None and memory_kv_cache.has_rope_kv:
            if _debug_dump.enabled():
                _debug_dump.dump("attn.memory_k_rope_prepend", memory_kv_cache.k_rope)
                _debug_dump.dump("attn.memory_v_rope_prepend", memory_kv_cache.v_rope)
            cached_k = torch.cat(
                [memory_kv_cache.k_rope, cached_k], dim=-3
            )
            cached_v = torch.cat(
                [memory_kv_cache.v_rope, cached_v], dim=-3
            )
        if _debug_dump.enabled():
            _debug_dump.dump("attn.cached_k_final", cached_k)
            _debug_dump.dump("attn.cached_v_final", cached_v)
        out_rope = self.attn_op(q_rope, cached_k, cached_v)
        out_rope = out_rope.reshape(batch_shape + (L, n * d))
        out_rope = self.o(out_rope)

        # --- PRoPE-branch attention -----------------------------------
        prope_cached_k = prope_kv_cache.cached_k()
        prope_cached_v = prope_kv_cache.cached_v()
        # Phase 2b.5b-part2: prepend the PRoPE-transformed memory K / V
        # (if any). Mirrors the same prepend on the camera branch as
        # upstream lines 172-173.
        if memory_kv_cache is not None and memory_kv_cache.has_prope_kv:
            prope_cached_k = torch.cat(
                [memory_kv_cache.k_prope, prope_cached_k], dim=-3
            )
            prope_cached_v = torch.cat(
                [memory_kv_cache.v_prope, prope_cached_v], dim=-3
            )
        out_prope = self.attn_op_prope(
            q_prope.transpose(1, 2),
            prope_cached_k,
            prope_cached_v,
        )
        # ``apply_fn_o`` expects ``[batch, num_heads, seqlen, head_dim]``;
        # we currently have ``[batch, seqlen, num_heads, head_dim]`` from
        # the cudnn-backed attn op output, so transpose for the matmul
        # and back for the final flatten + projection.
        out_prope = apply_fn_o(out_prope.transpose(1, 2)).transpose(1, 2)
        out_prope = out_prope.reshape(batch_shape + (L, n * d))
        out_prope = self.o_prope(out_prope)

        return out_rope + out_prope

    def prefill_memory_kv(
        self,
        x: Tensor,
        rope_freqs: Tensor,
        viewmats: Tensor,
        Ks: Tensor | None,
        memory_kv_cache: HyWorldPlayMemoryKVCache,
    ) -> Tensor:
        """Run the dual-branch self-attention at collapsed memory positions.

        Phase 2b.6.2 bug fix -- prior implementation only wrote K / V
        into ``memory_kv_cache`` and returned ``None``, leaving the
        per-block hidden state stuck at the patch-embedded /
        AdaLN-modulated input for every block in the prefill loop.
        Vendor's ``is_cache=True`` path in
        ``arwan_w_action_w_mem_relative_rope.py`` runs the *full*
        block (self-attn -> cross-attn -> FFN) for the cache-prefill
        pass, so each successive block's K / V projections see an
        already-attended hidden state. The previous native shortcut
        produced block-1+ memory K / V that diverged by 100-155%
        relative to vendor (rms diff ~1.0 on a tensor with rms ~0.9
        at e.g. block 1's value branch), which dominated the residual
        chunk-1 parity gap once the CFG and RNG mismatches landed.

        The fix runs the full attention pipeline:

        1. Project Q / K / V from the modulated ``x``.
        2. Apply RoPE / PRoPE at the collapsed positions.
        3. **Write** K / V (both branches) into ``memory_kv_cache``
           for chunk-1+'s ``forward_dual_branch`` to prepend.
        4. **Compute** attention output over the memory positions
           themselves (no cross-chunk K / V prepending -- this *is*
           the cross-chunk content) and return it so the caller can
           apply the post-attention residual + cross-attn + FFN.

        The returned tensor has the same shape as ``x`` and is
        bit-equivalent (modulo the PRoPE branch's zero-residual at
        random init) to what a single chunk-0 forward would have
        produced at the selected memory positions had it been driven
        at the collapsed RoPE positions instead of the original
        per-frame positions.

        Args:
            x: Pre-norm-modulated input for the selected memory frames,
                shape ``[..., L_mem, query_dim]`` where
                ``L_mem == K * tokens_per_frame``.
            rope_freqs: RoPE frequencies remapped to the collapsed
                positions, shape ``[L_mem, 1, 1, head_dim]``. The
                executor builds this from the per-rollout RoPE adapter
                using ``current_start=0`` /
                ``current_end=K * tokens_per_frame`` (mirrors upstream's
                ``rotary_emb[:, :, current_start:current_end, :]`` slice
                in ``arwan_w_action_w_mem_relative_rope.py`` line 914).
            viewmats: Per-memory-frame W2C matrices, shape
                ``[batch, K, 4, 4]``. Already sliced to
                ``selected_frame_indices`` by the executor.
            Ks: Optional per-memory-frame intrinsics
                ``[batch, K, 3, 3]``.
            memory_kv_cache: Cache to populate. Both branches are
                written.

        Returns:
            The attention output at the memory positions, summed over
            the standard-RoPE and PRoPE branches, shape
            ``[..., L_mem, query_dim]``. The block caller chains this
            into the post-attention residual + cross-attn + FFN to
            evolve the hidden state for the next block's prefill,
            mirroring vendor.
        """
        if self.is_context_parallel_enabled():
            raise NotImplementedError(
                "HyWorldPlayPRoPESelfAttention.prefill_memory_kv does "
                "not yet support context-parallel (cp_size > 1); CP "
                "wiring lands together with multi-rank action expansion "
                "in a follow-up."
            )

        batch_shape = x.shape[:-2]
        batch_size = math.prod(batch_shape)
        L, _ = x.shape[-2:]
        n, d = self.n_heads, self.head_dim

        q_raw = self.norm_q(self.q(x)).reshape(batch_size, L, n, d)
        k_raw = self.norm_k(self.k(x)).reshape(batch_size, L, n, d)
        v_raw = self.v(x).reshape(batch_size, L, n, d)

        from hy_worldplay import _debug_dump

        if _debug_dump.enabled():
            _debug_dump.dump("prefill.block.x_in", x)
            _debug_dump.dump("prefill.block.q_raw", q_raw)
            _debug_dump.dump("prefill.block.k_raw", k_raw)
            _debug_dump.dump("prefill.block.v_raw", v_raw)
            if rope_freqs is not None:
                _debug_dump.dump("prefill.block.rope_freqs", rope_freqs)

        # --- RoPE branch K (V is always raw) --------------------------
        k_for_rope = k_raw
        if rope_freqs is not None and self.apply_rope_before_kvcache:
            k_for_rope = apply_rope_freqs(
                k_for_rope, rope_freqs, interleaved=True
            )
        memory_kv_cache.write_rope(k_for_rope, v_raw)

        # --- PRoPE branch K / V ---------------------------------------
        # Same PRoPE pipeline as ``forward_dual_branch``: transpose to
        # ``[batch, num_heads, seqlen, head_dim]`` for the math, store
        # the post-transform K / V back in the cache layout
        # (``[batch, seqlen, num_heads, head_dim]``).
        q_prope, k_prope_bhsd, v_prope_bhsd, apply_fn_o = prope_qkv(
            q_raw.transpose(1, 2),
            k_raw.transpose(1, 2),
            v_raw.transpose(1, 2),
            viewmats=viewmats,
            Ks=Ks,
        )
        memory_kv_cache.write_prope(
            k_prope_bhsd.transpose(1, 2),
            v_prope_bhsd.transpose(1, 2),
        )

        if _debug_dump.enabled():
            _debug_dump.dump("prefill.block.k_rope_written", memory_kv_cache.k_rope)
            _debug_dump.dump("prefill.block.v_rope_written", memory_kv_cache.v_rope)
            _debug_dump.dump("prefill.block.k_prope_written", memory_kv_cache.k_prope)
            _debug_dump.dump("prefill.block.v_prope_written", memory_kv_cache.v_prope)

        # --- Standard RoPE-branch attention ---------------------------
        # Phase 2b.6.2 -- run attention *over the memory positions
        # themselves* (no cross-chunk K / V to prepend; the memory
        # tokens *are* the only sequence at the collapsed positions).
        # Mirrors vendor's ``is_cache=True`` attention call:
        # ``attn_op(q_rope, key_rope, value_rope)`` where key/value
        # are the just-computed memory K / V, not a concatenation.
        q_rope = q_raw
        if rope_freqs is not None:
            q_rope = apply_rope_freqs(q_rope, rope_freqs, interleaved=True)
        out_rope = self.attn_op(q_rope, k_for_rope, v_raw)
        out_rope = out_rope.reshape(batch_shape + (L, n * d))
        out_rope = self.o(out_rope)

        # --- PRoPE-branch attention -----------------------------------
        # ``prope_qkv`` returns Q / K / V transposed to
        # ``[batch, num_heads, seqlen, head_dim]``; attn_op_prope
        # consumes ``[batch, seqlen, num_heads, head_dim]`` (bshd
        # layout) so we transpose the K / V cache layout back. The
        # ``apply_fn_o`` post-attention map needs ``[batch, num_heads,
        # seqlen, head_dim]`` then we transpose back for the final
        # ``[..., L, n*d]`` flatten + projection.
        out_prope = self.attn_op_prope(
            q_prope.transpose(1, 2),
            k_prope_bhsd.transpose(1, 2),
            v_prope_bhsd.transpose(1, 2),
        )
        out_prope = apply_fn_o(out_prope.transpose(1, 2)).transpose(1, 2)
        out_prope = out_prope.reshape(batch_shape + (L, n * d))
        out_prope = self.o_prope(out_prope)

        return out_rope + out_prope


## ---------------------------------------------------------------------------
## Block + cache subclasses
## ---------------------------------------------------------------------------


@dataclass
class HyWorldPlayPRoPEBlockCache(BlockCache):
    """:class:`BlockCache` plus a PRoPE branch and a memory-prefill slot.

    Three caches per block:

    * ``self_attn`` -- inherited from :class:`BlockCache`, stores the
      standard RoPE-branch K / V for the *current chunk's* tokens.
      Reused across denoising steps within a chunk; reset at chunk
      start by the HY transformer's predict_flow.
    * ``prope_self_attn`` -- mirrors the layout of ``self_attn`` but
      stores the *already-PRoPE-transformed* K / V for the current
      chunk so each AR step only pays the per-frame projection cost
      once.
    * ``memory`` -- separate, flat per-block cache that stores the
      prefilled K / V from the selected memory frames at upstream's
      RoPE-collapsed positions ``[0, K)``. Wiped at chunk start by
      the prefill executor and repopulated from
      :class:`HyWorldPlayCtrl.memory_frame_indices`. The dual-branch
      attention prepends these K / V to ``self_attn`` /
      ``prope_self_attn`` for the actual attention call, so the
      total context is ``[memory K/V, current chunk K/V]`` along
      ``seq_dim=-3``.
    """

    prope_self_attn: BlockKVCache = None  # type: ignore[assignment]
    """PRoPE-branch KV cache (current-chunk K / V, dual of ``self_attn``)."""

    memory: HyWorldPlayMemoryKVCache = field(
        default_factory=HyWorldPlayMemoryKVCache
    )
    """Reconstituted-context memory cache (phase 2b.5b-part2). Holds K / V
    for the selected memory frames at RoPE-collapsed positions; empty
    on chunk 0, repopulated at the start of every chunk past the first
    by the prefill executor."""

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

    def reset_current_chunk(self) -> None:
        """Reset both per-chunk K / V caches to empty (filling) state.

        The memory cache is *not* touched here -- it has its own
        :meth:`HyWorldPlayMemoryKVCache.reset` that the prefill
        executor calls before repopulating. Splitting the resets
        keeps the two lifecycles independent: ``self_attn`` /
        ``prope_self_attn`` are reset every chunk so each chunk's
        denoising starts with an empty rolling window; the memory
        cache is only reset at chunks where the prefill actually
        runs.
        """
        self.self_attn.reset()
        self.prope_self_attn.reset()


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
            memory_kv_cache=cache.memory,
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

    def prefill_memory_kv(
        self,
        x: Tensor,
        e: Tensor,
        rope_freqs: Tensor,
        viewmats: Tensor,
        Ks: Tensor | None,
        cache: "HyWorldPlayPRoPEBlockCache",
    ) -> Tensor:
        """Run the full block forward at the collapsed memory positions.

        Phase 2b.6.2 bug fix -- previously this method only wrote K / V
        into ``cache.memory`` and returned ``None``, leaving the
        prefill executor to feed the same patch-embedded /
        AdaLN-modulated input into every block. Vendor's
        ``is_cache=True`` path in
        ``arwan_w_action_w_mem_relative_rope.py`` instead runs the
        complete block (self-attn -> cross-attn -> FFN with all three
        residual gates) so each successive block's K / V projections
        see an already-attended hidden state. The previous shortcut
        produced block-1+ memory K / V that diverged by 100-155%
        relative to vendor (rms diff ~1.0 on a tensor with rms ~0.9
        at e.g. block 1's value branch), which dominated the residual
        chunk-1 parity gap once the CFG (5.0 -> 1.0) and RNG
        (``HY_VENDOR_NOISE_MODE``) mismatches were patched.

        The fixed path now mirrors :meth:`forward` exactly:

        1. ``self.modulation + e`` -> 6 AdaLN coefficients.
        2. ``norm1`` + ``shift`` + ``(1 + scale)``-modulated self-attn.
        3. **Side-effect**: the dual-branch self-attention writes
           K / V into ``cache.memory`` (both branches) at the
           collapsed RoPE positions.
        4. ``gate`` * attention output + residual.
        5. ``cross_attn`` using ``cache.cross_attn``'s pre-cached text
           K / V (and image K / V on I2V). The text cache was
           populated by ``HyWorldPlayPRoPEBlock.initialize_cache``
           at runner-init, so the prefill call needs no extra plumbing.
        6. ``norm2`` + AdaLN + FFN + ``c_gate`` residual.
        7. Return the evolved hidden state for the next block's
           prefill input.

        Args:
            x: Pre-AdaLN input for the K selected memory frames,
                shape ``[..., L_mem, D]``.
            e: AdaLN modulation tensor for those frames (same contract
                as :meth:`forward`).
            rope_freqs: RoPE frequencies pre-sliced to the collapsed
                memory positions.
            viewmats: Per-memory-frame W2C extrinsics (already sliced
                to the selected indices).
            Ks: Optional per-memory-frame intrinsics.
            cache: The block's per-rollout cache. ``cache.memory`` is
                the only slot *written* by the self-attn side effect;
                ``cache.cross_attn`` is *read* for the cross-attention
                K / V (no write -- the text cache is static for the
                full rollout). ``cache.self_attn`` /
                ``cache.prope_self_attn`` are intentionally untouched
                (the prefill operates at collapsed positions which
                don't belong in the rolling current-chunk cache).

        Returns:
            Hidden state ``[..., L_mem, D]`` evolved through the full
            block. Caller threads this into the next block's prefill
            call (and the network's prefill driver discards the
            final-block output -- nothing past the last block reads
            it on the prefill code path).
        """
        if viewmats is None:
            raise ValueError(
                "HyWorldPlayPRoPEBlock.prefill_memory_kv requires viewmats; "
                "the prefill executor must slice the per-rollout viewmats "
                "by selected_frame_indices before calling."
            )
        e_chunks = [c.squeeze(-2) for c in (self.modulation + e).chunk(6, dim=-2)]

        y = self.norm1(x) * (1 + e_chunks[1]) + e_chunks[0]
        y = self.self_attn.prefill_memory_kv(
            y,
            rope_freqs=rope_freqs,
            viewmats=viewmats,
            Ks=Ks,
            memory_kv_cache=cache.memory,
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

