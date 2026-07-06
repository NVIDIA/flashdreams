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

"""HY-WorldPlay camera-projective attention (dual-branch self-attn, KV caches, DiT block)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributed import ProcessGroup

from flashdreams.core.attention import BlockKVCache, ContextParallelAttention
from flashdreams.core.attention.rope import apply_rope_freqs
from flashdreams.recipes.wan.transformer.impl.modules import (
    Block,
    BlockCache,
    SelfAttention,
)
from hy_worldplay._prope import prope_qkv


def _fp32_layer_norm(x: Tensor, norm: nn.LayerNorm) -> Tensor:
    """Run ``norm`` in float32 regardless of input / weight dtype, returning fp32.

    Casts both the input and any affine parameters to fp32 so callers can
    keep the subsequent AdaLN ``scale_shift`` blend in fp32 too.
    """
    weight = norm.weight.float() if norm.weight is not None else None
    bias = norm.bias.float() if norm.bias is not None else None
    return F.layer_norm(
        x.float(),
        norm.normalized_shape,
        weight,
        bias,
        norm.eps,
    )


__all__ = [
    "HyWorldPlayMemoryKVCache",
    "HyWorldPlayPRoPEBlock",
    "HyWorldPlayPRoPEBlockCache",
    "HyWorldPlayPRoPESelfAttention",
]


## Memory KV cache


@dataclass
class HyWorldPlayMemoryKVCache:
    """Per-block flat KV cache for HY-WorldPlay's reconstituted-context memory.

    Stores the selected memory frames' K / V (RoPE-modulated at the frames'
    true temporal positions) as a flat ``len(selected) * tokens_per_frame``
    sequence -- no rolling window. The prefill executor wipes and
    repopulates it at the start of every chunk past the first; within a
    chunk's denoising loop the contents are frozen.

    The standard-RoPE and PRoPE branches are stored independently so the
    dual-branch attention can address each without slicing a packed
    tensor. Layout ``[batch, S, n_heads, head_dim]`` with
    ``S == len(selected_frame_indices) * tokens_per_frame`` matches
    :meth:`BlockKVCache.cached_k`, so both caches concatenate along
    ``seq_dim=-3`` without a reshape.
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


## Dual-branch self-attention


class HyWorldPlayPRoPESelfAttention(SelfAttention):
    """Self-attention with a parallel PRoPE-projected branch.

    Owns a second output projection :attr:`o_prope` (zero-initialised so
    the PRoPE branch is a no-op until trained weights load) on top of the
    inherited Q / K / V / o projections. The PRoPE branch reuses the same
    Q / K / V tensors and only differs in how they are pre- and
    post-processed for attention.

    A second :class:`BlockKVCache` stores the *already-PRoPE-transformed*
    K / V from previous AR steps, so each step only transforms its current
    chunk's K / V. The query side is transformed fresh every call since it
    uses the current chunk's extrinsics + intrinsics.
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
        # Zero-init so the PRoPE branch adds zero residual until trained
        # weights load.
        self.o_prope = nn.Linear(self.inner_dim, self.query_dim)
        nn.init.zeros_(self.o_prope.weight)
        if self.o_prope.bias is not None:
            nn.init.zeros_(self.o_prope.bias)

        # Independent attention op so CP routing stays symmetric across
        # branches without sharing internal state.
        self.attn_op_prope = ContextParallelAttention(
            qkv_format="bshd", backend="cudnn"
        )

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
            prope_kv_cache: Second KV cache that stores the
                PRoPE-transformed K / V.
            rope_freqs: Standard-mode RoPE frequencies with shape
                ``[L, 1, 1, head_dim]``.
            viewmats: Per-frame W2C matrices for the *current* chunk,
                shape ``[batch, cameras, 4, 4]`` where ``cameras`` is the
                per-AR-step latent-frame count (``len_t``).
            Ks: Optional per-frame intrinsics ``[batch, cameras, 3, 3]``.
            memory_kv_cache: Optional reconstituted-context memory cache.
                When populated, the prefilled K / V are prepended along
                ``seq_dim=-3`` so the sequence becomes
                ``[memory K/V, current K/V]``. ``None`` or empty leaves
                the dual-branch path unchanged.

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

        # RoPE-branch K cache write.
        k_for_rope_cache = k_raw
        if rope_freqs_k is not None and self.apply_rope_before_kvcache:
            k_for_rope_cache = apply_rope_freqs(
                k_for_rope_cache, rope_freqs_k, interleaved=True
            )
        kv_cache.update(k_for_rope_cache, v_raw)

        # PRoPE math runs in bhsd; transpose in, then back to the bshd
        # cache layout before the write.
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

        # Standard RoPE-branch attention.
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
        # Prepend prefilled memory K / V (if any) -> ``[memory, current]``.
        if _debug_dump.enabled():
            _debug_dump.dump("attn.q_rope_post", q_rope)
            _debug_dump.dump("attn.cached_k_pre_mem_concat", cached_k)
            _debug_dump.dump("attn.cached_v_pre_mem_concat", cached_v)
        if memory_kv_cache is not None and memory_kv_cache.has_rope_kv:
            # Asserts narrow ``Tensor | None`` for ``torch.cat``.
            assert memory_kv_cache.k_rope is not None
            assert memory_kv_cache.v_rope is not None
            if _debug_dump.enabled():
                _debug_dump.dump("attn.memory_k_rope_prepend", memory_kv_cache.k_rope)
                _debug_dump.dump("attn.memory_v_rope_prepend", memory_kv_cache.v_rope)
            cached_k = torch.cat([memory_kv_cache.k_rope, cached_k], dim=-3)
            cached_v = torch.cat([memory_kv_cache.v_rope, cached_v], dim=-3)
        if _debug_dump.enabled():
            _debug_dump.dump("attn.cached_k_final", cached_k)
            _debug_dump.dump("attn.cached_v_final", cached_v)
        out_rope = self.attn_op(q_rope, cached_k, cached_v)
        out_rope = out_rope.reshape(batch_shape + (L, n * d))
        out_rope = self.o(out_rope)

        # PRoPE-branch attention; same memory prepend on the camera side.
        prope_cached_k = prope_kv_cache.cached_k()
        prope_cached_v = prope_kv_cache.cached_v()
        if memory_kv_cache is not None and memory_kv_cache.has_prope_kv:
            assert memory_kv_cache.k_prope is not None
            assert memory_kv_cache.v_prope is not None
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
        # ``apply_fn_o`` consumes bhsd; the cudnn attn op returns bshd.
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
        """Run the dual-branch self-attention over the selected memory frames.

        Projects, applies RoPE / PRoPE, writes both branches' K / V into
        ``memory_kv_cache``, then attends over the memory positions
        themselves. The K / V written here are what chunk-1+
        ``forward_dual_branch`` calls prepend.

        Args:
            x: Pre-norm-modulated input for the selected memory frames,
                shape ``[..., L_mem, query_dim]`` where
                ``L_mem == K * tokens_per_frame``.
            rope_freqs: RoPE frequencies at the memory frames' true temporal
                positions, shape ``[L_mem, 1, 1, head_dim]``. The executor
                builds this from the per-rollout RoPE adapter at the
                selected frames' history indices.
            viewmats: Per-memory-frame W2C matrices, shape
                ``[batch, K, 4, 4]``. Already sliced to
                ``selected_frame_indices`` by the executor.
            Ks: Optional per-memory-frame intrinsics
                ``[batch, K, 3, 3]``.
            memory_kv_cache: Cache to populate. Both branches are
                written.

        Returns:
            Attention output at the memory positions, summed over the
            standard-RoPE and PRoPE branches, shape
            ``[..., L_mem, query_dim]``.
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

        # RoPE-branch K write (V is always raw).
        k_for_rope = k_raw
        if rope_freqs is not None and self.apply_rope_before_kvcache:
            k_for_rope = apply_rope_freqs(k_for_rope, rope_freqs, interleaved=True)
        memory_kv_cache.write_rope(k_for_rope, v_raw)

        # PRoPE math runs in bhsd; store post-transform K / V back in the
        # bshd cache layout.
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

        # Standard RoPE-branch attention over the memory tokens themselves
        # -- they are the only sequence in this prefill, so K / V are the
        # just-computed tensors (no cross-chunk concatenation).
        q_rope = q_raw
        if rope_freqs is not None:
            q_rope = apply_rope_freqs(q_rope, rope_freqs, interleaved=True)
        out_rope = self.attn_op(q_rope, k_for_rope, v_raw)
        out_rope = out_rope.reshape(batch_shape + (L, n * d))
        out_rope = self.o(out_rope)

        # ``prope_qkv`` returns bhsd; attn_op_prope consumes bshd and
        # ``apply_fn_o`` consumes bhsd, hence the transposes.
        out_prope = self.attn_op_prope(
            q_prope.transpose(1, 2),
            k_prope_bhsd.transpose(1, 2),
            v_prope_bhsd.transpose(1, 2),
        )
        out_prope = apply_fn_o(out_prope.transpose(1, 2)).transpose(1, 2)
        out_prope = out_prope.reshape(batch_shape + (L, n * d))
        out_prope = self.o_prope(out_prope)

        return out_rope + out_prope


## Block + cache subclasses


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
      chunk so each AR step pays the per-frame projection cost once.
    * ``memory`` -- separate, flat per-block cache that stores the
      prefilled K / V from the selected memory frames, RoPE-modulated at
      the frames' true temporal positions. Wiped at chunk start by the
      prefill executor and repopulated from
      :class:`HyWorldPlayCtrl.memory_frame_indices`. The dual-branch
      attention prepends these K / V to ``self_attn`` /
      ``prope_self_attn`` for the actual attention call, so the total
      context is ``[memory K/V, current chunk K/V]`` along ``seq_dim=-3``.
    """

    prope_self_attn: BlockKVCache = None  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
    """PRoPE-branch KV cache (current-chunk K / V, dual of ``self_attn``)."""

    memory: HyWorldPlayMemoryKVCache = field(default_factory=HyWorldPlayMemoryKVCache)
    """Reconstituted-context memory cache. Empty on chunk 0; repopulated
    by the prefill executor at the start of every later chunk."""

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

        The memory cache is not touched here; it has its own
        :meth:`HyWorldPlayMemoryKVCache.reset` with an independent
        lifecycle driven by the prefill executor.
        """
        self.self_attn.reset()
        self.prope_self_attn.reset()


class HyWorldPlayPRoPEBlock(Block):
    """Transformer block whose self-attn runs the dual-branch RoPE + PRoPE path.

    Replaces the stock :class:`SelfAttention` with
    :class:`HyWorldPlayPRoPESelfAttention` and overrides
    :meth:`initialize_cache` / :meth:`forward` to create and thread the
    PRoPE branch's independent KV cache. Cross-attention + FFN are
    inherited unchanged.

    Accepts ``viewmats`` and ``Ks`` as forward kwargs. With ``o_prope``
    zero-initialised the block is a no-op versus the standard one until
    HY-WorldPlay weights are loaded.
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
        # Replace after super().__init__ so checkpoint weights addressed
        # by name (e.g. ``blocks.{i}.self_attn.q.weight``) still resolve
        # the inherited Q / K / V / o projections.
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
        # Check camera data before the cache-type assert so a missing
        # ``viewmats`` reports the real misconfiguration.
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

        # AdaLN runs in fp32: modulation table, norm + scale + shift,
        # residual gates, and FFN residual, cast back at each boundary.
        e_chunks = [
            c.squeeze(-2) for c in (self.modulation + e).float().chunk(6, dim=-2)
        ]

        y = (_fp32_layer_norm(x, self.norm1) * (1 + e_chunks[1]) + e_chunks[0]).type_as(
            x
        )
        y = self.self_attn.forward_dual_branch(
            y,
            kv_cache=cache.self_attn,
            prope_kv_cache=cache.prope_self_attn,
            rope_freqs=rope_freqs,
            viewmats=viewmats,
            Ks=Ks,
            memory_kv_cache=cache.memory,
        )
        x = (x.float() + y.float() * e_chunks[2]).type_as(x)

        # Cross-attn residual stays in bf16; only the norm before attn2
        # runs in fp32. The PRoPE block is always built with
        # ``cross_attn_norm=True``, so ``norm3`` is a ``LayerNorm`` here.
        assert isinstance(self.norm3, nn.LayerNorm)
        x = x + self.cross_attn(
            _fp32_layer_norm(x, self.norm3).type_as(x),
            kv_cache=cache.cross_attn,
        )
        y = (_fp32_layer_norm(x, self.norm2) * (1 + e_chunks[4]) + e_chunks[3]).type_as(
            x
        )
        y = self.ffn(y)
        x = (x.float() + y.float() * e_chunks[5]).type_as(x)
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
        """Run the full block forward over the selected memory frames.

        Mirrors :meth:`forward` so each successive block's K / V
        projections see an already-attended hidden state. The dual-branch
        self-attn writes both branches' K / V into ``cache.memory`` as a
        side effect; ``cache.cross_attn`` is read for the cross-attention
        text (and I2V image) K / V; ``cache.self_attn`` /
        ``cache.prope_self_attn`` are intentionally untouched -- the
        prefilled memory K / V don't belong in the rolling current-chunk
        cache.

        Args:
            x: Pre-AdaLN input for the K selected memory frames,
                shape ``[..., L_mem, D]``.
            e: AdaLN modulation tensor for those frames (same contract
                as :meth:`forward`).
            rope_freqs: RoPE frequencies at the memory frames' true temporal
                positions.
            viewmats: Per-memory-frame W2C extrinsics (already sliced
                to the selected indices).
            Ks: Optional per-memory-frame intrinsics.
            cache: The block's per-rollout cache.

        Returns:
            Hidden state ``[..., L_mem, D]`` evolved through the full
            block, for the next block's prefill call.
        """
        if viewmats is None:
            raise ValueError(
                "HyWorldPlayPRoPEBlock.prefill_memory_kv requires viewmats; "
                "the prefill executor must slice the per-rollout viewmats "
                "by selected_frame_indices before calling."
            )
        # FP32 AdaLN, matching :meth:`forward` so the memory K / V cache
        # carries no per-block bf16 rounding drift.
        e_chunks = [
            c.squeeze(-2) for c in (self.modulation + e).float().chunk(6, dim=-2)
        ]

        y = (_fp32_layer_norm(x, self.norm1) * (1 + e_chunks[1]) + e_chunks[0]).type_as(
            x
        )
        y = self.self_attn.prefill_memory_kv(
            y,
            rope_freqs=rope_freqs,
            viewmats=viewmats,
            Ks=Ks,
            memory_kv_cache=cache.memory,
        )
        x = (x.float() + y.float() * e_chunks[2]).type_as(x)

        # ``cross_attn_norm=True`` for the PRoPE block, so ``norm3`` is a
        # ``LayerNorm`` here.
        assert isinstance(self.norm3, nn.LayerNorm)
        x = x + self.cross_attn(
            _fp32_layer_norm(x, self.norm3).type_as(x),
            kv_cache=cache.cross_attn,
        )
        y = (_fp32_layer_norm(x, self.norm2) * (1 + e_chunks[4]) + e_chunks[3]).type_as(
            x
        )
        y = self.ffn(y)
        x = (x.float() + y.float() * e_chunks[5]).type_as(x)
        return x
