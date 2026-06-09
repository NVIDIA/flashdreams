"""VA-specific transformer building blocks.

``VASelfAttention`` adds a read-only attention path that attends to
cached + freshly-computed KV without mutating the cache — replacing
the upstream's write-then-rollback pattern with a compile-friendly op.

``VABlock`` wires ``VASelfAttention`` into the standard block and
exposes a ``persist`` flag to control cache writes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from flashdreams.core.attention import NativeAttention
from flashdreams.core.attention.rope import apply_rope_freqs
from flashdreams.recipes.wan.transformer.impl.modules import (
    Block,
    BlockCache,
    CrossAttnCache,
    CrossAttention,
    MultiHeadAttention,
)

from lingbot_va.transformer.impl.kvcache import DualChunkKVCache


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

@dataclass
class VABlockCache:
    """Per-block cache for a VA transformer block.

    Uses ``DualChunkKVCache`` for self-attention (shared between
    video and action) and ``CrossAttnCache`` for cross-attention
    (static text context).
    """

    self_attn: DualChunkKVCache
    cross_attn: CrossAttnCache


# ---------------------------------------------------------------------------
# VASelfAttention
# ---------------------------------------------------------------------------

class VASelfAttention(MultiHeadAttention):
    """Self-attention with read-only and write-commit modes.

    Extends ``MultiHeadAttention`` with ``forward_readonly`` for
    intermediate denoising steps and ``forward_persist`` for the
    final step that commits KV to the cache.
    """

    def forward_readonly(
        self,
        x: Tensor,
        kv_cache: DualChunkKVCache,
        rope_freqs: Tensor,
    ) -> Tensor:
        """Attend to cached + fresh KV without modifying the cache.

        Computes K, V from ``x``, concatenates with all committed
        cache tokens, runs attention, and returns the projected output.
        The cache is NOT mutated.

        Args:
            x: Input hidden states ``[batch, L, dim]``.
            kv_cache: The ``DualChunkKVCache`` to read from.
            rope_freqs: RoPE frequencies ``[L, 1, 1, head_dim]``.

        Returns:
            Attention output ``[batch, L, dim]``.
        """
        batch_size = x.shape[0]
        L = x.shape[1]
        n, d = self.n_heads, self.head_dim

        k_fresh = self.norm_k(self.k(x)).reshape(batch_size, L, n, d)
        v_fresh = self.v(x).reshape(batch_size, L, n, d)
        if self.apply_rope_before_kvcache:
            k_fresh = apply_rope_freqs(k_fresh, rope_freqs, interleaved=True)

        q = self.norm_q(self.q(x)).reshape(batch_size, L, n, d)
        q = apply_rope_freqs(q, rope_freqs, interleaved=True)

        full_k, full_v = kv_cache.cached_kv_plus_fresh(k_fresh, v_fresh)
        out = self.attn_op(q, full_k, full_v)
        out = out.reshape(batch_size, L, n * d)
        return self.o(out)

    def forward_persist(
        self,
        x: Tensor,
        kv_cache: DualChunkKVCache,
        rope_freqs: Tensor,
        write_mode: str,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Attend AND return projected K, V for cache write.

        Same attention as ``forward_readonly`` but also returns the
        projected K, V so the caller can write them to the cache.

        Args:
            x: Input hidden states ``[batch, L, dim]``.
            kv_cache: The ``DualChunkKVCache`` to read from.
            rope_freqs: RoPE frequencies ``[L, 1, 1, head_dim]``.
            write_mode: ``"primary"`` or ``"secondary"`` (unused here,
                passed through for the caller's convenience).

        Returns:
            ``(attn_output, k_to_write, v_to_write)``
        """
        batch_size = x.shape[0]
        L = x.shape[1]
        n, d = self.n_heads, self.head_dim

        k_fresh = self.norm_k(self.k(x)).reshape(batch_size, L, n, d)
        v_fresh = self.v(x).reshape(batch_size, L, n, d)
        if self.apply_rope_before_kvcache:
            k_fresh = apply_rope_freqs(k_fresh, rope_freqs, interleaved=True)

        q = self.norm_q(self.q(x)).reshape(batch_size, L, n, d)
        q = apply_rope_freqs(q, rope_freqs, interleaved=True)

        full_k, full_v = kv_cache.cached_kv_plus_fresh(k_fresh, v_fresh)
        out = self.attn_op(q, full_k, full_v)
        out = out.reshape(batch_size, L, n * d)
        return self.o(out), k_fresh, v_fresh


# ---------------------------------------------------------------------------
# VABlock
# ---------------------------------------------------------------------------

class VABlock(nn.Module):
    """Transformer block for video-action models.

    Mirrors the structure of ``Block`` but uses ``VASelfAttention``
    with ``DualChunkKVCache`` and supports a ``persist`` flag.
    """

    modulation: nn.Parameter

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        apply_rope_before_kvcache: bool = True,
        cp_method: str = "ring",
    ) -> None:
        super().__init__()
        self.dim = dim

        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.self_attn = VASelfAttention(
            query_dim=dim,
            n_heads=num_heads,
            head_dim=dim // num_heads,
            eps=eps,
            apply_rope_before_kvcache=apply_rope_before_kvcache,
            cp_method=cp_method,
        )
        self.norm3 = (
            nn.LayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )
        self.cross_attn = CrossAttention(
            query_dim=dim,
            n_heads=num_heads,
            head_dim=dim // num_heads,
            eps=eps,
            cp_method=cp_method,
        )
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self._parameters_updated_after_loading_checkpoint = False

    def update_parameters_after_loading_checkpoint(self) -> None:
        if self._parameters_updated_after_loading_checkpoint:
            return
        self.modulation.data = self.modulation.data.squeeze(0)
        self._parameters_updated_after_loading_checkpoint = True

    def forward(
        self,
        x: Tensor,
        e: Tensor,
        cache: VABlockCache,
        rope_freqs: Tensor,
        persist: bool = False,
        write_mode: str = "primary",
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        """Run one transformer block.

        Args:
            x: Hidden states ``[batch, L, dim]``.
            e: Modulation ``[batch, L, 6, dim]`` (per-token timestep).
            cache: Block cache (self-attn + cross-attn).
            rope_freqs: RoPE frequencies ``[L, 1, 1, head_dim]``.
            persist: If True, return ``(x, k, v)`` for cache commit.
                If False, run read-only attention.
            write_mode: ``"primary"`` or ``"secondary"`` (passed through).

        Returns:
            If ``persist=False``: updated hidden states ``[batch, L, dim]``.
            If ``persist=True``: ``(hidden_states, k_to_write, v_to_write)``.
        """
        assert self._parameters_updated_after_loading_checkpoint
        e_chunks = [c.squeeze(-2) for c in (self.modulation + e).chunk(6, dim=-2)]

        y = self.norm1(x) * (1 + e_chunks[1]) + e_chunks[0]

        if persist:
            attn_out, k_fresh, v_fresh = self.self_attn.forward_persist(
                y, cache.self_attn, rope_freqs, write_mode
            )
        else:
            attn_out = self.self_attn.forward_readonly(
                y, cache.self_attn, rope_freqs
            )

        x = x + (attn_out * e_chunks[2])

        x = x + self.cross_attn(self.norm3(x), kv_cache=cache.cross_attn)

        y = self.norm2(x) * (1 + e_chunks[4]) + e_chunks[3]
        y = self.ffn(y)
        x = x + (y * e_chunks[5])

        if persist:
            return x, k_fresh, v_fresh  # type: ignore[possibly-undefined]
        return x
