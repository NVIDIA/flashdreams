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

"""Building blocks for the Wan 2.1 video DiT."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributed import ProcessGroup

from flashdreams.core.attention import BlockKVCache, RingAttention

from .rope import apply_rope_freqs


def sinusoidal_embedding_1d(dim: int, position: Tensor) -> Tensor:
    """Create 1D sinusoidal embeddings.

    Args:
        dim: Embedding dimension. Must be even.
        position: Position tensor of shape [...].

    Returns:
        Tensor of shape [..., dim] with concatenated cos/sin features.
    """
    assert dim % 2 == 0, "dim must be even for sinusoidal embedding"
    half = dim // 2
    position = position.type(torch.float64)
    freqs = torch.pow(
        10000, -torch.arange(half, device=position.device, dtype=torch.float64) / half
    )
    sinusoid = position[..., None] * freqs
    out = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=-1)
    return out


class MLPProj(torch.nn.Module):
    """Project conditioning embeddings with a small normalized MLP."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim),
            torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(),
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Project input embeddings to the target dimension."""
        return self.proj(x)


class Head(nn.Module):
    """Final projection head with AdaLN-style modulation."""

    def __init__(
        self,
        dim: int,
        out_dim: int,
        patch_size: tuple[int, int, int],
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # Output projection
        out_dim = math.prod(patch_size) * out_dim
        self.norm = nn.LayerNorm(dim, eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim)

        # AdaLN-style modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

        self._parameters_updated_after_loading_checkpoint = False

    def update_parameters_after_loading_checkpoint(self) -> None:
        # This function should be called after loading the checkpoint
        if self._parameters_updated_after_loading_checkpoint:
            return

        # We squeeze the modulation tensor to [2, D] to support arbitrary batch sizes [...]
        self.modulation.data = self.modulation.data.squeeze(0)
        self._parameters_updated_after_loading_checkpoint = True

    def forward(self, x: Tensor, e: Tensor) -> Tensor:
        """Apply adaptive normalization and project to patch output.

        Args:
            x: Hidden states of shape [..., L, dim].
            e: Modulation tensor of shape [..., 1, dim].

        Returns:
            Tensor of shape [..., L, prod(patch_size) * out_dim].
        """
        assert self._parameters_updated_after_loading_checkpoint, (
            "We expect to have called update_parameters_after_loading_checkpoint() "
            "before running the forward pass"
        )
        assert x.ndim == e.ndim, "x and e must have the same number of dimensions"
        e = (self.modulation + e).chunk(
            2, dim=-2
        )  # [..., 1, D] each  # ty:ignore[invalid-assignment]
        x = self.norm(x) * (1 + e[1]) + e[0]  # [..., L, D]
        x = self.head(x)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head attention with KV cache and optional RoPE."""

    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        n_heads: int = 8,
        head_dim: int = 64,
        eps: float = 1e-6,
    ) -> None:
        """Initialize a multi-head attention module.

        Args:
            query_dim: Feature dimension of query tokens and projected output.
            context_dim: Feature dimension of key/value tokens. Defaults to ``query_dim``.
            n_heads: Number of attention heads.
            head_dim: Per-head feature dimension. Inner dimension is ``n_heads * head_dim``.
        """
        super().__init__()
        context_dim = query_dim if context_dim is None else context_dim
        inner_dim = head_dim * n_heads

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.inner_dim = inner_dim
        self.eps = eps

        self.q = nn.Linear(query_dim, inner_dim)
        self.k = nn.Linear(context_dim, inner_dim)
        self.v = nn.Linear(context_dim, inner_dim)
        self.o = nn.Linear(inner_dim, query_dim)

        self.norm_q = nn.RMSNorm(inner_dim, eps=eps)
        self.norm_k = nn.RMSNorm(inner_dim, eps=eps)

        self.attn_op = RingAttention(qkv_format="bshd", backend="cudnn")

    def set_context_parallel_group(self, cp_group: ProcessGroup | None) -> None:
        """Configure context-parallel process group for the underlying attention op."""
        self.attn_op.set_context_parallel_group(cp_group=cp_group)

    def is_context_parallel_enabled(self) -> bool:
        """Whether context parallelism is active for attention."""
        return self.attn_op.is_context_parallel_enabled()

    def context_parallel_size(self) -> int:
        """World size of the context-parallel group (1 if disabled)."""
        return self.attn_op.context_parallel_size()

    def _compute_or_update_kv_cache(
        self,
        context: Tensor,
        kv_cache: BlockKVCache | None = None,
        rope_freqs: Tensor | None = None,
    ) -> BlockKVCache:
        """Project ``context`` into K/V and optionally append to ``kv_cache``.

        Args:
            context: Context tensor of shape [..., L, context_dim].
            kv_cache: Existing cache to update, or ``None`` to create a new cache.
            rope_freqs: Optional RoPE frequencies for K, shape [L, 1, 1, d // 2].

        Returns:
            Updated ``BlockKVCache`` containing keys and values.
        """
        batch_shape = context.shape[:-2]
        batch_size = math.prod(batch_shape)
        L, D = context.shape[-2:]
        n, d = self.n_heads, self.head_dim

        k = self.norm_k(self.k(context)).reshape(batch_size, L, n, d)
        v = self.v(context).reshape(batch_size, L, n, d)
        if rope_freqs is not None:
            # rope_freqs = torch.repeat_interleave(rope_freqs, repeats=2, dim=-1)
            k = apply_rope_freqs(k, rope_freqs, interleaved=True)

        if kv_cache is None:
            kv_cache = BlockKVCache.from_tensor(k, v, seq_dim=-3)
        else:
            kv_cache.update(k, v)
        return kv_cache

    def compute_kv(
        self,
        x: Tensor,
        rope_freqs: Tensor | None = None,
    ) -> BlockKVCache:
        """Build a new KV cache from ``x``."""
        return self._compute_or_update_kv_cache(x, None, rope_freqs)

    def update_kv(
        self,
        x: Tensor,
        kv_cache: BlockKVCache,
        rope_freqs: Tensor | None = None,
    ) -> BlockKVCache:
        """Append K/V computed from ``x`` into an existing ``kv_cache``."""
        return self._compute_or_update_kv_cache(x, kv_cache, rope_freqs)

    def apply_kv(
        self,
        x: Tensor,
        kv_cache: BlockKVCache,
        rope_freqs: Tensor | None = None,
    ) -> Tensor:
        """Run attention with queries from ``x`` against cached K/V.

        Args:
            x: Query tensor of shape [..., L, query_dim].
            kv_cache: KV cache used as attention context.
            rope_freqs: Optional RoPE frequencies for Q, shape [L, 1, 1, d // 2].

        Returns:
            Tensor of shape [..., L, query_dim] after output projection.
        """
        batch_shape = x.shape[:-2]
        batch_size = math.prod(batch_shape)
        L, D = x.shape[-2:]
        n, d = self.n_heads, self.head_dim
        assert n * d == D, "n * d must be equal to D"

        q = self.norm_q(self.q(x)).reshape(batch_size, L, n, d)
        if rope_freqs is not None:
            # rope_freqs = torch.repeat_interleave(rope_freqs, repeats=2, dim=-1)
            q = apply_rope_freqs(q, rope_freqs, interleaved=True)

        cached_k = kv_cache.cached_k()
        cached_v = kv_cache.cached_v()

        out = self.attn_op(q, cached_k, cached_v)
        out = out.reshape(batch_shape + (L, n * d))
        return self.o(out)

    def forward(
        self,
        x: Tensor,
        kv_cache: BlockKVCache,
        rope_freqs: Tensor | None = None,
        update_kv_cache: bool = True,
    ) -> Tensor:
        """Optionally refresh cache from ``x`` and run attention.

        Args:
            x: Query tensor and, when updating, the source for new K/V ([..., L, n * d]).
            kv_cache: Cache read by attention; written when ``update_kv_cache`` is True.
            rope_freqs: Optional RoPE frequencies for Q and (when updating) K.
            update_kv_cache: If False, only run attention against the existing cache.

        Returns:
            Projected output tensor of shape [..., L, query_dim].
        """
        if update_kv_cache:
            kv_cache = self.update_kv(x, kv_cache, rope_freqs)
        return self.apply_kv(x, kv_cache, rope_freqs)


class SelfAttention(MultiHeadAttention):
    """Self-attention that always refreshes K/V cache from current ``x``."""

    def initialize_cache(
        self,
        batch_size: int,
        chunk_size: int,
        window_size: int,
        sink_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> BlockKVCache:
        """Initialize KV cache for streaming self-attention.

        Args:
            batch_size: Flattened batch size used by attention.
            chunk_size: Number of tokens appended per update step.
            window_size: Rolling-window capacity in tokens.
            sink_size: Sink-token capacity retained permanently.
            device: Device for cache tensors.
            dtype: Data type for cache tensors.

        Returns:
            An initialized ``BlockKVCache``.
        """
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

    def forward(
        self,
        x: Tensor,
        kv_cache: BlockKVCache,
        rope_freqs: Tensor,
    ) -> Tensor:
        """Update cache from ``x`` and return self-attention output."""
        return super().forward(x, kv_cache, rope_freqs=rope_freqs, update_kv_cache=True)


@dataclass
class CrossAttnCache:
    """Cache container for cross-attention."""

    text: BlockKVCache
    img: BlockKVCache | None = None  # Optional image cache (I2V).


class CrossAttention(MultiHeadAttention):
    """Cross-attention with static cached context."""

    def __init__(self, i2v: bool = False, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.i2v = i2v
        if self.i2v:
            self.k_img = nn.Linear(self.context_dim, self.inner_dim)
            self.v_img = nn.Linear(self.context_dim, self.inner_dim)
            self.norm_k_img = nn.RMSNorm(self.inner_dim, eps=self.eps)
            self.attn_op_image = RingAttention(qkv_format="bshd", backend="cudnn")

    def compute_kv_image(self, context: Tensor) -> BlockKVCache:
        """Compute K/V from image ``context``.

        Args:
            context: Tensor of shape [..., L, context_dim].

        Returns:
            ``BlockKVCache`` containing projected image keys and values.
        """
        batch_shape = context.shape[:-2]
        batch_size = math.prod(batch_shape)
        L, D = context.shape[-2:]
        n, d = self.n_heads, self.head_dim

        k = self.norm_k_img(self.k_img(context)).reshape(batch_size, L, n, d)
        v = self.v_img(context).reshape(batch_size, L, n, d)
        return BlockKVCache.from_tensor(k, v, seq_dim=-3)

    def initialize_cache(
        self,
        context_text: Tensor,
        context_img: Tensor | None = None,
    ) -> CrossAttnCache:
        """Initialize cross-attention cache.

        Args:
            context_text: Text context tensor [B, L_text, D].
            context_img: Optional image context tensor [B, L_img, D].

        Returns:
            ``CrossAttnCache`` with text K/V and optional image K/V.
        """
        text_cache = self.compute_kv(context_text)
        if self.i2v:
            img_cache = self.compute_kv_image(context_img)  # ty:ignore[invalid-argument-type]
        else:
            img_cache = None
        return CrossAttnCache(text=text_cache, img=img_cache)

    def forward(
        self,
        x: Tensor,
        kv_cache: CrossAttnCache,
    ) -> Tensor:
        """Run cross-attention with queries from ``x`` and cached context."""
        batch_shape = x.shape[:-2]
        batch_size = math.prod(batch_shape)
        L, D = x.shape[-2:]
        n, d = self.n_heads, self.head_dim
        assert n * d == D, "n * d must be equal to D"

        q = self.norm_q(self.q(x)).reshape(batch_size, L, n, d)
        out = self.attn_op(q, kv_cache.text.cached_k(), kv_cache.text.cached_v())
        if self.i2v:
            assert kv_cache.img is not None, (
                "kv_cache_img is expected to be provided for I2V cross-attention"
            )
            out_img = self.attn_op_image(
                q, kv_cache.img.cached_k(), kv_cache.img.cached_v()
            )
            out = out + out_img
        out = out.reshape(batch_shape + (L, n * d))
        return self.o(out)


@dataclass
class BlockCache:
    """Per-block cache container for self-attention and cross-attention."""

    self_attn: BlockKVCache
    cross_attn: CrossAttnCache

    def before_update(self, chunk_idx: int) -> None:
        """Run pre-update hook for self-attention cache."""
        self.self_attn.before_update(chunk_idx)

    def after_update(self, chunk_idx: int) -> None:
        """Run post-update hook for self-attention cache."""
        self.self_attn.after_update(chunk_idx)


class Block(nn.Module):
    """Transformer block with self-attn, cross-attn, and FFN branches."""

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        i2v: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # Core submodules
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.self_attn = SelfAttention(
            query_dim=dim,
            n_heads=num_heads,
            head_dim=dim // num_heads,
            eps=eps,
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
            i2v=i2v,
            eps=eps,
        )
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )

        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        self._parameters_updated_after_loading_checkpoint = False

    def initialize_cache(
        self,
        chunk_size: int,
        window_size: int,
        sink_size: int,
        context_text: Tensor,
        context_img: Tensor | None = None,
    ) -> BlockCache:
        """Initialize per-branch caches for this transformer block.

        Args:
            chunk_size: Number of tokens appended per streaming update step.
            window_size: Rolling-window capacity in tokens.
            sink_size: Sink-token capacity retained permanently.
            context_text: Text context tensor [..., L_text, D].
            context_img: Optional image context tensor [..., L_img, D].

        Returns:
            ``BlockCache`` initialized for this block.
        """
        batch_shape = context_text.shape[:-2]
        batch_size = math.prod(batch_shape)
        device = context_text.device
        dtype = context_text.dtype

        return BlockCache(
            self_attn=self.self_attn.initialize_cache(
                batch_size,
                chunk_size,
                window_size,
                sink_size,
                device=device,
                dtype=dtype,
            ),
            cross_attn=self.cross_attn.initialize_cache(context_text, context_img),
        )

    def set_context_parallel_group(self, cp_group: ProcessGroup | None) -> None:
        """Set context-parallel process group for self-attention."""
        self.self_attn.set_context_parallel_group(cp_group)

    def update_parameters_after_loading_checkpoint(self) -> None:
        # This function should be called after loading the checkpoint
        if self._parameters_updated_after_loading_checkpoint:
            return

        # We squeeze the modulation tensor to [6, D] to support arbitrary batch sizes [...]
        self.modulation.data = self.modulation.data.squeeze(0)
        self._parameters_updated_after_loading_checkpoint = True

    def forward(
        self,
        x: Tensor,
        e: Tensor,
        cache: BlockCache,
        rope_freqs: Tensor,
    ) -> Tensor:
        """Run one transformer block update.

        Args:
            x: Input tensor with shape [..., L, D].
            e: Modulation tensor with shape [..., 6, D].
            cache: KV cache container for this block.
            rope_freqs: RoPE frequencies with shape [L, 1, 1, head_dim // 2].

        Returns:
            Updated hidden states with shape [..., L, D].
        """
        assert self._parameters_updated_after_loading_checkpoint, (
            "We expect to have called update_parameters_after_loading_checkpoint() "
            "before running the forward pass"
        )
        e = (self.modulation + e).chunk(
            6, dim=-2
        )  # [..., 1, D] each  # ty:ignore[invalid-assignment]

        y = self.norm1(x) * (1 + e[1]) + e[0]  # [..., L, D]
        y = self.self_attn(
            y,
            rope_freqs=rope_freqs,
            kv_cache=cache.self_attn,
        )
        x = x + (y * e[2])  # [..., L, D]

        x = x + self.cross_attn(
            self.norm3(x),
            kv_cache=cache.cross_attn,
        )
        y = self.norm2(x) * (1 + e[4]) + e[3]  # [..., L, D]
        y = self.ffn(y)
        x = x + (y * e[5])  # [..., L, D]
        return x
