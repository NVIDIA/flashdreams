# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MemRoPE-specific Wan block variants."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from flashdreams.core.attention import MemRoPEKVCache, RingAttention
from flashdreams.recipes.wan.transformer.impl.modules import (
    Block,
    BlockCache,
    CrossAttnCache,
    SelfAttention,
)
from flashdreams.core.attention.rope import (
    RotaryPositionEmbedding3D,
)

try:
    import flash_attn  # type: ignore[import-not-found]
except (ImportError, ModuleNotFoundError):
    flash_attn = None


class MemRoPERMSNorm(nn.Module):
    """Official Wan RMSNorm semantics: fp32 norm, cast back, then scale."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return y.type_as(x) * self.weight


class MemRoPEFlashAttention(nn.Module):
    """Official Wan flash-attn varlen wrapper for cp_size=1 MemRoPE."""

    def __init__(self, *, deterministic: bool = False) -> None:
        super().__init__()
        self.deterministic = deterministic
        self.fallback = RingAttention(qkv_format="bshd", backend="flash")

    def set_context_parallel_group(self, cp_group) -> None:  # type: ignore[no-untyped-def]
        self.fallback.set_context_parallel_group(cp_group)

    def is_context_parallel_enabled(self) -> bool:
        return self.fallback.is_context_parallel_enabled()

    def context_parallel_size(self) -> int:
        return self.fallback.context_parallel_size()

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        if flash_attn is None or self.is_context_parallel_enabled():
            return self.fallback(query, key, value)

        half_dtypes = (torch.float16, torch.bfloat16)
        out_dtype = query.dtype
        batch, query_len, key_len = query.shape[0], query.shape[1], key.shape[1]

        q = query if query.dtype in half_dtypes else query.to(torch.bfloat16)
        k = key if key.dtype in half_dtypes else key.to(torch.bfloat16)
        v = value if value.dtype in half_dtypes else value.to(torch.bfloat16)
        q = q.flatten(0, 1).to(v.dtype)
        k = k.flatten(0, 1).to(v.dtype)
        v = v.flatten(0, 1)

        q_lens = torch.full(
            (batch,), query_len, dtype=torch.int32, device=query.device
        )
        k_lens = torch.full((batch,), key_len, dtype=torch.int32, device=key.device)
        cu_q = torch.cat([q_lens.new_zeros(1), q_lens]).cumsum(
            0, dtype=torch.int32
        )
        cu_k = torch.cat([k_lens.new_zeros(1), k_lens]).cumsum(
            0, dtype=torch.int32
        )
        out = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=query_len,
            max_seqlen_k=key_len,
            dropout_p=0.0,
            deterministic=self.deterministic,
        )
        return out.unflatten(0, (batch, query_len)).type(out_dtype)


def _official_complex_rope_params(
    max_seq_len: int,
    dim: int,
    *,
    device: torch.device,
) -> Tensor:
    freqs = torch.outer(
        torch.arange(max_seq_len, device=device, dtype=torch.float64),
        1.0
        / torch.pow(
            10000,
            torch.arange(0, dim, 2, device=device, dtype=torch.float64) / dim,
        ),
    )
    return torch.polar(torch.ones_like(freqs), freqs)


def memrope_apply_rope_official(
    x: Tensor,
    frame_indices: Tensor,
    *,
    len_h: int,
    len_w: int,
) -> Tensor:
    """Apply official MemRoPE/Wan complex RoPE for explicit frame indices."""
    batch, seq_len, n_heads, head_dim = x.shape
    half_dim = head_dim // 2
    dim_t = half_dim - 2 * (half_dim // 3)
    dim_h = half_dim // 3
    dim_w = half_dim // 3
    device = x.device
    frame_indices = frame_indices.to(device=device, dtype=torch.long)
    seq_h = torch.arange(len_h, device=device, dtype=torch.long)
    seq_w = torch.arange(len_w, device=device, dtype=torch.long)

    max_t = max(1024, int(frame_indices.max().item()) + 1)
    max_h = max(1024, len_h)
    max_w = max(1024, len_w)
    freqs_t = _official_complex_rope_params(max_t, 2 * dim_t, device=device)
    freqs_h = _official_complex_rope_params(max_h, 2 * dim_h, device=device)
    freqs_w = _official_complex_rope_params(max_w, 2 * dim_w, device=device)

    num_frames = frame_indices.numel()
    freqs_i = torch.cat(
        [
            freqs_t[frame_indices].view(num_frames, 1, 1, dim_t).expand(
                num_frames, len_h, len_w, dim_t
            ),
            freqs_h[seq_h].view(1, len_h, 1, dim_h).expand(
                num_frames, len_h, len_w, dim_h
            ),
            freqs_w[seq_w].view(1, 1, len_w, dim_w).expand(
                num_frames, len_h, len_w, dim_w
            ),
        ],
        dim=-1,
    ).reshape(seq_len, 1, half_dim)

    x_complex = torch.view_as_complex(
        x.to(torch.float64).reshape(batch, seq_len, n_heads, -1, 2)
    )
    x_complex = x_complex * freqs_i.unsqueeze(0)
    return torch.view_as_real(x_complex).flatten(3).type_as(x)


class MemRoPESelfAttention(SelfAttention):
    """Self-attention using MemRoPE raw-K cache and online RoPE indexing."""

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.attn_op = MemRoPEFlashAttention(deterministic=True)
        self.norm_q = MemRoPERMSNorm(self.inner_dim, eps=self.eps)
        self.norm_k = MemRoPERMSNorm(self.inner_dim, eps=self.eps)

    def initialize_cache(
        self,
        batch_size: int,
        chunk_size: int,
        window_size: int,
        sink_size: int,
        device: torch.device,
        dtype: torch.dtype,
        *,
        frame_size: int,
        recent_size: int,
        memory_frames: int,
        ema_alpha_long: float,
        ema_alpha_short: float,
    ) -> MemRoPEKVCache:
        total_size = sink_size + window_size
        return MemRoPEKVCache(
            k_shape=(batch_size, total_size, self.n_heads, self.head_dim),
            v_shape=(batch_size, total_size, self.n_heads, self.head_dim),
            seq_dim=-3,
            chunk_size=chunk_size,
            window_size=window_size,
            sink_size=sink_size,
            frame_size=frame_size,
            recent_size=recent_size,
            memory_frames=memory_frames,
            ema_alpha_long=ema_alpha_long,
            ema_alpha_short=ema_alpha_short,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: Tensor,
        kv_cache: MemRoPEKVCache,
        rope_adapter: RotaryPositionEmbedding3D,
        *,
        len_h: int,
        len_w: int,
    ) -> Tensor:
        batch_shape = x.shape[:-2]
        batch_size = math.prod(batch_shape)
        L, D = x.shape[-2:]
        n, d = self.n_heads, self.head_dim
        assert n * d == D, "n * d must be equal to D"

        q = self.norm_q(self.q(x)).reshape(batch_size, L, n, d)
        k = self.norm_k(self.k(x)).reshape(batch_size, L, n, d)
        v = self.v(x).reshape(batch_size, L, n, d)

        kv_cache.update(k, v)

        q = memrope_apply_rope_official(
            q,
            kv_cache.query_frame_indices(),
            len_h=len_h,
            len_w=len_w,
        )
        cached_k = memrope_apply_rope_official(
            kv_cache.cached_k(),
            kv_cache.cached_frame_indices(),
            len_h=len_h,
            len_w=len_w,
        )
        cached_v = kv_cache.cached_v()

        out = self.attn_op(q, cached_k, cached_v)
        out = out.reshape(batch_shape + (L, n * d))
        return self.o(out)


@dataclass
class MemRoPEBlockCache(BlockCache):
    """Per-block cache container with a MemRoPE self-attention cache."""

    self_attn: MemRoPEKVCache
    cross_attn: CrossAttnCache


class MemRoPEBlock(Block):
    """Wan transformer block with MemRoPE self-attention."""

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.self_attn = MemRoPESelfAttention(
            query_dim=self.dim,
            n_heads=self.num_heads,
            head_dim=self.dim // self.num_heads,
            eps=self.eps,
        )
        self.cross_attn.norm_q = MemRoPERMSNorm(self.dim, eps=self.eps)
        self.cross_attn.norm_k = MemRoPERMSNorm(self.dim, eps=self.eps)
        self.cross_attn.attn_op = MemRoPEFlashAttention(deterministic=False)

    def initialize_cache(
        self,
        chunk_size: int,
        window_size: int,
        sink_size: int,
        context_text: Tensor,
        context_img: Tensor | None = None,
        *,
        frame_size: int,
        recent_size: int,
        memory_frames: int,
        ema_alpha_long: float,
        ema_alpha_short: float,
    ) -> MemRoPEBlockCache:
        batch_shape = context_text.shape[:-2]
        batch_size = math.prod(batch_shape)
        device = context_text.device
        dtype = context_text.dtype
        assert isinstance(self.self_attn, MemRoPESelfAttention)

        return MemRoPEBlockCache(
            self_attn=self.self_attn.initialize_cache(
                batch_size,
                chunk_size,
                window_size,
                sink_size,
                device=device,
                dtype=dtype,
                frame_size=frame_size,
                recent_size=recent_size,
                memory_frames=memory_frames,
                ema_alpha_long=ema_alpha_long,
                ema_alpha_short=ema_alpha_short,
            ),
            cross_attn=self.cross_attn.initialize_cache(context_text, context_img),
        )

    def forward(
        self,
        x: Tensor,
        e: Tensor,
        cache: MemRoPEBlockCache,
        rope_adapter: RotaryPositionEmbedding3D,
        *,
        len_h: int,
        len_w: int,
    ) -> Tensor:
        assert self._parameters_updated_after_loading_checkpoint, (
            "We expect to have called update_parameters_after_loading_checkpoint() "
            "before running the forward pass"
        )
        assert isinstance(self.self_attn, MemRoPESelfAttention)
        e_chunks = (self.modulation + e).chunk(6, dim=-2)

        if e.ndim == x.ndim + 1:
            num_frames = e.shape[-3]
            frame_seqlen = x.shape[-2] // num_frames

            def split_frames(tensor: Tensor) -> Tensor:
                return tensor.unflatten(dim=-2, sizes=(num_frames, frame_seqlen))

            def merge_frames(tensor: Tensor) -> Tensor:
                return tensor.flatten(start_dim=-3, end_dim=-2)

            y = split_frames(self.norm1(x)) * (1 + e_chunks[1]) + e_chunks[0]
            y = merge_frames(y)
        else:
            y = self.norm1(x) * (1 + e_chunks[1]) + e_chunks[0]
        y = self.self_attn(
            y,
            kv_cache=cache.self_attn,
            rope_adapter=rope_adapter,
            len_h=len_h,
            len_w=len_w,
        )
        if e.ndim == x.ndim + 1:
            x = x + merge_frames(split_frames(y) * e_chunks[2])
        else:
            x = x + (y * e_chunks[2])

        x = x + self.cross_attn(
            self.norm3(x),
            kv_cache=cache.cross_attn,
        )
        if e.ndim == x.ndim + 1:
            y = split_frames(self.norm2(x)) * (1 + e_chunks[4]) + e_chunks[3]
            y = merge_frames(y)
        else:
            y = self.norm2(x) * (1 + e_chunks[4]) + e_chunks[3]
        y = self.ffn(y)
        if e.ndim == x.ndim + 1:
            x = x + merge_frames(split_frames(y) * e_chunks[5])
        else:
            x = x + (y * e_chunks[5])
        return x
