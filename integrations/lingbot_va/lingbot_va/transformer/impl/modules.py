# SPDX-FileCopyrightText: Copyright (c) 2026 Hongyu Zhou
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
"""VA-specific transformer building blocks.

All block computations take pre-extracted KV tensors as arguments (no cache
object access), enabling ``torch.compile(fullgraph=True)`` without graph breaks.
Cache read/write operations happen at the network level, outside the compiled graph.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from flashdreams.core.attention.rope import apply_rope_freqs
from flashdreams.recipes.wan.transformer.impl.modules import (
    CrossAttnCache,
    CrossAttention,
    MultiHeadAttention,
)

from lingbot_va.transformer.impl.kvcache import VAKVCache


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

@dataclass
class VABlockCache:
    """Per-block cache for a VA transformer block."""

    self_attn: VAKVCache
    cross_attn: CrossAttnCache


# ---------------------------------------------------------------------------
# VASelfAttention
# ---------------------------------------------------------------------------

class VASelfAttention(MultiHeadAttention):
    """Self-attention that takes committed KV as plain tensors.

    All methods receive pre-sliced committed K/V and cross-attn K/V as
    tensor arguments — no cache object access inside, fully compile-friendly.
    """

    def forward(
        self,
        x: Tensor,
        committed_k: Tensor,
        committed_v: Tensor,
        rope_freqs: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute self-attention against committed cache + fresh tokens.

        Args:
            x: Input hidden states ``[batch, L, dim]``.
            committed_k: Committed K from prior steps ``[batch, N, heads, head_dim]``.
            committed_v: Committed V from prior steps ``[batch, N, heads, head_dim]``.
            rope_freqs: RoPE frequencies ``[L, 1, 1, head_dim]``.

        Returns:
            ``(attn_output, k_fresh, v_fresh)`` — output + fresh KV for optional cache write.
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

        if committed_k.shape[1] > 0:
            full_k = torch.cat([committed_k, k_fresh], dim=1)
            full_v = torch.cat([committed_v, v_fresh], dim=1)
        else:
            full_k = k_fresh
            full_v = v_fresh

        out = self.attn_op(q, full_k, full_v)
        out = out.reshape(batch_size, L, n * d)
        return self.o(out), k_fresh, v_fresh


# ---------------------------------------------------------------------------
# VABlock
# ---------------------------------------------------------------------------

class VABlock(nn.Module):
    """Transformer block for video-action models.

    Takes committed KV and cross-attn KV as tensor arguments.
    No cache object access inside — fully compile-friendly.
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
        committed_k: Tensor,
        committed_v: Tensor,
        cross_k: Tensor,
        cross_v: Tensor,
        rope_freqs: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run one transformer block.

        Args:
            x: Hidden states ``[batch, L, dim]``.
            e: Modulation ``[batch, L, 6, dim]``.
            committed_k: Prior committed K ``[batch, N, heads, head_dim]``.
            committed_v: Prior committed V ``[batch, N, heads, head_dim]``.
            cross_k: Cross-attn text K ``[batch, T, heads, head_dim]``.
            cross_v: Cross-attn text V ``[batch, T, heads, head_dim]``.
            rope_freqs: RoPE frequencies ``[L, 1, 1, head_dim]``.

        Returns:
            ``(x, k_fresh, v_fresh)`` — updated hidden states + fresh KV.
        """
        assert self._parameters_updated_after_loading_checkpoint
        e_chunks = [c.squeeze(-2) for c in (self.modulation + e).chunk(6, dim=-2)]

        y = self.norm1(x) * (1 + e_chunks[1]) + e_chunks[0]
        attn_out, k_fresh, v_fresh = self.self_attn(
            y, committed_k, committed_v, rope_freqs
        )
        x = x + (attn_out * e_chunks[2])

        # Cross-attention with pre-extracted text KV
        B, L, D = x.shape
        n, d = self.self_attn.n_heads, self.self_attn.head_dim
        y2 = self.norm3(x)
        q2 = self.cross_attn.norm_q(self.cross_attn.q(y2)).reshape(B, L, n, d)
        out2 = self.cross_attn.attn_op(q2, cross_k, cross_v)
        out2 = out2.reshape(B, L, n * d)
        x = x + self.cross_attn.o(out2)

        # FFN
        y3 = self.norm2(x) * (1 + e_chunks[4]) + e_chunks[3]
        y3 = self.ffn(y3)
        x = x + (y3 * e_chunks[5])

        return x, k_fresh, v_fresh
