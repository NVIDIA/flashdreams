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

"""ArtiFixer cross-attention with a neighbor-frame KV bank + PRoPE.

Adds a third KV pathway (parallel to text and optional I2V image) for
attending over VAE-encoded neighbor frames. Module naming mirrors the
diffusers ``WanAttention`` extension in the ArtiFixer reference:

  - ``add_k_proj`` / ``add_v_proj`` — Linear projections from latent
    neighbor context to K / V
  - ``norm_added_k`` — RMSNorm on K (matches ``norm_k`` shape)
  - ``attn_op_neighbor`` — separate RingAttention op for the neighbor
    branch (lets q/k be PRoPE-transformed independently of the text
    branch)

``forward`` accepts optional ``neighbor_kv_cache`` + ``prope_src`` /
``prope_tgt`` modules. When all three are ``None`` (text-only path),
the call is identical to ``CrossAttention.forward``. Otherwise the
neighbor branch runs::

    q_pr = prope_src._apply_to_q(q)
    k_pr = prope_tgt._apply_to_kv(k_neighbor)
    v_pr = prope_tgt._apply_to_kv(v_neighbor)
    o_n  = prope_src._apply_to_o(attn_op_neighbor(q_pr, k_pr, v_pr))
    out  = out_text [+ out_img]  + (0 if ignore_neighbors else 1) * o_n

PRoPE math runs at fp32 with a ``.float() / .to(query.dtype)`` round
trip (matching the reference), since ``_rope_precompute_coeffs`` does
an int64 -> fp32 promotion internally.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

from flashdreams.core.attention.kvcache import BlockKVCache
from flashdreams.core.attention.ring import RingAttention
from flashdreams.recipes.wan.transformer.impl.modules import (
    CrossAttention,
    CrossAttnCache,
)


class ArtifixerCrossAttention(CrossAttention):
    """Cross-attention extended with a neighbor-frame KV branch."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # Parameter names match the ArtiFixer reference's WanAttention
        # (``add_k_proj`` / ``add_v_proj`` / ``norm_added_k``) so loading
        # a merged ArtiFixer DMD checkpoint only needs the diffusers
        # ``attn2 -> cross_attn`` regex remap, not a per-key rename.
        self.add_k_proj = nn.Linear(self.context_dim, self.inner_dim, bias=True)
        self.add_v_proj = nn.Linear(self.context_dim, self.inner_dim, bias=True)
        self.norm_added_k = nn.RMSNorm(self.inner_dim, eps=self.eps)

        # Zero-init add_v_proj only (matches the ArtiFixer reference).
        # Because attention is softmax(Q K^T) @ V, V=0 forces the neighbor
        # branch contribution to zero at load time even if add_k_proj is
        # non-zero, so the wrapped block is a no-op extension of base Wan
        # behavior.
        nn.init.zeros_(self.add_v_proj.weight)
        nn.init.zeros_(self.add_v_proj.bias)

        self.attn_op_neighbor = RingAttention(qkv_format="bshd", backend="cudnn")

        # Set once per rollout via :meth:`initialize_neighbor_cache`. Cleared
        # to ``None`` when no neighbor context is provided. Non-parameter
        # module state -- safe to mutate outside ``forward``.
        self.neighbor_kv_cache: BlockKVCache | None = None

    def set_context_parallel_group(self, cp_group: Any) -> None:
        """Set CP group on every attention op, including the neighbor branch."""
        super().set_context_parallel_group(cp_group)
        self.attn_op_neighbor.set_context_parallel_group(cp_group=cp_group)

    def compute_kv_neighbor(self, context: Tensor) -> BlockKVCache:
        """Project neighbor ``context`` into a static K/V cache.

        Args:
            context: Neighbor context, shape ``[..., L_neighbor, context_dim]``.

        Returns:
            ``BlockKVCache`` carrying ``(k, v)`` reshaped to
            ``[batch_size, L, n_heads, head_dim]``. Static (no rolling).
        """
        batch_shape = context.shape[:-2]
        batch_size = math.prod(batch_shape)
        L, _ = context.shape[-2:]
        n, d = self.n_heads, self.head_dim

        k = self.norm_added_k(self.add_k_proj(context)).reshape(batch_size, L, n, d)
        v = self.add_v_proj(context).reshape(batch_size, L, n, d)
        return BlockKVCache.from_tensor(k, v, seq_dim=-3)

    def initialize_neighbor_cache(self, context: Tensor | None) -> None:
        """Build and store the static neighbor KV cache for this module.

        Call once per rollout. ``context=None`` clears the cache so the
        forward path skips the neighbor branch (text-only mode).
        """
        self.neighbor_kv_cache = (
            None if context is None else self.compute_kv_neighbor(context)
        )

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        kv_cache: CrossAttnCache,
        *,
        prope_src: Any = None,
        prope_tgt: Any = None,
        ignore_neighbors: bool = False,
    ) -> Tensor:
        """Cross-attention with optional PRoPE neighbor branch.

        Uses the per-module ``self.neighbor_kv_cache`` populated by
        :meth:`initialize_neighbor_cache`. When that is ``None`` (or PRoPE
        modules are ``None``), behavior is identical to
        :meth:`CrossAttention.forward`. Otherwise the PRoPE-modulated
        neighbor branch is added to the output.

        Args:
            x: Query tokens ``[..., L, query_dim]``.
            kv_cache: Text (+ optional image) K/V cache from the base class.
            prope_src: PRoPE module for the source / target cameras
                (transforms q on read, o on write).
            prope_tgt: PRoPE module for the neighbor cameras (transforms
                k, v on read).
            ignore_neighbors: If ``True``, zero out the neighbor contribution.
                Matches the diffusion-forcing CFG dropout knob in the
                ArtiFixer reference.
        """
        neighbor_kv_cache = self.neighbor_kv_cache
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

        if neighbor_kv_cache is not None:
            assert prope_src is not None and prope_tgt is not None, (
                "prope_src and prope_tgt are required when neighbor_kv_cache is provided"
            )
            q_dtype = q.dtype

            # PRoPE expects (B, H, L, D); attn ops use (B, L, H, D) (bshd).
            # Run PRoPE math at fp32 with a dtype round-trip to mirror the
            # ArtiFixer reference.
            q_pr = (
                prope_src._apply_to_q(q.transpose(1, 2).float())
                .transpose(1, 2)
                .to(q_dtype)
            )
            k_n = neighbor_kv_cache.cached_k()
            v_n = neighbor_kv_cache.cached_v()
            k_pr = (
                prope_tgt._apply_to_kv(k_n.transpose(1, 2).float())
                .transpose(1, 2)
                .to(q_dtype)
            )
            v_pr = (
                prope_tgt._apply_to_kv(v_n.transpose(1, 2).float())
                .transpose(1, 2)
                .to(q_dtype)
            )

            out_neighbor = self.attn_op_neighbor(q_pr, k_pr, v_pr)
            out_neighbor = (
                prope_src._apply_to_o(out_neighbor.transpose(1, 2).float())
                .transpose(1, 2)
                .to(q_dtype)
            )

            if not ignore_neighbors:
                out = out + out_neighbor

        out = out.reshape(batch_shape + (L, n * d))
        return self.o(out)
