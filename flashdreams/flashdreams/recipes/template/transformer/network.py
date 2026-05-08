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

"""Dummy single-block DiT network used by the template recipe."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributed import ProcessGroup

from flashdreams.core.attention.kvcache import BlockKVCache
from flashdreams.core.attention.ring import RingAttention
from flashdreams.core.attention.rope import apply_rope_freqs
from flashdreams.infra.config import InstantiateConfig


@dataclass(kw_only=True)
class TemplateDiTCache:
    """Per-rollout network cache for :class:`TemplateDiT`.

    Holds the block's KV cache plus the one-shot context embedding
    injected as an additive bias every forward. Forwards the
    ``before_update`` / ``after_update`` protocol to :class:`BlockKVCache`.
    """

    kv_cache: BlockKVCache
    """Self-attention KV cache; shape ``[B, total_size, H, d_h]``."""

    context: Tensor
    """Per-rollout context tokens ``[B, N_ctx, D]``, injected as an
    additive bias on every forward."""

    def before_update(self, autoregressive_index: int) -> None:
        """Prepare the KV cache for writing chunk ``autoregressive_index``."""
        self.kv_cache.before_update(autoregressive_index)

    def after_update(self, autoregressive_index: int) -> None:
        """Commit bookkeeping after chunk ``autoregressive_index`` has been written."""
        self.kv_cache.after_update(autoregressive_index)


@dataclass(kw_only=True)
class TemplateDiTConfig(InstantiateConfig["TemplateDiT"]):
    """Config for the template recipe's dummy DiT."""

    _target: type["TemplateDiT"] = field(default_factory=lambda: TemplateDiT)

    in_channels: int = 4
    """Per-token channel width seen by the network — the
    **post-patchify** channel dim. Builders must set this to
    ``raw_channels * prod(TemplateTransformerConfig.patch_size)``."""

    context_channels: int = 16
    """Channel count of the pre-encoded context token tensor."""

    model_channels: int = 64
    """Hidden width used inside the block."""

    num_heads: int = 4
    """Attention head count. ``model_channels`` must be divisible by this."""

    ffn_mult: float = 2.0
    """Expansion factor applied to ``model_channels`` inside the FFN."""

    def __post_init__(self) -> None:
        assert self.model_channels % self.num_heads == 0, (
            f"model_channels ({self.model_channels}) must be divisible by "
            f"num_heads ({self.num_heads})."
        )


class TemplateDiT(nn.Module):
    """Minimal single-block DiT used as a reference recipe.

    Shape: per-token projection → time / context bias → self-attention
    through a :class:`~flashdreams.core.attention.kvcache.BlockKVCache`
    → FFN → output projection.

    Per-step usage:
        1. ``cache.before_update(autoregressive_index)`` — hoisted to
           :meth:`~flashdreams.recipes.template.transformer.TemplateTransformerCache.start`.
        2. ``forward(noisy_latent, timesteps, cache, control)``.
        3. ``cache.after_update(autoregressive_index)`` — hoisted to
           :meth:`~flashdreams.recipes.template.transformer.TemplateTransformerCache.finalize`.
    """

    def __init__(self, config: TemplateDiTConfig) -> None:
        super().__init__()
        self.config = config
        D = config.model_channels
        H = config.num_heads
        self._head_dim = D // H
        self._num_heads = H

        self.input_proj = nn.Linear(config.in_channels, D)
        self.context_proj = nn.Linear(config.context_channels, D)
        # Scalar timestep lifted to ``[1, D]``. Real recipes use a
        # sinusoidal embedding + MLP.
        self.timestep_encoder = nn.Sequential(
            nn.Linear(1, D),
            nn.SiLU(),
            nn.Linear(D, D),
        )

        self.norm1 = nn.LayerNorm(D)
        self.q_proj = nn.Linear(D, D)
        self.k_proj = nn.Linear(D, D)
        self.v_proj = nn.Linear(D, D)
        # ``bshd`` matches the native ``[B, S, H, d_h]`` layout;
        # RingAttention handles the CP gather + LSE merge.
        self.attn = RingAttention(qkv_format="bshd", backend="cudnn")
        self.attn_out = nn.Linear(D, D)

        self.norm2 = nn.LayerNorm(D)
        ffn_hidden = int(D * config.ffn_mult)
        self.ffn = nn.Sequential(
            nn.Linear(D, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, D),
        )

        self.output_proj = nn.Linear(D, config.in_channels)

    def set_context_parallel_group(self, cp_group: ProcessGroup | None) -> None:
        """Forward the CP group to :class:`RingAttention`.

        Args:
            cp_group: Context-parallel group; ``None`` disables CP.
        """
        self.attn.set_context_parallel_group(cp_group)

    def initialize_cache(
        self,
        *,
        chunk_size: int,
        window_size: int,
        sink_size: int,
        context: Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> TemplateDiTCache:
        """Allocate this block's KV cache and project the context tokens.

        Args:
            chunk_size: Per-rank tokens written per AR step.
            window_size: Per-rank sliding-window size, excluding sinks.
            sink_size: Per-rank sink-token count.
            context: ``[B, N_ctx, context_channels]`` tokens; projected
                to ``model_channels`` once here and reused every forward.
            batch_size: Leading batch dim of the KV buffer.
            device: Device for the KV buffer.
            dtype: Dtype for the KV buffer and the projected context.
        """
        total = sink_size + window_size
        k_shape = (batch_size, total, self._num_heads, self._head_dim)
        v_shape = (batch_size, total, self._num_heads, self._head_dim)
        kv_cache = BlockKVCache(
            k_shape=k_shape,
            v_shape=v_shape,
            seq_dim=1,
            chunk_size=chunk_size,
            window_size=window_size,
            sink_size=sink_size,
            device=device,
            dtype=dtype,
        )
        context = self.context_proj(context.to(dtype=dtype))
        return TemplateDiTCache(kv_cache=kv_cache, context=context)

    def forward(
        self,
        noisy_latent: Tensor,
        *,
        timesteps: Tensor,
        cache: TemplateDiTCache,
        rope_freqs: Tensor,
        control: Tensor | None = None,
    ) -> Tensor:
        """Predict flow for one per-rank AR chunk.

        Args:
            noisy_latent: ``[B, L/cp, in_channels]`` — already
                patchified and CP-split by the transformer wrapper.
            timesteps: Scalar timestep.
            cache: Per-rollout cache. AR-step bookkeeping is hoisted
                to :class:`TemplateTransformerCache`.
            rope_freqs: Self attn rope freqs. Shape [L/cp, 1, 1, d // 2].
            control: Per-AR-step control latent, same shape as
                ``noisy_latent``; ``None`` skips the control bias.

        Returns:
            ``[B, L/cp, in_channels]`` flow prediction.
        """
        B, L_local, _ = noisy_latent.shape
        D = self.config.model_channels

        x = self.input_proj(noisy_latent)
        if control is not None:
            x = x + self.input_proj(control)

        t_emb = self.timestep_encoder(timesteps.reshape(1, 1).to(x.dtype))
        ctx_bias = cache.context.mean(dim=1)  # [B, D]
        x = x + t_emb.view(1, 1, D) + ctx_bias.view(B, 1, D)

        x = self._self_attn_block(x, rope_freqs, kv_cache=cache.kv_cache)
        return self.output_proj(x)

    def _self_attn_block(
        self, x: Tensor, rope_freqs: Tensor, *, kv_cache: BlockKVCache
    ) -> Tensor:
        """Run one pre-norm self-attention + FFN residual block.

        Q is this rank's current chunk; K/V come from ``kv_cache``
        (filling or steady view). :class:`RingAttention` fuses the
        cross-rank KV gather with the SDPA call.
        """
        B, L_local, D = x.shape

        h = self.norm1(x)
        q = self.q_proj(h).view(B, L_local, self._num_heads, self._head_dim)
        k = self.k_proj(h).view(B, L_local, self._num_heads, self._head_dim)
        v = self.v_proj(h).view(B, L_local, self._num_heads, self._head_dim)

        q = apply_rope_freqs(q, rope_freqs)
        k = apply_rope_freqs(k, rope_freqs)

        kv_cache.update(k, v)
        k_local = kv_cache.cached_k()  # [B, S_local, H, d_h]
        v_local = kv_cache.cached_v()

        attn = self.attn(q, k_local, v_local).reshape(B, L_local, D)
        x = x + self.attn_out(attn)

        x = x + self.ffn(self.norm2(x))
        return x
