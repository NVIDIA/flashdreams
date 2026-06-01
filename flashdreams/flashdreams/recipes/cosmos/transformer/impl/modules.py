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

"""Cosmos DiT building blocks."""

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributed import ProcessGroup

from flashdreams.core.attention import (
    BlockKVCache,
    ContextParallelAttention,
    KVRange,
    PrefixBlockKVCache,
    RollingBlockKVCache,
)
from flashdreams.core.attention.rope import apply_rope_freqs


class GPT2FeedForward(nn.Module):
    """GPT-2 style feed-forward network with GELU activation."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.activation = nn.GELU()
        self.layer1 = nn.Linear(d_model, d_ff, bias=False)
        self.layer2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Apply feed-forward transformation.

        Args:
            x: Input tensor of shape (..., D).

        Returns:
            Output tensor of shape (..., D).
        """
        return self.layer2(self.activation(self.layer1(x)))


class Timesteps(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    SINUSOIDAL_FREQ_BASE = 10000

    emb: Tensor

    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.num_channels = num_channels

        half_dim = num_channels // 2
        exponent = -math.log(self.SINUSOIDAL_FREQ_BASE) * torch.arange(
            half_dim, dtype=torch.float32
        )
        exponent = exponent / half_dim
        emb = torch.exp(exponent)
        self.register_buffer("emb", emb, persistent=False)

    def forward(self, timesteps: Tensor) -> Tensor:
        """Embed timesteps into sinusoidal frequencies.

        Args:
            timesteps: Input tensor of shape (...).

        Returns:
            Embedded tensor of shape (..., num_channels).
        """
        emb = timesteps.unsqueeze(-1) * self.emb
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
        return emb


class TimestepEmbedding(nn.Module):
    """MLP for encoding timestep embeddings with optional AdaLN-LoRA."""

    def __init__(
        self, in_features: int, out_features: int, use_adaln_lora: bool = True
    ) -> None:
        super().__init__()
        self.use_adaln_lora = use_adaln_lora

        self.linear_1 = nn.Linear(in_features, out_features, bias=not use_adaln_lora)
        self.activation = nn.SiLU()

        out_dim = 3 * out_features if use_adaln_lora else out_features
        self.linear_2 = nn.Linear(out_features, out_dim, bias=False)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor | None]:
        """Encode timestep embedding.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Tuple of (emb, adaln_lora):
                - emb: Output tensor of shape (..., out_features).
                - adaln_lora: If use_adaln_lora, tensor of shape (..., 3 * out_features); otherwise None.
        """
        out = self.linear_2(self.activation(self.linear_1(x)))

        if self.use_adaln_lora:
            return x, out
        return out, None


class PatchEmbed(nn.Module):
    """Patch embedding module for video/image inputs.

    Note: The patchify operation (rearranging from spatial to patch tokens) is expected
    to be performed externally. This module expects post-patchified flattened input of shape (..., D)
    where D = in_channels * temporal_patch_size * spatial_patch_size^2.
    """

    def __init__(
        self,
        spatial_patch_size: int,
        temporal_patch_size: int,
        in_channels: int = 3,
        out_channels: int = 768,
    ) -> None:
        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels

        self.proj = nn.Sequential(
            nn.Identity(),  # Placeholder for checkpoint compatibility
            nn.Linear(self._compute_in_features(), out_channels, bias=False),
        )

    def _compute_in_features(self) -> int:
        """Compute the flattened patch dimension."""
        return self.in_channels * self.temporal_patch_size * self.spatial_patch_size**2

    def get_linear_in_channels(self) -> int:
        """Return input dimension for the linear projection (for external use)."""
        return self._compute_in_features()

    def forward(self, x: Tensor) -> Tensor:
        """Project flattened patches to embedding space.

        Args:
            x: Input tensor of shape (..., D) where D = in_channels * kt * kh * kw.

        Returns:
            Embedded patches of shape (..., out_channels).
        """
        expected_in_features = self._compute_in_features()
        assert x.shape[-1] == expected_in_features, (
            f"Expected input features to be {expected_in_features}, but got {x.shape[-1]}."
        )
        return self.proj(x)


class FinalLayer(nn.Module):
    """Final layer of the DiT network with AdaLN modulation."""

    NUM_ADALN_CHUNKS = 2

    def __init__(
        self,
        hidden_size: int,
        spatial_patch_size: int,
        temporal_patch_size: int,
        out_channels: int,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.use_adaln_lora = use_adaln_lora

        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        patch_dim = spatial_patch_size**2 * temporal_patch_size * out_channels
        self.linear = nn.Linear(hidden_size, patch_dim, bias=False)

        modulation_out_dim = self.NUM_ADALN_CHUNKS * hidden_size
        if use_adaln_lora:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, modulation_out_dim, bias=False),
            )
        else:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, modulation_out_dim, bias=False),
            )

    def forward(
        self, x: Tensor, emb: Tensor, adaln_lora: Tensor | None = None
    ) -> Tensor:
        """Apply final layer with adaptive layer normalization.

        Args:
            x: Input tensor of shape ``[..., L, D]``.
            emb: Conditioning embedding of shape ``[..., L or 1, D]``
            adaln_lora: Optional LoRA tensor of shape
                ``[..., L or 1, 3 * D]``.

        Returns:
            Output tensor of shape ``[..., L, patch_dim]``.
        """
        assert emb.ndim == x.ndim, "emb and x must have the same number of dimensions"
        if self.use_adaln_lora:
            assert adaln_lora is not None
            modulation = (
                self.adaln_modulation(emb) + adaln_lora[..., : 2 * self.hidden_size]
            )
            shift, scale = modulation.chunk(2, dim=-1)
        else:
            shift, scale = self.adaln_modulation(emb).chunk(2, dim=-1)

        x = self.layer_norm(x) * (1.0 + scale) + shift
        return self.linear(x)


class MultiHeadAttention(nn.Module):
    """Multi-head attention with KV cache and optional RoPE."""

    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        n_heads: int = 8,
        head_dim: int = 64,
        cp_method: Literal["ring", "ulysses"] = "ring",
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

        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.output_proj = nn.Linear(inner_dim, query_dim, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)

        self.attn_op = ContextParallelAttention(
            qkv_format="bshd", backend="cudnn", method=cp_method
        )

    def set_context_parallel_group(self, cp_group: ProcessGroup | None) -> None:
        """Configure context-parallel process group for the underlying attention op."""
        self.attn_op.set_context_parallel_group(cp_group=cp_group)

    def is_context_parallel_enabled(self) -> bool:
        """Whether context parallelism is active for attention."""
        return self.attn_op.is_context_parallel_enabled()

    def context_parallel_size(self) -> int:
        """World size of the context-parallel group (1 if disabled)."""
        return self.attn_op.context_parallel_size()

    def _project_kv(
        self,
        context: Tensor,
        rope_freqs: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Project ``context`` into cache-shaped K/V ``[batch, L, n, d]``.

        Single source for the K/V projection shared by :meth:`compute_kv`
        (prefix fill) and :meth:`update_kv` (rolling in-place write).
        """
        batch_shape = context.shape[:-2]
        batch_size = math.prod(batch_shape)
        L, D = context.shape[-2:]
        n, d = self.n_heads, self.head_dim

        k = self.k_norm(self.k_proj(context).reshape(batch_size, L, n, d))
        v = self.v_proj(context).reshape(batch_size, L, n, d)
        if rope_freqs is not None:
            k = apply_rope_freqs(k, rope_freqs)
        return k, v

    def compute_kv(
        self,
        x: Tensor,
        rope_freqs: Tensor | None = None,
    ) -> PrefixBlockKVCache:
        """Build a one-shot prefix KV cache from ``x`` (cross-attn prefix fill)."""
        k, v = self._project_kv(x, rope_freqs)
        return PrefixBlockKVCache.from_tensor(k, v, seq_dim=-3)

    def update_kv(
        self,
        x: Tensor,
        kv_cache: RollingBlockKVCache,
        kv_range: KVRange,
        rope_freqs: Tensor | None = None,
    ) -> RollingBlockKVCache:
        """Branchlessly write K/V computed from ``x`` into ``kv_cache`` at ``write_start``."""
        k, v = self._project_kv(x, rope_freqs)
        kv_cache.update_at(k, v, kv_range.write_start)
        return kv_cache

    def apply_kv(
        self,
        x: Tensor,
        kv_cache: BlockKVCache,
        kv_range: KVRange,
        rope_freqs: Tensor | None = None,
    ) -> Tensor:
        """Run attention using ``x`` as queries and ``kv_cache`` as K/V source.

        Args:
            x: Query tensor of shape [..., L, n * d].
            kv_cache: KV cache for inference.
            kv_range: Branchless write/read pair supplied by the caller; only
                ``valid_len`` (the read length) is consumed here.
            rope_freqs: RoPE frequencies, shape [L, 1, 1, d // 2].

        Returns:
            Output tensor of shape [..., L, n * d] after projection.
        """
        batch_shape = x.shape[:-2]
        batch_size = math.prod(batch_shape)
        L, D = x.shape[-2:]
        n, d = self.n_heads, self.head_dim
        assert n * d == D, "n * d must be equal to D"

        q = self.q_norm(self.q_proj(x).reshape(batch_size, L, n, d))
        if rope_freqs is not None:
            q = apply_rope_freqs(q, rope_freqs)

        cached_k = kv_cache.cached_k_at(kv_range.valid_len)
        cached_v = kv_cache.cached_v_at(kv_range.valid_len)

        out = self.attn_op(q, cached_k, cached_v)
        out = out.reshape(batch_shape + (L, n * d))
        return self.output_proj(out)


class SelfAttention(MultiHeadAttention):
    """Self-attention: queries and K/V are derived from the same ``x`` each step."""

    def initialize_cache(
        self,
        batch_size: int,
        chunk_size: int,
        window_size: int,
        sink_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> RollingBlockKVCache:
        """Initialize KV cache for streaming self-attention.

        Args:
            batch_size: Flattened batch size used by attention.
            chunk_size: Number of tokens appended per update step.
            window_size: Rolling-window capacity in tokens.
            sink_size: Sink-token capacity retained permanently.
            device: Device for cache tensors.
            dtype: Data type for cache tensors.

        Returns:
            An initialized ``RollingBlockKVCache``.
        """
        total_size = sink_size + window_size
        return RollingBlockKVCache(
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
        kv_cache: RollingBlockKVCache,
        kv_range: KVRange,
        rope_freqs: Tensor,
    ) -> Tensor:
        """Refresh the cache from ``x`` (``update_kv``) and run attention (``apply_kv``).

        ``kv_range`` carries the branchless write offset and read length.
        """
        self.update_kv(x, kv_cache, kv_range, rope_freqs)
        return self.apply_kv(x, kv_cache, kv_range, rope_freqs)


class CrossAttention(MultiHeadAttention):
    """Cross-attention: K/V live only in ``kv_cache``; ``forward`` does not refresh them."""

    def initialize_cache(
        self,
        context: Tensor,  # [..., L, D]
    ) -> PrefixBlockKVCache:
        """Initialize cross-attention cache from the provided context."""
        cache = self.compute_kv(context)
        return cache

    def forward(
        self,
        x: Tensor,
        kv_cache: PrefixBlockKVCache,
    ) -> Tensor:
        """Attend with queries from ``x`` against the prefix ``kv_cache``.

        The prefix cache is filled once via :meth:`compute_kv` and never updated,
        so its :attr:`PrefixBlockKVCache.range` is constant.
        """
        return self.apply_kv(x, kv_cache, kv_range=kv_cache.range)


@dataclass
class BlockCache:
    """Per-block cache container for self-attention and cross-attention."""

    self_attn: RollingBlockKVCache
    cross_attn: PrefixBlockKVCache

    def before_update(self, chunk_idx: int) -> None:
        """Run cache pre-update hook for the current chunk."""
        self.self_attn.before_update(chunk_idx)

    def after_update(self, chunk_idx: int) -> None:
        """Run cache post-update hook for the current chunk."""
        self.self_attn.after_update(chunk_idx)


class Block(nn.Module):
    """Cosmos transformer block with self-attn, cross-attn, and MLP branches."""

    def __init__(
        self,
        x_dim: int,
        context_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        cp_method: Literal["ring", "ulysses"] = "ring",
    ) -> None:
        super().__init__()
        self.x_dim = x_dim

        # Self-attention
        self.layer_norm_self_attn = nn.LayerNorm(
            x_dim, elementwise_affine=False, eps=1e-6
        )
        self.self_attn = SelfAttention(
            query_dim=x_dim,
            context_dim=None,
            n_heads=num_heads,
            head_dim=x_dim // num_heads,
            cp_method=cp_method,
        )

        # Cross-attention
        self.layer_norm_cross_attn = nn.LayerNorm(
            x_dim, elementwise_affine=False, eps=1e-6
        )
        self.cross_attn = CrossAttention(
            query_dim=x_dim,
            context_dim=context_dim,
            n_heads=num_heads,
            head_dim=x_dim // num_heads,
            cp_method=cp_method,
        )

        # MLP
        self.layer_norm_mlp = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = GPT2FeedForward(x_dim, int(x_dim * mlp_ratio))

        # AdaLN modulation
        self.use_adaln_lora = use_adaln_lora
        if use_adaln_lora:
            self.adaln_modulation_self_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_cross_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
        else:
            self.adaln_modulation_self_attn = nn.Sequential(
                nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False)
            )
            self.adaln_modulation_cross_attn = nn.Sequential(
                nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False)
            )
            self.adaln_modulation_mlp = nn.Sequential(
                nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False)
            )

    def set_context_parallel_group(
        self,
        cp_group: ProcessGroup | None,
    ) -> None:
        """Set hierarchical CP groups for self-attention.

        Args:
            cp_group: Context-parallel group.
        """
        self.self_attn.set_context_parallel_group(cp_group=cp_group)

    def initialize_cache(
        self,
        # self-attention
        chunk_size: int,
        window_size: int,
        sink_size: int,
        # cross-attention
        context: Tensor,  # [..., L, D]
    ) -> BlockCache:
        """Initialize per-branch caches for this transformer block."""
        batch_shape = context.shape[:-2]
        batch_size = math.prod(batch_shape)
        return BlockCache(
            self_attn=self.self_attn.initialize_cache(
                batch_size,
                chunk_size,
                window_size,
                sink_size,
                device=context.device,
                dtype=context.dtype,
            ),
            cross_attn=self.cross_attn.initialize_cache(context),
        )

    def forward(
        self,
        x: Tensor,
        emb: Tensor,
        cache: BlockCache,
        rope_freqs: Tensor,
        self_attn_range: KVRange,
        adaln_lora: Tensor | None = None,
    ) -> Tensor:
        """Run the full block update for one denoising step.

        Args:
            x: Input tensor with shape ``[..., L, D]``.
            emb: Timestep embedding with shape ``[..., L or 1, D]``.
            cache: KV cache container for this block.
            rope_freqs: RoPE frequencies with shape ``[L, 1, 1, D]``.
            self_attn_range: Branchless self-attn cache write/read pair,
                unpacked at :meth:`SelfAttention.forward`.
            adaln_lora: Optional AdaLN LoRA embedding with shape
                ``[..., L or 1, 3 * D]``.

        Returns:
            Updated hidden states with the same shape as ``x``.
        """
        assert emb.ndim == x.ndim, "emb and x must have the same number of dimensions"
        if self.use_adaln_lora:
            assert adaln_lora is not None, (
                "adaln_lora is required when use_adaln_lora is True"
            )
            shift_self, scale_self, gate_self = (
                self.adaln_modulation_self_attn(emb) + adaln_lora
            ).chunk(3, dim=-1)
            shift_cross, scale_cross, gate_cross = (
                self.adaln_modulation_cross_attn(emb) + adaln_lora
            ).chunk(3, dim=-1)
            shift_mlp, scale_mlp, gate_mlp = (
                self.adaln_modulation_mlp(emb) + adaln_lora
            ).chunk(3, dim=-1)
        else:
            shift_self, scale_self, gate_self = self.adaln_modulation_self_attn(
                emb
            ).chunk(3, dim=-1)
            shift_cross, scale_cross, gate_cross = self.adaln_modulation_cross_attn(
                emb
            ).chunk(3, dim=-1)
            shift_mlp, scale_mlp, gate_mlp = self.adaln_modulation_mlp(emb).chunk(
                3, dim=-1
            )

        # Self-attention
        normed_x = self.layer_norm_self_attn(x) * (1 + scale_self) + shift_self
        attn_out = self.self_attn(
            normed_x,
            rope_freqs=rope_freqs,
            kv_cache=cache.self_attn,
            kv_range=self_attn_range,
        )
        x = x + gate_self * attn_out

        # Cross-attention
        normed_x = self.layer_norm_cross_attn(x) * (1 + scale_cross) + shift_cross
        cross_out = self.cross_attn(
            normed_x,
            kv_cache=cache.cross_attn,
        )
        x = x + gate_cross * cross_out

        # MLP
        normed_x = self.layer_norm_mlp(x) * (1 + scale_mlp) + shift_mlp
        mlp_out = self.mlp(normed_x)
        x = x + gate_mlp * mlp_out

        return x
