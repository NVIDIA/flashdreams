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

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import Tensor
from torch.distributed import ProcessGroup

from flashdreams.core.attention import BlockKVCache, RingAttention
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
            x: Input tensor of shape (B, ..., D).
            emb: Conditioning embedding of shape (B, D).
            adaln_lora: Optional LoRA tensor of shape (B, 3 * D).

        Returns:
            Output tensor of shape (B, ..., D') where D' = patch_dim.
        """
        batch_size, *ellipsis_dims, hidden_dim = x.shape
        assert emb.shape == (batch_size, hidden_dim)

        emb = emb.reshape(batch_size, *([1] * len(ellipsis_dims)), hidden_dim)

        if self.use_adaln_lora:
            assert adaln_lora is not None and adaln_lora.shape == (
                batch_size,
                3 * hidden_dim,
            )
            adaln_lora = adaln_lora.reshape(
                batch_size, *([1] * len(ellipsis_dims)), 3 * hidden_dim
            )
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
        """Compute K/V from ``context`` and optionally merge into ``kv_cache``.

        Args:
            context: Tensor of shape [..., L, n * d] used to compute K/V.
            kv_cache: Existing cache to update, or None to allocate from this step.
            rope_freqs: RoPE frequencies, shape [L, 1, 1, d // 2].

        Returns:
            KV cache containing the merged keys and values.
        """
        batch_shape = context.shape[:-2]
        batch_size = math.prod(batch_shape)
        L, D = context.shape[-2:]
        n, d = self.n_heads, self.head_dim

        k = self.k_norm(self.k_proj(context).reshape(batch_size, L, n, d))
        v = self.v_proj(context).reshape(batch_size, L, n, d)
        if rope_freqs is not None:
            k = apply_rope_freqs(k, rope_freqs)

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
        """Run attention using ``x`` as queries and ``kv_cache`` as K/V source.

        Args:
            x: Query tensor of shape [..., L, n * d].
            kv_cache: KV cache for inference.
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

        cached_k = kv_cache.cached_k()
        cached_v = kv_cache.cached_v()

        out = self.attn_op(q, cached_k, cached_v)
        out = out.reshape(batch_shape + (L, n * d))
        return self.output_proj(out)

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
    """Self-attention: queries and K/V are derived from the same ``x`` each step."""

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
        """Same as base ``forward`` with ``update_kv_cache=True``."""
        return super().forward(x, kv_cache, rope_freqs=rope_freqs, update_kv_cache=True)


class CrossAttention(MultiHeadAttention):
    """Cross-attention: K/V live only in ``kv_cache``; ``forward`` does not refresh them."""

    def initialize_cache(
        self,
        context: Tensor,  # [B, V, L, D]
    ) -> BlockKVCache:
        """Initialize cross-attention cache from the provided context."""
        cache = self.compute_kv(context)
        return cache

    def forward(
        self,
        x: Tensor,
        kv_cache: BlockKVCache,
    ) -> Tensor:
        """Attend with queries from ``x``; populate or roll ``kv_cache`` outside this call."""
        return super().forward(x, kv_cache, rope_freqs=None, update_kv_cache=False)


@dataclass
class BlockCache:
    """Per-block cache container for self-attention and cross-attention."""

    self_attn: BlockKVCache
    cross_attn: BlockKVCache

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
        enable_cross_view_attn: bool = False,
    ) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.enable_cross_view_attn = enable_cross_view_attn

        # Self-attention
        self.layer_norm_self_attn = nn.LayerNorm(
            x_dim, elementwise_affine=False, eps=1e-6
        )
        self.self_attn = SelfAttention(
            query_dim=x_dim,
            context_dim=None,
            n_heads=num_heads,
            head_dim=x_dim // num_heads,
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

        if enable_cross_view_attn:
            # no modulation so we set elementwise_affine=True
            self.layer_norm_cross_view_attn = nn.LayerNorm(
                x_dim, elementwise_affine=True, eps=1e-6
            )
            # dense cross view attention
            self.cross_view_attn = CrossAttention(
                query_dim=x_dim,
                context_dim=x_dim,
                n_heads=num_heads,
                head_dim=x_dim // num_heads,
            )

    def set_context_parallel_group(
        self,
        self_attn_group: ProcessGroup | None,
        cross_view_attn_group: ProcessGroup | None = None,
    ) -> None:
        """Set hierarchical CP groups for self-attention and cross-view attention.

        Args:
            self_attn_group: Group for ranks processing the same view (T-axis gathering).
            cross_view_attn_group: Group for ranks at the same T slice (V-axis gathering).
        """
        # Self-attention uses self_attn_group (for T gathering)
        self.self_attn.set_context_parallel_group(cp_group=self_attn_group)
        # Cross-view attention uses cross_view_attn_group (for V gathering)
        if self.enable_cross_view_attn:
            self.cross_view_attn.set_context_parallel_group(
                cp_group=cross_view_attn_group
            )

    def initialize_cache(
        self,
        # self-attention
        chunk_size: int,
        window_size: int,
        sink_size: int,
        # cross-attention
        context: Tensor,  # [B, V, L, D]
    ) -> BlockCache:
        """Initialize per-branch caches for this transformer block."""
        device = context.device
        dtype = context.dtype
        batch_size = context.shape[0]
        num_views = context.shape[1]
        self_attn_batch_size = batch_size * num_views
        return BlockCache(
            self_attn=self.self_attn.initialize_cache(
                self_attn_batch_size,
                chunk_size,
                window_size,
                sink_size,
                device=device,
                dtype=dtype,
            ),
            cross_attn=self.cross_attn.initialize_cache(context),
        )

    def forward(
        self,
        x: Tensor,
        emb: Tensor,
        cache: BlockCache,
        rope_freqs: Tensor,
        adaln_lora: Tensor | None = None,
        view_embedding_proj: Tensor | None = None,
    ) -> Tensor:
        """Run the full block update for one denoising step.

        Args:
            x: Input tensor with shape [B, V, T, HW, D] or [B, V, L, D].
            emb: Timestep embedding with shape [B, D].
            cache: KV cache container for this block.
            rope_freqs: RoPE frequencies with shape [L, 1, 1, D].
            adaln_lora: Optional AdaLN LoRA embedding with shape [B, 3D].
            view_embedding_proj: Optional per-view modulation tensor [B, V, 9D].

        Returns:
            Updated hidden states with the same shape as ``x``.
        """
        if x.ndim == 5:
            B, V, T, HW, D = x.shape
            L = T * HW
            x = x.reshape(B, V, L, D)
        else:
            assert x.ndim == 4, "x must be a 4D tensor"
            B, V, L, D = x.shape
            # If x passes in as a 4D tensor, we don't know T and HW.
            T = HW = None

        # Reshape embeddings to be broadcastable with x.
        emb = emb.reshape(B, 1, 1, D)

        # Compute AdaLN modulation
        if self.use_adaln_lora:
            assert adaln_lora is not None, (
                "adaln_lora is required when use_adaln_lora is True"
            )
            adaln_lora = adaln_lora.reshape(B, 1, 1, 3 * D)
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

        if self.enable_cross_view_attn:
            assert view_embedding_proj is not None
            (
                view_shift_self,
                view_scale_self,
                view_gate_self,
                view_shift_cross,
                view_scale_cross,
                view_gate_cross,
                view_shift_mlp,
                view_scale_mlp,
                view_gate_mlp,
            ) = view_embedding_proj.chunk(9, dim=-1)

            def expand_view_mod(v_mod: Tensor) -> Tensor:
                return v_mod.reshape(B, V, 1, D)

            shift_self = shift_self + expand_view_mod(view_shift_self)
            scale_self = scale_self + expand_view_mod(view_scale_self)
            gate_self = gate_self + expand_view_mod(view_gate_self)

            shift_cross = shift_cross + expand_view_mod(view_shift_cross)
            scale_cross = scale_cross + expand_view_mod(view_scale_cross)
            gate_cross = gate_cross + expand_view_mod(view_gate_cross)

            shift_mlp = shift_mlp + expand_view_mod(view_shift_mlp)
            scale_mlp = scale_mlp + expand_view_mod(view_scale_mlp)
            gate_mlp = gate_mlp + expand_view_mod(view_gate_mlp)

        # Self-attention (API aligned with Attention.forward)
        normed_x = self.layer_norm_self_attn(x) * (1 + scale_self) + shift_self
        attn_out = self.self_attn(
            normed_x,
            rope_freqs=rope_freqs,
            kv_cache=cache.self_attn,
        ).reshape_as(normed_x)
        x = x + gate_self * attn_out

        # Cross-view attention: dense
        if self.enable_cross_view_attn:
            assert T is not None and HW is not None, (
                "T and HW must be available (x should be a 5D tensor) when cross-view attention is enabled"
            )
            normed_x_cv = self.layer_norm_cross_view_attn(x)
            x_cv = rearrange(normed_x_cv, "b v (t hw) d -> b t v hw d", t=T, hw=HW)
            if self.cross_view_attn.is_context_parallel_enabled():
                # When cross-view attention is CP-enabled, assume views are split
                # across GPUs in rank order. For 4 views on 2 GPUs, that is
                # [0, 1] on one rank group and [2, 3] on the other.
                if V == 1:
                    # If CP size equals number of views, each GPU processes one view.
                    # Ring attention gathers all K/V, so local context can remain unexpanded.
                    x_context = x_cv
                    # Effectively equivalent to the repeat() branch when V == 1.
                    # x_context = repeat(x, f"b t v hw d -> b t v2 (v hw) d", v2=V)
                else:
                    # If CP size is smaller than number of views, each GPU handles multiple
                    # views locally before global K/V gathering.
                    x_context = repeat(x_cv, "b t v hw d -> b t v2 (v hw) d", v2=V)
            else:
                # Without CP, repeat context so each view attends over all views.
                x_context = repeat(x_cv, "b t v hw d -> b t v2 (v hw) d", v2=V)
            cross_view_attn_kv_cache = self.cross_view_attn.compute_kv(x_context)
            cv_out = self.cross_view_attn(x_cv, kv_cache=cross_view_attn_kv_cache)
            cv_out = rearrange(cv_out, "b t v hw d -> b v (t hw) d")
            x = x + cv_out

        # Cross-attention
        normed_x = self.layer_norm_cross_attn(x) * (1 + scale_cross) + shift_cross
        cross_out = self.cross_attn(
            normed_x,
            kv_cache=cache.cross_attn,
        ).reshape_as(normed_x)
        x = x + gate_cross * cross_out

        # MLP
        normed_x = self.layer_norm_mlp(x) * (1 + scale_mlp) + shift_mlp
        mlp_out = self.mlp(normed_x)
        x = x + gate_mlp * mlp_out

        # reshape back to 5D if needed
        if T is not None and HW is not None:
            x = x.reshape(B, V, T, HW, D)  # [B, V, T, HW, D]

        return x
