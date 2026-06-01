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

"""Legacy (pre-split) wan-transformer cache layer for flashvsr / hy_worldplay.

Those two integrations were written against the **pre-refactor** self-contained
``BlockKVCache`` (``before_update`` / ``update`` / ``cached_k`` with an internal
cursor) and the old cache-construction chain. The core split
(:class:`RollingBlockKVCache` / :class:`PrefixBlockKVCache` +
:class:`KVCacheLifecycle`) changed that contract. Rather than migrate their
bespoke block-sparse / dual-branch-PRoPE forwards, this module provides
**same-named thin subclasses** of the live wan transformer that restore ONLY the
old self-attention *rolling* cache plus its construction + cascade.

What is reused unchanged from the live recipe (inherited, never copied):

- all model math: ``MultiHeadAttention`` projections / attention op, ``MLP``,
  RoPE, patchify / unpatchify, embeddings, ``Head``;
- the **cross-attention** prefix cache (it is immutable and already works on the
  new ``PrefixBlockKVCache`` / ``CrossAttnCache``).

flashvsr / hy_worldplay repoint their imports here -- import lines only; their
class bodies are unchanged. Bodies below are frozen copies of the pre-refactor
recipe (``cf0eb1dd``); do not add features here -- migrate the consumers instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor

from flashdreams.core.attention.legacy_kvcache import BlockKVCache
from flashdreams.core.attention.rope import (
    KVCacheRelativeRotaryPositionEmbedding3D,
    RotaryPositionEmbedding3D,
    apply_rope_freqs,
)
from flashdreams.infra.cuda_graph import CUDAGraphWrapper
from flashdreams.infra.diffusion.transformer import TransformerAutoregressiveCache
from flashdreams.recipes.wan.autoencoder.i2v import I2VCtrl

# Unchanged symbols re-exported so consumers can import everything from here.
from flashdreams.recipes.wan.transformer.impl.modules import (
    Block as _NewBlock,
)
from flashdreams.recipes.wan.transformer.impl.modules import (
    CrossAttention,
    CrossAttnCache,
    sinusoidal_embedding_1d,
)
from flashdreams.recipes.wan.transformer.impl.modules import (
    MultiHeadAttention as _NewMultiHeadAttention,
)
from flashdreams.recipes.wan.transformer.impl.network import (
    WanDiTNetwork as _NewWanDiTNetwork,
)
from flashdreams.recipes.wan.transformer.impl.network import (
    WanDiTNetwork1pt3BConfig,
    WanDiTNetwork14BConfig,
    WanDiTNetworkConfig,
    WanDiTNetworkTI2V5BConfig,
)
from flashdreams.recipes.wan.transformer.wan21 import (
    Wan21Transformer as _NewWan21Transformer,
)
from flashdreams.recipes.wan.transformer.wan21 import (
    Wan21TransformerConfig,
)

__all__ = [
    "BlockKVCache",
    "SelfAttention",
    "BlockCache",
    "Block",
    "CrossAttention",
    "CrossAttnCache",
    "MultiHeadAttention",
    "sinusoidal_embedding_1d",
    "WanDiTNetworkCache",
    "WanDiTNetwork",
    "WanDiTNetworkConfig",
    "WanDiTNetwork1pt3BConfig",
    "WanDiTNetwork14BConfig",
    "WanDiTNetworkTI2V5BConfig",
    "Wan21TransformerCache",
    "Wan21Transformer",
    "Wan21TransformerConfig",
]


class MultiHeadAttention(_NewMultiHeadAttention):
    """Legacy multi-head attention: pre-split self-contained ``BlockKVCache`` contract.

    Frozen copy (69e173c1) of the cache-facing methods the live
    :class:`MultiHeadAttention` replaced when it moved to the branchless
    ``RollingBlockKVCache`` + :class:`KVRange` API. The projections, attention
    op, RoPE and ``__init__`` are inherited from the live module unchanged; only
    the cursor-based cache read/write path (``before_update`` / ``update`` /
    ``cached_k`` / ``size`` / ``write_end``) is restored here verbatim.
    """

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
            rope_freqs: Optional RoPE frequencies for K before
                K cache write, shape ``[L, 1, 1, d]``.

        Returns:
            Updated ``BlockKVCache`` containing keys and values.
        """
        batch_shape = context.shape[:-2]
        batch_size = math.prod(batch_shape)
        L, D = context.shape[-2:]
        n, d = self.n_heads, self.head_dim

        k = self.norm_k(self.k(context)).reshape(batch_size, L, n, d)
        v = self.v(context).reshape(batch_size, L, n, d)
        if rope_freqs is not None and self.apply_rope_before_kvcache:
            k = apply_rope_freqs(k, rope_freqs, interleaved=True)

        if kv_cache is None:
            kv_cache = BlockKVCache.from_tensor(k, v, seq_dim=-3)
        else:
            kv_cache.update(k, v)
        return kv_cache

    def compute_kv(  # type: ignore[override]
        self,
        x: Tensor,
        rope_freqs: Tensor | None = None,
    ) -> BlockKVCache:
        """Build a new KV cache from ``x``."""
        return self._compute_or_update_kv_cache(x, None, rope_freqs)

    def update_kv(  # type: ignore[override]
        self,
        x: Tensor,
        kv_cache: BlockKVCache,
        rope_freqs: Tensor | None = None,
    ) -> BlockKVCache:
        """Append K/V computed from ``x`` into an existing ``kv_cache``."""
        return self._compute_or_update_kv_cache(x, kv_cache, rope_freqs)

    def apply_kv(  # type: ignore[override]
        self,
        x: Tensor,
        kv_cache: BlockKVCache,
        rope_freqs_q: Tensor | None = None,
        rope_freqs_k: Tensor | None = None,
    ) -> Tensor:
        """Run attention with queries from ``x`` against cached K/V.

        Args:
            x: Query tokens, shape ``[..., L, query_dim]``.
            kv_cache: KV cache used as attention context.
            rope_freqs_q: Optional RoPE frequencies for Q, shape
                ``[L, 1, 1, d]``.
            rope_freqs_k: Optional KV-cache-relative RoPE frequencies for
                cached K, shape ``[S_cache, 1, 1, d]``. Only used when
                K is stored without standard RoPE before the KV cache write.

        Returns:
            Output-projected attention, shape ``[..., L, query_dim]``.
        """
        batch_shape = x.shape[:-2]
        batch_size = math.prod(batch_shape)
        L, D = x.shape[-2:]
        n, d = self.n_heads, self.head_dim
        assert n * d == D, "n * d must be equal to D"

        q = self.norm_q(self.q(x)).reshape(batch_size, L, n, d)
        cached_k = kv_cache.cached_k()
        if rope_freqs_q is not None:
            q = apply_rope_freqs(q, rope_freqs_q, interleaved=True)
        if not self.apply_rope_before_kvcache:
            assert rope_freqs_k is not None, (
                "KV-cache-relative RoPE requires rope_freqs_k for cached K"
            )
            cached_k = cached_k.clone()
            cached_k = apply_rope_freqs(cached_k, rope_freqs_k, interleaved=True)

        cached_v = kv_cache.cached_v()

        out = self.attn_op(q, cached_k, cached_v)
        out = out.reshape(batch_shape + (L, n * d))
        return self.o(out)

    def _slice_rope_freqs(  # type: ignore[override]
        self,
        rope_freqs: Tensor | None,
        kv_cache: BlockKVCache,
    ) -> tuple[Tensor | None, Tensor | None]:
        """Select Q/K RoPE frequencies for standard or cache-relative mode."""
        if rope_freqs is None:
            return None, None
        if self.apply_rope_before_kvcache:
            return rope_freqs, rope_freqs

        write_end = kv_cache.write_end
        write_start = write_end - kv_cache.chunk_size
        rope_freqs_q = rope_freqs[write_start:write_end]
        rope_freqs_k = rope_freqs[: kv_cache.size]
        return rope_freqs_q, rope_freqs_k

    def forward(  # type: ignore[override]
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
            rope_freqs: Optional RoPE frequencies. Standard mode receives current-chunk
                frequencies. KV-cache-relative mode receives frequencies relative to the KV cache
                and applies the K slice on cache read.
            update_kv_cache: If False, only run attention against the existing cache.

        Returns:
            Projected output tensor of shape [..., L, query_dim].
        """
        rope_freqs_q, rope_freqs_k = self._slice_rope_freqs(rope_freqs, kv_cache)
        if update_kv_cache:
            kv_cache = self.update_kv(x, kv_cache, rope_freqs_k)
        return self.apply_kv(x, kv_cache, rope_freqs_q, rope_freqs_k)


class SelfAttention(MultiHeadAttention):
    """Self-attention that always refreshes K/V cache from current ``x`` (pre-split)."""

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
class BlockCache:
    """Per-block cache: legacy rolling self-attn + (new) immutable prefix cross-attn."""

    self_attn: BlockKVCache
    cross_attn: CrossAttnCache

    def before_update(self, chunk_idx: int) -> None:
        """Run pre-update hook for the self-attention cache."""
        self.self_attn.before_update(chunk_idx)

    def after_update(self, chunk_idx: int) -> None:
        """Run post-update hook for the self-attention cache."""
        self.self_attn.after_update(chunk_idx)


class Block(_NewBlock):
    """Legacy block: old-signature ``initialize_cache`` / ``forward``; cross-attn stays new.

    Self-attn cache construction goes through ``self.self_attn.initialize_cache``
    so consumer subclasses that swap in their own self-attn (flashvsr's
    ``SparseSelfAttention``, hy's PRoPE self-attn) keep their side-effect setup.
    """

    self_attn: SelfAttention  # type: ignore[assignment]
    """Legacy cursor-based self-attention (the live base builds a
    ``RollingBlockKVCache`` variant; we install the pre-split one below)."""

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        i2v: bool = False,
        apply_rope_before_kvcache: bool = True,
        cp_method: Literal["ring", "ulysses"] = "ring",
    ) -> None:
        super().__init__(
            dim=dim,
            ffn_dim=ffn_dim,
            num_heads=num_heads,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
            i2v=i2v,
            apply_rope_before_kvcache=apply_rope_before_kvcache,
            cp_method=cp_method,
        )
        # Replace the live RollingBlockKVCache self-attn with the legacy
        # cursor-based one so this block's initialize_cache / forward speak
        # the pre-split BlockKVCache contract. Parameter names are identical
        # (``self_attn.{q,k,v,o,norm_q,norm_k}``) so checkpoints load drop-in.
        self.self_attn = SelfAttention(
            query_dim=dim,
            n_heads=num_heads,
            head_dim=dim // num_heads,
            eps=eps,
            apply_rope_before_kvcache=apply_rope_before_kvcache,
            cp_method=cp_method,
        )

    def initialize_cache(  # type: ignore[override]
        self,
        chunk_size: int,
        window_size: int,
        sink_size: int,
        context_text: Tensor,
        context_img: Tensor | None = None,
    ) -> BlockCache:
        """Initialize per-branch caches for this transformer block."""
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

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        e: Tensor,
        cache: BlockCache,
        rope_freqs: Tensor,
    ) -> Tensor:
        """Run one transformer block update (pre-split, no ``self_attn_range``)."""
        assert self._parameters_updated_after_loading_checkpoint, (
            "We expect to have called update_parameters_after_loading_checkpoint() "
            "before running the forward pass"
        )
        e_chunks = [c.squeeze(-2) for c in (self.modulation + e).chunk(6, dim=-2)]

        y = self.norm1(x) * (1 + e_chunks[1]) + e_chunks[0]  # [..., L, D]
        y = self.self_attn(
            y,
            rope_freqs=rope_freqs,
            kv_cache=cache.self_attn,
        )
        x = x + (y * e_chunks[2])  # [..., L, D]

        x = x + self.cross_attn(
            self.norm3(x),
            kv_cache=cache.cross_attn,
        )
        y = self.norm2(x) * (1 + e_chunks[4]) + e_chunks[3]  # [..., L, D]
        y = self.ffn(y)
        x = x + (y * e_chunks[5])  # [..., L, D]
        return x


@dataclass
class WanDiTNetworkCache:
    """Cache container for all transformer blocks (legacy cascade)."""

    block_caches: list[BlockCache]
    """Per-transformer-block KV cache, indexed by block position."""

    def __getitem__(self, index: int) -> BlockCache:
        """Get cache for a specific block."""
        return self.block_caches[index]

    def before_update(self, chunk_idx: int) -> None:
        """Run pre-update hooks for all block caches."""
        for block_cache in self.block_caches:
            block_cache.before_update(chunk_idx)

    def after_update(self, chunk_idx: int) -> None:
        """Run post-update hooks for all block caches."""
        for block_cache in self.block_caches:
            block_cache.after_update(chunk_idx)


class WanDiTNetwork(_NewWanDiTNetwork):
    """Legacy network: builds legacy blocks, old-signature ``initialize_cache`` / ``forward``."""

    def _build_block(self, layer_idx: int) -> Block:
        """Construct one legacy (cursor-based self-attn) transformer block."""
        return Block(
            dim=self.dim,
            ffn_dim=self.ffn_dim,
            num_heads=self.num_heads,
            cross_attn_norm=self.cross_attn_norm,
            eps=self.eps,
            i2v=self.cross_attn_enable_img,
            apply_rope_before_kvcache=self.apply_rope_before_kvcache,
            cp_method=self.cp_method,
        )

    def initialize_cache(  # type: ignore[override]
        self,
        chunk_size: int,
        window_size: int,
        sink_size: int,
        text_embeddings: Tensor,
        img_embeddings: Tensor | None = None,
    ) -> WanDiTNetworkCache:
        """Initialize block caches from text/image context embeddings."""
        assert text_embeddings.shape[-2] == self.text_len
        context_text = self.text_embedding(text_embeddings)
        if self.cross_attn_enable_img:
            assert img_embeddings is not None, (
                "img_embeddings is required when cross_attn_enable_img=True"
            )
            context_img = self.img_emb(img_embeddings)
        else:
            context_img = None

        block_caches: list[BlockCache] = []
        for block in self.blocks:
            assert isinstance(block, Block)
            block_caches.append(
                block.initialize_cache(
                    chunk_size, window_size, sink_size, context_text, context_img
                )
            )
        return WanDiTNetworkCache(block_caches=block_caches)

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        timesteps: Tensor,
        cache: WanDiTNetworkCache,
        rope_freqs: Tensor,
        current_chunk_idx: int = 0,
        eager_mode: bool = True,
        block_extra_kwargs: dict[str, Any] = {},
    ) -> Tensor:
        """Run one denoising forward pass (pre-split eager cache cascade)."""
        assert self._parameters_updated_after_loading_checkpoint, (
            "We expect to have called update_parameters_after_loading_checkpoint() "
            "after loading the checkpoint"
        )
        batch_shape = x.shape[:-2]
        L = x.shape[-2]

        # Patch embedding
        if self.patch_embedding_type == "linear":
            x = self.patch_embedding(x)  # (..., L, D)
        elif self.patch_embedding_type == "conv3d":
            _weight = self.patch_embedding.weight.reshape(self.dim, -1)
            _bias = self.patch_embedding.bias
            x = torch.nn.functional.linear(x, _weight, _bias)
        else:
            raise ValueError(
                f"Invalid patch embedding type: {self.patch_embedding_type}"
            )

        per_token_timestep = (
            timesteps.ndim > len(batch_shape) and timesteps.shape[-1] == L
        )
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timesteps).type_as(x)
        )
        e0 = self.time_projection(e).unflatten(-1, (6, self.dim))

        if per_token_timestep:
            block_e_shape = batch_shape + (L, 6, self.dim)
            head_e = torch.broadcast_to(e, batch_shape + (L, self.dim)).unsqueeze(-2)
        else:
            block_e_shape = batch_shape + (6, self.dim)
            head_e = torch.broadcast_to(e, batch_shape + (self.dim,)).unsqueeze(-2)
        block_e = torch.broadcast_to(e0, block_e_shape)

        # Transformer blocks. The cache carries its own cursor, so eager mode
        # drives before_update / after_update here (the owning cache also does
        # so via its start / finalize cascade when eager_mode=False).
        if eager_mode:
            cache.before_update(current_chunk_idx)
        for block_idx, block in enumerate(self.blocks):
            assert isinstance(block, Block)
            x = block(
                x=x,
                e=block_e,
                rope_freqs=rope_freqs,
                cache=cache[block_idx],
                **block_extra_kwargs,
            )
        if eager_mode:
            cache.after_update(current_chunk_idx)

        # Final head
        x = self.head(x, head_e)  # (..., L, D)
        return x


@dataclass(kw_only=True)
class Wan21TransformerCache(TransformerAutoregressiveCache):
    """Per-rollout AR cache for the Wan 2.1 transformer (legacy, no lifecycle).

    Holds an always-present conditional network cache and an optional
    unconditional one for classifier-free guidance (``None`` disables CFG).
    The shared cursor lives per-cache (old ``before_update`` / ``after_update``
    cascade) rather than on a :class:`KVCacheLifecycle`.
    """

    network_cache: WanDiTNetworkCache
    """Conditional per-block KV / cross-attention caches."""

    network_cache_uncond: WanDiTNetworkCache | None = None
    """Unconditional caches; ``None`` disables CFG."""

    rope_adapter: RotaryPositionEmbedding3D | KVCacheRelativeRotaryPositionEmbedding3D
    """3D RoPE adapter for self-attention position frequencies."""

    rope_freqs: Tensor | None = None
    """Self-attention RoPE frequencies for the current AR step (recomputed in
    :meth:`start`, reused across cond/uncond and scheduler steps)."""

    autoregressive_index: int = -1
    """Current AR step index, set by :meth:`start`."""

    def start(self, autoregressive_index: int) -> None:
        self.rope_freqs = self.rope_adapter.shift_t(autoregressive_index)
        self.autoregressive_index = autoregressive_index
        self.network_cache.before_update(autoregressive_index)
        if self.network_cache_uncond is not None:
            self.network_cache_uncond.before_update(autoregressive_index)

    def finalize(self, autoregressive_index: int) -> None:
        self.network_cache.after_update(autoregressive_index)
        if self.network_cache_uncond is not None:
            self.network_cache_uncond.after_update(autoregressive_index)


class Wan21Transformer(_NewWan21Transformer):
    """Legacy transformer: old cache-construction chain (no lifecycle / self_attn_range)."""

    network: WanDiTNetwork  # type: ignore[assignment]
    """Legacy DiT network (the live base types this as the post-split network)."""

    @torch.no_grad()
    def _build_network_cache(  # type: ignore[override]
        self,
        *,
        text_embeddings: Tensor,
        image_embeddings: Tensor | None = None,
    ) -> WanDiTNetworkCache:
        """Build one network cache (cond or uncond branch)."""
        assert self._output_height is not None and self._output_width is not None, (
            "_build_network_cache called before height/width were stashed."
        )
        cfg = self.config
        kt, kh, kw = cfg.network.patch_size
        pHW = (self._output_height // kh) * (self._output_width // kw)
        cp_size = self._cp_size
        chunk_size = self.latent_shape[-2]  # already CP-divided
        window_size_t = cfg.window_size_t // kt
        sink_size_t = cfg.sink_size_t // kt
        assert (window_size_t * pHW) % cp_size == 0, (
            f"window_size_t * frame_token_count ({window_size_t * pHW}) must be "
            f"divisible by cp_size ({cp_size})"
        )
        assert (sink_size_t * pHW) % cp_size == 0, (
            f"sink_size_t * frame_token_count ({sink_size_t * pHW}) must be "
            f"divisible by cp_size ({cp_size})"
        )
        window_size = (window_size_t * pHW) // cp_size
        sink_size = (sink_size_t * pHW) // cp_size
        return self.network.initialize_cache(
            chunk_size=chunk_size,
            window_size=window_size,
            sink_size=sink_size,
            text_embeddings=text_embeddings,
            img_embeddings=image_embeddings,
        )

    @torch.no_grad()
    def initialize_autoregressive_cache(  # type: ignore[override]
        self,
        *,
        height: int,
        width: int,
        text_embeddings: Tensor,
        image_embeddings: Tensor | None = None,
        negative_text_embeddings: Tensor | None = None,
        **_unused: Any,
    ) -> Wan21TransformerCache:
        """Build a seeded transformer cache for a new rollout (legacy, no lifecycle)."""
        cfg = self.config
        kt, kh, kw = cfg.network.patch_size
        assert height % kh == 0 and width % kw == 0, (
            f"(height, width) = ({height}, {width}) must be divisible by "
            f"patch_size={cfg.network.patch_size[1:]}."
        )
        self._output_height = height
        self._output_width = width
        total_tokens = (cfg.len_t // kt) * (height // kh) * (width // kw)
        assert total_tokens % self._cp_size == 0, (
            f"Wan token length ({total_tokens} from len_t={cfg.len_t}, "
            f"height={height}, width={width}, "
            f"patch_size={cfg.network.patch_size}) must be divisible by "
            f"cp_size={self._cp_size}"
        )

        network_cache = self._build_network_cache(
            text_embeddings=text_embeddings,
            image_embeddings=image_embeddings,
        )
        network_cache_uncond: WanDiTNetworkCache | None = None
        if self.config.guidance_scale > 1.0:
            assert negative_text_embeddings is not None, (
                f"WanTransformerConfig.guidance_scale="
                f"{self.config.guidance_scale} > 1.0 requires "
                f"negative_text_embeddings."
            )
            network_cache_uncond = self._build_network_cache(
                text_embeddings=negative_text_embeddings,
                image_embeddings=image_embeddings,
            )

        head_dim = self.config.network.dim // self.config.network.num_heads
        rope_kwargs: dict[str, Any] = {
            "len_t": cfg.len_t // kt,
            "len_h": height // kh,
            "len_w": width // kw,
            "head_dim": head_dim,
            "h_extrapolation_ratio": self.config.h_extrapolation_ratio,
            "w_extrapolation_ratio": self.config.w_extrapolation_ratio,
            "interleaved": True,
            "device": self.device,
        }
        if cfg.network.apply_rope_before_kvcache:
            rope_adapter = RotaryPositionEmbedding3D(**rope_kwargs)
        else:
            rope_kwargs["sink_size_t"] = cfg.sink_size_t // kt
            rope_kwargs["window_size_t"] = cfg.window_size_t // kt
            rope_adapter = KVCacheRelativeRotaryPositionEmbedding3D(**rope_kwargs)
        rope_adapter.set_context_parallel_group(cp_group=self._cp_group)

        # Reset any prior CUDA graph: it refers to slot pointers from the
        # previous cache, which the new cache invalidates.
        if self._use_cuda_graph:
            assert isinstance(self._network_call, CUDAGraphWrapper)
            self._network_call.reset()
            assert isinstance(self._network_call_uncond, CUDAGraphWrapper)
            self._network_call_uncond.reset()

        return Wan21TransformerCache(
            network_cache=network_cache,
            network_cache_uncond=network_cache_uncond,
            rope_adapter=rope_adapter,
        )

    def _predict_flow(  # type: ignore[override]
        self,
        network_input: Tensor,
        timestep: Tensor,
        cache: Wan21TransformerCache,
        autoregressive_index: int,
        network_extra_kwargs: dict[str, Any],
        *,
        uncond: bool,
    ) -> Tensor:
        network_cache = cache.network_cache_uncond if uncond else cache.network_cache
        assert network_cache is not None, (
            "uncond=True requires cache.network_cache_uncond, but it is None "
            "(CFG was not enabled at cache build time)."
        )
        assert cache.rope_freqs is not None, (
            "Wan21TransformerCache.start() must populate rope_freqs before predict_flow"
        )
        return self._select_network(autoregressive_index, uncond=uncond)(
            x=network_input,
            timesteps=timestep,
            cache=network_cache,
            rope_freqs=cache.rope_freqs,
            current_chunk_idx=autoregressive_index,
            eager_mode=False,
            **network_extra_kwargs,
        )

    def predict_flow(  # type: ignore[override]
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        cache: Wan21TransformerCache,
        input: I2VCtrl | None = None,
        network_extra_kwargs: dict[str, Any] | None = None,
    ) -> Tensor:
        """Predict the flow for one denoising step (legacy cache type).

        Mirrors :meth:`Wan21Transformer.predict_flow` but is typed against the
        legacy :class:`Wan21TransformerCache` so consumers (hy_worldplay) can
        thread their own legacy-derived cache through ``super().predict_flow``.
        """
        ar_idx = cache.autoregressive_index
        assert ar_idx >= 0, (
            "Wan21TransformerCache.start(autoregressive_index) must be called "
            "before predict_flow (DiffusionModel.generate handles this)."
        )
        network_extra_kwargs = network_extra_kwargs or {}
        network_input = self._build_network_input(noisy_latent, input)
        timestep = self._maybe_build_per_token_timestep(
            timestep=timestep, input=input, autoregressive_index=ar_idx
        )

        flow_cond = self._predict_flow(
            network_input=network_input,
            timestep=timestep,
            cache=cache,
            autoregressive_index=ar_idx,
            network_extra_kwargs=network_extra_kwargs,
            uncond=False,
        )
        if cache.network_cache_uncond is None:
            return flow_cond
        flow_uncond = self._predict_flow(
            network_input=network_input,
            timestep=timestep,
            cache=cache,
            autoregressive_index=ar_idx,
            network_extra_kwargs=network_extra_kwargs,
            uncond=True,
        )
        return flow_uncond + self.config.guidance_scale * (flow_cond - flow_uncond)

    def finalize_kv_cache(  # type: ignore[override]
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        cache: Wan21TransformerCache,
        input: Any = None,
    ) -> None:
        """Advance the legacy AR cache via a single discarded ``predict_flow``."""
        _ = self.predict_flow(noisy_latent, timestep, cache, input)
