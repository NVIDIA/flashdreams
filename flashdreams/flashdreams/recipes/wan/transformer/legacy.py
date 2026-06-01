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
from typing import Any

import torch
from torch import Tensor

from flashdreams.core.attention.legacy_kvcache import BlockKVCache
from flashdreams.core.attention.rope import (
    KVCacheRelativeRotaryPositionEmbedding3D,
    RotaryPositionEmbedding3D,
)
from flashdreams.infra.cuda_graph import CUDAGraphWrapper
from flashdreams.infra.diffusion.transformer import TransformerAutoregressiveCache

# Unchanged symbols re-exported so consumers can import everything from here.
from flashdreams.recipes.wan.transformer.impl.modules import (
    Block as _NewBlock,
)
from flashdreams.recipes.wan.transformer.impl.modules import (
    CrossAttention,
    CrossAttnCache,
    MultiHeadAttention,
    sinusoidal_embedding_1d,
)
from flashdreams.recipes.wan.transformer.impl.modules import (
    SelfAttention as _NewSelfAttention,
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


class SelfAttention(_NewSelfAttention):
    """Legacy self-attention: builds the old self-contained rolling ``BlockKVCache``."""

    def initialize_cache(  # type: ignore[override]
        self,
        batch_size: int,
        chunk_size: int,
        window_size: int,
        sink_size: int,
        device: Any,
        dtype: Any,
    ) -> BlockKVCache:
        """Initialize the legacy rolling KV cache for streaming self-attention."""
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
    """Legacy block: old-signature ``initialize_cache``; cross-attn stays on the new cache.

    Self-attn cache construction goes through ``self.self_attn.initialize_cache``
    so consumer subclasses that swap in their own self-attn (flashvsr's
    ``SparseSelfAttention``, hy's PRoPE self-attn) keep their side-effect setup.
    """

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
    """Legacy network: old-signature ``initialize_cache`` building legacy block caches."""

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
            block_caches.append(
                block.initialize_cache(
                    chunk_size, window_size, sink_size, context_text, context_img
                )
            )
        return WanDiTNetworkCache(block_caches=block_caches)


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
