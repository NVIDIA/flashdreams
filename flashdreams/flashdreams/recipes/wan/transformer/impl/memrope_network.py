# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MemRoPE-specific Wan 2.1 DiT network."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
from einops import rearrange
from torch import Tensor

from flashdreams.recipes.wan.transformer.impl.memrope_modules import (
    MemRoPEBlock,
    MemRoPEBlockCache,
)
from flashdreams.recipes.wan.transformer.impl.modules import (
    Block,
    BlockCache,
    sinusoidal_embedding_1d,
)
from flashdreams.recipes.wan.transformer.impl.network import (
    WanDiTNetwork,
    WanDiTNetworkCache,
    WanDiTNetworkConfig,
)
from flashdreams.recipes.wan.transformer.impl.rope import RotaryPositionEmbedding3D


@dataclass
class MemRoPEWanDiTNetworkConfig(WanDiTNetworkConfig):
    """Configuration for the MemRoPE Wan DiT network."""

    _target: type["MemRoPEWanDiTNetwork"] = field(
        default_factory=lambda: MemRoPEWanDiTNetwork
    )


@dataclass
class MemRoPEWanDiTNetwork1pt3BConfig(MemRoPEWanDiTNetworkConfig):
    """Configuration for the 1.3B MemRoPE Wan DiT network."""

    dim: int = 1536
    ffn_dim: int = 8960
    num_heads: int = 12
    num_layers: int = 30


class MemRoPEWanDiTNetwork(WanDiTNetwork):
    """Wan diffusion backbone with MemRoPE self-attention blocks."""

    def _build_block(self, layer_idx: int) -> Block:
        return MemRoPEBlock(
            dim=self.dim,
            ffn_dim=self.ffn_dim,
            num_heads=self.num_heads,
            cross_attn_norm=self.cross_attn_norm,
            eps=self.eps,
            i2v=self.cross_attn_enable_img,
        )

    def _head_forward(self, x: Tensor, e: Tensor, batch_shape: tuple[int, ...]) -> Tensor:
        if e.ndim == len(batch_shape) + 2:
            assert self.head._parameters_updated_after_loading_checkpoint, (
                "We expect to have called update_parameters_after_loading_checkpoint() "
                "before running the forward pass"
            )
            num_frames = e.shape[-2]
            frame_seqlen = x.shape[-2] // num_frames
            e_chunks = (self.head.modulation + e.unsqueeze(-2)).chunk(2, dim=-2)
            y = self.head.norm(x).unflatten(
                dim=-2, sizes=(num_frames, frame_seqlen)
            )
            y = y * (1 + e_chunks[1]) + e_chunks[0]
            y = self.head.head(y)
            return y.flatten(start_dim=-3, end_dim=-2)
        return self.head(x, torch.broadcast_to(e, batch_shape + (1, e.shape[-1])))

    def initialize_cache(
        self,
        chunk_size: int,
        window_size: int,
        sink_size: int,
        text_embeddings: Tensor,
        img_embeddings: Tensor | None = None,
        *,
        frame_size: int,
        recent_size: int,
        memory_frames: int,
        ema_alpha_long: float,
        ema_alpha_short: float,
    ) -> WanDiTNetworkCache:
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
            assert isinstance(block, MemRoPEBlock)
            block_caches.append(
                block.initialize_cache(
                    chunk_size,
                    window_size,
                    sink_size,
                    context_text,
                    context_img,
                    frame_size=frame_size,
                    recent_size=recent_size,
                    memory_frames=memory_frames,
                    ema_alpha_long=ema_alpha_long,
                    ema_alpha_short=ema_alpha_short,
                )
            )
        return WanDiTNetworkCache(block_caches=block_caches)

    def forward(
        self,
        x: Tensor,
        timesteps: Tensor,
        cache: WanDiTNetworkCache,
        rope_adapter: RotaryPositionEmbedding3D,
        *,
        len_h: int,
        len_w: int,
        current_chunk_idx: int = 0,
        eager_mode: bool = True,
        block_extra_kwargs: dict[str, Any] = {},
    ) -> Tensor:
        assert self._parameters_updated_after_loading_checkpoint, (
            "We expect to have called update_parameters_after_loading_checkpoint() "
            "after loading the checkpoint"
        )
        batch_shape = x.shape[:-2]

        if self.patch_embedding_type == "linear":
            x = self.patch_embedding(x)
        elif self.patch_embedding_type == "conv3d":
            kt, kh, kw = self.patch_size
            L, D = x.shape[-2:]
            len_t = L // (len_h * len_w)
            channels = D // (kt * kh * kw)
            assert len_t * len_h * len_w == L, "invalid MemRoPE token grid"
            assert channels * kt * kh * kw == D, "invalid MemRoPE patch width"
            x_flat = x.reshape(math.prod(batch_shape), L, D)
            x_conv = rearrange(
                x_flat,
                "b (t h w) (c kt kh kw) -> b c (t kt) (h kh) (w kw)",
                t=len_t,
                h=len_h,
                w=len_w,
                c=channels,
                kt=kt,
                kh=kh,
                kw=kw,
            )
            x = self.patch_embedding(x_conv)
            x = rearrange(x, "b d t h w -> b (t h w) d")
            x = x.reshape(batch_shape + (L, self.dim))
        else:
            raise ValueError(
                f"Invalid patch embedding type: {self.patch_embedding_type}"
            )

        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timesteps).type_as(x)
        )
        e0 = self.time_projection(e).unflatten(-1, (6, self.dim))

        if eager_mode:
            cache.before_update(current_chunk_idx)
        for block_idx, block in enumerate(self.blocks):
            assert isinstance(block, MemRoPEBlock)
            block_cache = cache[block_idx]
            assert isinstance(block_cache, MemRoPEBlockCache)
            if e0.ndim == len(batch_shape) + 3:
                block_e = e0
            else:
                block_e = torch.broadcast_to(e0, batch_shape + e0.shape[-2:])
            x = block(
                x=x,
                e=block_e,
                cache=block_cache,
                rope_adapter=rope_adapter,
                len_h=len_h,
                len_w=len_w,
                **block_extra_kwargs,
            )
        if eager_mode:
            cache.after_update(current_chunk_idx)

        return self._head_forward(x, e, batch_shape)
