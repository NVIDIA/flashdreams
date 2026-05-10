# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MemRoPE Wan 2.1 transformer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from flashdreams.recipes.wan.autoencoder.i2v import I2VCtrl
from flashdreams.recipes.wan.transformer.impl.memrope_network import (
    MemRoPEWanDiTNetwork,
    MemRoPEWanDiTNetwork1pt3BConfig,
    MemRoPEWanDiTNetworkConfig,
)
from flashdreams.recipes.wan.transformer.impl.network import WanDiTNetworkCache
from flashdreams.recipes.wan.transformer.wan21 import (
    Wan21Transformer,
    Wan21TransformerCache,
    Wan21TransformerConfig,
)


@dataclass(kw_only=True)
class MemRoPEWan21TransformerConfig(Wan21TransformerConfig):
    """MemRoPE-specific Wan 2.1 transformer config."""

    _target: type["MemRoPEWan21Transformer"] = field(
        default_factory=lambda: MemRoPEWan21Transformer
    )

    network: MemRoPEWanDiTNetworkConfig = field(
        default_factory=MemRoPEWanDiTNetwork1pt3BConfig
    )
    recent_size_t: int = 13
    memory_size_t: int = 2
    ema_alpha_long: float = 0.01
    ema_alpha_short: float = 0.1
    height: int = 60
    width: int = 104
    cp_size: int = 1

    def __post_init__(self) -> None:
        assert self.network.patch_size[0] == 1, (
            "MemRoPE temporal cache sizing assumes patch_size_t=1"
        )
        assert not self.stamp_image_latent, "MemRoPE v1 is T2V-only"
        assert not self.concat_image_mask_to_latent, "MemRoPE v1 is T2V-only"
        assert self.recent_size_t >= 0, "recent_size_t must be non-negative"
        assert self.memory_size_t in (0, 2), (
            "memory_size_t must be 0 or 2 for the current MemRoPE layout"
        )
        expected_window_size_t = self.memory_size_t + self.recent_size_t + self.len_t
        assert self.window_size_t == expected_window_size_t, (
            "MemRoPE window_size_t must equal memory_size_t + "
            f"recent_size_t + len_t ({expected_window_size_t})"
        )
        assert 0.0 <= self.ema_alpha_long <= 1.0, (
            "ema_alpha_long must be in [0, 1]"
        )
        assert 0.0 <= self.ema_alpha_short <= 1.0, (
            "ema_alpha_short must be in [0, 1]"
        )


class MemRoPEWan21Transformer(Wan21Transformer):
    """Wan 2.1 DiT adapter using MemRoPE self-attention."""

    config: MemRoPEWan21TransformerConfig
    network: MemRoPEWanDiTNetwork

    def __init__(self, config: MemRoPEWan21TransformerConfig) -> None:
        super().__init__(config)
        assert self._cp_size == self.config.cp_size, (
            f"MemRoPE config cp_size={self.config.cp_size} does not match "
            f"distributed cp_size={self._cp_size}"
        )
        assert self._cp_size == 1, (
            "MemRoPE online RoPE indexing currently requires cp_size=1"
        )

    @torch.no_grad()
    def _build_network_cache(
        self,
        *,
        text_embeddings: Tensor,
        image_embeddings: Tensor | None = None,
    ) -> WanDiTNetworkCache:
        assert isinstance(self.network, MemRoPEWanDiTNetwork)
        assert self._output_height is not None and self._output_width is not None, (
            "_build_network_cache called before height/width were stashed."
        )
        _, kh, kw = self.config.network.patch_size
        len_h = self._output_height // kh
        len_w = self._output_width // kw
        chunk_size = self.latent_shape[-2]
        frame_size = len_h * len_w
        window_size = self.config.window_size_t * frame_size
        sink_size = self.config.sink_size_t * frame_size
        recent_size = self.config.recent_size_t * frame_size
        return self.network.initialize_cache(
            chunk_size=chunk_size,
            window_size=window_size,
            sink_size=sink_size,
            text_embeddings=text_embeddings,
            img_embeddings=image_embeddings,
            frame_size=frame_size,
            recent_size=recent_size,
            memory_frames=self.config.memory_size_t,
            ema_alpha_long=self.config.ema_alpha_long,
            ema_alpha_short=self.config.ema_alpha_short,
        )

    def predict_flow(
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        cache: Wan21TransformerCache,
        input: I2VCtrl | None = None,
        network_extra_kwargs: dict[str, Any] = {},
    ) -> Tensor:
        assert input is None, "MemRoPE v1 supports T2V only"
        ar_idx = cache.autoregressive_index
        assert ar_idx >= 0, (
            "Wan21TransformerCache.start(autoregressive_index) must be called "
            "before predict_flow (DiffusionModel.generate handles this)."
        )
        assert self._output_height is not None and self._output_width is not None, (
            "predict_flow called before height/width were stashed."
        )
        _, kh, kw = self.config.network.patch_size
        len_h = self._output_height // kh
        len_w = self._output_width // kw

        flow_cond = self.network(
            x=noisy_latent,
            timesteps=timestep,
            cache=cache.network_cache,
            rope_adapter=cache.rope_adapter,
            len_h=len_h,
            len_w=len_w,
            current_chunk_idx=ar_idx,
            eager_mode=False,
            **network_extra_kwargs,
        )
        if cache.network_cache_uncond is None:
            return flow_cond

        flow_uncond = self.network(
            x=noisy_latent,
            timesteps=timestep,
            cache=cache.network_cache_uncond,
            rope_adapter=cache.rope_adapter,
            len_h=len_h,
            len_w=len_w,
            current_chunk_idx=ar_idx,
            eager_mode=False,
            **network_extra_kwargs,
        )
        return flow_uncond + self.config.guidance_scale * (flow_cond - flow_uncond)
