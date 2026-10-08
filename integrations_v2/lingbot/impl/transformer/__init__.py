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

"""Wan 2.1 transformer adapter with Plücker camera control for Lingbot World."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast, overload

import torch
from torch import Tensor

from flashdreams.recipes.wan.autoencoder.i2v import I2VCtrl
from flashdreams.recipes.wan.transformer.wan21 import (
    Wan21Transformer,
    Wan21TransformerCache,
    Wan21TransformerConfig,
)
from lingbot.impl.encoder.camctrl import I2VCamCtrlEmbeddings

from .impl.network import (
    LingbotWorldDiTNetwork,
    LingbotWorldDiTNetwork14BConfig,
    LingbotWorldDiTNetworkCache,
    LingbotWorldDiTNetworkConfig,
)

LINGBOT_WORLD_MIN_CHECKPOINT_FREE_GB = 200.0
"""First-run storage budget documented for LingBot-World model caches."""


@dataclass(kw_only=True)
class LingbotWorldTransformerCache(Wan21TransformerCache):
    """Long-lived AR cache for the Lingbot World transformer.

    Narrows :class:`Wan21TransformerCache`'s network-cache slots to the
    Plücker-aware Lingbot variant. Inherits ``rope_adapter`` /
    ``rope_freqs`` / ``autoregressive_index`` from the parent and the
    same ``start`` / ``finalize`` lifecycle.
    """

    network_cache: LingbotWorldDiTNetworkCache
    """Conditional per-block KV / cross-attn cache."""

    network_cache_uncond: LingbotWorldDiTNetworkCache | None = None
    """Unconditional per-block caches; ``None`` disables CFG."""

    camera_cache_chunk_idx: int = -1
    """AR chunk whose camera modulation is cached; ``-1`` means stale."""

    reference_noise_bank: Tensor | None = None
    """Upstream-layout initial noise for every chunk in this rollout."""

    def reset(self) -> None:
        """Reset rollout and camera-cache bookkeeping."""
        super().reset()
        self.camera_cache_chunk_idx = -1
        self.reference_noise_bank = None


@dataclass(kw_only=True)
class LingbotWorldTransformerConfig(Wan21TransformerConfig):
    """Config for the Lingbot World transformer.

    Bakes in the temporal layout (``len_t``, ``window_size_t``,
    ``sink_size_t``); per-rollout spatial layout (``height``, ``width``)
    is supplied to
    :meth:`Wan21Transformer.initialize_autoregressive_cache`. CP size is
    auto-detected from ``torch.distributed.get_world_size()`` (see
    :class:`Wan21TransformerConfig`).
    """

    _target: type["LingbotWorldTransformer"] = field(
        default_factory=lambda: LingbotWorldTransformer
    )

    network: LingbotWorldDiTNetworkConfig = field(
        default_factory=LingbotWorldDiTNetwork14BConfig
    )
    checkpoint_min_free_gb: float | None = LINGBOT_WORLD_MIN_CHECKPOINT_FREE_GB
    reference_noise_steps: int | None = None
    """Preallocate this many initial-noise chunks using upstream RNG layout."""


class LingbotWorldTransformer(Wan21Transformer):
    """Lingbot World DiT (Wan 2.1 + per-block Plücker camera control)."""

    config: LingbotWorldTransformerConfig

    def initial_noise(
        self,
        *,
        latent_shape: tuple[int, ...],
        rng: torch.Generator | None,
        cache: LingbotWorldTransformerCache,
        input: Any = None,
    ) -> Tensor:
        """Draw one chunk or slice it from an upstream-compatible FP32 bank."""
        del input
        reference_noise_steps = self.config.reference_noise_steps
        if reference_noise_steps is None:
            return torch.randn(
                latent_shape,
                device=self.device,
                dtype=self.dtype,
                generator=rng,
            )
        if reference_noise_steps <= 0:
            raise ValueError("reference_noise_steps must be positive when set.")
        if cache.reference_noise_bank is None:
            assert cache.autoregressive_index == 0, (
                "reference noise must be initialized at autoregressive index 0"
            )
            assert len(latent_shape) >= 4, (
                f"reference noise requires trailing [T, C, H, W], got {latent_shape}"
            )
            *batch_shape, time, channels, height, width = latent_shape
            upstream_noise = torch.randn(
                (
                    *batch_shape,
                    channels,
                    time * reference_noise_steps,
                    height,
                    width,
                ),
                device=self.device,
                dtype=torch.float32,
                generator=rng,
            )
            cache.reference_noise_bank = upstream_noise.transpose(-4, -3)

        autoregressive_index = cache.autoregressive_index
        if not 0 <= autoregressive_index < reference_noise_steps:
            raise IndexError(
                f"autoregressive index {autoregressive_index} exceeds the "
                f"{reference_noise_steps}-chunk reference noise bank"
            )
        time = latent_shape[-4]
        return cache.reference_noise_bank.narrow(-4, autoregressive_index * time, time)

    def _build_network_input(
        self,
        noisy_latent: Tensor,
        input: I2VCtrl | None,
    ) -> Tensor:
        """Cast sampler and FP32-VAE tensors at the BF16 network boundary."""
        network_dtype = self.dtype
        if input is not None:
            input = I2VCtrl(
                latent=input.latent.to(dtype=network_dtype),
                mask=input.mask.to(dtype=network_dtype),
                _is_patchified=input._is_patchified,
            )
        return super()._build_network_input(
            noisy_latent.to(dtype=network_dtype),
            input,
        )

    @torch.no_grad()
    def initialize_autoregressive_cache(
        self,
        *,
        height: int,
        width: int,
        text_embeddings: Tensor,
        image_embeddings: Tensor | None = None,
        negative_text_embeddings: Tensor | None = None,
        **kwargs: Any,
    ) -> LingbotWorldTransformerCache:
        """Build a Lingbot rollout cache with camera-cache lifecycle state."""
        cache = super().initialize_autoregressive_cache(
            height=height,
            width=width,
            text_embeddings=text_embeddings,
            image_embeddings=image_embeddings,
            negative_text_embeddings=negative_text_embeddings,
            **kwargs,
        )
        return LingbotWorldTransformerCache(
            network_cache=cast(LingbotWorldDiTNetworkCache, cache.network_cache),
            network_cache_uncond=cast(
                LingbotWorldDiTNetworkCache | None, cache.network_cache_uncond
            ),
            rope_adapter=cache.rope_adapter,
            rope_freqs=cache.rope_freqs,
            autoregressive_index=cache.autoregressive_index,
        )

    @torch.no_grad()
    def replace_text_embeddings(
        self,
        cache: LingbotWorldTransformerCache,
        text_embeddings: Tensor,
    ) -> None:
        """Swap the rollout's conditional cross-attention text context."""
        network = getattr(self.network, "_orig_mod", self.network)
        assert isinstance(network, LingbotWorldDiTNetwork)
        network.replace_text_embeddings(cache.network_cache, text_embeddings)
        if self._use_cuda_graph:
            self._cuda_graph_dispatch.reset()

    def predict_flow(
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        cache: LingbotWorldTransformerCache,
        input: I2VCamCtrlEmbeddings,
    ) -> Tensor:
        if cache.camera_cache_chunk_idx != cache.autoregressive_index:
            network = getattr(self.network, "_orig_mod", self.network)
            assert isinstance(network, LingbotWorldDiTNetwork)
            network.prepare_camera_cache(
                input.plucker.to(dtype=self.dtype),
                cache.network_cache,
                cache.network_cache_uncond,
            )
            cache.camera_cache_chunk_idx = cache.autoregressive_index
        return super().predict_flow(
            noisy_latent=noisy_latent,
            timestep=timestep,
            cache=cache,
            input=input.i2v,
        )

    @overload
    def patchify_and_maybe_split_cp(self, x: Tensor) -> Tensor: ...
    @overload
    def patchify_and_maybe_split_cp(
        self, x: I2VCamCtrlEmbeddings
    ) -> I2VCamCtrlEmbeddings: ...
    def patchify_and_maybe_split_cp(
        self, x: Tensor | I2VCamCtrlEmbeddings
    ) -> Tensor | I2VCamCtrlEmbeddings:
        """Patchify and (optionally) split for context parallelism."""
        if isinstance(x, I2VCamCtrlEmbeddings):
            if x._is_patchified:
                return x
            return I2VCamCtrlEmbeddings(
                i2v=super().patchify_and_maybe_split_cp(x.i2v),
                plucker=super().patchify_and_maybe_split_cp(x.plucker),
                _is_patchified=True,
            )
        return super().patchify_and_maybe_split_cp(x)
