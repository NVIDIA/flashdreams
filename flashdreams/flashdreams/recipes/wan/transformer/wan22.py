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

"""Wan 2.2 MoE DiT (two Wan 2.1 networks + timestep-based dispatch)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from torch import Tensor

from flashdreams.infra.config import InstantiateConfig
from flashdreams.infra.diffusion.transformer import (
    Transformer,
    TransformerAutoregressiveCache,
)
from flashdreams.recipes.wan.transformer.impl.network import WanDiTNetwork14BConfig
from flashdreams.recipes.wan.transformer.wan21 import (
    Wan21Transformer,
    Wan21TransformerCache,
    Wan21TransformerConfig,
)

## HF diffusers → bare WanDiTNetwork key remap

# Wan 2.2 ships in the HF diffusers layout, which differs from the bare
# WanDiTNetwork.state_dict() keys. This is the same mapping the legacy
# projects.causal_wan2_2.dit.model.WanDiT used.
CHECKPOINT_KEY_MAPPING: dict[str, str] = {
    # Global embedding/head remaps
    r"^condition_embedder\.text_embedder\.linear_1\.(.*)$": r"text_embedding.0.\1",
    r"^condition_embedder\.text_embedder\.linear_2\.(.*)$": r"text_embedding.2.\1",
    r"^condition_embedder\.time_embedder\.linear_1\.(.*)$": r"time_embedding.0.\1",
    r"^condition_embedder\.time_embedder\.linear_2\.(.*)$": r"time_embedding.2.\1",
    r"^condition_embedder\.time_proj\.(.*)$": r"time_projection.1.\1",
    r"^scale_shift_table$": r"head.modulation",
    r"^proj_out\.(.*)$": r"head.head.\1",
    # Block attention projections
    r"^blocks\.(\d+)\.attn1\.to_q\.(.*)$": r"blocks.\1.self_attn.q.\2",
    r"^blocks\.(\d+)\.attn1\.to_k\.(.*)$": r"blocks.\1.self_attn.k.\2",
    r"^blocks\.(\d+)\.attn1\.to_v\.(.*)$": r"blocks.\1.self_attn.v.\2",
    r"^blocks\.(\d+)\.attn1\.to_out\.0\.(.*)$": r"blocks.\1.self_attn.o.\2",
    r"^blocks\.(\d+)\.attn2\.to_q\.(.*)$": r"blocks.\1.cross_attn.q.\2",
    r"^blocks\.(\d+)\.attn2\.to_k\.(.*)$": r"blocks.\1.cross_attn.k.\2",
    r"^blocks\.(\d+)\.attn2\.to_v\.(.*)$": r"blocks.\1.cross_attn.v.\2",
    r"^blocks\.(\d+)\.attn2\.to_out\.0\.(.*)$": r"blocks.\1.cross_attn.o.\2",
    # Block norm/modulation remaps
    r"^blocks\.(\d+)\.attn1\.norm_q\.(.*)$": r"blocks.\1.self_attn.norm_q.\2",
    r"^blocks\.(\d+)\.attn1\.norm_k\.(.*)$": r"blocks.\1.self_attn.norm_k.\2",
    r"^blocks\.(\d+)\.attn2\.norm_q\.(.*)$": r"blocks.\1.cross_attn.norm_q.\2",
    r"^blocks\.(\d+)\.attn2\.norm_k\.(.*)$": r"blocks.\1.cross_attn.norm_k.\2",
    r"^blocks\.(\d+)\.norm2\.(.*)$": r"blocks.\1.norm3.\2",
    r"^blocks\.(\d+)\.scale_shift_table$": r"blocks.\1.modulation",
    # Block FFN remaps
    r"^blocks\.(\d+)\.ffn\.fc_in\.(.*)$": r"blocks.\1.ffn.0.\2",
    r"^blocks\.(\d+)\.ffn\.fc_out\.(.*)$": r"blocks.\1.ffn.2.\2",
    r"^blocks\.(\d+)\.ffn\.net\.0\.proj\.(.*)$": r"blocks.\1.ffn.0.\2",
    r"^blocks\.(\d+)\.ffn\.net\.2\.(.*)$": r"blocks.\1.ffn.2.\2",
}


## Defaults for the two branches


def _default_branch_config() -> Wan21TransformerConfig:
    return Wan21TransformerConfig(
        network=WanDiTNetwork14BConfig(),
        len_t=3,
    )


## Autoregressive cache (per-rollout, mutated across AR steps)


@dataclass(kw_only=True)
class Wan22TransformerCache(TransformerAutoregressiveCache):
    """Per-rollout AR cache for the Wan 2.2 transformer.

    Wraps two independent Wan 2.1 caches — one per branch — because the
    residual stream diverges between high- and low-noise stacks. ``start``
    / ``finalize`` advance both in lock-step.
    """

    transformer_high_noise: Wan21TransformerCache
    """AR cache for the high-noise branch."""

    transformer_low_noise: Wan21TransformerCache
    """AR cache for the low-noise branch."""

    def start(self, autoregressive_index: int) -> None:
        self.transformer_high_noise.start(autoregressive_index)
        self.transformer_low_noise.start(autoregressive_index)

    def finalize(self, autoregressive_index: int) -> None:
        self.transformer_high_noise.finalize(autoregressive_index)
        self.transformer_low_noise.finalize(autoregressive_index)


## Transformer


@dataclass(kw_only=True)
class Wan22TransformerConfig(InstantiateConfig["Wan22Transformer"]):
    """Config for the Wan 2.2 dual-network transformer.

    Wan 2.2 dispatches to one of two Wan 2.1 networks based on whether the
    timestep is above or below ``boundary_ratio * num_train_timesteps``.
    Both branches must agree on patch_size / in_dim / dim / batch_shape /
    len_t / guidance_scale (asserted in ``__post_init__``). The per-rollout
    spatial layout (``height``, ``width``) is supplied to
    :meth:`Wan22Transformer.initialize_autoregressive_cache` and forwarded
    to both branches. The CP size is auto-detected from
    ``torch.distributed.get_world_size()``, same as Wan 2.1. Wan 2.2 has
    no CFG or I2V support here.
    """

    _target: type["Wan22Transformer"] = field(default_factory=lambda: Wan22Transformer)

    transformer_high_noise: Wan21TransformerConfig = field(
        default_factory=_default_branch_config
    )
    """Sub-config for the high-noise branch (timestep > boundary)."""

    transformer_low_noise: Wan21TransformerConfig = field(
        default_factory=_default_branch_config
    )
    """Sub-config for the low-noise branch (timestep <= boundary)."""

    boundary_ratio: float = 0.875
    """Fraction of ``num_train_timesteps`` separating the two branches.
    Default ``0.875`` matches upstream Wan 2.2."""

    num_train_timesteps: int = 1000

    def __post_init__(self) -> None:
        hi, lo = self.transformer_high_noise, self.transformer_low_noise

        # Per-branch network must agree on the token layout.
        assert hi.network.patch_size == lo.network.patch_size, (
            "high/low noise networks must share patch_size; got "
            f"{hi.network.patch_size} vs {lo.network.patch_size}"
        )
        assert hi.network.in_dim == lo.network.in_dim, (
            "high/low noise networks must share in_dim; got "
            f"{hi.network.in_dim} vs {lo.network.in_dim}"
        )
        assert hi.network.dim == lo.network.dim, (
            "high/low noise networks must share dim (head sizing); got "
            f"{hi.network.dim} vs {lo.network.dim}"
        )

        # guidance_scale is part of this list because the unified pipeline
        # reads it off this config to decide whether to build an uncond text
        # branch — the two sub-configs can't disagree.
        for key in (
            "batch_shape",
            "len_t",
            "guidance_scale",
        ):
            assert getattr(hi, key) == getattr(lo, key), (
                f"high/low noise sub-configs must share {key}; got "
                f"{getattr(hi, key)} vs {getattr(lo, key)}"
            )

    @property
    def boundary_timestep(self) -> float:
        """Absolute timestep boundary."""
        return self.boundary_ratio * self.num_train_timesteps

    ## Shared-field aliases (mirror the fields both branches must agree on)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.transformer_high_noise.batch_shape

    @property
    def len_t(self) -> int:
        return self.transformer_high_noise.len_t

    @property
    def dtype(self) -> torch.dtype:
        return self.transformer_high_noise.dtype

    @property
    def guidance_scale(self) -> float:
        return self.transformer_high_noise.guidance_scale


_NetworkChoice = Literal["high_noise", "low_noise"]


class Wan22Transformer(Transformer[Wan22TransformerCache]):
    """Wan 2.2 dual-network DiT.

    ``predict_flow`` dispatches to the branch selected by the timestep;
    ``finalize_kv_cache`` re-runs both branches once at the context noise so
    neither KV cache lags behind.
    """

    transformer_high_noise: Wan21Transformer
    transformer_low_noise: Wan21Transformer

    def __init__(self, config: Wan22TransformerConfig) -> None:
        super().__init__(config)
        self.config: Wan22TransformerConfig = config
        self.transformer_high_noise = Wan21Transformer(config.transformer_high_noise)
        self.transformer_low_noise = Wan21Transformer(config.transformer_low_noise)

    @property
    def latent_shape(self) -> tuple[int, ...]:
        """Per-rank latent shape (both branches share this, asserted by config)."""
        return self.transformer_high_noise.latent_shape

    @torch.no_grad()
    def initialize_autoregressive_cache(
        self,
        *,
        height: int,
        width: int,
        text_embeddings: Tensor,
        image_embeddings: Tensor | None = None,
        **_unused: Any,
    ) -> Wan22TransformerCache:
        """Build a seeded transformer cache for a new rollout.

        Both branches see the same text/image conditioning and the same
        per-rollout spatial layout, matching upstream Wan 2.2. CFG is not
        supported here.
        """
        return Wan22TransformerCache(
            transformer_high_noise=self.transformer_high_noise.initialize_autoregressive_cache(
                height=height,
                width=width,
                text_embeddings=text_embeddings,
                image_embeddings=image_embeddings,
            ),
            transformer_low_noise=self.transformer_low_noise.initialize_autoregressive_cache(
                height=height,
                width=width,
                text_embeddings=text_embeddings,
                image_embeddings=image_embeddings,
            ),
        )

    def _choose_network(self, timestep: Tensor) -> _NetworkChoice:
        """High-noise branch above the boundary, low-noise at or below."""
        scalar = timestep.flatten()[0] if timestep.numel() > 0 else timestep
        return "high_noise" if scalar > self.config.boundary_timestep else "low_noise"

    def predict_flow(
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        cache: Wan22TransformerCache,
        input: Any = None,
    ) -> Tensor:
        # Wan 2.2 (FastVideo T2V) has no per-AR-step encoder input; accept
        # and ignore ``input`` to satisfy the DiffusionModel.generate contract.
        if self._choose_network(timestep) == "high_noise":
            return self.transformer_high_noise.predict_flow(
                noisy_latent, timestep, cache.transformer_high_noise
            )
        else:
            return self.transformer_low_noise.predict_flow(
                noisy_latent, timestep, cache.transformer_low_noise
            )

    def finalize_kv_cache(
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        cache: Wan22TransformerCache,
        input: Any = None,
    ) -> None:
        """Refresh both networks' KV caches at the context-noise step.

        Each Wan 2.2 denoising step only touches one branch, so the other
        lags by the end of the loop. Re-running both at ``context_noise``
        keeps them in lock-step; flow outputs are discarded.
        """
        _ = self.transformer_high_noise.predict_flow(
            noisy_latent, timestep, cache.transformer_high_noise
        )
        _ = self.transformer_low_noise.predict_flow(
            noisy_latent, timestep, cache.transformer_low_noise
        )

    def patchify_and_maybe_split_cp(self, x: Tensor) -> Tensor:
        return self.transformer_high_noise.patchify_and_maybe_split_cp(x)

    def unpatchify_and_maybe_gather_cp(self, x: Tensor) -> Tensor:
        return self.transformer_high_noise.unpatchify_and_maybe_gather_cp(x)
