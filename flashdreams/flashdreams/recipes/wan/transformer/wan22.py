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

from flashdreams.infra.diffusion.transformer import (
    Transformer,
    TransformerAutoregressiveCache,
    TransformerConfig,
)

from .impl.network import WanDiTNetwork14BConfig
from .wan21 import (
    Wan21Transformer,
    Wan21TransformerCache,
    Wan21TransformerConfig,
)

# ---------------------------------------------------------------------------
# HF (diffusers) → official Wan key remap. Wan 2.2 ships in a HF layout that
# differs from the bare ``WanDiTNetwork.state_dict()`` keys; this mapping is
# the same one used by the legacy ``projects.causal_wan2_2.dit.model.WanDiT``.
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Defaults for the two branches. Wan 2.2 is 14B + len_t=3 + compile by
# default, with plain T2V semantics (no CFG, no I2V) on each branch.
# ---------------------------------------------------------------------------


def _default_branch_config() -> Wan21TransformerConfig:
    return Wan21TransformerConfig(
        network=WanDiTNetwork14BConfig(),
        len_t=3,
        compile_network=True,
    )


# ---------------------------------------------------------------------------
# Autoregressive cache (per-rollout, mutated across AR steps)
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class Wan22TransformerCache(TransformerAutoregressiveCache):
    """Long-lived AR cache for :class:`Wan22Transformer`.

    Wraps two independent :class:`Wan21TransformerCache` instances — one
    per network — because the residual stream diverges between high- and
    low-noise stacks. Both are advanced in lock-step: every AR step
    calls each network at least once during finalize so neither cache
    lags. ``start`` / ``finalize`` delegate to both subcaches.
    """

    transformer_high_noise: Wan21TransformerCache
    """Full Wan 2.1 AR cache for the high-noise stack (KV / cross-attn / rope)."""

    transformer_low_noise: Wan21TransformerCache
    """Full Wan 2.1 AR cache for the low-noise stack (KV / cross-attn / rope)."""

    def start(self, autoregressive_index: int) -> None:
        self.transformer_high_noise.start(autoregressive_index)
        self.transformer_low_noise.start(autoregressive_index)

    def finalize(self, autoregressive_index: int) -> None:
        self.transformer_high_noise.finalize(autoregressive_index)
        self.transformer_low_noise.finalize(autoregressive_index)


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class Wan22TransformerConfig(TransformerConfig):
    """Configuration for :class:`Wan22Transformer`.

    Wan 2.2 is two Wan 2.1 networks with a timestep-based dispatch, so
    this config simply holds two :class:`Wan21TransformerConfig`
    instances (one per branch) plus the Wan-2.2-specific boundary
    hyperparameters. Each sub-config drives an independent
    :class:`Wan21Transformer` submodule.

    Both branches must agree on patch_size / in_dim / dim / batch_shape /
    height / width / len_t / cp_size (asserted in ``__post_init__``), and
    must be configured as plain T2V (``guidance_scale == 1.0``, no I2V
    stamping / concat) since Wan 2.2 has no CFG or I2V support in this
    port.
    """

    _target: type["Wan22Transformer"] = field(default_factory=lambda: Wan22Transformer)

    transformer_high_noise: Wan21TransformerConfig = field(
        default_factory=_default_branch_config
    )
    """Wan 2.1 sub-config for the high-noise network (``timestep > boundary``)."""

    transformer_low_noise: Wan21TransformerConfig = field(
        default_factory=_default_branch_config
    )
    """Wan 2.1 sub-config for the low-noise network (``timestep <= boundary``)."""

    # Network choice boundary -------------------------------------------
    boundary_ratio: float = 0.875
    """Fraction of ``num_train_timesteps`` above which the *high*-noise
    network is used and below which the *low*-noise network is used.
    Default ``0.875`` matches the upstream Wan 2.2 release."""
    num_train_timesteps: int = 1000
    """Used together with ``boundary_ratio`` to compute the actual
    timestep boundary used in :meth:`Wan22Transformer.predict_flow`."""

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

        # Per-rollout layout must match so both sub-caches share a shape.
        # ``guidance_scale`` is included because the CFG branch is a
        # whole-pipeline decision (the unified pipeline reads it off this
        # config to decide whether to build an uncond text branch), so
        # the two networks can't disagree on it.
        for key in (
            "batch_shape",
            "height",
            "width",
            "len_t",
            "cp_size",
            "guidance_scale",
        ):
            assert getattr(hi, key) == getattr(lo, key), (
                f"high/low noise sub-configs must share {key}; got "
                f"{getattr(hi, key)} vs {getattr(lo, key)}"
            )

    @property
    def boundary_timestep(self) -> float:
        """Absolute timestep boundary: ``boundary_ratio * num_train_timesteps``."""
        return self.boundary_ratio * self.num_train_timesteps

    # ------------------------------------------------------------------
    # Shared-field aliases
    #
    # These mirror the per-rollout layout fields that both sub-configs
    # agree on (asserted in ``__post_init__``) so external callers can
    # read video dimensions / dtype / batch shape off ``Wan22TransformerConfig``
    # without reaching into either branch.
    # ------------------------------------------------------------------

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.transformer_high_noise.batch_shape

    @property
    def height(self) -> int:
        return self.transformer_high_noise.height

    @property
    def width(self) -> int:
        return self.transformer_high_noise.width

    @property
    def len_t(self) -> int:
        return self.transformer_high_noise.len_t

    @property
    def cp_size(self) -> int:
        return self.transformer_high_noise.cp_size

    @property
    def dtype(self) -> torch.dtype:
        return self.transformer_high_noise.dtype

    @property
    def guidance_scale(self) -> float:
        return self.transformer_high_noise.guidance_scale


_NetworkChoice = Literal["high_noise", "low_noise"]


class Wan22Transformer(Transformer[Wan22TransformerCache]):
    """Wan 2.2 dual-network DiT built from two :class:`Wan21Transformer`s.

    Each branch is a plain Wan 2.1 T2V transformer (``guidance_scale=1.0``,
    no I2V). :meth:`predict_flow` dispatches to the branch selected by
    ``timestep`` crossing ``boundary_timestep``; :meth:`finalize_kv_cache`
    re-runs *both* branches at the AR step's context-noise so neither
    KV cache lags.
    """

    transformer_high_noise: Wan21Transformer
    transformer_low_noise: Wan21Transformer

    def __init__(
        self,
        config: Wan22TransformerConfig,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(config)
        self.config: Wan22TransformerConfig = config
        self.transformer_high_noise = Wan21Transformer(
            config.transformer_high_noise, device=device
        )
        self.transformer_low_noise = Wan21Transformer(
            config.transformer_low_noise, device=device
        )

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    @property
    def latent_shape(self) -> tuple[int, ...]:
        """Per-rank latent shape. Both branches agree (asserted by config)."""
        return self.transformer_high_noise.latent_shape

    @torch.no_grad()
    def initialize_autoregressive_cache(  # type: ignore[override]
        self,
        *,
        text_embeddings: Tensor,
        image_embeddings: Tensor | None = None,
        **_unused: Any,
    ) -> Wan22TransformerCache:
        """Build a fully seeded :class:`Wan22TransformerCache` for a new rollout.

        Both branches see the same text/image conditioning (mirrors the
        upstream Wan 2.2 reference inference). Wan 2.2 has no CFG support
        in this port (the upstream FastVideo I2V checkpoint is distilled
        to a single conditional forward); to add CFG, subclass and build
        paired uncond caches manually.
        """
        return Wan22TransformerCache(
            transformer_high_noise=self.transformer_high_noise.initialize_autoregressive_cache(
                text_embeddings=text_embeddings,
                image_embeddings=image_embeddings,
            ),
            transformer_low_noise=self.transformer_low_noise.initialize_autoregressive_cache(
                text_embeddings=text_embeddings,
                image_embeddings=image_embeddings,
            ),
        )

    def _choose_network(self, timestep: Tensor) -> _NetworkChoice:
        """High-noise above the boundary, low-noise at or below.

        ``timestep`` is a 0-d or 1-d Tensor. Compares the (first) scalar
        against the absolute boundary (``boundary_ratio *
        num_train_timesteps``) — same convention as upstream Wan 2.2.
        """
        scalar = timestep.flatten()[0] if timestep.numel() > 0 else timestep
        return "high_noise" if scalar > self.config.boundary_timestep else "low_noise"

    def predict_flow(
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        cache: Wan22TransformerCache,
        input: Any = None,
    ) -> Tensor:
        # Wan 2.2 (T2V, FastVideo) has no per-AR-step encoder input (the
        # causal_wan22 recipe config wires ``encoder=None``); accept and
        # ignore ``input`` so the base ``DiffusionModel.generate``
        # contract is satisfied.
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
        """Refresh *both* networks' KV caches at ``context_noise``.

        Each Wan 2.2 denoising step touches only the network selected by
        the timestep boundary, so the unselected network's KV cache lags
        a chunk behind by the end of the loop. Re-running both branches
        once at the AR-step's ``context_noise`` keeps them in lock-step.
        Flow outputs are discarded — only the per-block KV side effect
        matters.
        """
        _ = self.transformer_high_noise.predict_flow(
            noisy_latent, timestep, cache.transformer_high_noise
        )
        _ = self.transformer_low_noise.predict_flow(
            noisy_latent, timestep, cache.transformer_low_noise
        )

    def patchify_and_maybe_split_cp(self, x: Tensor) -> Tensor:
        return self.transformer_high_noise.patchify_and_maybe_split_cp(x)  # ty:ignore[invalid-return-type]

    def unpatchify_and_maybe_gather_cp(self, x: Tensor) -> Tensor:
        return self.transformer_high_noise.unpatchify_and_maybe_gather_cp(x)
