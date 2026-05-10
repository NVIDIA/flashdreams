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

"""Reference :class:`Transformer` subclass for the FlashVSR recipe.

Mirrors the ``template/transformer/__init__.py`` role: this module owns
the :class:`Wan21Transformer` subclass (including its config) that wraps
the raw :class:`FlashVSRDiTNetwork` from :mod:`.network` and exposes the
streaming inference contract (``predict_flow``, autoregressive cache
lifecycle, KV-cache-aware patchify hook).

The CFG branch is structurally supported by the parent class but FlashVSR
asserts ``guidance_scale == 1.0`` in ``__post_init__`` (the legacy
distilled checkpoint does not provide negative-prompt embeddings); kept
for future I2V experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from torch import Tensor

from flashdreams.recipes.flashvsr.transformer.network import (
    _SELF_ATTN_WINDOW,
    _SELF_ATTN_WINDOW_TOKENS,
    FlashVSRDiTNetwork,
    FlashVSRDiTNetworkConfig,
)
from flashdreams.recipes.wan.transformer.impl.network import WanDiTNetworkConfig
from flashdreams.recipes.wan.transformer.wan21 import (
    Wan21Transformer,
    Wan21TransformerCache,
    Wan21TransformerConfig,
)

__all__ = [
    "FlashVSRTransformer",
    "FlashVSRTransformerConfig",
]


@dataclass(kw_only=True)
class FlashVSRTransformerConfig(Wan21TransformerConfig):
    """Configuration for :class:`FlashVSRTransformer`.

    Wraps :class:`Wan21TransformerConfig` with the FlashVSR-specific knobs:

    - ``topk_ratio``: top-k block budget multiplier (the legacy DiT computes
      ``topk = int(window_size**2 * topk_ratio) - 1``).
    - ``kv_ratio``: number of prior chunks retained in the streaming
      self-attention KV cache (the just-written chunk is also visible at
      attention time, so the buffer holds ``kv_ratio + 1`` chunks).
    - ``local_range``: spatial window radius for the local-block mask.

    ``__post_init__`` enforces the parent's invariants and additionally:

    - Sets ``window_size_t = (kv_ratio + 1) * len_t`` and
      ``sink_size_t = 0`` so ``Wan21Transformer._build_network_cache``
      sizes the per-block ``BlockKVCache`` to the (kv_ratio+1)-chunk
      capacity FlashVSR expects.
    """

    _target: type["FlashVSRTransformer"] = field(  # type: ignore[assignment]
        default_factory=lambda: FlashVSRTransformer
    )

    network: WanDiTNetworkConfig = field(default_factory=FlashVSRDiTNetworkConfig)
    len_t: int = 2
    """FlashVSR processes 2 latent frames per DiT iteration."""

    topk_ratio: float = 2.0
    """Multiplier on the per-chunk window count squared; sets the top-k block budget."""

    kv_ratio: int = 3
    """Number of prior chunks retained in the streaming self-attention KV cache.

    The buffer holds ``kv_ratio + 1`` chunks at attention time -- the
    ``kv_ratio`` cached prior chunks plus the just-written current one.
    ``__post_init__`` translates this into
    ``window_size_t = (kv_ratio + 1) * len_t`` for the inherited
    ``Wan21Transformer._build_network_cache``, which sizes the per-block
    :class:`BlockKVCache` accordingly."""

    local_range: int = 11
    """Local-block window radius (in window units) for the draft mask."""

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.guidance_scale == 1.0, (
            "FlashVSR does not support classifier-free guidance; "
            f"set guidance_scale=1.0 (got {self.guidance_scale})."
        )
        # FlashVSR's KV cache holds ``kv_ratio + 1`` chunks at attention time
        # (the cached prior chunks plus the just-written one). Map this onto
        # the parent's pre-patchify frame-window knob so the inherited
        # ``_build_network_cache`` sizes the buffer correctly.
        self.window_size_t = (self.kv_ratio + 1) * self.len_t
        self.sink_size_t = 0


class FlashVSRTransformer(Wan21Transformer):
    """Wan 2.1 transformer specialised for the FlashVSR streaming VSR DiT."""

    config: FlashVSRTransformerConfig
    network: FlashVSRDiTNetwork

    def finalize_kv_cache(self, *args: Any, **kwargs: Any) -> None:
        """No-op: FlashVSR keys its KV cache from the **noisy** forward."""

    def patchify_and_maybe_split_cp(self, x):  # type: ignore[override]
        """Pass through ``list`` payloads; defer tensors to the standard path.

        ``FlashVSREncoder.forward`` returns the per-block low-resolution
        latent slices as a ``list[Tensor]`` already in ``[B, L, D]``
        post-patchify space (the projector emits them that way). The
        infra ``DiffusionModel.generate`` calls
        ``transformer.patchify_and_maybe_split_cp(input)`` unconditionally
        on the encoder output; we pass the list straight through so the
        per-block contract survives, while plain tensor inputs (e.g. an
        unpatchified noisy latent in tests) still take the parent's
        rearrange + linear path.
        """
        if isinstance(x, list):
            return x
        return super().patchify_and_maybe_split_cp(x)

    def predict_flow(  # type: ignore[override]
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        cache: Wan21TransformerCache,
        input: Any = None,
        network_extra_kwargs: Optional[dict[str, Any]] = None,
    ) -> Tensor:
        """Predict the FlashVSR flow at ``timestep``.

        Args:
            noisy_latent: Patchified noisy latent for this AR step
                (i.e. ``patchify_and_maybe_split_cp(...)``-output shape).
            timestep: Scalar timestep tensor; FlashVSR fixes this at 1000.
            cache: Per-rollout AR cache.
            input: Optional list of per-block patchified low-resolution
                latents from the LR projector. ``input[i]`` is added to the
                hidden state at the start of block ``i``.

        Returns:
            Tensor with the same shape as ``noisy_latent`` (the predicted flow).
        """
        ar_idx = cache.autoregressive_index
        assert ar_idx >= 0, (
            "FlashVSRTransformerCache.start(autoregressive_index) must be called "
            "before predict_flow (DiffusionModel.generate handles this)."
        )

        # FlashVSR temporal-RoPE rule. The legacy WanModel keeps two distinct
        # ``RotaryPositionEmbedding3D`` instances (``rope_freq_first`` for
        # ``ar_idx==0``, ``rope_freq_other`` otherwise) but both are
        # constructed from identical inputs, so a single instance with two
        # different offsets is bit-equivalent.
        if ar_idx == 0:
            rope_freqs = cache.rope_adapter.shift_t(0)
        else:
            rope_freqs = cache.rope_adapter.shift_t(2 + ar_idx * 2)

        # Match the legacy topk computation (see ``WanModel.forward``):
        #   block_n_per_chunk = win[0] * h * w / 128 = 2 * pH * pW / 128
        #   topk = int(block_n_per_chunk**2 * topk_ratio) - 1
        cfg = self.config
        block_n_per_chunk = (
            _SELF_ATTN_WINDOW[0] * cache.len_h * cache.len_w
        ) // _SELF_ATTN_WINDOW_TOKENS
        topk = int(block_n_per_chunk * block_n_per_chunk * cfg.topk_ratio) - 1

        block_extra_kwargs = {
            "f": cache.len_t,
            "h": cache.len_h,
            "w": cache.len_w,
            "topk": topk,
            "local_range": cfg.local_range,
        }

        flow_cond = self.network(
            x=noisy_latent,
            timesteps=timestep,
            cache=cache.network_cache_cond,
            rope_freqs=rope_freqs,
            current_chunk_idx=ar_idx,
            eager_mode=True,
            block_extra_kwargs=block_extra_kwargs,
            lq_latents=input,
        )

        if cache.network_cache_uncond is None:
            return flow_cond

        # CFG path is structurally supported by the parent class but FlashVSR
        # asserts ``guidance_scale==1.0`` in ``__post_init__``, so this branch
        # is never reached in practice. Keep it for future I2V experiments.
        flow_uncond = self.network(
            x=noisy_latent,
            timesteps=timestep,
            cache=cache.network_cache_uncond,
            rope_freqs=rope_freqs,
            current_chunk_idx=ar_idx,
            eager_mode=True,
            block_extra_kwargs=block_extra_kwargs,
            lq_latents=input,
        )
        return flow_uncond + cfg.guidance_scale * (flow_cond - flow_uncond)
