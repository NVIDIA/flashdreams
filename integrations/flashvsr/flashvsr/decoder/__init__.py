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

"""FlashVSR decoder: TC decoder + AdaIN color corrector.

Wraps :class:`FlashVSR_TAEHV` (in :mod:`.network`) + :class:`FlashVSRColorCorrector`
for the :class:`flashdreams.infra.decoder.StreamingDecoder` interface.
One ``forward()`` call ingests a chunk of clean latents plus the
bicubic upres stashed by :meth:`FlashVSRPipeline.generate` on
:attr:`FlashVSRDecoderCache.last_upres`, and returns RGB frames in
``[-1, 1]``. The upres threads through the cache rather than the
standard ``input=`` arg because both the TC decoder (as ``cond``) and
the color corrector (as the AdaIN reference) consume it. See the
sibling ``README.md`` for the streaming chunk contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from flashdreams.infra.decoder import (
    DecoderConfig,
    StreamingDecoder,
    StreamingDecoderCache,
)
from flashdreams.infra.profiler import EventProfiler, record_event
from flashdreams.recipes.taehv.checkpoint import (
    StateDictTransform,
    legacy_to_blocks_keys,
)
from flashvsr.corrector import (
    ColorCorrectorImplementation,
    FlashVSRColorCorrector,
)
from flashvsr.decoder.network import FlashVSR_TAEHV, FlashVSR_TAEHV_Cache

__all__ = [
    "FlashVSRDecoder",
    "FlashVSRDecoderConfig",
    "FlashVSRDecoderCache",
    "tcdecoder_state_dict_transform",
]


tcdecoder_state_dict_transform: StateDictTransform = legacy_to_blocks_keys
"""Per-checkpoint remap for FlashVSR's ``TCDecoder.ckpt``. Rewrites the
flat ``decoder.<i>.*`` keys to the current ``decoder.blocks.<i>.*``
layout; the FlashVSR weights already ship with the right ``TGrow``
strides (and the identity-deepened body shifts the indices anyway), so
no ``truncate_oversize_tgrow_weights`` step is composed here. Lives
next to :attr:`FlashVSRDecoderConfig.tcdecoder_checkpoint_path` so each
checkpoint owns its own remap."""


@dataclass(kw_only=True)
class FlashVSRDecoderConfig(DecoderConfig):
    """Configuration for :class:`FlashVSRDecoder`."""

    _target: type["FlashVSRDecoder"] = field(default_factory=lambda: FlashVSRDecoder)

    tcdecoder_checkpoint_path: str | None = None
    """Path / URL to the TC decoder ``.ckpt``. Required at runtime;
    :class:`FlashVSRDecoder.__init__` asserts when this is ``None``."""

    state_dict_transform: StateDictTransform | None = tcdecoder_state_dict_transform
    """Pre-load state-dict remap for the TC decoder checkpoint; defaults
    to :data:`tcdecoder_state_dict_transform` (module-level) which is the
    canonical FlashVSR ``TCDecoder.ckpt`` remap. Override per-variant
    when wiring a different checkpoint that needs its own key remap."""

    tcdecoder_channels: tuple[int, int, int, int] = (512, 256, 128, 128)
    """TAEHV decoder block channel widths (FlashVSR-tiny default)."""

    tcdecoder_latent_channels: int = 16 + 768
    """TAEHV input channels: 16 noise channels + 768 conditioning channels
    (the conditioning slice comes from the bicubic upres pixel-shuffle)."""

    use_compile: bool = False
    """``torch.compile`` the TAEHV decode path."""

    use_cuda_graph: bool = False
    """Wrap the TAEHV decode in ``CUDAGraphWrapper`` for steady-state replay.

    Defaults to ``False`` and matches :attr:`FlashVSREncoderConfig.use_cuda_graph`.
    :func:`build_flashvsr_v1_1` hard-codes both knobs to ``True`` for
    production wiring (the cudagraph paths for both components are
    well-tested), so this default only matters when assembling sub-configs
    by hand outside the builder -- e.g. in the equivalence pytest cases."""

    color_corrector_levels: int = 5
    """Wavelet decomposition levels for the (torch) wavelet color corrector."""

    color_corrector_implementation: ColorCorrectorImplementation = "cuda"
    """Backend for :class:`FlashVSRColorCorrector`. ``"cuda"`` (default)
    dispatches to the hand-rolled AdaIN extension; ``"torch"`` runs the
    pure-torch reference (wavelet + AdaIN) on the same device."""

    dtype: torch.dtype = torch.bfloat16
    """Decoder compute dtype. ``bfloat16`` matches FlashVSR-tiny weights."""


@dataclass(kw_only=True)
class FlashVSRDecoderCache(StreamingDecoderCache):
    """Per-rollout decoder cache.

    Holds the TC decoder's streaming state plus a slot for the bicubic
    upres of the current AR step (set by
    :meth:`FlashVSRPipeline.generate` from
    ``cache.encoder_cache.last_upres`` between the encoder and decoder
    calls).

    Per-AR-step profiling lives on the parent
    :attr:`StreamInferencePipelineCache.event_profiler`; the pipeline
    forwards that single instance to :meth:`FlashVSRDecoder.forward` as
    an explicit kwarg, so we do not duplicate the slot here.
    """

    tdec_cache: FlashVSR_TAEHV_Cache
    """TAEHV streaming state."""

    last_upres: Tensor | None = None
    """``[B, 3, T, target_H, target_W]`` bicubic upres of the current chunk,
    used as ``cond`` for the TC decoder and as the AdaIN reference for the
    color corrector."""

    def reset(self) -> None:
        """Reset temporal decoder state for a new video rollout."""
        self.tdec_cache.dec_state.clear()
        self.last_upres = None


class FlashVSRDecoder(StreamingDecoder[FlashVSRDecoderCache]):
    """TC decoder + AdaIN color corrector for FlashVSR."""

    def __init__(self, config: FlashVSRDecoderConfig) -> None:
        super().__init__(config)
        self.config: FlashVSRDecoderConfig = config

        assert config.tcdecoder_checkpoint_path is not None, (
            "FlashVSRDecoderConfig.tcdecoder_checkpoint_path must be set "
            "(point at the staged TCDecoder.ckpt -- see "
            "internal/upsampler/scripts/download_flashvsr_weights.sh)."
        )
        self.tcdecoder = (
            FlashVSR_TAEHV(
                channels=config.tcdecoder_channels,
                latent_channels=config.tcdecoder_latent_channels,
                checkpoint_path=config.tcdecoder_checkpoint_path,
                use_cuda_graph=config.use_cuda_graph,
                use_compile=config.use_compile,
                state_dict_transform=config.state_dict_transform,
            )
            .to(dtype=config.dtype)
            .eval()
            .requires_grad_(False)
        )
        self.color_corrector = FlashVSRColorCorrector(
            levels=config.color_corrector_levels,
            implementation=config.color_corrector_implementation,
        )

    def initialize_autoregressive_cache(self, **_unused: Any) -> FlashVSRDecoderCache:
        return FlashVSRDecoderCache(
            tdec_cache=self.tcdecoder.prepare_cache(),
            last_upres=None,
        )

    def forward(  # type: ignore[override]
        self,
        input: Tensor,
        autoregressive_index: int = 0,
        cache: FlashVSRDecoderCache | None = None,
        event_profiler: EventProfiler | None = None,
    ) -> Tensor:
        """Decode clean latents to RGB and color-correct against the upres.

        Args:
            input: Clean latents from the diffusion model with shape
                ``[B, T, C, H, W]`` (the layout produced by
                ``Wan21Transformer.unpatchify_and_maybe_gather_cp``;
                ``T`` is the latent-frame count, ``C=16``).
            autoregressive_index: AR step index (unused; the decoder is
                purely streaming and reads its state from ``cache``).
            cache: Per-rollout decoder cache. ``cache.last_upres`` must have
                been set by ``FlashVSRPipeline.generate`` before this call.
            event_profiler: Optional per-AR-step profiler (forwarded by
                :meth:`FlashVSRPipeline.generate` from the parent
                :attr:`StreamInferencePipelineCache.event_profiler`).
                The decoder records the ``decoder`` and ``color`` sub-stages
                against it.

        Returns:
            RGB frames ``[B, 3, T_out, target_H, target_W]`` in ``[-1, 1]``,
            with ``T_out`` matching the un-padded upres time dimension.
        """
        assert cache is not None, "FlashVSRDecoder requires a cache"
        upres = cache.last_upres
        assert upres is not None, (
            "FlashVSRDecoder.forward called before encoder ran: "
            "cache.last_upres must be populated by FlashVSRPipeline.generate."
        )
        del autoregressive_index  # unused; streaming state lives on cache

        # Wan21Transformer.unpatchify -> [B, T, C, H, W]; FlashVSR's TC
        # decoder + downstream color corrector expect [B, C, T, H, W]
        # framewise (FlashVSR / VAE convention).
        clean_latents = input.transpose(1, 2)

        # TAEHV.forward signature mirrors the legacy upsampler's call.
        # Note the double transpose: the legacy code passed
        # ``cur_latents.transpose(1, 2)`` (= [B, T, C, H, W]) into TAEHV and
        # then ``.transpose(1, 2)`` the output back. We replicate that
        # here exactly.
        cur_frames = (
            self.tcdecoder(
                clean_latents.transpose(1, 2),
                parallel=True,
                show_progress_bar=False,
                cond=upres,
                cache=cache.tdec_cache,
            )
            .transpose(1, 2)
            .mul(2)
            .sub(1)
        )

        record_event(event_profiler, "decoder")

        out = self.color_corrector(
            cur_frames,
            upres,
            clip_range=(-1, 1),
            chunk_size=None,
            method="adain",
        )

        record_event(event_profiler, "color")

        return out
