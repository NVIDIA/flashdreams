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

"""Autoregressive inference pipeline: encode → diffuse → decode."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, cast

import torch
import torch.nn as nn
from loguru import logger
from torch import Tensor

from flashdreams.infra.config import InstantiateConfig
from flashdreams.infra.decoder import (
    DecCacheT,
    Decoder,
)
from flashdreams.infra.diffusion.model import (
    DiffusionModel,
    DiffusionModelConfig,
)
from flashdreams.infra.diffusion.transformer import (
    TransformerCacheT,
)
from flashdreams.infra.encoder import (
    EncCacheT,
    Encoder,
)
from flashdreams.infra.profiler import EventProfiler


@dataclass(kw_only=True)
class StreamInferencePipelineConfig(InstantiateConfig["StreamInferencePipeline"]):
    """Config for the streaming inference pipeline.

    Set ``encoder=None`` when the pipeline has no per-AR-step control input
    (pure T2V). Set ``decoder=None`` to return the clean latent directly
    (training, latent-space evaluation, or pipelines that own decoding).
    """

    _target: type["StreamInferencePipeline"] = field(
        default_factory=lambda: StreamInferencePipeline
    )

    diffusion_model: DiffusionModelConfig
    """Transformer + scheduler config."""

    decoder: InstantiateConfig[Any] | None = None
    """Optional output decoder."""

    encoder: InstantiateConfig[Any] | None = None
    """Optional per-AR-step input encoder."""

    enable_sync_and_profile: bool = False
    """Record per-stage CUDA events and log timing per AR step. Calls
    ``torch.cuda.synchronize()`` once per step, which hurts throughput."""


@dataclass(kw_only=True)
class StreamInferencePipelineCache(Generic[EncCacheT, TransformerCacheT, DecCacheT]):
    """Per-rollout cache held by the pipeline."""

    transformer_cache: TransformerCacheT
    """Long-lived transformer AR cache (always present)."""

    encoder_cache: EncCacheT | None = None
    """Encoder AR cache; ``None`` iff the pipeline has no encoder."""

    decoder_cache: DecCacheT | None = None
    """Decoder AR cache; ``None`` iff the pipeline has no decoder."""

    final_state: "DiffusionModel.FinalState[TransformerCacheT] | None" = None
    """Diffusion-model state from the most recent ``generate``, consumed
    by ``finalize``."""

    autoregressive_index: int | None = None
    """AR step index of the most recent ``generate``."""

    event_profiler: EventProfiler | None = None
    """Per-step profiler, populated only when profiling is on."""


class StreamInferencePipeline(
    nn.Module,
    Generic[
        EncCacheT,
        TransformerCacheT,
        DecCacheT,
    ],
):
    """End-to-end streaming inference pipeline.

    Generic over the encoder, transformer, and decoder cache types. The
    encoder's input/output types are forwarded as ``Any`` so the
    transformer's ``predict_flow`` / ``postprocess_clean_latent`` overrides
    own the typing on the ``input`` argument they receive.

    Typical usage example:

        cache = pipeline.initialize_cache(transformer_context={...})
        output = pipeline.generate(0, cache, input=...)
        pipeline.finalize(0, cache)
        output = pipeline.generate(1, cache, input=...)
        pipeline.finalize(1, cache)  # optional for the last rollout
    """

    encoder: Encoder[EncCacheT] | None
    decoder: Decoder[DecCacheT] | None
    diffusion_model: DiffusionModel[TransformerCacheT]

    def __init__(self, config: StreamInferencePipelineConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = config.encoder.setup() if config.encoder is not None else None
        self.decoder = config.decoder.setup() if config.decoder is not None else None
        self.diffusion_model = cast(
            "DiffusionModel[TransformerCacheT]",
            config.diffusion_model.setup(),
        )

    @property
    def device(self) -> torch.device:
        return self.diffusion_model.device

    def initialize_cache(
        self,
        transformer_context: dict[str, Any] | None = None,
        encoder_context: dict[str, Any] | None = None,
        decoder_context: dict[str, Any] | None = None,
    ) -> StreamInferencePipelineCache[EncCacheT, TransformerCacheT, DecCacheT]:
        """Build a fresh per-rollout cache.

        Each ``*_context`` dict is forwarded as keyword arguments to the
        corresponding component's ``initialize_autoregressive_cache``.

        Args:
            transformer_context: Per-rollout state for the transformer
                (e.g. ``{"text_embeddings": ..., "image_embeddings": ...}``).
            encoder_context: Per-rollout state for the encoder. Ignored
                when there is no encoder.
            decoder_context: Per-rollout state for the decoder. Ignored
                when there is no decoder.

        Returns:
            A fresh cache to thread through ``generate`` / ``finalize``.
        """
        transformer_context = transformer_context or {}
        encoder_context = encoder_context or {}
        decoder_context = decoder_context or {}
        return StreamInferencePipelineCache(
            encoder_cache=(
                self.encoder.initialize_autoregressive_cache(**encoder_context)
                if self.encoder is not None
                else None
            ),
            decoder_cache=(
                self.decoder.initialize_autoregressive_cache(**decoder_context)
                if self.decoder is not None
                else None
            ),
            transformer_cache=self.diffusion_model.transformer.initialize_autoregressive_cache(
                **transformer_context
            ),
        )

    @torch.no_grad()
    def generate(
        self,
        autoregressive_index: int,
        cache: StreamInferencePipelineCache[EncCacheT, TransformerCacheT, DecCacheT],
        input: Any = None,
    ) -> Tensor:
        """Generate one chunk for this AR step.

        Args:
            autoregressive_index: Must be ``cache.autoregressive_index + 1``,
                or ``0`` for the first call after ``initialize_cache``.
            cache: Per-rollout cache from ``initialize_cache``.
            input: Raw input fed to the encoder. Required when an encoder
                is configured, must be ``None`` otherwise. Use
                ``NullEncoderConfig`` to pass an already-encoded tensor
                straight through.

        Returns:
            Decoded tensor (e.g. RGB video) when a decoder is configured;
            otherwise the unpatchified clean latent from the diffusion model.
        """
        prev = cache.autoregressive_index
        expected = (prev + 1) if prev is not None else 0
        assert autoregressive_index == expected, (
            f"AR step out of order: previous step was {prev}, expected next "
            f"{expected}, got {autoregressive_index}"
        )
        cache.autoregressive_index = autoregressive_index

        events: EventProfiler | None = None
        if self.config.enable_sync_and_profile:
            events = EventProfiler()
            cache.event_profiler = events

        if input is not None:
            assert self.encoder is not None, (
                "input was provided but the pipeline has no encoder. "
                "Configure StreamInferencePipelineConfig.encoder (e.g. with "
                "NullEncoderConfig() for an identity passthrough)."
            )
            assert cache.encoder_cache is not None  # invariant: paired with encoder
            input = self.encoder(
                input=input,
                autoregressive_index=autoregressive_index,
                cache=cache.encoder_cache,
            )

        if events is not None:
            events.record("encode")

        clean_latent, final_state = self.diffusion_model.generate(
            autoregressive_index=autoregressive_index,
            cache=cache.transformer_cache,
            input=input,
        )
        cache.final_state = final_state

        if events is not None:
            events.record("diffuse")

        if self.decoder is not None:
            assert cache.decoder_cache is not None  # invariant: paired with decoder
            output = self.decoder(
                input=clean_latent,
                autoregressive_index=autoregressive_index,
                cache=cache.decoder_cache,
            )
        else:
            output = clean_latent

        if events is not None:
            events.record("decode")

        return output

    @torch.no_grad()
    def finalize(
        self,
        autoregressive_index: int,
        cache: StreamInferencePipelineCache[EncCacheT, TransformerCacheT, DecCacheT],
    ) -> dict[str, float] | None:
        """Advance the diffusion AR cache for the next AR step.

        Args:
            autoregressive_index: Must match the index passed to the most
                recent ``generate`` (asserted).
            cache: Same cache used by ``generate``. Consumes
                ``cache.final_state``.

        Returns:
            ``None`` when profiling is disabled. Otherwise a snapshot of this
            AR step's per-stage timings (ms) and GPU memory (GiB):
            ``{<stage>_ms, total_ms, total_ms_wo_finalize, mem_alloc_gib,
            mem_reserved_gib, mem_peak_gib}``. The same numbers are also
            logged via ``logger.info``.
        """
        assert cache.autoregressive_index == autoregressive_index, (
            f"autoregressive_index mismatch: generate() ran with "
            f"{cache.autoregressive_index} but finalize() was called with "
            f"{autoregressive_index}."
        )
        assert cache.final_state is not None, (
            "finalize() called before generate() — no FinalState on the cache."
        )
        self.diffusion_model.finalize(final_state=cache.final_state)
        if not self.config.enable_sync_and_profile:
            return None

        assert cache.event_profiler is not None, (
            "finalize() called before any generate() — no EventProfiler on the cache."
        )
        cache.event_profiler.record("finalize")
        stats_ms = cache.event_profiler.sync_and_summarize()
        total_ms = sum(stats_ms.values())
        total_ms_wo_finalize = total_ms - stats_ms.get("finalize", 0.0)
        stages_str = " ".join(f"{stage} {ms:.3f} ms" for stage, ms in stats_ms.items())

        stats: dict[str, float] = {f"{stage}_ms": ms for stage, ms in stats_ms.items()}
        stats["total_ms"] = total_ms
        stats["total_ms_wo_finalize"] = total_ms_wo_finalize

        mem_str = ""
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            gib = 1024**3
            mem_alloc_gib = torch.cuda.memory_allocated(device) / gib
            mem_reserved_gib = torch.cuda.memory_reserved(device) / gib
            mem_peak_gib = torch.cuda.max_memory_allocated(device) / gib
            stats["mem_alloc_gib"] = mem_alloc_gib
            stats["mem_reserved_gib"] = mem_reserved_gib
            stats["mem_peak_gib"] = mem_peak_gib
            mem_str = (
                f" | GPU mem alloc {mem_alloc_gib:.3f} GiB "
                f"reserved {mem_reserved_gib:.3f} GiB "
                f"peak {mem_peak_gib:.3f} GiB"
            )
        logger.info(
            f"AR {autoregressive_index} {stages_str} | "
            f"total(w/o finalize) {total_ms_wo_finalize:.3f} ms "
            f"total {total_ms:.3f} ms"
            f"{mem_str}"
        )
        return stats
