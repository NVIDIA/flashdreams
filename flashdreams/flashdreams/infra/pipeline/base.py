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
from typing import Any, Generic

import torch
import torch.nn as nn
from loguru import logger
from torch import Tensor

from flashdreams.infra.config import InstantiateConfig
from flashdreams.infra.decoder import (
    DecCacheT,
    Decoder,
    DecoderConfig,
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
    EncoderConfig,
)
from flashdreams.infra.profiler import EventProfiler


@dataclass(kw_only=True)
class StreamInferencePipelineConfig(InstantiateConfig["StreamInferencePipeline"]):
    """Configuration for :class:`StreamInferencePipeline`.

    Both ``encoder`` and ``decoder`` are optional. Use ``encoder=None``
    when there is no per-AR-step control input (e.g. pure T2V).
    Use ``decoder=None`` to return the clean latent directly (useful for
    training, latent-space evaluation, or pipelines that own decoding).
    """

    _target: type["StreamInferencePipeline"] = field(
        default_factory=lambda: StreamInferencePipeline
    )

    diffusion_model: DiffusionModelConfig
    """Diffusion model config (transformer + scheduler)."""

    decoder: DecoderConfig | None = None
    """Optional decoder. When ``None``, :meth:`generate` returns the unpatchified clean latent."""

    encoder: EncoderConfig | None = None
    """Optional encoder. When ``None``, :meth:`generate` must be called with ``input=None``."""

    enable_sync_and_profile: bool = False
    """If ``True``, record per-stage CUDA events and log a per-AR-step breakdown.

    Warning: enabling this calls ``torch.cuda.synchronize()`` once per AR
    step, which serializes the host against in-flight CUDA work and
    hurts throughput.
    """


@dataclass(kw_only=True)
class StreamInferencePipelineCache(Generic[EncCacheT, TransformerCacheT, DecCacheT]):
    """Per-rollout cache held by the pipeline."""

    transformer_cache: TransformerCacheT
    """Long-lived transformer AR cache (always present)."""

    encoder_cache: EncCacheT | None = None
    """Encoder AR cache. ``None`` iff the pipeline has no encoder."""

    decoder_cache: DecCacheT | None = None
    """Decoder AR cache. ``None`` iff the pipeline has no decoder."""

    final_state: "DiffusionModel.FinalState[TransformerCacheT] | None" = None
    """:class:`DiffusionModel.FinalState` from the most recent :meth:`generate`. ``None`` until then."""

    autoregressive_index: int | None = None
    """AR step index of the most recent :meth:`generate` (used to assert generate/finalize pairing)."""

    event_profiler: EventProfiler | None = None
    """Current AR step's :class:`EventProfiler` (only when profiling is enabled)."""


class StreamInferencePipeline(
    nn.Module,
    Generic[
        EncCacheT,
        TransformerCacheT,
        DecCacheT,
    ],
):
    """End-to-end inference pipeline for one AR step.

    Generic over the encoder, transformer, and decoder cache types.
    The encoder's input/output types are *not* part of the generic —
    they are forwarded as :class:`typing.Any` so the transformer's
    ``predict_flow`` / ``postprocess_clean_latent`` overrides own the
    typing on the ``input`` argument they receive.

    Per-AR-step usage::

        cache = pipeline.initialize_cache(transformer_context={...})
        for i in range(num_ar_steps):
            output = pipeline.generate(autoregressive_index=i, cache=cache, input=...)
            pipeline.finalize(autoregressive_index=i, cache=cache)
    """

    encoder: Encoder[EncCacheT] | None
    decoder: Decoder[DecCacheT] | None
    diffusion_model: DiffusionModel[TransformerCacheT]

    def __init__(self, config: StreamInferencePipelineConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = config.encoder.setup() if config.encoder is not None else None  # ty:ignore[invalid-assignment]
        self.decoder = config.decoder.setup() if config.decoder is not None else None  # ty:ignore[invalid-assignment]
        self.diffusion_model = config.diffusion_model.setup()  # ty:ignore[invalid-assignment]

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
            A fresh :class:`StreamInferencePipelineCache`.
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
                or ``0`` on the first call after :meth:`initialize_cache`.
            cache: Per-rollout cache returned by :meth:`initialize_cache`.
            input: Raw input fed to the encoder. Required when
                ``self.encoder is not None``; must be ``None`` otherwise.
                To pipe an *already encoded* tensor straight through, use
                :class:`NullEncoderConfig` (an identity encoder).

        Returns:
            Decoded tensor (e.g. RGB video) when a decoder is configured;
            otherwise the unpatchified clean latent straight from the
            diffusion model.

        Note: stashes the :class:`DiffusionModel.FinalState` and the AR
        step index on ``cache`` so the matching :meth:`finalize` can
        consume them. When :attr:`StreamInferencePipelineConfig.enable_sync_and_profile`
        is ``True``, also records per-stage CUDA events on
        ``cache.event_profiler`` for :meth:`finalize` to summarize.
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
                recent :meth:`generate` (asserted to catch drift).
            cache: Same cache used in :meth:`generate`. Consumes
                ``cache.final_state`` (the :class:`DiffusionModel.FinalState`
                stashed there).

        Returns:
            ``None`` when :attr:`StreamInferencePipelineConfig.enable_sync_and_profile`
            is ``False``. Otherwise a flat ``dict[str, float]`` snapshot of
            this AR step's per-stage timings (ms) and current GPU memory
            (GiB), with keys::

                {<stage>_ms for stage in stats},   # e.g. encode_ms, diffuse_ms, ...
                "total_ms",
                "total_ms_wo_finalize",
                "mem_alloc_gib",                    # only if torch.cuda available
                "mem_reserved_gib",
                "mem_peak_gib",

            The same numbers are also emitted via ``logger.info``; the
            return value lets callers persist them (CSV, JSONL, MLflow,
            ...) without re-parsing the log line.
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
