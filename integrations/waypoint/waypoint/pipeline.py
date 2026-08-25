# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Waypoint pipeline initialization for an image-established world state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import torch
from torch import Tensor

from flashdreams.infra.pipeline import (
    StreamInferencePipeline,
    StreamInferencePipelineCache,
    StreamInferencePipelineConfig,
)
from flashdreams.recipes.taehv import Hy15TAEHVEncoder, Hy15TAEHVEncoderConfig
from waypoint.controls import WaypointControl
from waypoint.decoder import WaypointTAEHVDecoder
from waypoint.transformer.impl import WaypointTransformerCache


@dataclass(kw_only=True)
class WaypointInferencePipelineConfig(StreamInferencePipelineConfig):
    """Configuration for an image-established Waypoint rollout."""

    _target: type["WaypointInferencePipeline"] = field(
        default_factory=lambda: WaypointInferencePipeline
    )

    seed_encoder: Hy15TAEHVEncoderConfig = field(default_factory=Hy15TAEHVEncoderConfig)
    """Codec encoder that converts the initial displayed image into history."""


class WaypointInferencePipeline(StreamInferencePipeline):
    """Initialize persistent model state from the image that establishes the world."""

    seed_encoder: Hy15TAEHVEncoder

    def __init__(self, config: WaypointInferencePipelineConfig) -> None:
        super().__init__(config)
        self.config: WaypointInferencePipelineConfig = config
        self.seed_encoder = config.seed_encoder.setup()

    @torch.no_grad()
    def initialize_cache(
        self,
        *,
        seed_pixels: Tensor,
        transformer_context: dict[str, Any] | None = None,
        encoder_context: dict[str, Any] | None = None,
        decoder_context: dict[str, Any] | None = None,
    ) -> StreamInferencePipelineCache:
        """Create a cache whose first historical action is the seed image.

        ``seed_pixels`` is four identical RGB frames in the codec's native
        ``[0, 1]`` domain and ``[B, 4, 3, 512, 1024]`` layout. It is committed
        as action zero; callers therefore begin generated actions at index one.
        """
        if seed_pixels.ndim != 5 or tuple(seed_pixels.shape[1:]) != (4, 3, 512, 1024):
            raise ValueError(
                "seed_pixels must have [B, 4, 3, 512, 1024] layout, got "
                f"{tuple(seed_pixels.shape)}"
            )
        if seed_pixels.device != self.device:
            raise ValueError(
                f"seed_pixels is on {seed_pixels.device}, expected {self.device}"
            )

        batch_size = seed_pixels.shape[0]
        transformer_context = dict(transformer_context or {})
        supplied_batch_size = transformer_context.setdefault("batch_size", batch_size)
        if supplied_batch_size != batch_size:
            raise ValueError(
                "transformer_context batch_size must match seed_pixels batch size, got "
                f"{supplied_batch_size} and {batch_size}"
            )
        cache = super().initialize_cache(
            transformer_context=transformer_context,
            encoder_context=encoder_context,
            decoder_context=decoder_context,
        )

        seed_latent = self.seed_encoder.taehv.encode(seed_pixels)
        transformer = self.diffusion_model.transformer
        transformer_cache = cast(WaypointTransformerCache, cache.transformer_cache)
        transformer_cache.start(0)
        transformer_cache.kv_cache.set_frozen(False)
        try:
            transformer.predict_flow(
                seed_latent,
                torch.zeros((), device=seed_latent.device, dtype=seed_latent.dtype),
                transformer_cache,
                WaypointControl(),
            )
        finally:
            transformer_cache.kv_cache.set_frozen(True)

        if not isinstance(self.decoder, WaypointTAEHVDecoder):
            raise TypeError("Waypoint pipeline requires a WaypointTAEHVDecoder")
        if cache.decoder_cache is None:
            raise RuntimeError("Waypoint pipeline requires a decoder cache")
        decoder_input = seed_latent.permute(0, 2, 1, 3, 4).contiguous()
        self.decoder(
            decoder_input,
            autoregressive_index=0,
            cache=cache.decoder_cache,
        )
        cache.autoregressive_index = 0
        return cache
