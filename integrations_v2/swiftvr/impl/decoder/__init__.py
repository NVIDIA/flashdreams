# SPDX-FileCopyrightText: Copyright (c) 2026 SwiftVR Authors.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SwiftVR streaming restoration decoder."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from flashdreams.core.checkpoint.load import load_checkpoint
from flashdreams.infra.decoder import (
    DecoderConfig,
    StreamingDecoder,
    StreamingDecoderCache,
)
from swiftvr.impl.autoencoder import apply_with_boundaries, split_reae_state_dict
from swiftvr.impl.decoder.network import SwiftVRDecoderNetwork


@dataclass(kw_only=True)
class SwiftVRDecoderConfig(DecoderConfig):
    """Configuration for :class:`SwiftVRDecoder`."""

    _target: type["SwiftVRDecoder"] = field(default_factory=lambda: SwiftVRDecoder)

    checkpoint_path: str | None = None
    """Path to ``reae.safetensors``; ``None`` leaves random test weights."""

    dtype: torch.dtype = torch.bfloat16
    """Decoder compute dtype."""


@dataclass(kw_only=True)
class SwiftVRDecoderCache(StreamingDecoderCache):
    """Per-stream decoder boundary state and cold-start trim flag."""

    output_height: int
    output_width: int
    state: dict[str, Tensor | None] | None = None
    first_decode: bool = True


class SwiftVRDecoder(StreamingDecoder[SwiftVRDecoderCache]):
    """Stream ReAE latents into cropped RGB output frames."""

    def __init__(self, config: SwiftVRDecoderConfig) -> None:
        super().__init__(config)
        self.config: SwiftVRDecoderConfig = config
        self.network = SwiftVRDecoderNetwork()
        if config.checkpoint_path is not None:
            _, decoder_state = split_reae_state_dict(
                load_checkpoint(config.checkpoint_path, map_location="cpu")
            )
            self.network.load_state_dict(decoder_state, strict=True)
        self.network.to(dtype=config.dtype).eval().requires_grad_(False)

    def initialize_autoregressive_cache(
        self,
        *,
        output_height: int,
        output_width: int,
        **_unused: Any,
    ) -> SwiftVRDecoderCache:
        """Create temporal decoder state for one output shape."""
        return SwiftVRDecoderCache(
            output_height=output_height,
            output_width=output_width,
        )

    def forward(  # type: ignore[override]
        self,
        input: Tensor,
        autoregressive_index: int = 0,
        cache: SwiftVRDecoderCache | None = None,
    ) -> Tensor | None:
        """Decode ``[B,C,T,H,W]`` latents to ``[B,T,3,H,W]`` frames."""
        del autoregressive_index
        assert cache is not None, "SwiftVRDecoder requires a cache"
        tensor = input.permute(0, 2, 1, 3, 4).contiguous()
        decoded, cache.state = apply_with_boundaries(self.network, tensor, cache.state)
        if decoded is None:
            return None
        decoded = decoded.clamp_(0, 1)
        batch, time, channels, height, width = decoded.shape
        decoded = F.pixel_shuffle(
            decoded.reshape(batch * time, channels, height, width),
            self.network.patch_size,
        ).reshape(
            batch,
            time,
            3,
            height * self.network.patch_size,
            width * self.network.patch_size,
        )
        if cache.first_decode:
            decoded = decoded[:, self.network.frames_to_trim :]
            cache.first_decode = False
        return decoded[:, :, :, : cache.output_height, : cache.output_width]


__all__ = [
    "SwiftVRDecoder",
    "SwiftVRDecoderCache",
    "SwiftVRDecoderConfig",
]
