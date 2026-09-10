# SPDX-FileCopyrightText: Copyright (c) 2026 SwiftVR Authors.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# This file has been modified by NVIDIA CORPORATION & AFFILIATES.

"""SwiftVR streaming restoration decoder."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

from flashdreams.infra.decoder import (
    DecoderConfig,
    StreamingDecoder,
    StreamingDecoderCache,
)
from flashdreams.recipes.taehv.impl import TAEHVCache
from swiftvr.impl.decoder.network import SwiftVRTAEHV


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
    taehv: TAEHVCache


class SwiftVRDecoder(StreamingDecoder[SwiftVRDecoderCache]):
    """Stream ReAE latents into cropped RGB output frames."""

    def __init__(self, config: SwiftVRDecoderConfig) -> None:
        super().__init__(config)
        self.config: SwiftVRDecoderConfig = config
        self.network = SwiftVRTAEHV(config.checkpoint_path)
        if config.checkpoint_path is None:
            self.network.to_empty(device="cpu")
            for module in self.network.modules():
                if isinstance(module, (nn.Conv2d, nn.Conv3d)):
                    module.reset_parameters()
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
            taehv=self.network.prepare_cache(),
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
        decoded = self.network.decode(
            input.permute(0, 2, 1, 3, 4).contiguous(),
            cache=cache.taehv,
        )
        return decoded[:, :, :, : cache.output_height, : cache.output_width]


__all__ = [
    "SwiftVRDecoder",
    "SwiftVRDecoderCache",
    "SwiftVRDecoderConfig",
]
