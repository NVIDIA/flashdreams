# SPDX-FileCopyrightText: Copyright (c) 2026 SwiftVR Authors.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SwiftVR streaming restoration encoder."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from flashdreams.core.checkpoint.load import load_checkpoint
from flashdreams.infra.encoder import (
    EncoderConfig,
    StreamingEncoder,
    StreamingEncoderCache,
)
from swiftvr.impl.autoencoder import apply_with_boundaries, split_reae_state_dict
from swiftvr.impl.encoder.network import SwiftVREncoderNetwork


@dataclass(kw_only=True)
class SwiftVREncoderConfig(EncoderConfig):
    """Configuration for :class:`SwiftVREncoder`."""

    _target: type["SwiftVREncoder"] = field(default_factory=lambda: SwiftVREncoder)

    checkpoint_path: str | None = None
    """Path to ``reae.safetensors``; ``None`` leaves random test weights."""

    dtype: torch.dtype = torch.bfloat16
    """Encoder compute dtype."""


@dataclass(kw_only=True)
class SwiftVREncoderCache(StreamingEncoderCache):
    """Per-stream encoder boundary state and incomplete temporal group."""

    output_height: int
    output_width: int
    pad_height: int
    pad_width: int
    state: dict[str, Tensor | None] | None = None
    tail: Tensor | None = None


class SwiftVREncoder(StreamingEncoder[SwiftVREncoderCache]):
    """Resize input frames and stream them through the ReAE encoder."""

    spatial_compression_ratio = 16

    def __init__(self, config: SwiftVREncoderConfig) -> None:
        super().__init__(config)
        self.config: SwiftVREncoderConfig = config
        self.network = SwiftVREncoderNetwork()
        if config.checkpoint_path is not None:
            encoder_state, _ = split_reae_state_dict(
                load_checkpoint(config.checkpoint_path, map_location="cpu")
            )
            self.network.load_state_dict(encoder_state, strict=True)
        self.network.to(dtype=config.dtype).eval().requires_grad_(False)

    @property
    def device(self) -> torch.device:
        return next(self.network.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.network.parameters()).dtype

    def initialize_autoregressive_cache(
        self,
        *,
        output_height: int,
        output_width: int,
        **_unused: Any,
    ) -> SwiftVREncoderCache:
        """Create temporal state for one output shape."""
        if output_height <= 0 or output_width <= 0:
            raise ValueError(
                "SwiftVR output dimensions must be positive, got "
                f"{output_height}x{output_width}."
            )
        return SwiftVREncoderCache(
            output_height=output_height,
            output_width=output_width,
            pad_height=(-output_height) % 32,
            pad_width=(-output_width) % 32,
        )

    def preprocess(self, frames: Tensor, cache: SwiftVREncoderCache) -> Tensor:
        """Convert ``[T,H,W,3]`` uint8 input to padded ``[B,T,C,H,W]``."""
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(
                "SwiftVR encoder expects [T,H,W,3] frames, got "
                f"shape={tuple(frames.shape)}."
            )
        frames = frames.to(self.device)
        frames = frames.permute(0, 3, 1, 2).to(dtype=self.dtype)
        frames = F.interpolate(
            frames,
            size=(cache.output_height, cache.output_width),
            mode="bilinear",
            align_corners=False,
        ).div_(255)
        if cache.pad_height or cache.pad_width:
            frames = F.pad(frames, (0, cache.pad_width, 0, cache.pad_height))
        return frames.unsqueeze(0)

    def forward(  # type: ignore[override]
        self,
        input: Tensor,
        autoregressive_index: int = 0,
        cache: SwiftVREncoderCache | None = None,
    ) -> Tensor | None:
        """Encode every complete four-frame group and retain the tail."""
        del autoregressive_index
        assert cache is not None, "SwiftVREncoder requires a cache"
        tensor = self.preprocess(input, cache)
        batch, time, channels, height, width = tensor.shape
        tensor = F.pixel_unshuffle(
            tensor.reshape(batch * time, channels, height, width),
            self.network.patch_size,
        ).reshape(
            batch,
            time,
            -1,
            height // self.network.patch_size,
            width // self.network.patch_size,
        )
        if cache.tail is not None:
            tensor = torch.cat([cache.tail, tensor], dim=1)
        remainder = tensor.shape[1] % 4
        if remainder:
            cache.tail = tensor[:, -remainder:].detach().clone()
            tensor = tensor[:, :-remainder]
        else:
            cache.tail = None
        if tensor.shape[1] == 0:
            return None
        encoded, cache.state = apply_with_boundaries(self.network, tensor, cache.state)
        return encoded

    def flush(self, cache: SwiftVREncoderCache) -> Tensor | None:
        """Replicate-pad and encode the final partial temporal group."""
        if cache.tail is None:
            return None
        tensor = cache.tail
        cache.tail = None
        padding = (-tensor.shape[1]) % 4
        if padding:
            tensor = torch.cat(
                [tensor, tensor[:, -1:].expand(-1, padding, -1, -1, -1)], dim=1
            )
        encoded, cache.state = apply_with_boundaries(self.network, tensor, cache.state)
        return encoded


__all__ = [
    "SwiftVREncoder",
    "SwiftVREncoderCache",
    "SwiftVREncoderConfig",
]
