# SPDX-FileCopyrightText: Copyright (c) 2026 SwiftVR Authors.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# This file has been modified by NVIDIA CORPORATION & AFFILIATES.

"""SwiftVR streaming restoration encoder."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from flashdreams.core.checkpoint.load import load_checkpoint
from flashdreams.infra.encoder import (
    EncoderConfig,
    StreamingEncoder,
    StreamingEncoderCache,
)
from flashdreams.recipes.taehv.checkpoint import legacy_to_blocks_keys
from flashdreams.recipes.taehv.impl import Encoder as TAEHVEncoder
from flashdreams.recipes.taehv.impl import MemBlock


def _encode_complete_groups(
    network: TAEHVEncoder,
    tensor: Tensor,
    state: dict[int, Tensor],
) -> Tensor:
    """Run a complete ReAE temporal group through the shared TAEHV blocks."""
    batch, time, channels, height, width = tensor.shape
    assert time > 0 and time % 4 == 0

    # ponytail: This fast path requires complete four-frame groups; use
    # TAEHVEncoder.forward with TAEHVEncoderCache if that contract changes.
    tensor = tensor.reshape(batch * time, channels, height, width)
    for block in network.blocks:
        if isinstance(block, MemBlock):
            _, channels, height, width = tensor.shape
            key = id(block)
            if key not in state:
                state[key] = tensor.new_zeros(batch, 1, channels, height, width)
            tensor = block.cache_step(tensor, state, batch)
        else:
            tensor = block(tensor)

    _, channels, height, width = tensor.shape
    return tensor.reshape(batch, -1, channels, height, width)


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
    state: dict[int, Tensor] = field(default_factory=dict)
    tail: Tensor | None = None


class SwiftVREncoder(StreamingEncoder[SwiftVREncoderCache]):
    """Resize input frames and stream them through the ReAE encoder."""

    spatial_compression_ratio = 16
    patch_size = 2

    def __init__(self, config: SwiftVREncoderConfig) -> None:
        super().__init__(config)
        self.config: SwiftVREncoderConfig = config
        self.network = TAEHVEncoder(
            latent_channels=48,
            image_channels=3,
            patch_size=self.patch_size,
            act_func=nn.ReLU(inplace=True),
        )
        if config.checkpoint_path is not None:
            state_dict = legacy_to_blocks_keys(
                load_checkpoint(config.checkpoint_path, map_location="cpu")
            )
            encoder_state = {
                key.removeprefix("encoder."): value
                for key, value in state_dict.items()
                if key.startswith("encoder.")
            }
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
            self.patch_size,
        ).reshape(
            batch,
            time,
            -1,
            height // self.patch_size,
            width // self.patch_size,
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
        return _encode_complete_groups(self.network, tensor, cache.state)

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
        return _encode_complete_groups(self.network, tensor, cache.state)


__all__ = [
    "SwiftVREncoder",
    "SwiftVREncoderCache",
    "SwiftVREncoderConfig",
]
