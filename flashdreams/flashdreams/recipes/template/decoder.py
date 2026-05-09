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

"""Tiny latent-to-pixel decoder for the template recipe."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from flashdreams.infra.decoder import (
    DecoderConfig,
    StreamingDecoder,
    StreamingDecoderCache,
)


@dataclass(kw_only=True)
class TemplateDecoderConfig(DecoderConfig):
    """Config for the template decoder.

    Point-wise ``Conv3d`` mapping latent channels to output channels.
    Real recipes replace with a VAE / TAEHV / etc.
    """

    _target: type = field(default_factory=lambda: TemplateDecoder)

    in_channels: int = 4
    """Channels of the clean latent produced by the transformer."""

    out_channels: int = 3
    """Channels emitted by the decoder (RGB-like)."""

    dtype: torch.dtype = torch.bfloat16
    """Parameter and activation dtype for the projection."""


class TemplateDecoder(StreamingDecoder[StreamingDecoderCache]):
    """Stateless per-token latent-to-pixel decoder.

    ``[B, C_latent, T, H, W] → [B, C_out, T, H, W]``. Stateless; the
    cache exists only to satisfy the :class:`StreamingDecoder` contract.
    """

    def __init__(self, config: TemplateDecoderConfig) -> None:
        super().__init__(config)
        self.config: TemplateDecoderConfig = config
        self.proj = nn.Conv3d(
            config.in_channels, config.out_channels, kernel_size=1
        ).to(dtype=config.dtype)
        self.proj.eval()

    def initialize_autoregressive_cache(self, **_context: Any) -> StreamingDecoderCache:
        """Return an empty cache (stateless decoder)."""
        return StreamingDecoderCache()

    @torch.no_grad()
    def forward(
        self,
        input: Tensor,
        autoregressive_index: int = 0,
        cache: StreamingDecoderCache | None = None,
    ) -> Tensor:
        """Project ``[B, C_latent, T, H, W]`` latent to ``[B, C_out, T, H, W]``."""
        assert input.ndim == 5, (
            f"TemplateDecoder expects [B, C, T, H, W], got {tuple(input.shape)}."
        )
        _ = autoregressive_index, cache  # stateless
        return self.proj(input.to(dtype=self.config.dtype))
