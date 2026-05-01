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

"""Frame-selection + pixel-unshuffle pseudo-VAE encoder.

Stateless drop-in for a learned VAE: select temporal frames per AR step
and unshuffle each frame's 8x8 spatial blocks into channels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from einops import rearrange
from torch import Tensor

from flashdreams.infra.config import InstantiateConfig
from flashdreams.infra.encoder import (
    Encoder,
    EncoderAutoregressiveCache,
)


@dataclass(kw_only=True)
class PixelShuffleVAEEncoderCache(EncoderAutoregressiveCache):
    """AR cache that tracks step index for frame selection."""

    autoregressive_index: int = -1


@dataclass(kw_only=True)
class PixelShuffleVAEEncoderConfig(InstantiateConfig["PixelShuffleVAEEncoder"]):
    """Config for the pixel-shuffle pseudo-VAE encoder."""

    _target: type["PixelShuffleVAEEncoder"] = field(
        default_factory=lambda: PixelShuffleVAEEncoder
    )

    frame_selection_mode: Literal["first_frame", "last_frame"] = "last_frame"
    """Which frame in each 4-frame window to keep."""


class PixelShuffleVAEEncoder(Encoder[PixelShuffleVAEEncoderCache]):
    """Stateless pseudo-VAE: frame select + 8x8 spatial unshuffle.

    Input is a video of shape ``[..., T, C, H, W]`` in ``[-1, 1]``.
    Output latent is ``[..., Tl, C * 64, H/8, W/8]``; ``Tl`` depends on
    the frame-selection mode and the current AR step.
    """

    TEMPORAL_COMPRESSION_RATIO = 4
    SPATIAL_COMPRESSION_RATIO = 8

    def __init__(self, config: PixelShuffleVAEEncoderConfig) -> None:
        super().__init__(config)
        self.config: PixelShuffleVAEEncoderConfig = config

    def initialize_autoregressive_cache(self) -> PixelShuffleVAEEncoderCache:
        return PixelShuffleVAEEncoderCache()

    def forward(
        self,
        input: Tensor,
        autoregressive_index: int = 0,
        cache: PixelShuffleVAEEncoderCache | None = None,
    ) -> Tensor:
        if cache is None:
            cache = self.initialize_autoregressive_cache()
        cache.autoregressive_index = autoregressive_index

        T = input.shape[-4]

        if self.config.frame_selection_mode == "first_frame":
            if autoregressive_index == 0:
                indices = [0] + list(range(1, T, 4))
            else:
                indices = list(range(0, T, 4))
        elif self.config.frame_selection_mode == "last_frame":
            if autoregressive_index == 0:
                indices = [0] + list(range(4, T, 4))
            else:
                indices = list(range(3, T, 4))
        else:
            raise ValueError(
                f"Invalid frame selection mode: {self.config.frame_selection_mode}"
            )

        x = input[..., indices, :, :, :]
        return rearrange(x, "... t c (h h8) (w w8) -> ... t (c h8 w8) h w", h8=8, w8=8)

    @property
    def temporal_compression_ratio(self) -> int:
        return self.TEMPORAL_COMPRESSION_RATIO

    @property
    def spatial_compression_ratio(self) -> int:
        return self.SPATIAL_COMPRESSION_RATIO


if __name__ == "__main__":
    import tyro

    config = tyro.cli(PixelShuffleVAEEncoderConfig)
    model = config.setup()
    print(model)
