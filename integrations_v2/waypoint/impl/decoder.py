# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TAEHV layout adapter for the Waypoint video pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

from torch import Tensor

from flashdreams.recipes.taehv import Hy15TAEHVDecoder, Hy15TAEHVDecoderConfig
from flashdreams.recipes.taehv.impl import TAEHVCache


@dataclass(kw_only=True)
class WaypointTAEHVDecoderConfig(Hy15TAEHVDecoderConfig):
    """Config for the matching TAEHV decoder behind the Waypoint pipeline."""

    _target: type["WaypointTAEHVDecoder"] = field(
        default_factory=lambda: WaypointTAEHVDecoder
    )


class WaypointTAEHVDecoder(Hy15TAEHVDecoder):
    """Decode FlashDreams video latents using TAEHV's frame-first layout."""

    def forward(
        self,
        input: Tensor,
        autoregressive_index: int = 0,
        cache: TAEHVCache | None = None,
    ) -> Tensor:
        """Decode a Waypoint action into ``[B, T, C, H, W]`` RGB frames.

        Args:
            input: FlashDreams latent video in ``[B, C, T, H, W]`` layout.
            autoregressive_index: Current latent action index.
            cache: Long-lived causal TAEHV state.

        Returns:
            Decoded RGB video in ``[B, T, C, H, W]`` layout and ``[-1, 1]`` range.

        Raises:
            ValueError: The latent does not use FlashDreams' five-dimensional layout.
        """
        if input.ndim != 5:
            raise ValueError(
                "Waypoint TAEHV input must have [B, C, T, H, W] layout, got "
                f"{tuple(input.shape)}"
            )
        return super().forward(
            input.permute(0, 2, 1, 3, 4).contiguous(),
            autoregressive_index=autoregressive_index,
            cache=cache,
        )
