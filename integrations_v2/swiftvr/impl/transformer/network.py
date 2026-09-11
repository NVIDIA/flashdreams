# SPDX-FileCopyrightText: Copyright (c) 2026 SwiftVR Authors.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# This file has been modified by NVIDIA CORPORATION & AFFILIATES.

"""FlashDreams WAN network specialized with SwiftVR attention blocks."""

from __future__ import annotations

from dataclasses import dataclass, field

from flashdreams.recipes.wan.transformer.impl.modules import Block
from flashdreams.recipes.wan.transformer.impl.network import (
    WanDiTNetwork,
    WanDiTNetworkTI2V5BConfig,
)
from swiftvr.impl.attention import SwiftVRBlock


@dataclass(kw_only=True)
class SwiftVRDiTNetworkConfig(WanDiTNetworkTI2V5BConfig):
    """WAN TI2V-5B dimensions with SwiftVR shifted-window attention."""

    _target: type["SwiftVRDiTNetwork"] = field(
        default_factory=lambda: SwiftVRDiTNetwork
    )

    attention_window: tuple[int, int] = (16, 16)
    """Spatial latent-token window used by alternating shifted blocks."""


class SwiftVRDiTNetwork(WanDiTNetwork):
    """FlashDreams TI2V-5B WAN network specialized with SwiftVR blocks."""

    def __init__(self, config: SwiftVRDiTNetworkConfig) -> None:
        if any(size <= 0 for size in config.attention_window):
            raise ValueError(
                "SwiftVR attention_window values must be positive, got "
                f"{config.attention_window}."
            )
        self.attention_window = config.attention_window
        super().__init__(config)

    def _build_block(self, layer_idx: int) -> Block:
        return SwiftVRBlock(
            dim=self.dim,
            ffn_dim=self.ffn_dim,
            num_heads=self.num_heads,
            cross_attn_norm=self.cross_attn_norm,
            eps=self.eps,
            window=self.attention_window,
            shifted=bool(layer_idx % 2),
        )


__all__ = ["SwiftVRDiTNetwork", "SwiftVRDiTNetworkConfig"]
