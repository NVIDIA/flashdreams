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

"""Tiny per-AR-step control encoder for the template recipe."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from flashdreams.infra.config import InstantiateConfig
from flashdreams.infra.encoder import StreamingEncoder, StreamingEncoderCache


@dataclass(kw_only=True)
class TemplateControlEncoderConfig(InstantiateConfig["TemplateControlEncoder"]):
    """Config for the template control encoder.

    Point-wise ``Conv3d`` projecting a dummy control channel stack to
    the transformer's ``in_channels``. Real recipes swap in an
    HDMap / camera-control / depth encoder here.
    """

    _target: type["TemplateControlEncoder"] = field(
        default_factory=lambda: TemplateControlEncoder
    )

    control_channels: int = 8
    """Channels of the raw control tensor passed to :meth:`TemplateControlEncoder.forward`."""

    out_channels: int = 4
    """Channels emitted; must match the transformer's ``network.in_channels``."""

    dtype: torch.dtype = torch.bfloat16
    """Parameter and activation dtype for the projection."""


class TemplateControlEncoder(StreamingEncoder[StreamingEncoderCache]):
    """Stateless per-token control encoder.

    ``[B, C_ctrl, T, H, W] → [B, C_latent, T, H, W]`` — the
    pre-patchify shape consumed by the transformer. Stateless; the
    cache exists only to satisfy the :class:`Encoder` contract.
    """

    def __init__(self, config: TemplateControlEncoderConfig) -> None:
        super().__init__(config)
        self.config: TemplateControlEncoderConfig = config
        self.proj = nn.Conv3d(
            config.control_channels, config.out_channels, kernel_size=1
        ).to(dtype=config.dtype)
        self.proj.eval()

    def initialize_autoregressive_cache(self, **_context: Any) -> StreamingEncoderCache:
        """Return an empty cache (stateless encoder)."""
        return StreamingEncoderCache()

    @torch.no_grad()
    def forward(
        self,
        input: Tensor,
        autoregressive_index: int = 0,
        cache: StreamingEncoderCache | None = None,
    ) -> Tensor:
        """Project ``[B, C_ctrl, T, H, W]`` control to ``[B, C_latent, T, H, W]``."""
        assert input.ndim == 5, (
            f"TemplateControlEncoder expects [B, C, T, H, W], got {tuple(input.shape)}."
        )
        _ = autoregressive_index, cache  # stateless
        return self.proj(input.to(dtype=self.config.dtype))
