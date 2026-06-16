# SPDX-FileCopyrightText: Copyright (c) 2026 Hongyu Zhou
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
"""Robotwin action normalization and channel selection for LingBot-VA."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from lingbot_va.constants import (
    ROBOTWIN_ACTION_DIM,
    ROBOTWIN_ACTION_Q01,
    ROBOTWIN_ACTION_Q99,
    ROBOTWIN_USED_ACTION_CHANNEL_IDS,
)


@dataclass(frozen=True)
class LingbotVAActionProcessorConfig:
    """Pure-Python config for Robotwin action preprocessing/postprocessing."""

    action_dim: int = ROBOTWIN_ACTION_DIM
    used_action_channel_ids: tuple[int, ...] = ROBOTWIN_USED_ACTION_CHANNEL_IDS
    q01: tuple[float, ...] = ROBOTWIN_ACTION_Q01
    q99: tuple[float, ...] = ROBOTWIN_ACTION_Q99
    norm_method: str = "quantiles"


@dataclass
class LingbotVAActionProcessor:
    """Normalize, mask, and denormalize Robotwin actions exactly like upstream."""

    config: LingbotVAActionProcessorConfig = field(
        default_factory=LingbotVAActionProcessorConfig
    )

    @property
    def inverse_used_action_channel_ids(self) -> tuple[int, ...]:
        inverse = [len(self.config.used_action_channel_ids)] * self.config.action_dim
        for i, channel in enumerate(self.config.used_action_channel_ids):
            inverse[channel] = i
        return tuple(inverse)

    def action_mask(self, *, device: torch.device | None = None) -> Tensor:
        mask = torch.zeros([self.config.action_dim], dtype=torch.bool, device=device)
        mask[list(self.config.used_action_channel_ids)] = True
        return mask

    def q01_tensor(self, *, device: torch.device | None = None) -> Tensor:
        return torch.tensor(self.config.q01, dtype=torch.float32, device=device).reshape(-1, 1, 1)

    def q99_tensor(self, *, device: torch.device | None = None) -> Tensor:
        return torch.tensor(self.config.q99, dtype=torch.float32, device=device).reshape(-1, 1, 1)

    def preprocess(self, action: Tensor) -> Tensor:
        """Normalize a raw action tensor of shape ``[C_used_or_full, F, H]``.

        Mirrors upstream: pad one action channel, expand selected channels
        to the model's full 30-channel order via ``inverse_used_action_channel_ids``,
        quantile-normalize to roughly ``[-1, 1]``, then return
        ``[1, action_dim, F, H, 1]``.
        """
        assert action.ndim == 3, f"expected [C, F, H], got {tuple(action.shape)}"
        padded = torch.nn.functional.pad(action, [0, 0, 0, 0, 0, 1], mode="constant", value=0)
        expanded = padded[list(self.inverse_used_action_channel_ids)]
        if self.config.norm_method != "quantiles":
            raise NotImplementedError(self.config.norm_method)
        q01 = self.q01_tensor(device=expanded.device)
        q99 = self.q99_tensor(device=expanded.device)
        expanded = (expanded - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        return expanded.unsqueeze(0).unsqueeze(-1)

    def zero_unused_channels(self, action: Tensor) -> Tensor:
        """Return a copy with all non-Robotwin-used channels set to zero."""
        masked = action.clone()
        masked[:, ~self.action_mask(device=masked.device)] = 0
        return masked

    def postprocess(self, action: Tensor) -> Tensor:
        """Denormalize model action output and return selected Robotwin channels.

        Args:
            action: Tensor with shape ``[B, 30, F, H, 1]``.

        Returns:
            Tensor with shape ``[len(used_action_channel_ids), F, H]``.
        """
        assert action.ndim == 5, f"expected [B, C, F, H, W], got {tuple(action.shape)}"
        action_cpu = action[0, ..., 0].detach().cpu()
        if self.config.norm_method != "quantiles":
            raise NotImplementedError(self.config.norm_method)
        q01 = self.q01_tensor()
        q99 = self.q99_tensor()
        denorm = (action_cpu + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        return denorm[list(self.config.used_action_channel_ids)]
