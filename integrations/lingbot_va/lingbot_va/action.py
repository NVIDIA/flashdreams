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

    def __post_init__(self) -> None:
        """Reject inconsistent action schemas before tensor processing starts."""
        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if len(self.q01) != self.action_dim or len(self.q99) != self.action_dim:
            raise ValueError("q01 and q99 must each contain action_dim values")
        if len(set(self.used_action_channel_ids)) != len(self.used_action_channel_ids):
            raise ValueError("used_action_channel_ids must be unique")
        if any(
            channel < 0 or channel >= self.action_dim
            for channel in self.used_action_channel_ids
        ):
            raise ValueError("used_action_channel_ids must be within action_dim")
        if self.norm_method != "quantiles":
            raise ValueError(f"unsupported norm_method: {self.norm_method!r}")


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
        return torch.tensor(
            self.config.q01,
            dtype=torch.float32,
            device=device,
        ).reshape(-1, 1, 1)

    def q99_tensor(self, *, device: torch.device | None = None) -> Tensor:
        return torch.tensor(
            self.config.q99,
            dtype=torch.float32,
            device=device,
        ).reshape(-1, 1, 1)

    def preprocess(self, action: Tensor) -> Tensor:
        """Normalize a raw action tensor of shape ``[C_used_or_full, F, H]``.

        Mirrors upstream: pad one action channel, expand selected channels
        to the model's full 30-channel order via ``inverse_used_action_channel_ids``,
        quantile-normalize to roughly ``[-1, 1]``, then return
        ``[1, action_dim, F, H, 1]``.
        """
        expected_channels = len(self.config.used_action_channel_ids)
        if action.ndim != 3 or action.shape[0] != expected_channels:
            raise ValueError(
                f"expected [{expected_channels}, F, H], got {tuple(action.shape)}"
            )
        padded = torch.nn.functional.pad(
            action,
            [0, 0, 0, 0, 0, 1],
            mode="constant",
            value=0,
        )
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
            Tensor with shape ``[F * H, len(used_action_channel_ids)]``.
        """
        if (
            action.ndim != 5
            or action.shape[0] != 1
            or action.shape[1] != self.config.action_dim
            or action.shape[-1] != 1
        ):
            raise ValueError(
                f"expected [1, {self.config.action_dim}, F, H, 1], "
                f"got {tuple(action.shape)}"
            )
        action_cpu = action[0, ..., 0].detach().cpu()
        if self.config.norm_method != "quantiles":
            raise NotImplementedError(self.config.norm_method)
        q01 = self.q01_tensor()
        q99 = self.q99_tensor()
        denorm = (action_cpu + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        selected = denorm[list(self.config.used_action_channel_ids)]
        return selected.permute(1, 2, 0).reshape(
            -1, len(self.config.used_action_channel_ids)
        )
