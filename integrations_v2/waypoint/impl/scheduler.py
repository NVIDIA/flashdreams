# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Waypoint's checkpoint-specific rectified-flow Euler schedule."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import Tensor

from flashdreams.infra.diffusion.scheduler.fm_euler import (
    FlowMatchEulerDiscreteScheduler,
    FlowMatchEulerDiscreteSchedulerConfig,
)


@dataclass(kw_only=True)
class WaypointEulerSchedulerConfig(FlowMatchEulerDiscreteSchedulerConfig):
    """Fixed four-step scheduler whose BF16 arithmetic matches Waypoint."""

    _target: type["WaypointEulerScheduler"] = field(
        default_factory=lambda: WaypointEulerScheduler
    )


class WaypointEulerScheduler(FlowMatchEulerDiscreteScheduler):
    """Apply Euler steps from a BF16-quantized fixed sigma schedule.

    The checkpoint's inference path stores its schedule in the same BF16 dtype
    as its latent. Computing adjacent differences after that quantization is
    part of the learned four-step trajectory, not merely a storage choice.
    """

    def sample(
        self,
        initial_noise: Tensor,
        predict_flow: Callable[[Tensor, Tensor], Tensor],
        rng: torch.Generator | None = None,
    ) -> Tensor:
        """Denoise one action with checkpoint-equivalent Euler updates."""
        del rng
        schedule = self.sigmas.to(
            device=initial_noise.device, dtype=initial_noise.dtype
        )
        noisy = initial_noise
        for sigma, next_sigma in zip(schedule[:-1], schedule[1:], strict=True):
            flow = predict_flow(noisy, sigma)
            noisy = (noisy.float() + (next_sigma - sigma).float() * flow.float()).to(
                dtype=initial_noise.dtype
            )
        return noisy
