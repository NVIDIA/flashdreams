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

"""Diffusion scheduler interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol

import torch
import torch.nn as nn
from torch import Tensor

from flashdreams.infra.config import InstantiateConfig


class FlowPredictor(Protocol):
    """Closure ``(noisy_latent, timestep) -> predicted_flow``.

    Built by ``DiffusionModel.generate`` by binding the per-AR-step ``cache``
    / ``input`` to the transformer's ``predict_flow``. A scheduler invokes
    it once per denoising iteration. The scheduler decides the timestep
    dtype (UniPC uses int64, flow-match uses float).
    """

    def __call__(self, noisy_latent: Tensor, timestep: Tensor) -> Tensor:
        """Predict the flow at ``timestep``."""
        ...


class Scheduler(nn.Module, ABC):
    """Denoising scheduler.

    Owns the entire denoising loop. Callers see only ``noise → clean``;
    the loop shape (renoise / multistep / plain ODE) is private.

    Concrete configs inherit ``SchedulerConfig`` and declare their own
    ``num_inference_steps`` / ``shift`` fields (the base holds no
    shared dataclass fields).

    Examples:

        scheduler = config.setup()
        clean = scheduler.sample(initial_noise=noise, predict_flow=predictor)
        noisy = scheduler.add_noise(clean_input=clean, timestep=t)
    """

    def __init__(self, config: SchedulerConfig) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def sample(
        self,
        initial_noise: Tensor,
        predict_flow: FlowPredictor,
        rng: torch.Generator | None = None,
    ) -> Tensor:
        """Run the full denoising loop and return the clean latent.

        Schedulers are shape-agnostic: every internal op broadcasts against
        per-step scalar sigmas. In practice ``initial_noise`` is a video
        latent ``[B, C, T, H, W]``, conventionally treated as a sample at
        ``sigma=1``.

        Args:
            initial_noise: Gaussian noise on the caller's device/dtype.
            predict_flow: Per-step closure invoked
                ``num_inference_steps`` times.
            rng: Generator on the same device. Used by self-forcing renoise
                loops; pure ODE solvers ignore it.

        Returns:
            Clean latent with the same shape, device, and dtype as
            ``initial_noise``.
        """

    @abstractmethod
    def add_noise(
        self,
        clean_input: Tensor,
        timestep: Tensor,
        rng: torch.Generator | None = None,
    ) -> Tensor:
        """Apply the forward corruption ``x_t = (1 - sigma(t)) * x_0 + sigma(t) * eps``.

        Timestep value semantics are scheduler-specific: FlowMatch snaps to
        the nearest entry of its 1000-step training table; FlowUniPC
        requires exact membership in its inference schedule.
        """


@dataclass(kw_only=True)
class SchedulerConfig(InstantiateConfig):
    """Category base for every denoising-scheduler config."""

    _target: type = field(default_factory=lambda: Scheduler)
