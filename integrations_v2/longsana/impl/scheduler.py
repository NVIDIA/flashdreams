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

"""LongSana parity wrapper around the shared self-forcing flow scheduler."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor
from tqdm import tqdm

from flashdreams.infra.diffusion.scheduler.fm import (
    FlowMatchScheduler,
    FlowMatchSchedulerConfig,
)
from flashdreams.infra.diffusion.scheduler import FlowPredictor


@dataclass(kw_only=True)
class LongSanaFlowMatchSchedulerConfig(FlowMatchSchedulerConfig):
    """Flow scheduler preserving upstream LongSana precision and RNG layout."""

    _target: type["LongSanaFlowMatchScheduler"] = field(
        default_factory=lambda: LongSanaFlowMatchScheduler
    )


class LongSanaFlowMatchScheduler(FlowMatchScheduler):
    """Four-step self-forcing scheduler with upstream B/T/C noise ordering."""

    config: LongSanaFlowMatchSchedulerConfig

    def sample(
        self,
        initial_noise: Tensor,
        predict_flow: FlowPredictor,
        rng: torch.Generator | None = None,
    ) -> Tensor:
        """Denoise TCHW/BCTHW using upstream double-precision x0 conversion."""
        if initial_noise.ndim not in (4, 5):
            raise ValueError(
                "LongSana scheduler expects TCHW or BCTHW latents, "
                f"got shape {tuple(initial_noise.shape)}."
            )
        input_dtype = initial_noise.dtype
        noisy = initial_noise
        clean: Tensor | None = None
        for index in tqdm(
            range(self.denoising_step_list.shape[0]),
            disable=not self.config.enable_tqdm,
            desc="LongSanaFlowMatchScheduler",
        ):
            sigma = self.denoising_sigmas[index]
            timestep = self.denoising_step_list[index].to(dtype=input_dtype)
            if index > 0:
                if clean is None:
                    raise RuntimeError("LongSana scheduler lost its previous x0.")
                noise = _upstream_renoise_tensor(noisy, rng)
                noisy = ((1.0 - sigma) * clean + sigma * noise).to(input_dtype)
            flow = predict_flow(noisy, timestep)
            clean = (noisy.double() - sigma.double() * flow.double()).to(input_dtype)
        if clean is None:
            raise RuntimeError("LongSana denoising timestep list is empty.")
        return clean


def _upstream_renoise_tensor(
    like: Tensor,
    rng: torch.Generator | None,
) -> Tensor:
    """Draw noise in upstream's flattened B/T/C/H/W element order."""
    unbatched = like.ndim == 4
    if unbatched:
        frames, channels, height, width = like.shape
        batch = 1
    else:
        batch, channels, frames, height, width = like.shape
    noise = torch.randn(
        (batch * frames, channels, height, width),
        device=like.device,
        dtype=like.dtype,
        generator=rng,
    )
    if unbatched:
        return noise
    noise = noise.unflatten(0, (batch, frames)).permute(0, 2, 1, 3, 4).contiguous()
    return noise
