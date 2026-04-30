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

"""Autoregressive diffusion model: scheduler + transformer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic

import torch
import torch.nn as nn
from torch import Tensor

from flashdreams.infra.config import InstantiateConfig
from flashdreams.infra.diffusion.scheduler import (
    Scheduler,
    SchedulerConfig,
)
from flashdreams.infra.diffusion.transformer import (
    Transformer,
    TransformerCacheT,
    TransformerConfig,
)


@dataclass(kw_only=True)
class DiffusionModelConfig(InstantiateConfig["DiffusionModel"]):
    """Hyperparameters for a :class:`DiffusionModel`."""

    _target: type["DiffusionModel"] = field(default_factory=lambda: DiffusionModel)

    transformer: TransformerConfig
    """Flow-prediction network config."""

    scheduler: SchedulerConfig
    """Denoising-loop config."""

    seed: int | None = None
    """RNG seed for the initial-noise draw and scheduler sampling. ``None`` uses the global RNG."""

    context_noise: int = 0
    """Timestep used by :meth:`DiffusionModel.finalize` for the AR cache-update forward. ``0`` skips :meth:`Scheduler.add_noise`."""


class DiffusionModel(nn.Module, Generic[TransformerCacheT]):
    """Autoregressive diffusion model: scheduler + transformer.

    Generic over the concrete ``TransformerAutoregressiveCache`` subclass
    so user-facing typing on ``cache`` is preserved end-to-end.

    Example::

        model = config.setup().to("cuda")
        cache = model.transformer.initialize_autoregressive_cache(...)
        clean, final_state = model.generate(autoregressive_index=0, cache=cache)
        model.finalize(final_state)  # advance KV cache for the next AR step
    """

    @dataclass(kw_only=True)
    class FinalState(Generic[TransformerCacheT]):  # ty:ignore[shadowed-type-variable]
        """State passed from :meth:`generate` to :meth:`finalize`."""

        clean_latent: Tensor
        """Patchified clean latent from the end of the denoising loop."""

        autoregressive_index: int
        """AR step index this state was produced at."""

        cache: TransformerCacheT
        """Long-lived AR cache used during generation."""

        input: Any = None
        """Per-AR-step encoder output (already patchified), or ``None`` if the pipeline has no encoder."""

    transformer: Transformer[TransformerCacheT]
    scheduler: Scheduler

    def __init__(self, config: DiffusionModelConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = self.config.transformer.setup()  # ty:ignore[invalid-assignment]
        self.scheduler = self.config.scheduler.setup()
        self._rng: torch.Generator | None = None

    @property
    def device(self) -> torch.device:
        return self.transformer.device

    @property
    def dtype(self) -> torch.dtype:
        return self.transformer.dtype

    @property
    def rng(self) -> torch.Generator | None:
        """Per-model :class:`torch.Generator` (lazily built on the current device).

        Returns ``None`` when :attr:`DiffusionModelConfig.seed` is ``None``.
        Rebuilt the first time the model's device changes after a ``.to(...)``.

        Warning: a device move resets the RNG stream — fine for the
        usual "construct on CPU, ``.to(gpu)`` once" workflow, but
        mid-rollout device hops will lose RNG state.
        """
        if self.config.seed is None:
            return None
        if self._rng is None or self._rng.device != self.device:
            self._rng = torch.Generator(device=self.device).manual_seed(
                self.config.seed
            )
        return self._rng

    @property
    def latent_shape(self) -> tuple[int, ...]:
        return self.transformer.latent_shape

    def generate(
        self,
        autoregressive_index: int,
        cache: TransformerCacheT,
        input: Any = None,
    ) -> tuple[Tensor, "DiffusionModel.FinalState[TransformerCacheT]"]:
        """Run the denoising loop for one AR step.

        Args:
            autoregressive_index: Index of this AR step.
            cache: Long-lived AR cache; mutated in place across AR steps.
            input: Optional per-AR-step encoder output. When provided it
                is patchified via :meth:`Transformer.patchify_and_maybe_split_cp`
                and forwarded to :meth:`Transformer.predict_flow` /
                :meth:`Transformer.postprocess_clean_latent`, then saved
                on the returned :class:`FinalState` for re-use in
                :meth:`finalize`.

        Returns:
            ``(clean_latent, final_state)`` where ``clean_latent`` is the
            unpatchified clean latent and ``final_state`` should be passed
            to :meth:`finalize` to advance the AR cache.
        """
        if input is not None:
            input = self.transformer.patchify_and_maybe_split_cp(input)
        cache.start(autoregressive_index)

        initial_noise = torch.randn(
            self.latent_shape,
            device=self.device,
            dtype=self.dtype,
            generator=self.rng,
        )

        def predict_flow(noisy_latent: Tensor, timestep: Tensor) -> Tensor:
            return self.transformer.predict_flow(
                noisy_latent=noisy_latent,
                timestep=timestep,
                cache=cache,
                input=input,
            )

        clean_latent = self.scheduler.sample(
            initial_noise=initial_noise,
            predict_flow=predict_flow,
            rng=self.rng,
        )

        clean_latent = self.transformer.postprocess_clean_latent(
            clean_latent=clean_latent,
            cache=cache,
            input=input,
        )

        # Postpone KV cache update to the finalization step.
        final_state = DiffusionModel.FinalState[TransformerCacheT](
            clean_latent=clean_latent,
            autoregressive_index=autoregressive_index,
            cache=cache,
            input=input,
        )

        clean_latent = self.transformer.unpatchify_and_maybe_gather_cp(clean_latent)
        return clean_latent, final_state  # ty:ignore[invalid-return-type]

    def finalize(
        self,
        final_state: "DiffusionModel.FinalState[TransformerCacheT]",
    ) -> None:
        """Advance the AR cache using the clean latent from :meth:`generate`.

        Renoises the clean latent to
        :attr:`DiffusionModelConfig.context_noise`, then defers the
        actual cache-update forward(s) to
        :meth:`Transformer.finalize_kv_cache` (one network for vanilla
        transformers, both for Wan 2.2's dual-network DiT, etc.).

        Args:
            final_state: The :class:`FinalState` returned by :meth:`generate`.

        Note: ``context_noise == 0`` skips :meth:`Scheduler.add_noise`
        (sigma=0 is the identity) and feeds the clean latent directly,
        which also avoids needing schedulers to support a ``t=0`` lookup
        (e.g. UniPC's inference schedule does not contain ``t=0``).
        """
        context_noise = self.config.context_noise
        timestep = torch.tensor(context_noise, device=self.device, dtype=self.dtype)
        if context_noise > 0.0:
            noisy_latent = self.scheduler.add_noise(
                clean_input=final_state.clean_latent,
                timestep=timestep,
                rng=self.rng,
            )
        else:
            noisy_latent = final_state.clean_latent
        self.transformer.finalize_kv_cache(
            noisy_latent=noisy_latent,
            timestep=timestep,
            cache=final_state.cache,
            input=final_state.input,
        )
        final_state.cache.finalize(final_state.autoregressive_index)
