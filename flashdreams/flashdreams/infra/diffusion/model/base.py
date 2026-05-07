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
from typing_extensions import TypeVar

from flashdreams.infra.config import InstantiateConfig
from flashdreams.infra.diffusion.scheduler import Scheduler
from flashdreams.infra.diffusion.transformer import (
    Transformer,
    TransformerAutoregressiveCache,
    TransformerCacheT,
)

# Distinct TypeVar for ``DiffusionModel.FinalState`` so the nested generic
# owns its own parameter instead of shadowing the outer ``TransformerCacheT``
# (which would make ty resolve ``FinalState`` calls to the TypeVar's default).
_FinalStateCacheT = TypeVar(
    "_FinalStateCacheT",
    bound=TransformerAutoregressiveCache,
    default=TransformerAutoregressiveCache,
)


@dataclass(kw_only=True)
class DiffusionModelConfig(InstantiateConfig["DiffusionModel"]):
    """Config for the autoregressive diffusion model."""

    _target: type["DiffusionModel"] = field(default_factory=lambda: DiffusionModel)

    transformer: InstantiateConfig[Any]
    """Flow-prediction network config."""

    scheduler: InstantiateConfig[Any]
    """Denoising-loop config."""

    seed: int | None = None
    """RNG seed for initial-noise draws and scheduler sampling.
    ``None`` uses the global RNG."""

    context_noise: int = 0
    """Timestep used by ``finalize`` for the AR cache-update forward.
    ``0`` skips ``add_noise``."""

    _noise_in_unpatchified_shape: bool = False
    """Debug only: Create the noise in the unpatchified shape (to align with the open-source codebases).
    Note this will hurt performance so only use it for debugging."""


class DiffusionModel(nn.Module, Generic[TransformerCacheT]):
    """Autoregressive diffusion model (scheduler + transformer).

    Generic over the transformer's AR cache type so user-facing typing on
    ``cache`` is preserved end-to-end.

    Examples:

        model = config.setup().to("cuda")
        cache = model.transformer.initialize_autoregressive_cache(...)
        clean, final_state = model.generate(autoregressive_index=0, cache=cache)
        model.finalize(final_state)
    """

    @dataclass(kw_only=True)
    class FinalState(Generic[_FinalStateCacheT]):
        """State passed from ``generate`` to ``finalize``.

        Uses its own ``_FinalStateCacheT`` rather than the enclosing class's
        ``TransformerCacheT`` because nested classes don't inherit outer-scope
        type parameters; reusing the same TypeVar object would only shadow
        it and confuse the type checker.
        """

        clean_latent: Tensor
        """Patchified clean latent at the end of denoising."""

        autoregressive_index: int
        """AR step this state was produced at."""

        cache: _FinalStateCacheT
        """Long-lived AR cache used during generation."""

        input: Any = None
        """Patchified per-AR-step encoder output, or ``None``."""

    transformer: Transformer[TransformerCacheT]
    scheduler: Scheduler

    def __init__(self, config: DiffusionModelConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = self.config.transformer.setup()
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
        """Per-model generator, lazily built on the current device.

        Returns ``None`` when ``config.seed`` is ``None``. Rebuilt the first
        time the model's device changes after a ``.to(...)``. A device move
        resets the RNG stream — fine for "construct on CPU, ``.to(gpu)``
        once" but mid-rollout device hops lose RNG state.
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
            autoregressive_index: AR step index.
            cache: Long-lived AR cache, mutated in place.
            input: Optional per-AR-step encoder output. Patchified here and
                forwarded to ``predict_flow`` / ``postprocess_clean_latent``,
                then stashed on the returned ``FinalState`` for ``finalize``.

        Returns:
            ``(clean_latent, final_state)``. ``clean_latent`` is unpatchified;
            ``final_state`` should be passed to ``finalize``.
        """
        if input is not None:
            input = self.transformer.patchify_and_maybe_split_cp(input)
        cache.start(autoregressive_index)

        if self.config._noise_in_unpatchified_shape:
            # The `self.latent_shape` is for patchified latent. To align with the OSS codebases
            # we here want to create the noise with the *unpatchified* shape, then patchify it back.
            dummy_latent = torch.empty(
                self.latent_shape, device=self.device, dtype=self.dtype
            )
            dummy_latent = self.transformer.unpatchify_and_maybe_gather_cp(dummy_latent)
            initial_noise = torch.randn(
                dummy_latent.shape,
                device=self.device,
                dtype=self.dtype,
                generator=self.rng,
            )  # shape of an unpatchified latent

        else:
            initial_noise = torch.randn(
                self.latent_shape,
                device=self.device,
                dtype=self.dtype,
                generator=self.rng,
            )  # shape of a patchified latent

        def predict_flow(noisy_latent: Tensor, timestep: Tensor) -> Tensor:
            if self.config._noise_in_unpatchified_shape:
                # If the noise is in the unpatchified shape, we need to patchify it first
                # because the transformer expects a patchified latent.
                noisy_latent = self.transformer.patchify_and_maybe_split_cp(
                    noisy_latent
                )

            output = self.transformer.predict_flow(
                noisy_latent=noisy_latent,
                timestep=timestep,
                cache=cache,
                input=input,
            )  # patchified output

            if self.config._noise_in_unpatchified_shape:
                # The we need to unpatchify it again, to make sure scheduler operates
                # on the unpatchified latent.
                output = self.transformer.unpatchify_and_maybe_gather_cp(output)
            return output

        clean_latent = self.scheduler.sample(
            initial_noise=initial_noise,
            predict_flow=predict_flow,
            rng=self.rng,
        )

        if self.config._noise_in_unpatchified_shape:
            # If this option is enabled, the scheduler operates on the unpatchified latent.
            # So the clean latent it outputs is in the unpatchified shape. Here we patchify it.
            clean_latent = self.transformer.patchify_and_maybe_split_cp(clean_latent)

        clean_latent = self.transformer.postprocess_clean_latent(
            clean_latent=clean_latent,
            cache=cache,
            input=input,
        )

        # Postpone KV cache update to the finalization step. No runtime
        # subscript: ``_FinalStateCacheT`` is bound from ``cache``'s type.
        final_state = DiffusionModel.FinalState(
            clean_latent=clean_latent,
            autoregressive_index=autoregressive_index,
            cache=cache,
            input=input,
        )

        clean_latent = self.transformer.unpatchify_and_maybe_gather_cp(clean_latent)
        return clean_latent, final_state

    def finalize(
        self,
        final_state: "DiffusionModel.FinalState[TransformerCacheT]",
    ) -> None:
        """Advance the AR cache using the clean latent from ``generate``.

        Re-noises the clean latent to ``config.context_noise`` and runs the
        transformer's ``finalize_kv_cache`` (one forward for vanilla
        transformers, multiple for dual-network DiTs).

        ``context_noise == 0`` skips ``add_noise`` (sigma=0 is identity) and
        feeds the clean latent directly. This also dodges the requirement
        for schedulers to support a ``t=0`` lookup (UniPC's inference
        schedule has no ``t=0`` entry).
        """
        context_noise = self.config.context_noise
        timestep = torch.tensor(context_noise, device=self.device, dtype=self.dtype)
        if context_noise > 0.0:
            clean_latent = final_state.clean_latent
            if self.config._noise_in_unpatchified_shape:
                # If this option is enabled, we want the scheduler to operate on the unpatchified latent.
                clean_latent = self.transformer.unpatchify_and_maybe_gather_cp(
                    final_state.clean_latent
                )
            noisy_latent = self.scheduler.add_noise(
                clean_input=clean_latent,
                timestep=timestep,
                rng=self.rng,
            )
            if self.config._noise_in_unpatchified_shape:
                # If this option is enabled, then the scheduler outputs an unpatchified latent so we
                # need to patchify it.
                noisy_latent = self.transformer.patchify_and_maybe_split_cp(
                    noisy_latent
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
