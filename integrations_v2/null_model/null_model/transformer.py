# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Input-conditioned deterministic RGB transformer for the NULL model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from flashdreams.infra.diffusion.transformer import (
    Transformer,
    TransformerAutoregressiveCache,
    TransformerConfig,
)


@dataclass(kw_only=True)
class NullTransformerCache(TransformerAutoregressiveCache):
    """Minimal per-rollout state demonstrating minimal transformer cache lifecycle without ``finalize``."""

    autoregressive_index: int = -1
    """Current AR step; ``-1`` before the first :meth:`start` call."""

    def start(self, autoregressive_index: int) -> None:
        """Record the AR step that determines the output value.

        The diffusion model calls this before :meth:`NullTransformer.predict_flow`,
        so every denoising invocation observes the correct step.

        Args:
            autoregressive_index: Current zero-based AR step.
        """
        self.autoregressive_index = autoregressive_index


@dataclass(kw_only=True)
class NullTransformerConfig(TransformerConfig):
    """Config selecting the deterministic NULL transformer implementation."""

    _target: type["NullTransformer"] = field(default_factory=lambda: NullTransformer)


class NullTransformer(Transformer[NullTransformerCache]):
    """Deterministic RGB transformer driven by the scalar input and AR step."""

    def __init__(self, config: NullTransformerConfig) -> None:
        """
        Args:
            config: Transformer instantiation config supplied by the inherited
                ``setup()`` mechanism.
        """
        super().__init__(config)

        self._device_anchor = torch.nn.Parameter(
            torch.zeros(()),
            requires_grad=False,
        )

    @property
    def latent_shape(self) -> tuple[int, ...]:
        """Return the fixed one-pixel RGB chunk shape ``[B, C, T, H, W]``.

        The NULL model uses ``[1, 3, 1, 1, 1]`` as both its diffusion latent and
        final output shape.
        """
        return (1, 3, 1, 1, 1)

    def initialize_autoregressive_cache(self) -> NullTransformerCache:
        """Return a fresh step-index cache for one rollout.

        The pipeline calls this method once from ``initialize_cache`` and sends the
        result through every ``generate`` / ``finalize`` pair to keep track of the autoregressive step.
        """
        return NullTransformerCache()

    def initial_noise(
        self,
        *,
        latent_shape: tuple[int, ...],
        rng: torch.Generator | None,
        cache: NullTransformerCache,
        input: Any = None,
    ) -> Tensor:
        """Return a deterministic starting latent so that the scheduler does not have any 'noise' to 'denoise' from the :meth:`predict_flow` method result.

        Args:
            latent_shape: Shape requested by the diffusion model.
            rng: Framework RNG; unused because the tensor is deterministic.
            cache: Per-rollout transformer cache; not needed to initialize zeros.
            input: Encoded per-step scalar; not needed until flow prediction.

        Returns:
            A zero tensor on the transformer's current device and dtype.

        Note:
            A real diffusion integration normally inherits the base Gaussian
            implementation.
        """
        del rng, cache, input
        return torch.zeros(latent_shape, device=self.device, dtype=self.dtype)

    def predict_flow(
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        cache: NullTransformerCache,
        input: Tensor | None = None,
    ) -> Tensor:
        """Predict flow from random noise to the input value plus AR step.

        Args:
            noisy_latent: Current noisy RGB output tensor.
            timestep: Ignored
            cache: Tracks AR step via ``cache.autoregressive_index``.
            input: Encoded-input with shape ``[1, 1]``.

        Returns:
            Flow that produces a tensor filled with ``input + cache.autoregressive_index``.
        """
        del timestep
        assert isinstance(input, Tensor), (
            f"expected input to be a Tensor, got {type(input).__name__}"
        )
        assert input.shape == (1, 1), (
            f"expected input tensor shape (1, 1), got {tuple(input.shape)}"
        )

        return noisy_latent - (input + cache.autoregressive_index)

    def patchify_and_maybe_split_cp(self, x: Any) -> Any:
        """Stubbed out for compatibility with the transformer interface. No patching of inputs is needed for the NULL model."""
        return x

    def unpatchify_and_maybe_gather_cp(self, x: Tensor) -> Tensor:
        """Stubbed out for compatibility with the transformer interface. No unpatchifying of outputs is needed for the NULL model."""
        return x
