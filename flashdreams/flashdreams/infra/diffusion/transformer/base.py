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

"""Diffusion transformer interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic

import torch
import torch.nn as nn
from torch import Tensor
from typing_extensions import TypeVar

from flashdreams.infra.config import InstantiateConfig


@dataclass(kw_only=True)
class TransformerConfig(InstantiateConfig["Transformer"]):
    """Configuration for a diffusion transformer network.

    Concrete subclasses are responsible for exposing :attr:`Transformer.latent_shape`
    on the runtime object (typically as a property derived from runtime CP
    group sizes), since the per-rank shape depends on hierarchical
    context-parallel splits that are only known after distributed init.
    """

    _target: type["Transformer"] = field(default_factory=lambda: Transformer)


@dataclass(kw_only=True)
class TransformerAutoregressiveCache:
    """Cache that persists across an AR rollout.

    Empty by default; safe to instantiate directly for non-AR transformers
    (the default :meth:`start` / :meth:`finalize` are no-ops). Subclass and
    add fields plus AR bookkeeping for real per-rollout state.

    Example::

        cache.start(autoregressive_index)
        # ... use the cache with the transformer (one or more denoising steps)
        cache.finalize(autoregressive_index)
    """

    def start(self, autoregressive_index: int) -> None:
        """Mark the start of an AR step. Default is a no-op.

        Args:
            autoregressive_index: Index of the AR step being started.
        """

    def finalize(self, autoregressive_index: int) -> None:
        """Finalize bookkeeping after use at this AR step. Default is a no-op.

        Args:
            autoregressive_index: Index of the AR step just finalized.
        """


TransformerCacheT = TypeVar(
    "TransformerCacheT",
    bound=TransformerAutoregressiveCache,
    default=TransformerAutoregressiveCache,
)


class Transformer(nn.Module, ABC, Generic[TransformerCacheT]):
    """Flow-prediction transformer, generic over its AR cache subclass.

    Subclasses implement :meth:`predict_flow` and the patchify hooks; AR
    transformers additionally subclass :class:`TransformerAutoregressiveCache`
    and override :meth:`initialize_autoregressive_cache`.

    Example (subclassing for an AR transformer)::

        class MyTransformer(Transformer[MyCache]):
            def predict_flow(self, noisy_latent, timestep, cache, input=None):
                ...
            def initialize_autoregressive_cache(self, **context) -> MyCache:
                ...
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    @abstractmethod
    def latent_shape(self) -> tuple[int, ...]:
        """Shape of the input/output latent tensor for this rank.

        Includes batch dims. May depend on hierarchical context-parallel
        group sizes (V/T/HW), so subclasses typically derive this from
        ``self.cp_groups`` rather than from static config alone.
        """

    @abstractmethod
    def predict_flow(
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        cache: TransformerCacheT,
        input: Any = None,
    ) -> Tensor:
        """Predict the flow at ``timestep``.

        Args:
            noisy_latent: Patchified noisy latent for this denoising step.
            timestep: Scalar timestep tensor.
            cache: Long-lived AR cache for this rollout.
            input: Encoder output for this AR step (already patchified
                by :meth:`patchify_and_maybe_split_cp`), or ``None``
                when the pipeline has no encoder. Subclasses should
                narrow the type to their encoder's output type.

        Returns:
            Predicted flow tensor (same shape as ``noisy_latent``).
        """

    def finalize_kv_cache(
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        cache: TransformerCacheT,
        input: Any = None,
    ) -> None:
        """Advance the AR cache so it is ready for the next AR step.

        Called once by :meth:`DiffusionModel.finalize` after the denoising
        loop; the resulting flow is discarded -- only the cache side
        effect matters. Default runs a single :meth:`predict_flow` forward,
        which is the right thing for transformers with one network.
        Subclasses with multiple parallel networks/caches that must stay
        in lock-step (e.g. Wan 2.2's dual-network DiT) should override
        this to forward through *each* network once per AR step.

        Args:
            noisy_latent: Patchified latent at the AR-step's
                ``context_noise`` (or the clean latent when
                ``context_noise == 0``).
            timestep: 0-d ``context_noise`` timestep tensor.
            cache: Long-lived AR cache for this rollout.
            input: Same per-AR-step encoder output passed to
                :meth:`predict_flow` (already patchified).
        """
        _ = self.predict_flow(noisy_latent, timestep, cache, input)

    def initialize_autoregressive_cache(self, **context: Any) -> TransformerCacheT:
        """Build a fresh AR cache for a new rollout.

        Default returns the empty :class:`TransformerAutoregressiveCache`
        sentinel (correct for non-AR transformers). Subclasses with a
        custom ``TransformerCacheT`` must override this and may declare
        typed per-rollout context, e.g. ``text_embeddings`` for T2V or
        ``text_embeddings`` + ``image_embeddings`` for I2V.

        Args:
            context: Per-rollout state forwarded as keyword arguments.

        Returns:
            A fresh :class:`TransformerAutoregressiveCache` (or subclass).
        """
        return TransformerAutoregressiveCache()  # type: ignore[return-value]  # ty:ignore[invalid-return-type]

    def postprocess_clean_latent(
        self,
        clean_latent: Tensor,
        cache: TransformerCacheT,
        input: Any = None,
    ) -> Tensor:
        """Optional postprocessing hook for the predicted clean latent.

        Default returns ``clean_latent`` unchanged. Override to clamp or
        re-inject regions whose clean value is known a priori (e.g. I2V
        first-frame pinning). Called once at the end of
        :meth:`DiffusionModel.generate`, before the result is unpatchified
        and stashed on :class:`DiffusionModel.FinalState`.

        Args:
            clean_latent: Patchified ``x0`` from the denoising loop.
            cache: Long-lived AR cache for this rollout.
            input: Same per-AR-step encoder output passed to
                :meth:`predict_flow` (already patchified).

        Returns:
            Postprocessed clean latent (same shape as ``clean_latent``).
        """
        return clean_latent

    @abstractmethod
    def patchify_and_maybe_split_cp(self, x: Any) -> Any:
        """Patchify and (optionally) split for context parallelism.

        Used for both the noisy latent and the per-AR-step ``input``
        payload. Subclasses dispatch on type:

        - ``Tensor``: patchify + CP-split.
        - Structured payloads (e.g. ``ImageCtrl`` with ``latent`` and
          ``mask`` Tensors): patchify each Tensor field independently
          and return the same structured type.

        Implement as identity when neither token packing nor CP sharding
        applies. Output preserves the input *type* — only shapes change.

        Args:
            x: Tensor or structured payload to patchify.

        Returns:
            Patchified value with the same Python type as ``x``.
        """

    @abstractmethod
    def unpatchify_and_maybe_gather_cp(self, x: Tensor) -> Tensor:
        """Inverse of :meth:`patchify_and_maybe_split_cp` for the network output.

        Always called on a plain :class:`Tensor` (the predicted clean
        latent), so the structured-payload concern from
        :meth:`patchify_and_maybe_split_cp` does not apply.

        Args:
            x: Patchified (and possibly CP-split) latent.

        Returns:
            Unpatchified, gathered latent.
        """
