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
from typing import Any, Generic, cast

import torch
import torch.nn as nn
from torch import Tensor
from typing_extensions import TypeVar

from flashdreams.infra.config import InstantiateConfig


@dataclass(kw_only=True)
class TransformerAutoregressiveCache:
    """Cache that persists across an AR rollout.

    Empty by default; safe to instantiate directly for non-AR transformers
    (default ``start`` / ``finalize`` are no-ops). Subclass and add fields
    plus AR bookkeeping for real per-rollout state.

    Examples:

        cache.start(autoregressive_index)
        # one or more denoising steps...
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

    Subclasses implement ``predict_flow`` and the patchify hooks. AR
    transformers also subclass ``TransformerAutoregressiveCache`` and
    override ``initialize_autoregressive_cache``.

    Examples:

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
            cache: Per-rollout AR cache.
            input: Patchified encoder output for this AR step, or ``None``
                when the pipeline has no encoder. Subclasses should narrow
                the type to their encoder's output type.

        Returns:
            Predicted flow tensor with the same shape as ``noisy_latent``.
        """

    def finalize_kv_cache(
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        cache: TransformerCacheT,
        input: Any = None,
    ) -> None:
        """Advance the AR cache so it is ready for the next AR step.

        Called by ``DiffusionModel.finalize`` after the denoising loop; the
        flow is discarded — only the cache side effect matters. Default
        runs a single ``predict_flow`` forward. Override for transformers
        with multiple parallel networks that must stay in lock-step.

        Args:
            noisy_latent: Patchified latent at the AR-step's context noise
                (or the clean latent when ``context_noise == 0``).
            timestep: 0-d context-noise timestep tensor.
            cache: Per-rollout AR cache.
            input: Same patchified encoder output passed to ``predict_flow``.
        """
        _ = self.predict_flow(noisy_latent, timestep, cache, input)

    def initialize_autoregressive_cache(self, **context: Any) -> TransformerCacheT:
        """Build a fresh AR cache for a new rollout.

        Default returns an empty cache, correct for non-AR transformers.
        Subclasses with custom cache types must override this and may
        declare typed per-rollout context (e.g. ``text_embeddings``).

        Args:
            context: Per-rollout state forwarded as keyword arguments.

        Returns:
            Fresh AR cache.
        """
        return cast("TransformerCacheT", TransformerAutoregressiveCache())

    def postprocess_clean_latent(
        self,
        clean_latent: Tensor,
        cache: TransformerCacheT,
        input: Any = None,
    ) -> Tensor:
        """Optional postprocessing hook for the predicted clean latent.

        Default is identity. Override to clamp or re-inject regions whose
        clean value is known a priori (e.g. I2V first-frame pinning).
        Called at the end of ``DiffusionModel.generate``.

        Args:
            clean_latent: Patchified ``x0`` from the denoising loop.
            cache: Per-rollout AR cache.
            input: Same patchified encoder output passed to ``predict_flow``.

        Returns:
            Postprocessed clean latent with the same shape.
        """
        return clean_latent

    @abstractmethod
    def patchify_and_maybe_split_cp(self, x: Any) -> Any:
        """Patchify and (optionally) CP-split a noisy latent or encoder payload.

        Tensors patchify and split. Structured payloads (e.g. an image-control
        struct with ``latent`` + ``mask``) patchify each tensor field and
        return the same struct type. Implement as identity when neither
        token packing nor CP sharding applies. Output preserves the input
        Python type — only shapes change.
        """

    @abstractmethod
    def unpatchify_and_maybe_gather_cp(self, x: Tensor) -> Tensor:
        """Inverse of ``patchify_and_maybe_split_cp`` for the network output."""


@dataclass(kw_only=True)
class TransformerConfig(InstantiateConfig):
    """Category base for every flow-prediction transformer config."""

    _target: type = field(default_factory=lambda: Transformer)
