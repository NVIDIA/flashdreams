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

"""Encoder interface."""

from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Generic

import torch.nn as nn
from typing_extensions import TypeVar

from flashdreams.infra.config import InstantiateConfig


@dataclass(kw_only=True)
class EncoderConfig(InstantiateConfig["Encoder"]):
    """Encoder configuration."""

    _target: type["Encoder"] = field(default_factory=lambda: Encoder)


@dataclass(kw_only=True)
class EncoderAutoregressiveCache:
    """Per-rollout encoder cache. Empty by default; subclass to add fields."""


EncCacheT = TypeVar(
    "EncCacheT",
    bound=EncoderAutoregressiveCache,
    default=EncoderAutoregressiveCache,
)


class Encoder(ABC, nn.Module, Generic[EncCacheT]):
    """Encoder interface, generic over the per-rollout cache type ``EncCacheT``.

    ``forward`` is intentionally not pinned by the base; subclasses pick
    whatever signature their data needs. Stateful encoders called by
    :class:`StreamInferencePipeline` must match the pipeline call shape::

        def forward(self, input, autoregressive_index=0, cache=None) -> Tensor

    Stateless encoders not called by the pipeline can use a slimmer
    signature, e.g. ``def forward(self, input) -> Tensor``.
    """

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config

    def initialize_autoregressive_cache(self, **context: Any) -> EncCacheT:
        """Build a fresh per-rollout cache.

        Default returns the empty :class:`EncoderAutoregressiveCache`
        sentinel and ignores ``**context``. Subclasses with a custom
        ``EncCacheT`` must override this.

        Args:
            context: Per-rollout state forwarded as keyword arguments.

        Returns:
            A fresh :class:`EncoderAutoregressiveCache` (or subclass).
        """
        return EncoderAutoregressiveCache()  # type: ignore[return-value]  # ty:ignore[invalid-return-type]


@dataclass(kw_only=True)
class NullEncoderConfig(EncoderConfig):
    """Configuration for :class:`NullEncoder`."""

    _target: type["Encoder"] = field(default_factory=lambda: NullEncoder)


class NullEncoder(Encoder[EncoderAutoregressiveCache]):
    """Identity encoder: returns its input unchanged.

    Use as :attr:`StreamInferencePipelineConfig.encoder` to pass
    already-encoded tensors straight to the diffusion model.

    Example::

        config = StreamInferencePipelineConfig(
            encoder=NullEncoderConfig(),
            ...,
        )
    """

    def forward(
        self,
        input: Any,
        autoregressive_index: int = 0,
        cache: EncoderAutoregressiveCache | None = None,
    ) -> Any:
        return input
