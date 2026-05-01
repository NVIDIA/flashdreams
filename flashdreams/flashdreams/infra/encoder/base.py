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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic

import torch.nn as nn
from typing_extensions import TypeVar

from flashdreams.infra.config import InstantiateConfig


@dataclass(kw_only=True)
class EncoderAutoregressiveCache:
    """Per-rollout encoder cache. Empty by default; subclass to add fields."""


EncCacheT = TypeVar(
    "EncCacheT",
    bound=EncoderAutoregressiveCache,
    default=EncoderAutoregressiveCache,
)


class Encoder(ABC, nn.Module, Generic[EncCacheT]):
    """Encoder interface, generic over the per-rollout cache type.

    ``forward`` is not pinned by the base: subclasses pick whatever
    signature fits. Encoders called by ``StreamInferencePipeline`` must
    match its call shape: ``forward(self, input, autoregressive_index=0,
    cache=None)``. Stateless one-shot encoders (text, CLIP image) can use
    a slimmer ``forward(self, input)``.
    """

    def __init__(self, config: InstantiateConfig[Any]) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def initialize_autoregressive_cache(self, **context: Any) -> EncCacheT:
        """Build a fresh per-rollout cache.

        Override to return the encoder's concrete cache type. Stateless
        encoders (e.g. text, CLIP image) return a fresh
        ``EncoderAutoregressiveCache``.
        """


@dataclass(kw_only=True)
class NullEncoderConfig(InstantiateConfig["NullEncoder"]):
    """Config for the identity encoder."""

    _target: type["NullEncoder"] = field(default_factory=lambda: NullEncoder)


class NullEncoder(Encoder[EncoderAutoregressiveCache]):
    """Identity encoder: returns its input unchanged.

    Wire as the pipeline's ``encoder`` slot to pass already-encoded tensors
    straight to the diffusion model.

    Typical usage example:

        config = StreamInferencePipelineConfig(
            encoder=NullEncoderConfig(),
            ...,
        )
    """

    def initialize_autoregressive_cache(
        self, **context: Any
    ) -> EncoderAutoregressiveCache:
        return EncoderAutoregressiveCache()

    def forward(
        self,
        input: Any,
        autoregressive_index: int = 0,
        cache: EncoderAutoregressiveCache | None = None,
    ) -> Any:
        return input
