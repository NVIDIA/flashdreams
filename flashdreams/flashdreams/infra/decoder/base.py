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

"""Decoder interface."""

from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Generic

import torch.nn as nn
from typing_extensions import TypeVar

from flashdreams.infra.config import InstantiateConfig


@dataclass(kw_only=True)
class DecoderConfig(InstantiateConfig["Decoder"]):
    """Decoder configuration."""

    _target: type["Decoder"] = field(default_factory=lambda: Decoder)


@dataclass(kw_only=True)
class DecoderAutoregressiveCache:
    """Per-rollout decoder cache. Empty by default; subclass to add fields."""


DecCacheT = TypeVar(
    "DecCacheT",
    bound=DecoderAutoregressiveCache,
    default=DecoderAutoregressiveCache,
)


class Decoder(ABC, nn.Module, Generic[DecCacheT]):
    """Decoder interface, generic over the per-rollout cache type ``DecCacheT``.

    Input is a diffusion latent ``Tensor``; output is the decoded sample
    ``Tensor`` (e.g. RGB video). ``forward`` is intentionally not pinned
    by the base; subclasses pick whatever signature their data needs.
    Stateful decoders called by :class:`StreamInferencePipeline` must
    match the pipeline call shape::

        def forward(self, input, autoregressive_index=0, cache=None) -> Tensor

    Stateless decoders not called by the pipeline can use a slimmer
    signature, e.g. ``def forward(self, input) -> Tensor``.
    """

    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        self.config = config

    def initialize_autoregressive_cache(self, **context: Any) -> DecCacheT:
        """Build a fresh per-rollout cache.

        Default returns the empty :class:`DecoderAutoregressiveCache`
        sentinel and ignores ``**context``. Subclasses with a custom
        ``DecCacheT`` must override this.

        Args:
            context: Per-rollout state forwarded as keyword arguments.

        Returns:
            A fresh :class:`DecoderAutoregressiveCache` (or subclass).
        """
        return DecoderAutoregressiveCache()  # type: ignore[return-value]  # ty:ignore[invalid-return-type]
