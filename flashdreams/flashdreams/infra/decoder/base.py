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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic

import torch.nn as nn
from typing_extensions import TypeVar

from flashdreams.infra.config import InstantiateConfig


@dataclass(kw_only=True)
class DecoderAutoregressiveCache:
    """Per-rollout decoder cache. Empty by default; subclass to add fields."""


DecCacheT = TypeVar(
    "DecCacheT",
    bound=DecoderAutoregressiveCache,
    default=DecoderAutoregressiveCache,
)


class Decoder(ABC, nn.Module, Generic[DecCacheT]):
    """Decoder interface, generic over the per-rollout cache type.

    Input is a latent tensor; output is the decoded sample (e.g. RGB video).
    ``forward`` isn't pinned by the base. Decoders called by
    ``StreamInferencePipeline`` must match its call shape:
    ``forward(self, input, autoregressive_index=0, cache=None)``.
    """

    def __init__(self, config: InstantiateConfig[Any]) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def initialize_autoregressive_cache(self, **context: Any) -> DecCacheT:
        """Build a fresh per-rollout cache.

        Override to return the decoder's concrete cache type.
        """
