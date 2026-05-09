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

"""Decoder interfaces.

Two flavours:

- :class:`StreamingDecoder` is stateful. ``forward(self, input,
  autoregressive_index, cache)`` plus a per-rollout
  :class:`StreamingDecoderCache`. Use for chunk-by-chunk streaming
  decoders (e.g. the WAN VAE that maintains a temporal cache across AR
  steps); also fine for stateless decoders — just return an empty
  :class:`StreamingDecoderCache` and ignore ``autoregressive_index`` /
  ``cache`` in ``forward``.
- :class:`StreamingVideoDecoder` extends :class:`StreamingDecoder` with
  the contracts a streaming pixel-video decoder always needs: spatial
  and temporal compression ratios, plus AR-step-aware temporal size
  mappers between latent and pixel space.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic

import torch.nn as nn
from typing_extensions import TypeVar

from flashdreams.infra.config import InstantiateConfig


@dataclass(kw_only=True)
class StreamingDecoderCache:
    """Per-rollout cache for :class:`StreamingDecoder`.

    Empty by default; subclass to add fields (e.g. temporal feature
    buffers carried across AR steps).
    """


StreamingDecoderCacheT = TypeVar(
    "StreamingDecoderCacheT",
    bound=StreamingDecoderCache,
    default=StreamingDecoderCache,
)


class StreamingDecoder(ABC, nn.Module, Generic[StreamingDecoderCacheT]):
    """Streaming decoder, generic over the per-rollout cache type.

    ``forward`` is not pinned by the base. Streaming decoders called by
    :class:`StreamInferencePipeline` must match its call shape:
    ``forward(self, input, autoregressive_index=0, cache=None)``.
    """

    def __init__(self, config: "DecoderConfig") -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def initialize_autoregressive_cache(self, **context: Any) -> StreamingDecoderCacheT:
        """Build a fresh per-rollout cache.

        Override to return the decoder's concrete cache type.
        """


class StreamingVideoDecoder(StreamingDecoder[StreamingDecoderCacheT]):
    """Streaming pixel-video decoder.

    Pins down the contracts that every streaming latent→pixel video
    decoder satisfies in addition to :class:`StreamingDecoder`:

    - Spatial and temporal compression ratios between the latent and
      pixel grids (constants of the architecture).
    - AR-step-aware temporal size mappers, so a pipeline can size its
      inputs and outputs without knowing the decoder's concrete
      temporal cache topology (causal first-frame padding, sliding
      windows, etc.).

    Spatial scaling is trivially ``side * spatial_compression_ratio``
    in either direction; the AR-step-asymmetric piece is the temporal
    size, which gets its own mapper. Typically AR 0 produces fewer
    pixel frames per latent frame than AR ≥ 1 because of causal
    first-frame padding.
    """

    @property
    @abstractmethod
    def spatial_compression_ratio(self) -> int:
        """Pixel side ÷ latent side. Constant across AR steps."""

    @property
    @abstractmethod
    def temporal_compression_ratio(self) -> int:
        """Pixel frames ÷ latent frames in steady state (AR ≥ 1).

        AR 0 typically yields fewer pixel frames per latent frame
        because of causal first-frame padding; that asymmetry lives
        inside :meth:`get_output_temporal_size` /
        :meth:`get_input_temporal_size`.
        """

    @abstractmethod
    def get_output_temporal_size(
        self,
        autoregressive_index: int,
        input_temporal_size: int,
    ) -> int:
        """Pixel frame count produced by ``input_temporal_size`` latent frames.

        Args:
            autoregressive_index: AR step index (0-based).
            input_temporal_size: Number of latent frames fed at this step.

        Returns:
            Number of pixel frames emitted at this step.
        """

    @abstractmethod
    def get_input_temporal_size(
        self,
        autoregressive_index: int,
        output_temporal_size: int,
    ) -> int:
        """Latent frame count needed to produce ``output_temporal_size`` pixels.

        Inverse of :meth:`get_output_temporal_size`. Implementations
        should assert ``output_temporal_size`` is achievable at this AR
        step (i.e. divisible by the right ratio after subtracting any
        causal padding).

        Args:
            autoregressive_index: AR step index (0-based).
            output_temporal_size: Desired number of pixel frames.

        Returns:
            Number of latent frames needed at this step.
        """


@dataclass(kw_only=True)
class DecoderConfig(InstantiateConfig):
    """Category base for every decoder config."""

    _target: type = field(default_factory=lambda: StreamingDecoder)
