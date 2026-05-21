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

"""Encoder interfaces.

Two flavours, distinguished by whether the encoder carries per-rollout
state across AR steps:

- :class:`Encoder` is stateless. ``forward(self, input)``. Use for
  one-shot context encoders (text encoders, CLIP image encoders, the
  identity :class:`NullEncoder`) that run once per rollout, typically as
  the transformer's ``context_encoder`` slot.
- :class:`StreamingEncoder` is stateful. ``forward(self, input,
  autoregressive_index, cache)`` plus a per-rollout
  :class:`StreamingEncoderCache`. Use for per-AR-step control
  encoders (HDMap, camera trajectory, I2V first-frame VAE) wired into
  the pipeline's ``encoder`` slot.
- :class:`StreamingVideoEncoder` extends :class:`StreamingEncoder` with
  the contracts a streaming pixel-video encoder always needs: spatial
  and temporal compression ratios, plus AR-step-aware temporal size
  mappers between pixel and latent space.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic

import torch.nn as nn
from typing_extensions import TypeVar

from flashdreams.infra.config import InstantiateConfig


class Encoder(ABC, nn.Module):
    """Stateless encoder.

    ``forward`` is not pinned by the base. Encoders used as a
    ``context_encoder`` (one-shot, called once inside
    :meth:`Transformer.initialize_autoregressive_cache`) must match the
    slim call shape ``forward(self, input)``.

    For per-AR-step encoders that need a per-rollout cache, inherit
    from :class:`StreamingEncoder` instead.
    """

    def __init__(self, config: "EncoderConfig") -> None:
        super().__init__()
        self.config = config


@dataclass(kw_only=True)
class EncoderConfig(InstantiateConfig):
    """Category base for every encoder config (stateless or streaming)."""

    _target: type = field(default_factory=lambda: Encoder)


@dataclass(kw_only=True)
class StreamingEncoderCache:
    """Per-rollout cache for :class:`StreamingEncoder`.

    Empty by default; subclass to add fields (e.g. last-frame latent,
    cross-step accumulators).
    """


StreamingEncoderCacheT = TypeVar(
    "StreamingEncoderCacheT",
    bound=StreamingEncoderCache,
    default=StreamingEncoderCache,
)


class StreamingEncoder(ABC, nn.Module, Generic[StreamingEncoderCacheT]):
    """Streaming encoder, generic over the per-rollout cache type.

    ``forward`` is not pinned by the base. Streaming encoders called by
    :class:`StreamInferencePipeline` must match its call shape:
    ``forward(self, input, autoregressive_index=0, cache=None)``.
    """

    def __init__(self, config: "EncoderConfig") -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def initialize_autoregressive_cache(self, **context: Any) -> StreamingEncoderCacheT:
        """Build a fresh per-rollout cache.

        Override to return the encoder's concrete cache type.
        """


class StreamingVideoEncoder(StreamingEncoder[StreamingEncoderCacheT]):
    """Streaming pixel-video encoder.

    Pins down the contracts that every streaming pixel→latent video
    encoder satisfies in addition to :class:`StreamingEncoder`:

    - Spatial and temporal compression ratios between the pixel and
      latent grids (constants of the architecture).
    - AR-step-aware temporal size mappers, so a pipeline can size its
      inputs and outputs without knowing the encoder's concrete
      temporal cache topology (causal first-frame padding, sliding
      windows, etc.).

    Spatial scaling is trivially ``side // spatial_compression_ratio``
    in either direction; the AR-step-asymmetric piece is the temporal
    size, which gets its own mapper. Typically AR 0 takes
    ``1 + (T_lat - 1) * r`` pixel frames to produce ``T_lat`` latent
    frames because of causal first-frame padding, while AR ≥ 1 takes
    ``T_lat * r`` pixel frames.
    """

    @property
    @abstractmethod
    def spatial_compression_ratio(self) -> int:
        """Pixel side ÷ latent side. Constant across AR steps."""

    @property
    @abstractmethod
    def temporal_compression_ratio(self) -> int:
        """Pixel frames ÷ latent frames in steady state (AR ≥ 1).

        AR 0 typically takes one extra (un-grouped) pixel frame to
        produce its first latent frame because of causal first-frame
        padding; that asymmetry lives inside
        :meth:`get_output_temporal_size` /
        :meth:`get_input_temporal_size`.
        """

    @abstractmethod
    def get_output_temporal_size(
        self,
        autoregressive_index: int,
        input_temporal_size: int,
    ) -> int:
        """Latent frame count produced from ``input_temporal_size`` pixel frames.

        Args:
            autoregressive_index: AR step index (0-based).
            input_temporal_size: Number of pixel frames fed at this step.

        Returns:
            Number of latent frames emitted at this step.
        """

    @abstractmethod
    def get_input_temporal_size(
        self,
        autoregressive_index: int,
        output_temporal_size: int,
    ) -> int:
        """Pixel frame count needed to produce ``output_temporal_size`` latents.

        Inverse of :meth:`get_output_temporal_size`. Implementations
        should assert ``output_temporal_size`` is achievable at this AR
        step (i.e. the corresponding pixel count comes out as a
        positive integer).

        Args:
            autoregressive_index: AR step index (0-based).
            output_temporal_size: Desired number of latent frames.

        Returns:
            Number of pixel frames needed at this step.
        """


@dataclass(kw_only=True)
class NullEncoderConfig(EncoderConfig):
    """Config for the identity encoder."""

    _target: type = field(default_factory=lambda: NullEncoder)


class NullEncoder(Encoder):
    """Identity encoder: returns its input unchanged.

    Wire as the transformer's ``context_encoder`` slot to pass already-
    encoded tensors straight to the diffusion model.

    Example:

    .. code-block:: python

        config = TransformerConfig(
            context_encoder=NullEncoderConfig(),
            ...,
        )
    """

    def forward(self, input: Any) -> Any:
        return input
