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

"""Independently deployable encoder, diffusion, and decoder pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic

import torch
from torch import Tensor, nn

from flashdreams.infra.decoder import (
    DecoderConfig,
    StreamingDecoder,
    StreamingDecoderCacheT,
)
from flashdreams.infra.diffusion.model import DiffusionModel, DiffusionModelConfig
from flashdreams.infra.diffusion.transformer import TransformerCacheT
from flashdreams.infra.encoder import (
    EncoderConfig,
    StreamingEncoder,
    StreamingEncoderCacheT,
)


class StreamingEncoderStage(nn.Module, Generic[StreamingEncoderCacheT]):
    """Own only a pipeline's per-AR-step encoder."""

    encoder: StreamingEncoder[StreamingEncoderCacheT]

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.encoder = config.setup()

    def initialize_cache(self, **context: Any) -> StreamingEncoderCacheT:
        """Build the encoder's per-rollout cache."""
        return self.encoder.initialize_autoregressive_cache(**context)

    @torch.no_grad()
    def encode(
        self,
        input: Any,
        autoregressive_index: int,
        cache: StreamingEncoderCacheT,
    ) -> Any:
        """Encode one raw control chunk."""
        return self.encoder(
            input=input,
            autoregressive_index=autoregressive_index,
            cache=cache,
        )


@dataclass(kw_only=True)
class DiffusionStageCache(Generic[TransformerCacheT]):
    """Per-rollout state owned by a diffusion stage worker."""

    transformer_cache: TransformerCacheT
    """Long-lived transformer cache pinned to this worker."""

    final_state: DiffusionModel.FinalState[TransformerCacheT] | None = None
    """Most recent denoising result consumed by :meth:`DiffusionStage.finalize`."""

    autoregressive_index: int | None = None
    """Most recent AR index, or ``None`` before generation starts."""


class DiffusionStage(nn.Module, Generic[TransformerCacheT]):
    """Own only the scheduler and denoising transformer (DiT)."""

    diffusion_model: DiffusionModel[TransformerCacheT]

    def __init__(self, config: DiffusionModelConfig) -> None:
        super().__init__()
        self.diffusion_model = config.setup()

    @property
    def device(self) -> torch.device:
        """Return the DiT device."""
        return self.diffusion_model.device

    def initialize_cache(
        self, **context: Any
    ) -> DiffusionStageCache[TransformerCacheT]:
        """Build the transformer cache from encoder-stage context."""
        transformer_cache = (
            self.diffusion_model.transformer.initialize_autoregressive_cache(**context)
        )
        return DiffusionStageCache(transformer_cache=transformer_cache)

    @torch.no_grad()
    def generate(
        self,
        autoregressive_index: int,
        cache: DiffusionStageCache[TransformerCacheT],
        input: Any = None,
    ) -> Tensor:
        """Denoise one AR chunk and retain the state needed for finalization."""
        previous = cache.autoregressive_index
        expected = previous + 1 if previous is not None else 0
        assert autoregressive_index == expected, (
            f"AR step out of order: previous step was {previous}, expected "
            f"{expected}, got {autoregressive_index}."
        )
        clean_latent, final_state = self.diffusion_model.generate(
            autoregressive_index=autoregressive_index,
            cache=cache.transformer_cache,
            input=input,
        )
        cache.autoregressive_index = autoregressive_index
        cache.final_state = final_state
        return clean_latent

    @torch.no_grad()
    def finalize(
        self,
        autoregressive_index: int,
        cache: DiffusionStageCache[TransformerCacheT],
    ) -> None:
        """Advance the resident DiT cache after one generated chunk."""
        assert cache.autoregressive_index == autoregressive_index, (
            f"autoregressive_index mismatch: generate() ran with "
            f"{cache.autoregressive_index}, finalize() got {autoregressive_index}."
        )
        assert cache.final_state is not None, (
            "finalize() called before generate() produced a final state."
        )
        self.diffusion_model.finalize(cache.final_state)
        cache.final_state = None


class DecoderStage(nn.Module, Generic[StreamingDecoderCacheT]):
    """Own only a pipeline's streaming decoder."""

    decoder: StreamingDecoder[StreamingDecoderCacheT]

    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        self.decoder = config.setup()

    def initialize_cache(self, **context: Any) -> StreamingDecoderCacheT:
        """Build the decoder's per-rollout cache."""
        return self.decoder.initialize_autoregressive_cache(**context)

    @torch.no_grad()
    def decode(
        self,
        input: Tensor,
        autoregressive_index: int,
        cache: StreamingDecoderCacheT,
    ) -> Tensor:
        """Decode one clean latent chunk."""
        return self.decoder(
            input=input,
            autoregressive_index=autoregressive_index,
            cache=cache,
        )
