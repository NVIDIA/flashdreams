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

"""I2V control encoder for the causal Wan 2.1 pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

import torch
from torch import Tensor

from flashdreams.infra.config import InstantiateConfig
from flashdreams.infra.encoder import Encoder
from flashdreams.recipes.wan.autoencoder.vae import (
    WanVAECache,
    WanVAEEncoder,
    WanVAEEncoderConfig,
)

I2VCtrlEncoderCache: TypeAlias = WanVAECache
"""Per-AR-step I2V control encoder cache.

Aliased to ``WanVAECache``: the I2V encoder runs the inner VAE encoder
and threads its cache directly, so the two are structurally identical."""


@dataclass(kw_only=True)
class I2VCtrl:
    """I2V control payload (image latent + injection mask)."""

    latent: Tensor
    """VAE-encoded image latent ``[*batch_shape, len_t, in_dim, Hl, Wl]``
    before patchify, ``[*batch_shape, L, in_dim*K]`` after."""

    mask: Tensor
    """Same shape as ``latent``, values in ``{0, 1}``; ``1`` marks positions
    re-injected into the noisy latent / ``x0``."""

    _is_patchified: bool = False


@dataclass(kw_only=True)
class I2VCtrlEncoderConfig(InstantiateConfig["I2VCtrlEncoder"]):
    """Config for the I2V control encoder."""

    _target: type["I2VCtrlEncoder"] = field(default_factory=lambda: I2VCtrlEncoder)

    encoder: WanVAEEncoderConfig = field(default_factory=WanVAEEncoderConfig)
    """Streaming Wan VAE encoder. Pin its checkpoint to the decoder's so
    the encoded latent matches the network's input distribution."""


class I2VCtrlEncoder(Encoder[I2VCtrlEncoderCache]):
    """Per-AR-step I2V control encoder.

    Forward takes the AR-step pixel chunk ``[B, T_pixel, 3, H, W]`` in
    ``[-1, 1]``:

    - AR step 0: the user's first frame plus zeros along T; the streaming
      VAE produces ``len_t`` latent frames with the encoded image at index 0,
      and the mask is one-hot on the first frame.
    - AR step > 0: pure zeros to flush the VAE's temporal context; the mask
      is all-zeros so the network ignores the resulting latent.
    """

    encoder: WanVAEEncoder

    def __init__(self, config: I2VCtrlEncoderConfig) -> None:
        super().__init__(config)
        self.config: I2VCtrlEncoderConfig = config
        self.encoder = config.encoder.setup()

    def initialize_autoregressive_cache(self) -> I2VCtrlEncoderCache:
        return self.encoder.initialize_autoregressive_cache()

    @torch.no_grad()
    def forward(
        self,
        input: Tensor,
        autoregressive_index: int = 0,
        cache: I2VCtrlEncoderCache | None = None,
    ) -> I2VCtrl:
        # TODO: the Wan VAE encoder is identity after chunk 3, so for I2V we
        # could cache and skip the VAE call past that point.
        latent = self.encoder(
            input,
            autoregressive_index=autoregressive_index,
            cache=cache,
        )
        # Mask shape matches latent so they patchify identically and the
        # downstream blend is a plain elementwise multiply.
        mask = torch.zeros_like(latent)
        if autoregressive_index == 0:
            mask[..., 0, :, :, :] = 1.0
        return I2VCtrl(latent=latent, mask=mask)

    @property
    def temporal_compression_ratio(self) -> int:
        return self.encoder.temporal_compression_ratio

    @property
    def spatial_compression_ratio(self) -> int:
        return self.encoder.spatial_compression_ratio
