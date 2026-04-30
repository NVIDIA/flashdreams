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

import torch
from torch import Tensor

from flashdreams.infra.encoder import (
    Encoder,
    EncoderConfig,
)

from .vae import (
    WanVAECache,
    WanVAEEncoder,
    WanVAEEncoderConfig,
)


@dataclass(kw_only=True)
class I2VCtrl:
    """I2V control payload produced by the per-AR-step encoder.

    Attributes:
        latent: ``[*batch_shape, len_t, in_dim, Hl, Wl]`` (unpatchified)
            or ``[*batch_shape, L, in_dim * K]`` (patchified +
            CP-split). The transformer's underlying
            :class:`WanDiTNetwork` must have a matching ``in_dim``.
        mask: same shape as ``latent``. Values in ``{0, 1}``: ``1`` at
            (in_dim-channel-replicated) positions whose latent value
            should be re-injected into ``noisy_latent`` / ``x0``, ``0``
            everywhere else.
    """

    latent: Tensor
    mask: Tensor

    _is_patchified: bool = False


@dataclass(kw_only=True)
class I2VCtrlEncoderConfig(EncoderConfig):
    """Configuration for :class:`I2VCtrlEncoder`."""

    _target: type["I2VCtrlEncoder"] = field(default_factory=lambda: I2VCtrlEncoder)

    encoder: WanVAEEncoderConfig = field(default_factory=WanVAEEncoderConfig)
    """Streaming Wan VAE encoder. Pin its checkpoint to the same Wan VAE
    used by the decoder so the encoded latent matches the network's
    input distribution exactly."""


@dataclass(kw_only=True)
class I2VCtrlEncoderCache(WanVAECache):
    """Per-AR-step I2V control encoder cache."""


class I2VCtrlEncoder(Encoder[I2VCtrlEncoderCache]):
    """Per-AR-step I2V control encoder.

    Forward input is the raw pixel chunk for this AR step
    (``[B, T_pixel, 3, H, W]`` in ``[-1, 1]``):

      * AR step 0: the user's first-frame image followed by zeros along
        T so the streaming VAE produces ``len_t`` latent frames whose
        first frame is the encoded image. The mask is ``[1, 0, 0, ...]``
        along T.
      * AR step ``> 0``: pure zeros so the streaming VAE flushes its
        temporal context. The mask is all-zeros (the network ignores
        these latent values).

    The :class:`WanVAECache` lives on
    :attr:`StreamInferencePipelineCache.encoder_cache` and advances in place
    across AR steps.
    """

    encoder: WanVAEEncoder

    def __init__(self, config: I2VCtrlEncoderConfig) -> None:
        super().__init__(config)
        self.config: I2VCtrlEncoderConfig = config
        self.encoder = config.encoder.setup()  # ty:ignore[invalid-assignment]

    def initialize_autoregressive_cache(self) -> I2VCtrlEncoderCache:
        return self.encoder.initialize_autoregressive_cache()  # ty:ignore[invalid-return-type]

    @torch.no_grad()
    def forward(
        self,
        input: Tensor,
        autoregressive_index: int = 0,
        cache: I2VCtrlEncoderCache | None = None,
    ) -> I2VCtrl:
        # TODO: Wan VAE encoder returns all the same after chunk3. So
        # we can cache and skip VAE here as an optimization for I2V encoding.
        latent = self.encoder(  # [*batch_shape, len_t, in_dim, Hl, Wl]
            input,
            autoregressive_index=autoregressive_index,
            cache=cache,
        )
        # Mask matches ``latent`` shape exactly so they patchify to the
        # same layout and inject with a plain elementwise multiply.
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
