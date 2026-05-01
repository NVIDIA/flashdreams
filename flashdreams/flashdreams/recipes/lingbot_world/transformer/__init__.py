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

"""Wan 2.1 transformer adapter with Plücker camera control for Lingbot World."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import overload

import torch
from torch import Tensor

from flashdreams.recipes.lingbot_world.encoder.camctrl import I2VCamCtrlEmbeddings
from flashdreams.recipes.wan.transformer.wan21 import (
    Wan21Transformer,
    Wan21TransformerCache,
    Wan21TransformerConfig,
)

from .impl.network import (
    LingbotWorldDiTNetwork14BConfig,
    LingbotWorldDiTNetworkCache,
    LingbotWorldDiTNetworkConfig,
)


@dataclass(kw_only=True)
class LingbotWorldTransformerCache(Wan21TransformerCache):
    """Long-lived AR cache for the Lingbot World transformer.

    The guidance scale lives on the config because it is a model-level
    hyperparameter, not per-rollout state.
    """

    network_cache_cond: LingbotWorldDiTNetworkCache
    """Conditional per-block KV / cross-attn cache."""

    network_cache_uncond: LingbotWorldDiTNetworkCache | None = None
    """Unconditional per-block caches; ``None`` disables CFG."""


@dataclass(kw_only=True)
class LingbotWorldTransformerConfig(Wan21TransformerConfig):
    """Config for the Lingbot World transformer.

    Each instance is bound to one ``(batch_shape, height, width,
    len_t)`` layout and one ``cp_size``. The I2V channel-concat ``in_dim``
    is enforced in ``__post_init__``.
    """

    _target: type["LingbotWorldTransformer"] = field(
        default_factory=lambda: LingbotWorldTransformer
    )

    network: LingbotWorldDiTNetworkConfig = field(
        default_factory=LingbotWorldDiTNetwork14BConfig
    )


class LingbotWorldTransformer(Wan21Transformer):
    """Lingbot World DiT (Wan 2.1 + per-block Plücker camera control)."""

    def __init__(
        self,
        config: LingbotWorldTransformerConfig,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(config, device=device)

    def predict_flow(
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        cache: LingbotWorldTransformerCache,
        input: I2VCamCtrlEmbeddings,
    ) -> Tensor:
        return super().predict_flow(
            noisy_latent=noisy_latent,
            timestep=timestep,
            cache=cache,
            input=input.i2v,
            network_extra_kwargs={"plucker": input.plucker},
        )

    @overload
    def patchify_and_maybe_split_cp(self, x: Tensor) -> Tensor: ...
    @overload
    def patchify_and_maybe_split_cp(
        self, x: I2VCamCtrlEmbeddings
    ) -> I2VCamCtrlEmbeddings: ...
    def patchify_and_maybe_split_cp(
        self, x: Tensor | I2VCamCtrlEmbeddings
    ) -> Tensor | I2VCamCtrlEmbeddings:
        """Patchify and (optionally) split for context parallelism."""
        if isinstance(x, I2VCamCtrlEmbeddings):
            if x._is_patchified:
                return x
            return I2VCamCtrlEmbeddings(
                i2v=super().patchify_and_maybe_split_cp(x.i2v),
                plucker=super().patchify_and_maybe_split_cp(x.plucker),
                _is_patchified=True,
            )
        return super().patchify_and_maybe_split_cp(x)
