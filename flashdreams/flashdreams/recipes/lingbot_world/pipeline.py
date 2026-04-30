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

"""Lingbot World streaming inference pipeline (camera-controlled I2V)."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from flashdreams.infra.pipeline import StreamInferencePipeline
from flashdreams.recipes.lingbot_world.encoder.camctrl import (
    CamCtrlInput,
    I2VCamCtrlInput,
)
from flashdreams.recipes.wan.pipeline import (
    WanInferencePipeline,
    WanInferencePipelineCache,
    WanInferencePipelineConfig,
)


@dataclass(kw_only=True)
class LingbotWorldInferencePipelineConfig(WanInferencePipelineConfig):
    """Config for :class:`LingbotWorldInferencePipeline`.

    Identical in shape to :class:`WanInferencePipelineConfig`; only
    ``_target`` is overridden so ``.setup()`` instantiates the Lingbot
    subclass.
    """

    _target: type["LingbotWorldInferencePipeline"] = field(
        default_factory=lambda: LingbotWorldInferencePipeline
    )


class LingbotWorldInferencePipeline(WanInferencePipeline):
    """Streaming camera-controlled I2V pipeline for Lingbot World.

    The only behavioral difference from :class:`WanInferencePipeline`
    is :meth:`generate`'s signature: the caller hands in a
    :class:`CamCtrlInput` (intrinsics + poses + world scale) which
    this pipeline packs with the per-AR-step image chunk into the
    :class:`I2VCamCtrlInput` that :class:`I2VCamCtrlEncoder` expects.
    """

    @torch.no_grad()
    def generate(  # type: ignore[override]
        self,
        autoregressive_index: int,
        cache: WanInferencePipelineCache,
        input: CamCtrlInput,
    ) -> Tensor:
        """Generate one decoded video chunk for AR step ``autoregressive_index``.

        Args:
            autoregressive_index: AR step index — ``0`` on the first
                call after :meth:`initialize_cache`, then strictly
                increasing by 1 per call.
            cache: Per-rollout cache returned by
                :meth:`initialize_cache`. ``cache.image`` must be
                populated (Lingbot World is I2V-only).
            input: Camera payload for this AR step — intrinsics,
                poses, and world scale. The matching first-frame pixel
                chunk is constructed internally from ``cache.image``.

        Returns:
            Decoded video of shape ``[*batch_shape, T, C, H, W]`` in
            ``[-1, 1]``.
        """
        assert cache.image is not None, (
            "LingbotWorldInferencePipeline is I2V-only; pass ``image=...`` "
            "to ``initialize_cache``."
        )
        i2v_chunk = self._preprocess_i2v_input(autoregressive_index, cache.image)
        camctrl_input = I2VCamCtrlInput(i2v=i2v_chunk, camctrl=input)

        # Dispatch to the infra base class directly:
        # ``WanInferencePipeline.generate`` would rebuild ``input`` from
        # ``cache.image`` and discard our composite payload.
        return StreamInferencePipeline.generate(
            self,
            autoregressive_index=autoregressive_index,
            cache=cache,  # ty:ignore[invalid-argument-type]
            input=camctrl_input,
        )
