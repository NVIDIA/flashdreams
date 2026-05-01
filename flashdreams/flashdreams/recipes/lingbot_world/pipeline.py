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
    """Config for the Lingbot World streaming pipeline.

    Same shape as the Wan I2V config; only the target class is overridden.
    """

    _target: type["LingbotWorldInferencePipeline"] = field(
        default_factory=lambda: LingbotWorldInferencePipeline
    )


class LingbotWorldInferencePipeline(WanInferencePipeline):
    """Streaming camera-controlled I2V pipeline for Lingbot World.

    The only difference from the base Wan I2V pipeline is ``generate``:
    the caller passes a camera payload (intrinsics, poses, world scale),
    and the per-AR-step image chunk is constructed internally from the
    cached first frame.
    """

    @torch.no_grad()
    def generate(
        self,
        autoregressive_index: int,
        cache: WanInferencePipelineCache,
        input: CamCtrlInput,
    ) -> Tensor:
        """Generate one decoded video chunk for AR step ``autoregressive_index``.

        Args:
            autoregressive_index: AR step index, starting at 0.
            cache: Cache from ``initialize_cache``; ``cache.image`` must
                be populated (Lingbot World is I2V-only).
            input: Per-AR-step camera payload (intrinsics, poses, world
                scale). The first-frame pixel chunk is built internally
                from ``cache.image``.

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

        # Skip ``WanInferencePipeline.generate`` -- it would rebuild ``input``
        # from ``cache.image`` and discard the composite camera payload.
        # Using ``super(WanInferencePipeline, self)`` jumps directly to
        # ``StreamInferencePipeline.generate`` while keeping ``self``'s
        # generic-parameter bindings.
        return super(WanInferencePipeline, self).generate(
            autoregressive_index=autoregressive_index,
            cache=cache,
            input=camctrl_input,
        )
