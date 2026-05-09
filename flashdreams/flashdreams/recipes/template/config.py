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

"""Pre-built ``StreamInferencePipelineConfig`` literals for the template recipe."""

from __future__ import annotations

from typing import cast

import torch

from flashdreams.infra.config import derive_config
from flashdreams.infra.diffusion.model import DiffusionModelConfig
from flashdreams.infra.diffusion.scheduler.fm import FlowMatchSchedulerConfig
from flashdreams.infra.pipeline import StreamInferencePipelineConfig
from flashdreams.recipes.template.decoder import TemplateDecoderConfig
from flashdreams.recipes.template.encoder import TemplateControlEncoderConfig
from flashdreams.recipes.template.transformer import TemplateTransformerConfig
from flashdreams.recipes.template.transformer.network import TemplateDiTConfig

## Offline (bidirectional, one-shot) reference rollout.
##
## Single AR step over the full temporal window
## (``window_size_t == len_t == 8``), CFG off, per-step control encoded
## into the latent channel count, clean latent decoded to 3 channels.
## ``head_dim = 128 // 2 = 64`` so cuDNN flash-attention picks a stable
## kernel; smaller head_dims (16/8) silently NaN. The network's
## ``in_channels`` is the post-patch width
## ``4 * (2 * 2 * 2) = 32``; ``patch_size = (2, 2, 2)`` must match
## :attr:`TemplateTransformerConfig.patch_size`.
TEMPLATE_OFFLINE = StreamInferencePipelineConfig(
    recipe_name="template-offline",
    encoder=TemplateControlEncoderConfig(
        control_channels=8,
        out_channels=4,
        dtype=torch.bfloat16,
    ),
    decoder=TemplateDecoderConfig(
        in_channels=4,
        out_channels=3,
        dtype=torch.bfloat16,
    ),
    diffusion_model=DiffusionModelConfig(
        seed=42,
        context_noise=0,
        transformer=TemplateTransformerConfig(
            network=TemplateDiTConfig(
                in_channels=4 * (2 * 2 * 2),
                context_channels=16,
                model_channels=128,
                num_heads=2,
            ),
            patch_size=(2, 2, 2),
            len_t=8,
            window_size_t=8,
            sink_size_t=0,
            guidance_scale=1.0,
            dtype=torch.bfloat16,
        ),
        scheduler=FlowMatchSchedulerConfig(
            num_inference_steps=2,
            denoising_timesteps=[1000, 500],
            warp_denoising_step=True,
            shift=5.0,
            num_train_timesteps=1000,
        ),
    ),
)

## Streaming AR variant: smaller per-chunk ``len_t`` (2) and a larger
## ``window_size_t`` (4 = 2 * len_t) so the KV cache fills over multiple
## AR steps before rolling. CFG still off; patch ``guidance_scale > 1.0``
## via :func:`derive_config` to enable it.
TEMPLATE_AUTOREGRESSIVE = cast(
    StreamInferencePipelineConfig,
    derive_config(
        TEMPLATE_OFFLINE,
        recipe_name="template-autoregressive",
        diffusion_model=dict(
            transformer=dict(
                len_t=2,
                window_size_t=4,
            ),
            scheduler=dict(
                num_inference_steps=1,
                denoising_timesteps=[500],
            ),
        ),
    ),
)

## Streaming AR with ``torch.compile`` + ``CUDAGraphWrapper`` enabled on
## the DiT network. The fast deployment path: keep ``TEMPLATE_AUTOREGRESSIVE``
## as the easy-to-debug default and reach for this when measuring
## inference latency.
TEMPLATE_AUTOREGRESSIVE_COMPILED = cast(
    StreamInferencePipelineConfig,
    derive_config(
        TEMPLATE_AUTOREGRESSIVE,
        recipe_name="template-autoregressive-compiled",
        diffusion_model=dict(
            transformer=dict(
                compile_network=True,
                use_cuda_graph=True,
            ),
        ),
    ),
)

TEMPLATE_CONFIGS: dict[str, StreamInferencePipelineConfig] = {
    cfg.recipe_name: cfg
    for cfg in (
        TEMPLATE_OFFLINE,
        TEMPLATE_AUTOREGRESSIVE,
        TEMPLATE_AUTOREGRESSIVE_COMPILED,
    )
}
"""All shipped template-recipe variants, keyed by ``recipe_name``."""
