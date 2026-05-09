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

"""Pre-built pipeline configs for non-streaming Wan 2.1.

One module-level literal per shipped variant. Per-rollout latent
``(height, width)`` is supplied to
:meth:`WanInferencePipeline.initialize_cache`; for 480p use
``height=60, width=104`` (``480/8, 832/8``) for the T2V variant.
"""

from __future__ import annotations

from flashdreams.infra.diffusion.model import DiffusionModelConfig
from flashdreams.infra.diffusion.scheduler import (
    FlowMatchUniPCSchedulerConfig,
)
from flashdreams.infra.encoder.image.clip import CLIPImageEncoderConfig
from flashdreams.recipes.wan.autoencoder.i2v import (
    I2VCtrlEncoderConfig,
)
from flashdreams.recipes.wan.autoencoder.vae import (
    AVAILABLE_WAN_VAE_CHECKPOINT_PATHS,
    WanVAEDecoderConfig,
    WanVAEEncoderConfig,
)
from flashdreams.recipes.wan.pipeline import WanInferencePipelineConfig
from flashdreams.recipes.wan.transformer.impl.network import (
    WanDiTNetwork1pt3BConfig,
    WanDiTNetwork14BConfig,
)
from flashdreams.recipes.wan.transformer.wan21 import Wan21TransformerConfig

## Wan 2.1 1.3B T2V (official Wan-AI checkpoint, 480p).
##
## ``len_t == window_size_t == 21`` -> single-AR-step rollout: the whole
## 81-frame video is one chunk.
WAN21_T2V_1PT3B_480P = WanInferencePipelineConfig(
    recipe_name="wan21-t2v-1.3b-480p",
    enable_sync_and_profile=True,
    encoder=None,
    decoder=WanVAEDecoderConfig(),
    diffusion_model=DiffusionModelConfig(
        seed=42,
        transformer=Wan21TransformerConfig(
            network=WanDiTNetwork1pt3BConfig(),
            checkpoint_path=(
                "https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/blob/main/"
                "diffusion_pytorch_model.safetensors"
            ),
            batch_shape=(),
            len_t=21,
            window_size_t=21,
            guidance_scale=6.0,
        ),
        scheduler=FlowMatchUniPCSchedulerConfig(
            num_inference_steps=50,
            shift=8.0,
        ),
    ),
)

## Wan 2.1 14B I2V (official Wan-AI checkpoint, 480p).
##
## Per-rollout latent ``(height, width)`` is derived from the input
## image's pixel size in :meth:`WanInferencePipeline.initialize_cache`.
## ``in_dim = 16 + 4 + 16``: 16 noise channels + 4-channel mask +
## 16-channel image latent (channel-concat I2V layout). Must match
## ``concat_image_mask_to_latent=True``.
WAN21_I2V_14B_480P = WanInferencePipelineConfig(
    recipe_name="wan21-i2v-14b-480p",
    enable_sync_and_profile=True,
    encoder=I2VCtrlEncoderConfig(
        encoder=WanVAEEncoderConfig(
            checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"],
        ),
    ),
    decoder=WanVAEDecoderConfig(
        checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"],
    ),
    diffusion_model=DiffusionModelConfig(
        seed=42,
        transformer=Wan21TransformerConfig(
            network=WanDiTNetwork14BConfig(
                cross_attn_enable_img=True,
                in_dim=16 + 4 + 16,
            ),
            checkpoint_path=(
                "https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P/blob/main/"
                "diffusion_pytorch_model.safetensors.index.json"
            ),
            batch_shape=(),
            len_t=21,
            window_size_t=21,
            guidance_scale=5.0,
            concat_image_mask_to_latent=True,
        ),
        scheduler=FlowMatchUniPCSchedulerConfig(
            num_inference_steps=40,
            shift=3.0,
        ),
    ),
    image_encoder=CLIPImageEncoderConfig(
        model_id_or_local_path="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
    ),
)

WAN21_CONFIGS: dict[str, WanInferencePipelineConfig] = {
    cfg.recipe_name: cfg
    for cfg in (
        WAN21_T2V_1PT3B_480P,
        WAN21_I2V_14B_480P,
    )
}
"""All shipped non-streaming Wan 2.1 variants, keyed by ``recipe_name``."""
