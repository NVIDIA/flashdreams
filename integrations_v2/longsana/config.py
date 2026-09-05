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

"""Public LongSana Runtime V2 pipeline configuration."""

from __future__ import annotations

import torch

from flashdreams.infra.diffusion.model import DiffusionModelConfig
from flashdreams.recipes.wan.autoencoder.vae import WanVAEDecoderConfig
from longsana.impl.constants import (
    DEFAULT_DENOISING_TIMESTEPS,
    LONGSANA_TEXT_CONFIG_PATH,
    LONGSANA_VAE_CHECKPOINT_PATH,
)
from longsana.impl.pipeline import LongSanaPipelineConfig
from longsana.impl.scheduler import LongSanaFlowMatchSchedulerConfig
from longsana.impl.transformer import LongSanaTransformerConfig
from sana_wm.impl.conditioning import SanaWMTextPromptEncoderConfig

PIPELINE_LONGSANA_2B_480P = LongSanaPipelineConfig(
    name="longsana-2b-480p",
    enable_sync_and_profile=True,
    encoder=None,
    prompt_encoder=SanaWMTextPromptEncoderConfig(
        config_path=LONGSANA_TEXT_CONFIG_PATH,
        offload_text_encoder=True,
    ),
    decoder=WanVAEDecoderConfig(
        checkpoint_path=LONGSANA_VAE_CHECKPOINT_PATH,
        dtype=torch.float32,
        use_cuda_graph=False,
        use_compile=False,
    ),
    diffusion_model=DiffusionModelConfig(
        seed=0,
        context_noise=0,
        transformer=LongSanaTransformerConfig(),
        scheduler=LongSanaFlowMatchSchedulerConfig(
            num_inference_steps=4,
            shift=7.0,
            denoising_timesteps=list(DEFAULT_DENOISING_TIMESTEPS),
            warp_denoising_step=False,
            num_train_timesteps=1000,
            sigma_max=1.0,
            sigma_min=0.0,
            extra_one_step=True,
            timestep_dtype=torch.float32,
            enable_tqdm=True,
        ),
    ),
)
"""Official four-step LongSana 2B 480p pipeline."""

LONGSANA_CONFIGS: dict[str, LongSanaPipelineConfig] = {
    PIPELINE_LONGSANA_2B_480P.name: PIPELINE_LONGSANA_2B_480P,
}
"""All public LongSana pipeline configurations."""
