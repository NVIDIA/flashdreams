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

"""Configs for non-streaming Wan 2.1."""

from __future__ import annotations

from flashdreams.infra.diffusion.model import DiffusionModelConfig
from flashdreams.infra.diffusion.scheduler import (
    FlowMatchUniPCSchedulerConfig,
)
from flashdreams.infra.encoder.image.clip import CLIPImageEncoderConfig
from flashdreams.infra.runner import RunnerConfig
from flashdreams.recipes.wan import (
    Wan21TransformerConfig,
    WanDiTNetwork1pt3BConfig,
    WanDiTNetwork14BConfig,
    WanI2VCtrlEncoderConfig,
    WanInferencePipelineConfig,
    WanVAEDecoderConfig,
    WanVAEEncoderConfig,
)
from wan21.runner import Wan21I2VRunnerConfig, Wan21T2VRunnerConfig

CHECKPOINT_PATH_T2V_1PT3B = (
    "https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/blob/main/"
    "diffusion_pytorch_model.safetensors"
)
CHECKPOINT_PATH_I2V_14B_480P = (
    "https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P/blob/main/"
    "diffusion_pytorch_model.safetensors.index.json"
)

PIPELINE_WAN21_T2V_1PT3B_480P = WanInferencePipelineConfig(
    recipe_name="wan21-t2v-1.3b-480p",
    enable_sync_and_profile=True,
    encoder=None,
    decoder=WanVAEDecoderConfig(),
    diffusion_model=DiffusionModelConfig(
        seed=42,
        transformer=Wan21TransformerConfig(
            network=WanDiTNetwork1pt3BConfig(),
            checkpoint_path=CHECKPOINT_PATH_T2V_1PT3B,
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
RUNNER_WAN21_T2V_1PT3B_480P = Wan21T2VRunnerConfig(
    runner_name=PIPELINE_WAN21_T2V_1PT3B_480P.recipe_name,
    description="Wan 2.1 T2V 1.3B at 480p (single AR step, prompt-only).",
    pipeline=PIPELINE_WAN21_T2V_1PT3B_480P,
)

PIPELINE_WAN21_I2V_14B_480P = WanInferencePipelineConfig(
    recipe_name="wan21-i2v-14b-480p",
    enable_sync_and_profile=True,
    encoder=WanI2VCtrlEncoderConfig(
        encoder=WanVAEEncoderConfig(),
    ),
    decoder=WanVAEDecoderConfig(),
    diffusion_model=DiffusionModelConfig(
        seed=42,
        transformer=Wan21TransformerConfig(
            network=WanDiTNetwork14BConfig(
                cross_attn_enable_img=True,
                in_dim=16 + 4 + 16,
            ),
            checkpoint_path=CHECKPOINT_PATH_I2V_14B_480P,
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
RUNNER_WAN21_I2V_14B_480P = Wan21I2VRunnerConfig(
    runner_name=PIPELINE_WAN21_I2V_14B_480P.recipe_name,
    description="Wan 2.1 I2V 14B at 480p (single AR step, prompt + first-frame).",
    pipeline=PIPELINE_WAN21_I2V_14B_480P,
)

RUNNER_CONFIGS: dict[str, RunnerConfig] = {
    cfg.runner_name: cfg
    for cfg in (
        RUNNER_WAN21_T2V_1PT3B_480P,
        RUNNER_WAN21_I2V_14B_480P,
    )
}
