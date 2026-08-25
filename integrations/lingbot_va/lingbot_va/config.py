# SPDX-FileCopyrightText: Copyright (c) 2026 Hongyu Zhou
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
"""Static configs for the LingBot-VA Robotwin I2AV integration."""

from __future__ import annotations

from flashdreams.infra.diffusion.model import DiffusionModelConfig
from lingbot_va.constants import (
    DEFAULT_CHECKPOINT_ROOT,
    ROBOTWIN_ACTION_INFERENCE_STEPS,
    ROBOTWIN_ACTION_PER_FRAME,
    ROBOTWIN_ACTION_SNR_SHIFT,
    ROBOTWIN_ACTION_TOKEN_PER_CHUNK,
    ROBOTWIN_ATTENTION_WINDOW,
    ROBOTWIN_FRAME_CHUNK_SIZE,
    ROBOTWIN_HEIGHT,
    ROBOTWIN_LATENT_CHANNELS,
    ROBOTWIN_LATENT_HEIGHT,
    ROBOTWIN_LATENT_TOKEN_PER_CHUNK,
    ROBOTWIN_LATENT_WIDTH,
    ROBOTWIN_SNR_SHIFT,
    ROBOTWIN_VIDEO_INFERENCE_STEPS,
    ROBOTWIN_WIDTH,
    RUNNER_NAME_ROBOTWIN_I2AV,
)
from lingbot_va.pipeline import LingbotVAInferencePipelineConfig
from lingbot_va.scheduler import LingbotVAFlowMatchSchedulerConfig
from lingbot_va.transformer import LingbotVATransformerConfig

PIPELINE_LINGBOT_VA_ROBOTWIN_I2AV = LingbotVAInferencePipelineConfig(
    name=RUNNER_NAME_ROBOTWIN_I2AV,
    enable_sync_and_profile=True,
    encoder=None,
    decoder=None,
    checkpoint_root=DEFAULT_CHECKPOINT_ROOT,
    robotwin_height=ROBOTWIN_HEIGHT,
    robotwin_width=ROBOTWIN_WIDTH,
    frame_chunk_size=ROBOTWIN_FRAME_CHUNK_SIZE,
    action_per_frame=ROBOTWIN_ACTION_PER_FRAME,
    attn_window=ROBOTWIN_ATTENTION_WINDOW,
    latent_height=ROBOTWIN_LATENT_HEIGHT,
    latent_width=ROBOTWIN_LATENT_WIDTH,
    latent_channels=ROBOTWIN_LATENT_CHANNELS,
    latent_token_per_chunk=ROBOTWIN_LATENT_TOKEN_PER_CHUNK,
    action_token_per_chunk=ROBOTWIN_ACTION_TOKEN_PER_CHUNK,
    diffusion_model=DiffusionModelConfig(
        seed=42,
        transformer=LingbotVATransformerConfig(
            checkpoint_root=DEFAULT_CHECKPOINT_ROOT,
            guidance_scale=5.0,
            action_guidance_scale=1.0,
            latent_height=ROBOTWIN_LATENT_HEIGHT,
            latent_width=ROBOTWIN_LATENT_WIDTH,
            frame_chunk_size=ROBOTWIN_FRAME_CHUNK_SIZE,
            action_per_frame=ROBOTWIN_ACTION_PER_FRAME,
            attn_window=ROBOTWIN_ATTENTION_WINDOW,
        ),
        scheduler=LingbotVAFlowMatchSchedulerConfig(
            num_inference_steps=ROBOTWIN_VIDEO_INFERENCE_STEPS,
            shift=ROBOTWIN_SNR_SHIFT,
        ),
    ),
    action_scheduler=LingbotVAFlowMatchSchedulerConfig(
        num_inference_steps=ROBOTWIN_ACTION_INFERENCE_STEPS,
        shift=ROBOTWIN_ACTION_SNR_SHIFT,
    ),
)
PIPELINE_CONFIGS: dict[str, LingbotVAInferencePipelineConfig] = {
    PIPELINE_LINGBOT_VA_ROBOTWIN_I2AV.name: PIPELINE_LINGBOT_VA_ROBOTWIN_I2AV,
}
