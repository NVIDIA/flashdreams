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

from dataclasses import dataclass
from pathlib import Path

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
from flashdreams.runtime.video_runner import (
    ImageConditionedVideoRunnerConfig,
    VideoRunnerConfig,
)

DEFAULT_PROMPT = (
    "Summer beach vacation style, a white cat wearing sunglasses sits on "
    "a surfboard. The fluffy-furred feline gazes directly at the camera "
    "with a relaxed expression. Blurred beach scenery forms the background "
    "featuring crystal-clear waters, distant green hills, and a blue sky "
    "dotted with white clouds. The cat assumes a naturally relaxed posture, "
    "as if savoring the sea breeze and warm sunlight. A close-up shot "
    "highlights the feline's intricate details and the refreshing "
    "atmosphere of the seaside."
)

DEFAULT_I2V_IMAGE_URL = (
    "https://raw.githubusercontent.com/Wan-Video/Wan2.1/main/examples/i2v_input.JPG"
)


@dataclass(kw_only=True)
class Wan21T2VRunnerConfig(VideoRunnerConfig):
    """Runner config for the Wan 2.1 T2V variant."""

    prompt: str | Path = DEFAULT_PROMPT
    pixel_height: int = 480
    pixel_width: int = 832
    fps: int = 16


@dataclass(kw_only=True)
class Wan21I2VRunnerConfig(ImageConditionedVideoRunnerConfig, Wan21T2VRunnerConfig):
    """Runner config for the Wan 2.1 I2V variant."""

    image_path: str | Path = DEFAULT_I2V_IMAGE_URL
    image_cache_subdir = "wan21"
    pixel_height: int = 832
    pixel_width: int = 480


CHECKPOINT_PATH_T2V_1PT3B = (
    "https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/blob/main/"
    "diffusion_pytorch_model.safetensors"
)
CHECKPOINT_PATH_I2V_14B_480P = (
    "https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P/blob/main/"
    "diffusion_pytorch_model.safetensors.index.json"
)

PIPELINE_WAN21_T2V_1PT3B_480P = WanInferencePipelineConfig(
    name="wan21-t2v-1.3b-480p",
    enable_sync_and_profile=True,
    encoder=None,
    decoder=WanVAEDecoderConfig(),
    diffusion_model=DiffusionModelConfig(
        seed=42,
        transformer=Wan21TransformerConfig(
            network=WanDiTNetwork1pt3BConfig(cp_method="ring"),
            checkpoint_path=CHECKPOINT_PATH_T2V_1PT3B,
            batch_shape=(),
            len_t=21,
            window_size_t=21,
            guidance_scale=6.0,
        ),
        scheduler=FlowMatchUniPCSchedulerConfig(
            num_inference_steps=50,
            shift=8.0,
            enable_tqdm=True,
        ),
    ),
)
RUNNER_WAN21_T2V_1PT3B_480P = Wan21T2VRunnerConfig(
    runner_name=PIPELINE_WAN21_T2V_1PT3B_480P.name,
    description="Wan 2.1 T2V 1.3B at 480p (single AR step, prompt-only).",
    pipeline=PIPELINE_WAN21_T2V_1PT3B_480P,
)

PIPELINE_WAN21_I2V_14B_480P = WanInferencePipelineConfig(
    name="wan21-i2v-14b-480p",
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
                cp_method="ring",
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
            enable_tqdm=True,
        ),
    ),
    image_encoder=CLIPImageEncoderConfig(
        model_id_or_local_path="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
    ),
)
RUNNER_WAN21_I2V_14B_480P = Wan21I2VRunnerConfig(
    runner_name=PIPELINE_WAN21_I2V_14B_480P.name,
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
