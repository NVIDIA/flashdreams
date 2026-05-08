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

"""Pipeline-config builders for streaming Wan 2.2."""

from __future__ import annotations

from collections.abc import Callable

import torch

from flashdreams.core.checkpoint.remap import remap_checkpoint_keys
from flashdreams.infra.diffusion.model import DiffusionModelConfig
from flashdreams.infra.diffusion.scheduler.fm import FlowMatchSchedulerConfig
from flashdreams.recipes.wan.autoencoder.vae import (
    AVAILABLE_WAN_VAE_CHECKPOINT_PATHS,
    WanVAEDecoderConfig,
)
from flashdreams.recipes.wan.pipeline import WanInferencePipelineConfig
from flashdreams.recipes.wan.transformer.impl.network import (
    WanDiTNetwork14BConfig,
)
from flashdreams.recipes.wan.transformer.wan21 import Wan21TransformerConfig
from flashdreams.recipes.wan.transformer.wan22 import (
    CHECKPOINT_KEY_MAPPING,
    Wan22TransformerConfig,
)

AVAILABLE_CAUSAL_WAN22_CHECKPOINT_PATHS: dict[str, dict[str, str]] = {
    "fastvideo": {
        "high_noise": "https://huggingface.co/FastVideo/CausalWan2.2-I2V-A14B-Preview-Diffusers/blob/main/transformer/diffusion_pytorch_model.safetensors",
        "low_noise": "https://huggingface.co/FastVideo/CausalWan2.2-I2V-A14B-Preview-Diffusers/blob/main/transformer_2/diffusion_pytorch_model.safetensors",
    },
}


## Checkpoint remap


def _remap_diffusers_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remap an HF diffusers Wan 2.2 state-dict to the WanDiTNetwork layout."""
    return remap_checkpoint_keys(state_dict, CHECKPOINT_KEY_MAPPING)


## Canonical Wan 2.2 streaming defaults

# FastVideo 8-step distillation schedule.
_DEFAULT_DENOISING_TIMESTEPS = [1000, 850, 700, 550, 350, 275, 200, 125]
_DEFAULT_NUM_TRAIN_TIMESTEPS = 1000
_DEFAULT_BOUNDARY_RATIO = 0.875

_DEFAULT_BATCH_SHAPE: tuple[int, ...] = (1,)
# Canonical pixel-space defaults; callers pass the matching latent
# (height, width) into :meth:`WanInferencePipeline.initialize_cache`.
DEFAULT_VIDEO_HEIGHT = 480
DEFAULT_VIDEO_WIDTH = 832
_DEFAULT_LEN_T_LATENT = 3
WAN_VAE_SPATIAL_COMPRESSION = 8


def _wan_vae_decoder_config() -> WanVAEDecoderConfig:
    """Wan VAE decoder config."""
    return WanVAEDecoderConfig(
        checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"],
    )


def _scheduler_config(
    num_inference_steps: int = len(_DEFAULT_DENOISING_TIMESTEPS),
) -> FlowMatchSchedulerConfig:
    """FastVideo Wan 2.2 flow-match scheduler defaults."""
    timesteps = _DEFAULT_DENOISING_TIMESTEPS[:num_inference_steps]
    return FlowMatchSchedulerConfig(
        num_inference_steps=num_inference_steps,
        denoising_timesteps=timesteps,
        warp_denoising_step=True,
        shift=5.0,
        sigma_min=0.0,
        extra_one_step=True,
        num_train_timesteps=_DEFAULT_NUM_TRAIN_TIMESTEPS,
    )


def _transformer_config(
    *,
    checkpoint_path: dict[str, str],
    compile_network: bool,
) -> Wan22TransformerConfig:
    """Wan 2.2 dual-14B transformer defaults for causal/streaming T2V."""

    def _branch(ckpt: str) -> Wan21TransformerConfig:
        return Wan21TransformerConfig(
            network=WanDiTNetwork14BConfig(
                patch_embedding_type="conv3d",
            ),
            checkpoint_path=ckpt,
            state_dict_transform=_remap_diffusers_state_dict,
            batch_shape=_DEFAULT_BATCH_SHAPE,
            len_t=_DEFAULT_LEN_T_LATENT,
            guidance_scale=1.0,
            window_size_t=21,
            sink_size_t=0,
            compile_network=compile_network,
        )

    return Wan22TransformerConfig(
        transformer_high_noise=_branch(checkpoint_path["high_noise"]),
        transformer_low_noise=_branch(checkpoint_path["low_noise"]),
        boundary_ratio=_DEFAULT_BOUNDARY_RATIO,
        num_train_timesteps=_DEFAULT_NUM_TRAIN_TIMESTEPS,
    )


## Builders


def build_fastvideo(
    *,
    compile_network: bool = True,
    seed: int = 42,
    enable_sync_and_profile: bool = False,
) -> WanInferencePipelineConfig:
    """FastVideo CausalWan2.2 distilled T2V config (Wan VAE decoder)."""
    return WanInferencePipelineConfig(
        enable_sync_and_profile=enable_sync_and_profile,
        encoder=None,
        decoder=_wan_vae_decoder_config(),
        diffusion_model=DiffusionModelConfig(
            seed=seed,
            transformer=_transformer_config(
                checkpoint_path=AVAILABLE_CAUSAL_WAN22_CHECKPOINT_PATHS["fastvideo"],
                compile_network=compile_network,
            ),
            scheduler=_scheduler_config(),
        ),
    )


CAUSAL_WAN22_CONFIG_BUILDERS: dict[str, Callable[..., WanInferencePipelineConfig]] = {
    "fastvideo": build_fastvideo,
}
