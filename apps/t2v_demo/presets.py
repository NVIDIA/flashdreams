# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""App-owned T2V pipeline presets, independent of workspace integrations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import Tensor

from flashdreams.infra.config import derive_config
from flashdreams.infra.diffusion.model import DiffusionModelConfig
from flashdreams.infra.diffusion.scheduler import FlowMatchUniPCSchedulerConfig
from flashdreams.infra.diffusion.scheduler.fm import FlowMatchSchedulerConfig
from flashdreams.recipes.cosmos.pipeline import CosmosInferencePipelineConfig
from flashdreams.recipes.cosmos.transformer import CosmosTransformerConfig
from flashdreams.recipes.cosmos.transformer.impl.network import (
    CosmosDiTNetworkConfig,
)
from flashdreams.recipes.cosmos.transformer.impl.network import (
    state_dict_transform as cosmos_state_dict_transform,
)
from flashdreams.recipes.taehv import TeahvVAEDecoderConfig
from flashdreams.recipes.wan import (
    Wan21TransformerConfig,
    WanDiTNetwork1pt3BConfig,
    WanInferencePipelineConfig,
    WanVAEDecoderConfig,
)


@dataclass(frozen=True, slots=True)
class T2VPreset:
    """One app-supported text-to-video pipeline configuration."""

    name: str
    pipeline: Any
    prompt: str
    total_blocks: int
    pixel_height: int
    pixel_width: int
    fps: int


CAUSAL_FORCING_PROMPT = "A cinematic closeup of a reindeer in a snowy forest at sunset."
SELF_FORCING_PROMPT = "A stylish woman strolls down a neon-lit Tokyo street at night."
COSMOS_PROMPT = (
    "A high-definition video of a robotic arm welding in an industrial workshop."
)


def _wan_state_dict_transform(state_dict: dict[str, Any]) -> dict[str, Tensor]:
    """Normalize upstream Causal/Self-Forcing checkpoint wrapper keys."""
    state_dict = state_dict.get(
        "generator_ema", state_dict.get("generator", state_dict)
    )
    out: dict[str, Tensor] = {}
    for key, value in state_dict.items():
        key = key.removeprefix("model.").removeprefix("net.")
        out[key.removeprefix("_fsdp_wrapped_module.")] = value
    return out


CAUSAL_FORCING_PIPELINE = WanInferencePipelineConfig(
    name="causal-forcing-wan2.1-t2v-1.3b-chunkwise",
    enable_sync_and_profile=True,
    encoder=None,
    decoder=WanVAEDecoderConfig(),
    diffusion_model=DiffusionModelConfig(
        seed=42,
        transformer=Wan21TransformerConfig(
            network=WanDiTNetwork1pt3BConfig(
                patch_embedding_type="conv3d", cp_method="ring"
            ),
            checkpoint_path="https://huggingface.co/zhuhz22/Causal-Forcing/blob/main/chunkwise/causal_forcing.pt",
            state_dict_transform=_wan_state_dict_transform,
            batch_shape=(),
            len_t=3,
            guidance_scale=1.0,
            window_size_t=21,
            sink_size_t=0,
            stamp_image_latent=False,
            compile_network=True,
        ),
        scheduler=FlowMatchSchedulerConfig(
            num_inference_steps=4,
            denoising_timesteps=[1000, 750, 500, 250],
            warp_denoising_step=True,
            shift=5.0,
            sigma_min=0.0,
            extra_one_step=True,
            num_train_timesteps=1000,
        ),
    ),
)
CAUSAL_FORCING_FRAMEWISE_PIPELINE = derive_config(
    CAUSAL_FORCING_PIPELINE,
    name="causal-forcing-wan2.1-t2v-1.3b-framewise",
    diffusion_model=dict(
        transformer=dict(
            checkpoint_path="https://huggingface.co/zhuhz22/Causal-Forcing/blob/main/framewise/causal_forcing.pt",
            len_t=1,
        ),
    ),
)

SELF_FORCING_PIPELINE = WanInferencePipelineConfig(
    name="self-forcing-wan2.1-t2v-1.3b",
    enable_sync_and_profile=True,
    encoder=None,
    decoder=WanVAEDecoderConfig(),
    diffusion_model=DiffusionModelConfig(
        seed=42,
        transformer=Wan21TransformerConfig(
            network=WanDiTNetwork1pt3BConfig(
                patch_embedding_type="conv3d", cp_method="ring"
            ),
            checkpoint_path="https://huggingface.co/gdhe17/Self-Forcing/blob/main/checkpoints/self_forcing_dmd.pt",
            state_dict_transform=_wan_state_dict_transform,
            batch_shape=(),
            len_t=3,
            guidance_scale=1.0,
            window_size_t=21,
            sink_size_t=0,
            stamp_image_latent=False,
            compile_network=True,
        ),
        scheduler=FlowMatchSchedulerConfig(
            num_inference_steps=4,
            denoising_timesteps=[1000, 750, 500, 250],
            warp_denoising_step=True,
            shift=8.0,
            sigma_min=0.0,
            extra_one_step=True,
            num_train_timesteps=1000,
        ),
    ),
)
SELF_FORCING_TAEHV_PIPELINE = derive_config(
    SELF_FORCING_PIPELINE,
    name="self-forcing-wan2.1-t2v-1.3b-taehv",
    decoder=TeahvVAEDecoderConfig(),
)
SELF_FORCING_REROPE_PIPELINE = derive_config(
    SELF_FORCING_PIPELINE,
    name="self-forcing-wan2.1-t2v-1.3b-sink5-window7-rerope",
    diffusion_model=dict(
        seed=0,
        transformer=dict(
            window_size_t=7,
            sink_size_t=5,
            compile_network=False,
            use_cuda_graph=False,
            network=dict(apply_rope_before_kvcache=False),
        ),
    ),
)

COSMOS_PIPELINE = CosmosInferencePipelineConfig(
    name="cosmos2-t2v-2b-720p",
    enable_sync_and_profile=True,
    encoder=None,
    decoder=WanVAEDecoderConfig(),
    diffusion_model=DiffusionModelConfig(
        seed=42,
        transformer=CosmosTransformerConfig(
            network=CosmosDiTNetworkConfig(cp_method="ring"),
            checkpoint_path="https://huggingface.co/nvidia/Cosmos-Predict2.5-2B/blob/main/base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt",
            state_dict_transform=cosmos_state_dict_transform,
            batch_shape=(),
            len_t=24,
            window_size_t=24,
            guidance_scale=8.0,
            compile_network=True,
            use_cuda_graph=False,
        ),
        scheduler=FlowMatchUniPCSchedulerConfig(
            num_inference_steps=35, shift=5.0, use_kerras_sigma=True, enable_tqdm=True
        ),
    ),
)

PRESETS: dict[str, T2VPreset] = {
    preset.name: preset
    for preset in (
        T2VPreset(
            name=CAUSAL_FORCING_PIPELINE.name,
            pipeline=CAUSAL_FORCING_PIPELINE,
            prompt=CAUSAL_FORCING_PROMPT,
            total_blocks=60,
            pixel_height=480,
            pixel_width=832,
            fps=16,
        ),
        T2VPreset(
            name=CAUSAL_FORCING_FRAMEWISE_PIPELINE.name,
            pipeline=CAUSAL_FORCING_FRAMEWISE_PIPELINE,
            prompt=CAUSAL_FORCING_PROMPT,
            total_blocks=60,
            pixel_height=480,
            pixel_width=832,
            fps=16,
        ),
        T2VPreset(
            name=SELF_FORCING_PIPELINE.name,
            pipeline=SELF_FORCING_PIPELINE,
            prompt=SELF_FORCING_PROMPT,
            total_blocks=60,
            pixel_height=480,
            pixel_width=832,
            fps=16,
        ),
        T2VPreset(
            name=SELF_FORCING_TAEHV_PIPELINE.name,
            pipeline=SELF_FORCING_TAEHV_PIPELINE,
            prompt=SELF_FORCING_PROMPT,
            total_blocks=60,
            pixel_height=480,
            pixel_width=832,
            fps=16,
        ),
        T2VPreset(
            name=SELF_FORCING_REROPE_PIPELINE.name,
            pipeline=SELF_FORCING_REROPE_PIPELINE,
            prompt=SELF_FORCING_PROMPT,
            total_blocks=80,
            pixel_height=480,
            pixel_width=832,
            fps=16,
        ),
        T2VPreset(
            name=COSMOS_PIPELINE.name,
            pipeline=COSMOS_PIPELINE,
            prompt=COSMOS_PROMPT,
            total_blocks=1,
            pixel_height=720,
            pixel_width=1280,
            fps=16,
        ),
    )
}
