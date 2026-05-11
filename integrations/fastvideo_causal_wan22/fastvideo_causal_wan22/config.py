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

"""Configs for the FastVideo CausalWan 2.2 distilled model."""

from __future__ import annotations

import torch

from fastvideo_causal_wan22.runner import FastvideoCausalWan22T2VRunnerConfig
from flashdreams.core.checkpoint.remap import remap_checkpoint_keys
from flashdreams.infra.diffusion.model import DiffusionModelConfig
from flashdreams.infra.diffusion.scheduler.fm import FlowMatchSchedulerConfig
from flashdreams.infra.runner import RunnerConfig
from flashdreams.recipes.wan import (
    Wan21TransformerConfig,
    Wan22TransformerConfig,
    WanDiTNetwork14BConfig,
    WanInferencePipelineConfig,
    WanVAEDecoderConfig,
)

CHECKPOINT_PATH_HIGH_NOISE = "https://huggingface.co/FastVideo/CausalWan2.2-I2V-A14B-Preview-Diffusers/blob/main/transformer/diffusion_pytorch_model.safetensors"
CHECKPOINT_PATH_LOW_NOISE = "https://huggingface.co/FastVideo/CausalWan2.2-I2V-A14B-Preview-Diffusers/blob/main/transformer_2/diffusion_pytorch_model.safetensors"

## HF diffusers → bare WanDiTNetwork key remap

# Wan 2.2 ships in the HF diffusers layout, which differs from the bare
# WanDiTNetwork.state_dict() keys. This is the same mapping the legacy
# projects.causal_wan2_2.dit.model.WanDiT used.
CHECKPOINT_KEY_MAPPING: dict[str, str] = {
    # Global embedding/head remaps
    r"^condition_embedder\.text_embedder\.linear_1\.(.*)$": r"text_embedding.0.\1",
    r"^condition_embedder\.text_embedder\.linear_2\.(.*)$": r"text_embedding.2.\1",
    r"^condition_embedder\.time_embedder\.linear_1\.(.*)$": r"time_embedding.0.\1",
    r"^condition_embedder\.time_embedder\.linear_2\.(.*)$": r"time_embedding.2.\1",
    r"^condition_embedder\.time_proj\.(.*)$": r"time_projection.1.\1",
    r"^scale_shift_table$": r"head.modulation",
    r"^proj_out\.(.*)$": r"head.head.\1",
    # Block attention projections
    r"^blocks\.(\d+)\.attn1\.to_q\.(.*)$": r"blocks.\1.self_attn.q.\2",
    r"^blocks\.(\d+)\.attn1\.to_k\.(.*)$": r"blocks.\1.self_attn.k.\2",
    r"^blocks\.(\d+)\.attn1\.to_v\.(.*)$": r"blocks.\1.self_attn.v.\2",
    r"^blocks\.(\d+)\.attn1\.to_out\.0\.(.*)$": r"blocks.\1.self_attn.o.\2",
    r"^blocks\.(\d+)\.attn2\.to_q\.(.*)$": r"blocks.\1.cross_attn.q.\2",
    r"^blocks\.(\d+)\.attn2\.to_k\.(.*)$": r"blocks.\1.cross_attn.k.\2",
    r"^blocks\.(\d+)\.attn2\.to_v\.(.*)$": r"blocks.\1.cross_attn.v.\2",
    r"^blocks\.(\d+)\.attn2\.to_out\.0\.(.*)$": r"blocks.\1.cross_attn.o.\2",
    # Block norm/modulation remaps
    r"^blocks\.(\d+)\.attn1\.norm_q\.(.*)$": r"blocks.\1.self_attn.norm_q.\2",
    r"^blocks\.(\d+)\.attn1\.norm_k\.(.*)$": r"blocks.\1.self_attn.norm_k.\2",
    r"^blocks\.(\d+)\.attn2\.norm_q\.(.*)$": r"blocks.\1.cross_attn.norm_q.\2",
    r"^blocks\.(\d+)\.attn2\.norm_k\.(.*)$": r"blocks.\1.cross_attn.norm_k.\2",
    r"^blocks\.(\d+)\.norm2\.(.*)$": r"blocks.\1.norm3.\2",
    r"^blocks\.(\d+)\.scale_shift_table$": r"blocks.\1.modulation",
    # Block FFN remaps
    r"^blocks\.(\d+)\.ffn\.fc_in\.(.*)$": r"blocks.\1.ffn.0.\2",
    r"^blocks\.(\d+)\.ffn\.fc_out\.(.*)$": r"blocks.\1.ffn.2.\2",
    r"^blocks\.(\d+)\.ffn\.net\.0\.proj\.(.*)$": r"blocks.\1.ffn.0.\2",
    r"^blocks\.(\d+)\.ffn\.net\.2\.(.*)$": r"blocks.\1.ffn.2.\2",
}


def state_dict_transform(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remap an HF diffusers Wan 2.2 state-dict to the WanDiTNetwork layout."""
    return remap_checkpoint_keys(state_dict, CHECKPOINT_KEY_MAPPING)


def _wan22_branch(checkpoint_path: str) -> Wan21TransformerConfig:
    """Build one of the two Wan 2.2 MoE branches (high-noise / low-noise).

    Both branches share every Wan 2.1 14B knob; only the checkpoint
    differs. Kept as a tiny helper so the literal below stays
    readable -- inlining would duplicate ~12 lines per branch.
    """
    return Wan21TransformerConfig(
        network=WanDiTNetwork14BConfig(
            patch_embedding_type="conv3d",
        ),
        checkpoint_path=checkpoint_path,
        state_dict_transform=state_dict_transform,
        batch_shape=(),
        len_t=3,
        guidance_scale=1.0,
        window_size_t=21,
        sink_size_t=0,
        compile_network=True,
    )


# Official FastVideo CausalWan 2.2 14B MoE T2V pipeline config.
PIPELINE_WAN22_T2V_14B = WanInferencePipelineConfig(
    recipe_name="fastvideo-causal-wan2.2-t2v-14b",
    # Warning: This will slow down the e2e latency.
    enable_sync_and_profile=True,
    encoder=None,
    decoder=WanVAEDecoderConfig(),
    diffusion_model=DiffusionModelConfig(
        seed=42,
        transformer=Wan22TransformerConfig(
            transformer_high_noise=_wan22_branch(CHECKPOINT_PATH_HIGH_NOISE),
            transformer_low_noise=_wan22_branch(CHECKPOINT_PATH_LOW_NOISE),
            # ``high_noise`` runs above the boundary
            # (``timestep / num_train_timesteps >= boundary_ratio``);
            # ``low_noise`` runs below.
            boundary_ratio=0.875,
            num_train_timesteps=1000,
        ),
        scheduler=FlowMatchSchedulerConfig(
            num_inference_steps=8,
            denoising_timesteps=[1000, 850, 700, 550, 350, 275, 200, 125],
            warp_denoising_step=True,
            shift=5.0,
            sigma_min=0.0,
            extra_one_step=True,
            num_train_timesteps=1000,
        ),
    ),
)
RUNNER_WAN22_T2V_14B = FastvideoCausalWan22T2VRunnerConfig(
    runner_name=PIPELINE_WAN22_T2V_14B.recipe_name,
    description="FastVideo CausalWan 2.2 14B MoE T2V (Wan VAE decoder, 8-step).",
    pipeline=PIPELINE_WAN22_T2V_14B,
)

RUNNER_CONFIGS: dict[str, RunnerConfig] = {
    cfg.runner_name: cfg for cfg in (RUNNER_WAN22_T2V_14B,)
}
