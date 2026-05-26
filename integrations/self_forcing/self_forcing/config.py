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

"""Configs for the Self-Forcing distilled model."""

from __future__ import annotations

from typing import Any, cast

from torch import Tensor

from flashdreams.infra.config import derive_config
from flashdreams.infra.diffusion.model import DiffusionModelConfig
from flashdreams.infra.diffusion.scheduler.fm import FlowMatchSchedulerConfig
from flashdreams.infra.runner import RunnerConfig
from flashdreams.recipes.taehv import TeahvVAEDecoderConfig
from flashdreams.recipes.wan import (
    Wan21TransformerConfig,
    WanDiTNetwork1pt3BConfig,
    WanInferencePipelineConfig,
    WanVAEDecoderConfig,
)
from self_forcing.runner import SelfForcingT2VRunnerConfig

CHECKPOINT_PATH = "https://huggingface.co/gdhe17/Self-Forcing/blob/main/checkpoints/self_forcing_dmd.pt"


def state_dict_transform(state_dict: dict[str, Any]) -> dict[str, Tensor]:
    """Strip Self-Forcing wrapper prefixes from the checkpoint state-dict."""
    if "generator_ema" in state_dict:
        state_dict = state_dict["generator_ema"]
    elif "generator" in state_dict:
        state_dict = state_dict["generator"]

    out: dict[str, Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_k = k[len("model.") :]
        elif k.startswith("net."):
            new_k = k[len("net.") :]
        else:
            new_k = k
        if new_k.startswith("_fsdp_wrapped_module."):
            new_k = new_k[len("_fsdp_wrapped_module.") :]
        out[new_k] = v
    return out


# Official Self-Forcing Wan 2.1 1.3B T2V pipeline config.
PIPELINE_WAN21_T2V_1PT3B = WanInferencePipelineConfig(
    name="self-forcing-wan2.1-t2v-1.3b",
    # Warning: This will slow down the e2e latency.
    enable_sync_and_profile=True,
    encoder=None,
    decoder=WanVAEDecoderConfig(),
    diffusion_model=DiffusionModelConfig(
        seed=42,
        transformer=Wan21TransformerConfig(
            network=WanDiTNetwork1pt3BConfig(
                patch_embedding_type="conv3d",
                cp_method="ring",
            ),
            checkpoint_path=CHECKPOINT_PATH,
            state_dict_transform=state_dict_transform,
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
RUNNER_WAN21_T2V_1PT3B = SelfForcingT2VRunnerConfig(
    runner_name=PIPELINE_WAN21_T2V_1PT3B.name,
    description="Self-Forcing distilled Wan 2.1 1.3B T2V (Wan VAE decoder, 4-step).",
    pipeline=PIPELINE_WAN21_T2V_1PT3B,
)

# Faster version with changes:
# - Use faster VAE decoder.
PIPELINE_WAN21_T2V_1PT3B_FLASH = cast(
    WanInferencePipelineConfig,
    derive_config(
        PIPELINE_WAN21_T2V_1PT3B,
        name="self-forcing-wan2.1-t2v-1.3b-flash",
        decoder=TeahvVAEDecoderConfig(),
    ),
)  # ty:ignore[redundant-cast]
RUNNER_WAN21_T2V_1PT3B_FLASH = SelfForcingT2VRunnerConfig(
    runner_name=PIPELINE_WAN21_T2V_1PT3B_FLASH.name,
    description="Self-Forcing distilled Wan 2.1 1.3B T2V (TAEHV decoder, 4-step).",
    pipeline=PIPELINE_WAN21_T2V_1PT3B_FLASH,
)

# Anti-drift Wan preset: static sink=5, rolling window=7
# (recent=4 + current=3), with KVCache-relative RoPE.
PIPELINE_WAN21_T2V_1PT3B_ANTI_DRIFT = cast(
    WanInferencePipelineConfig,
    derive_config(
        PIPELINE_WAN21_T2V_1PT3B,
        name="self-forcing-wan2.1-t2v-1.3b-anti-drift",
        diffusion_model=dict(
            seed=0,
            transformer=dict(
                window_size_t=7,
                sink_size_t=5,
                compile_network=False,
                use_cuda_graph=False,
                network=dict(
                    apply_rope_before_kvcache=False,
                ),
            ),
        ),
    ),
)  # ty:ignore[redundant-cast]
RUNNER_WAN21_T2V_1PT3B_ANTI_DRIFT = SelfForcingT2VRunnerConfig(
    runner_name=PIPELINE_WAN21_T2V_1PT3B_ANTI_DRIFT.name,
    description=(
        "Self-Forcing distilled Wan 2.1 1.3B T2V "
        "(KVCache-relative RoPE, static sink/window, 4-step)."
    ),
    pipeline=PIPELINE_WAN21_T2V_1PT3B_ANTI_DRIFT,
    total_blocks=80,
)

RUNNER_CONFIGS: dict[str, RunnerConfig] = {
    cfg.runner_name: cfg
    for cfg in (
        RUNNER_WAN21_T2V_1PT3B,
        RUNNER_WAN21_T2V_1PT3B_FLASH,
        RUNNER_WAN21_T2V_1PT3B_ANTI_DRIFT,
    )
}
