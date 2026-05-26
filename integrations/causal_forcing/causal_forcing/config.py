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

"""Configs for the Causal-Forcing distilled model (chunkwise + framewise)."""

from __future__ import annotations

from typing import Any, cast

from torch import Tensor

from causal_forcing.runner import (
    CausalForcingI2VRunnerConfig,
    CausalForcingT2VRunnerConfig,
)
from flashdreams.infra.config import derive_config
from flashdreams.infra.diffusion.model import DiffusionModelConfig
from flashdreams.infra.diffusion.scheduler.fm import FlowMatchSchedulerConfig
from flashdreams.infra.runner import RunnerConfig
from flashdreams.recipes.wan import (
    Wan21TransformerConfig,
    WanDiTNetwork1pt3BConfig,
    WanI2VCtrlEncoderConfig,
    WanInferencePipelineConfig,
    WanVAEDecoderConfig,
    WanVAEEncoderConfig,
)

CHECKPOINT_PATH_CHUNKWISE = "https://huggingface.co/zhuhz22/Causal-Forcing/blob/main/chunkwise/causal_forcing.pt"
CHECKPOINT_PATH_FRAMEWISE = "https://huggingface.co/zhuhz22/Causal-Forcing/blob/main/framewise/causal_forcing.pt"


def state_dict_transform(state_dict: dict[str, Any]) -> dict[str, Tensor]:
    """Strip Causal-Forcing wrapper prefixes from the checkpoint state-dict.

    Drops the ``generator_ema`` / ``generator`` container, the ``model.``
    / ``net.`` outer prefix, and the ``_fsdp_wrapped_module.`` inner
    prefix (framewise variant) so keys match a bare ``WanDiTNetwork``.
    """
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


# Causal-Forcing chunkwise Wan 2.1 1.3B T2V pipeline.
PIPELINE_WAN21_T2V_1PT3B_CHUNKWISE = WanInferencePipelineConfig(
    name="causal-forcing-wan2.1-t2v-1.3b-chunkwise",
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
            checkpoint_path=CHECKPOINT_PATH_CHUNKWISE,
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
            shift=5.0,
            sigma_min=0.0,
            extra_one_step=True,
            num_train_timesteps=1000,
        ),
    ),
)
RUNNER_WAN21_T2V_1PT3B_CHUNKWISE = CausalForcingT2VRunnerConfig(
    runner_name=PIPELINE_WAN21_T2V_1PT3B_CHUNKWISE.name,
    description="Causal-Forcing chunkwise Wan 2.1 1.3B T2V (Wan VAE decoder, 4-step).",
    pipeline=PIPELINE_WAN21_T2V_1PT3B_CHUNKWISE,
)

# Framewise variant: one latent frame per AR chunk.
PIPELINE_WAN21_T2V_1PT3B_FRAMEWISE = cast(
    WanInferencePipelineConfig,
    derive_config(
        PIPELINE_WAN21_T2V_1PT3B_CHUNKWISE,
        name="causal-forcing-wan2.1-t2v-1.3b-framewise",
        diffusion_model=dict(
            transformer=dict(
                checkpoint_path=CHECKPOINT_PATH_FRAMEWISE,
                len_t=1,
            ),
        ),
    ),
)  # ty:ignore[redundant-cast]
RUNNER_WAN21_T2V_1PT3B_FRAMEWISE = CausalForcingT2VRunnerConfig(
    runner_name=PIPELINE_WAN21_T2V_1PT3B_FRAMEWISE.name,
    description="Causal-Forcing framewise Wan 2.1 1.3B T2V (len_t=1, Wan VAE).",
    pipeline=PIPELINE_WAN21_T2V_1PT3B_FRAMEWISE,
)


# I2V variant: framewise T2V model can naturally support I2V, by stamping
# the image latent into the KV cache of the first rollout (stamp_image_latent=True)
PIPELINE_WAN21_I2V_1PT3B_FRAMEWISE = cast(
    WanInferencePipelineConfig,
    derive_config(
        PIPELINE_WAN21_T2V_1PT3B_FRAMEWISE,
        name="causal-forcing-wan2.1-i2v-1.3b-framewise",
        encoder=WanI2VCtrlEncoderConfig(
            encoder=WanVAEEncoderConfig(),
        ),
        diffusion_model=dict(
            transformer=dict(stamp_image_latent=True),
        ),
    ),
)  # ty:ignore[redundant-cast]
RUNNER_WAN21_I2V_1PT3B_FRAMEWISE = CausalForcingI2VRunnerConfig(
    runner_name=PIPELINE_WAN21_I2V_1PT3B_FRAMEWISE.name,
    description="Causal-Forcing framewise Wan 2.1 1.3B I2V (len_t=1, Wan VAE).",
    pipeline=PIPELINE_WAN21_I2V_1PT3B_FRAMEWISE,
)


RUNNER_CONFIGS: dict[str, RunnerConfig] = {
    cfg.runner_name: cfg
    for cfg in (
        RUNNER_WAN21_T2V_1PT3B_CHUNKWISE,
        RUNNER_WAN21_T2V_1PT3B_FRAMEWISE,
        RUNNER_WAN21_I2V_1PT3B_FRAMEWISE,
    )
}
