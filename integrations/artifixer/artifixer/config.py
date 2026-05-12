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

"""Configs for the ArtiFixer DMD-distilled inference recipe.

ArtiFixer is a reconstruction-enhanced T2V model built on Wan 2.1 1.3B that
adds (a) per-block opacity and Plucker-camera-ray MLPs, (b) neighbor cross-
attention with PRoPE, and (c) opacity-weighted latent mixing.

Phase 2.1: the network now uses :class:`ArtifixerDiTNetwork` whose blocks
carry the opacity + camera-ray MLPs (zero-initialized so behavior matches
vanilla Wan when ``opacity_extra`` / ``camera_extra`` are not provided).
The recipe still loads vanilla Wan 2.1 1.3B base weights from HuggingFace;
:func:`zero_pad_artifixer_keys` pads the state dict so strict-mode load
succeeds.

Later commits add the neighbor cross-attention third KV bank (Phase 2.2),
PRoPE (Phase 2.3), the opacity-weighted latent mixing pipeline (Phase 3),
and the ``state_dict_transform`` for the merged ArtiFixer DMD safetensors
(Phase 5).
"""

from __future__ import annotations

import torch
from artifixer.checkpoint import zero_pad_artifixer_keys
from artifixer.network.dit import ArtifixerDiTNetwork1pt3BConfig
from artifixer.runner import ArtifixerDmdT2VRunnerConfig
from artifixer.transformer import ArtifixerWanTransformerConfig

from flashdreams.infra.diffusion.model import DiffusionModelConfig
from flashdreams.infra.diffusion.scheduler.fm import FlowMatchSchedulerConfig
from flashdreams.infra.runner import RunnerConfig
from flashdreams.recipes.wan import WanInferencePipelineConfig, WanVAEDecoderConfig

# Mirrors the dreamfix stage-3 run config (see
# wandb/.../run-*/files/config.yaml in
# artifixer-runs/artifixer-s3-dmd-1p3b-from-s2-10000-s1-15000-128g):
#   model_id = Wan-AI/Wan2.1-T2V-1.3B-Diffusers
#   frames_per_block = 7           -> len_t
#   local_attn_size = 21           -> window_size_t
#   sink_size = 7                  -> sink_size_t
#   num_inference_steps = 4
#   timestep_shift = 5             -> FlowMatchSchedulerConfig.shift
ARTIFIXER_LEN_T = 7
ARTIFIXER_WINDOW_SIZE_T = 21
ARTIFIXER_SINK_SIZE_T = 7
ARTIFIXER_NUM_INFERENCE_STEPS = 4
ARTIFIXER_TIMESTEP_SHIFT = 5.0

# Phase 1: load vanilla Wan 2.1 1.3B base weights from HF. Phase 5 replaces
# this with the merged ArtiFixer DMD safetensors plus a state_dict_transform
# that remaps diffusers naming and absorbs the ArtiFixer-only keys.
BASE_WAN_T2V_1PT3B_CHECKPOINT_PATH = (
    "https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/blob/main/diffusion_pytorch_model.safetensors"
)


_BASE_NETWORK_CONFIG = ArtifixerDiTNetwork1pt3BConfig(
    patch_embedding_type="conv3d",
)
_BASE_TRANSFORMER_DTYPE = torch.bfloat16

PIPELINE_ARTIFIXER_DMD_T2V_1PT3B = WanInferencePipelineConfig(
    recipe_name="artifixer-dmd-wan2.1-t2v-1.3b",
    enable_sync_and_profile=True,
    encoder=None,
    decoder=WanVAEDecoderConfig(),
    diffusion_model=DiffusionModelConfig(
        seed=42,
        transformer=ArtifixerWanTransformerConfig(
            network=_BASE_NETWORK_CONFIG,
            dtype=_BASE_TRANSFORMER_DTYPE,
            checkpoint_path=BASE_WAN_T2V_1PT3B_CHECKPOINT_PATH,
            state_dict_transform=zero_pad_artifixer_keys(
                num_layers=_BASE_NETWORK_CONFIG.num_layers,
                dim=_BASE_NETWORK_CONFIG.dim,
                patch_size=_BASE_NETWORK_CONFIG.patch_size,
                dtype=_BASE_TRANSFORMER_DTYPE,
            ),
            batch_shape=(),
            len_t=ARTIFIXER_LEN_T,
            guidance_scale=1.0,
            window_size_t=ARTIFIXER_WINDOW_SIZE_T,
            sink_size_t=ARTIFIXER_SINK_SIZE_T,
            stamp_image_latent=False,
            compile_network=True,
        ),
        scheduler=FlowMatchSchedulerConfig(
            num_inference_steps=ARTIFIXER_NUM_INFERENCE_STEPS,
            denoising_timesteps=[1000, 750, 500, 250],
            warp_denoising_step=True,
            shift=ARTIFIXER_TIMESTEP_SHIFT,
            sigma_min=0.0,
            extra_one_step=True,
            num_train_timesteps=1000,
        ),
    ),
)
RUNNER_ARTIFIXER_DMD_T2V_1PT3B = ArtifixerDmdT2VRunnerConfig(
    runner_name=PIPELINE_ARTIFIXER_DMD_T2V_1PT3B.recipe_name,
    description=(
        "ArtiFixer reconstruction-enhanced T2V (Wan 2.1 1.3B + opacity/camera/neighbor "
        "extensions, 4-step DMD). Phase 1 scaffold: loads vanilla Wan base weights."
    ),
    pipeline=PIPELINE_ARTIFIXER_DMD_T2V_1PT3B,
)


RUNNER_CONFIGS: dict[str, RunnerConfig] = {
    cfg.runner_name: cfg
    for cfg in (RUNNER_ARTIFIXER_DMD_T2V_1PT3B,)
}
