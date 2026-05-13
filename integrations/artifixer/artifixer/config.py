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

The network is :class:`ArtifixerDiTNetwork` whose blocks carry the
opacity + camera-ray MLPs (zero-initialized so behavior matches vanilla
Wan when ``opacity_extra`` / ``camera_extra`` are not provided).

By default the recipe loads the merged ArtiFixer DMD safetensors via
:func:`artifixer_dmd_state_dict_transform`; set
``ARTIFIXER_USE_BASE_WAN_WEIGHTS=1`` to fall back to vanilla Wan 2.1
1.3B base weights, in which case :func:`zero_pad_artifixer_keys` pads
the state dict (opacity / camera / neighbor cross-attn keys) so the
strict-mode load succeeds.
"""

from __future__ import annotations

import os

import torch
from artifixer.checkpoint import (
    artifixer_dmd_state_dict_transform,
    zero_pad_artifixer_keys,
)
from artifixer.network.dit import ArtifixerDiTNetwork1pt3BConfig
from artifixer.pipeline import ArtifixerInferencePipelineConfig
from artifixer.runner import ArtifixerDmdT2VRunnerConfig
from artifixer.transformer import ArtifixerWanTransformerConfig

from flashdreams.infra.diffusion.model import DiffusionModelConfig
from flashdreams.infra.diffusion.scheduler.fm import FlowMatchSchedulerConfig
from flashdreams.infra.runner import RunnerConfig
from flashdreams.recipes.wan import WanVAEDecoderConfig

# Mirrors the ArtiFixer DMD stage-3 1.3B training config:
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

# Default: load the merged ArtiFixer DMD safetensors (a consolidated
# single-file checkpoint built from the sharded FSDP training output).
# There is no committed default location, since the merged file is large
# and lives outside the repo; users MUST set
# ``ARTIFIXER_DMD_CHECKPOINT_PATH`` to the merged safetensors path, or
# set ``ARTIFIXER_USE_BASE_WAN_WEIGHTS=1`` to fall back to vanilla Wan
# 2.1 1.3B HuggingFace weights.
ARTIFIXER_DMD_CHECKPOINT_PATH: str | None = os.environ.get(
    "ARTIFIXER_DMD_CHECKPOINT_PATH"
)

# Fallback: vanilla Wan 2.1 1.3B base weights from HF, paired with
# ``zero_pad_artifixer_keys`` so ``load_state_dict`` succeeds in strict
# mode (the ArtiFixer extension paths sit at zero until trained weights
# are loaded). Useful for smoke-testing the recipe wiring when the
# merged safetensors are unavailable; flip ``ARTIFIXER_USE_BASE_WAN_WEIGHTS=1``.
BASE_WAN_T2V_1PT3B_CHECKPOINT_PATH = (
    "https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/blob/main/diffusion_pytorch_model.safetensors"
)


_BASE_NETWORK_CONFIG = ArtifixerDiTNetwork1pt3BConfig(
    patch_embedding_type="conv3d",
)
_BASE_TRANSFORMER_DTYPE = torch.bfloat16

_USE_BASE_WAN_WEIGHTS = os.environ.get("ARTIFIXER_USE_BASE_WAN_WEIGHTS") == "1"
if _USE_BASE_WAN_WEIGHTS:
    _CHECKPOINT_PATH: str = BASE_WAN_T2V_1PT3B_CHECKPOINT_PATH
    _STATE_DICT_TRANSFORM = zero_pad_artifixer_keys(
        num_layers=_BASE_NETWORK_CONFIG.num_layers,
        dim=_BASE_NETWORK_CONFIG.dim,
        patch_size=_BASE_NETWORK_CONFIG.patch_size,
        dtype=_BASE_TRANSFORMER_DTYPE,
    )
else:
    if ARTIFIXER_DMD_CHECKPOINT_PATH is None:
        raise RuntimeError(
            "ArtiFixer recipe requires a checkpoint path. Set "
            "``ARTIFIXER_DMD_CHECKPOINT_PATH`` to the merged DMD "
            "safetensors file, or set "
            "``ARTIFIXER_USE_BASE_WAN_WEIGHTS=1`` to fall back to "
            "vanilla Wan 2.1 1.3B base weights."
        )
    _CHECKPOINT_PATH = ARTIFIXER_DMD_CHECKPOINT_PATH
    _STATE_DICT_TRANSFORM = artifixer_dmd_state_dict_transform

PIPELINE_ARTIFIXER_DMD_T2V_1PT3B = ArtifixerInferencePipelineConfig(
    recipe_name="artifixer-dmd-wan2.1-t2v-1.3b",
    # Profiling is off for the initial bring-up. The base
    # ``StreamInferencePipeline.generate`` is what installs
    # ``cache.event_profiler`` when this is True, but our custom
    # ``ArtifixerInferencePipeline.generate`` bypasses that path. Re-enable
    # only alongside the corresponding ``EventProfiler`` setup in
    # ``ArtifixerInferencePipeline.generate``.
    enable_sync_and_profile=False,
    encoder=None,
    decoder=WanVAEDecoderConfig(),
    diffusion_model=DiffusionModelConfig(
        seed=42,
        transformer=ArtifixerWanTransformerConfig(
            network=_BASE_NETWORK_CONFIG,
            dtype=_BASE_TRANSFORMER_DTYPE,
            checkpoint_path=_CHECKPOINT_PATH,
            state_dict_transform=_STATE_DICT_TRANSFORM,
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
        "extensions, 4-step DMD)."
    ),
    pipeline=PIPELINE_ARTIFIXER_DMD_T2V_1PT3B,
)


RUNNER_CONFIGS: dict[str, RunnerConfig] = {
    cfg.runner_name: cfg
    for cfg in (RUNNER_ARTIFIXER_DMD_T2V_1PT3B,)
}
