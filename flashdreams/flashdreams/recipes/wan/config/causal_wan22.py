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

"""User-facing configs for streaming Wan 2.2.

Hosts both the pre-built :class:`WanInferencePipelineConfig` literals
and the per-slug :class:`CausalWan22RunnerConfig` literals that drive
``flashdreams-run``. Wan 2.2 currently ships only the FastVideo
distilled T2V preset; the dual 14B MoE backbone is expressed as two
``Wan21TransformerConfig`` branches inside
:class:`Wan22TransformerConfig`. The runner-config literal
self-registers with :mod:`flashdreams.configs.registry` at import
time.
"""

from __future__ import annotations

import torch

from flashdreams.configs.registry import register_runner
from flashdreams.core.checkpoint.remap import remap_checkpoint_keys
from flashdreams.infra.diffusion.model import DiffusionModelConfig
from flashdreams.infra.diffusion.scheduler.fm import FlowMatchSchedulerConfig
from flashdreams.infra.runner import RunnerConfig
from flashdreams.recipes.wan.autoencoder.vae import (
    AVAILABLE_WAN_VAE_CHECKPOINT_PATHS,
    WanVAEDecoderConfig,
)
from flashdreams.recipes.wan.pipeline import WanInferencePipelineConfig
from flashdreams.recipes.wan.runner_causal_wan22 import CausalWan22RunnerConfig
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

# Canonical pixel-space defaults; callers pass the matching latent
# (height, width) into :meth:`WanInferencePipeline.initialize_cache`.
DEFAULT_VIDEO_HEIGHT = 480
DEFAULT_VIDEO_WIDTH = 832


def _remap_diffusers_state_dict(
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
        state_dict_transform=_remap_diffusers_state_dict,
        batch_shape=(1,),
        len_t=3,
        guidance_scale=1.0,
        window_size_t=21,
        sink_size_t=0,
        compile_network=True,
    )


FASTVIDEO_T2V = WanInferencePipelineConfig(
    recipe_name="causal-wan22-fastvideo-t2v",
    enable_sync_and_profile=True,
    encoder=None,
    decoder=WanVAEDecoderConfig(
        checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"],
    ),
    diffusion_model=DiffusionModelConfig(
        seed=42,
        transformer=Wan22TransformerConfig(
            transformer_high_noise=_wan22_branch(
                AVAILABLE_CAUSAL_WAN22_CHECKPOINT_PATHS["fastvideo"]["high_noise"]
            ),
            transformer_low_noise=_wan22_branch(
                AVAILABLE_CAUSAL_WAN22_CHECKPOINT_PATHS["fastvideo"]["low_noise"]
            ),
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
"""FastVideo CausalWan2.2 distilled T2V (Wan VAE decoder).

Two-branch MoE: ``high_noise`` runs above the boundary
(``timestep / num_train_timesteps >= boundary_ratio``), ``low_noise``
below. FastVideo 8-step distillation schedule
``[1000, 850, 700, 550, 350, 275, 200, 125]``.
"""


CAUSAL_WAN22_CONFIGS: dict[str, WanInferencePipelineConfig] = {
    cfg.recipe_name: cfg for cfg in (FASTVIDEO_T2V,)
}
"""All shipped streaming Wan 2.2 variants, keyed by ``recipe_name``."""


## Per-variant runner-config literals (slug == ``recipe_name``).

CAUSAL_WAN22_FASTVIDEO_T2V_RUNNER = CausalWan22RunnerConfig(
    runner_name=FASTVIDEO_T2V.recipe_name,
    description="FastVideo distilled CausalWan 2.2 14B MoE T2V (8-step schedule).",
    pipeline=FASTVIDEO_T2V,
)
"""FastVideo distilled streaming CausalWan 2.2 14B MoE T2V runner."""


CAUSAL_WAN22_RUNNERS: dict[str, RunnerConfig] = {
    cfg.runner_name: cfg for cfg in (CAUSAL_WAN22_FASTVIDEO_T2V_RUNNER,)
}
"""All shipped streaming causal Wan 2.2 runners, keyed by ``runner_name``."""

for _name, _cfg in CAUSAL_WAN22_RUNNERS.items():
    register_runner(_name, _cfg, source="builtin")
