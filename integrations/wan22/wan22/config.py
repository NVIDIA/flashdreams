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

"""Pre-rolled Wan pipeline configs.

The Wan 2.2 TI2V-5B recipe, shipped as importable config constants.
TI2V mode reuses :class:`WanInferencePipeline`: the I2V control encoder
(over the 5B VAE) seeds the first frame, and the transformer conditions
on it via ``stamp_image_latent`` and ``ti2v_first_frame_per_token_timestep``
(frame-0 tokens see ``t=0``, the rest denoise at the scheduler step). The
``Wan-AI/Wan2.2-TI2V-5B-Diffusers`` checkpoints load through the DiT and
VAE remap transforms. Downstream runners (e.g. ``hy_worldplay``) layer the
I/O wrapper on top.
"""

from __future__ import annotations

import torch

from flashdreams.core.checkpoint.remap import remap_checkpoint_keys
from flashdreams.infra.diffusion.model import DiffusionModelConfig
from flashdreams.infra.diffusion.scheduler import (
    FlowMatchUniPCSchedulerConfig,
)
from flashdreams.recipes.wan.autoencoder.i2v import WanI2VCtrlEncoderConfig
from flashdreams.recipes.wan.autoencoder.vae import (
    Wan22TI2V5BVAEDecoderConfig,
    Wan22TI2V5BVAEEncoderConfig,
)
from flashdreams.recipes.wan.pipeline import WanInferencePipelineConfig
from flashdreams.recipes.wan.transformer.impl.network import (
    WanDiTNetworkTI2V5BConfig,
)
from flashdreams.recipes.wan.transformer.wan21 import Wan21TransformerConfig

WAN22_TI2V_5B_DIT_DIFFUSERS_PATH = (
    "https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers/resolve/main/"
    "transformer/diffusion_pytorch_model.safetensors"
)
"""HF diffusers checkpoint for the Wan 2.2 TI2V-5B DiT (``transformer/`` subfolder)."""


# Diffusers ``WanTransformer3DModel`` -> ``WanDiTNetwork`` key remap:
# condition embedders, scale/shift table, attention projections
# (``attn1``/``attn2``), and FFN.
_WAN22_TI2V_5B_DIT_KEY_REMAP: dict[str, str] = {
    r"^condition_embedder\.text_embedder\.linear_1\.(.*)$": r"text_embedding.0.\1",
    r"^condition_embedder\.text_embedder\.linear_2\.(.*)$": r"text_embedding.2.\1",
    r"^condition_embedder\.time_embedder\.linear_1\.(.*)$": r"time_embedding.0.\1",
    r"^condition_embedder\.time_embedder\.linear_2\.(.*)$": r"time_embedding.2.\1",
    r"^condition_embedder\.time_proj\.(.*)$": r"time_projection.1.\1",
    r"^scale_shift_table$": r"head.modulation",
    r"^proj_out\.(.*)$": r"head.head.\1",
    r"^blocks\.(\d+)\.attn1\.to_q\.(.*)$": r"blocks.\1.self_attn.q.\2",
    r"^blocks\.(\d+)\.attn1\.to_k\.(.*)$": r"blocks.\1.self_attn.k.\2",
    r"^blocks\.(\d+)\.attn1\.to_v\.(.*)$": r"blocks.\1.self_attn.v.\2",
    r"^blocks\.(\d+)\.attn1\.to_out\.0\.(.*)$": r"blocks.\1.self_attn.o.\2",
    r"^blocks\.(\d+)\.attn2\.to_q\.(.*)$": r"blocks.\1.cross_attn.q.\2",
    r"^blocks\.(\d+)\.attn2\.to_k\.(.*)$": r"blocks.\1.cross_attn.k.\2",
    r"^blocks\.(\d+)\.attn2\.to_v\.(.*)$": r"blocks.\1.cross_attn.v.\2",
    r"^blocks\.(\d+)\.attn2\.to_out\.0\.(.*)$": r"blocks.\1.cross_attn.o.\2",
    r"^blocks\.(\d+)\.attn1\.norm_q\.(.*)$": r"blocks.\1.self_attn.norm_q.\2",
    r"^blocks\.(\d+)\.attn1\.norm_k\.(.*)$": r"blocks.\1.self_attn.norm_k.\2",
    r"^blocks\.(\d+)\.attn2\.norm_q\.(.*)$": r"blocks.\1.cross_attn.norm_q.\2",
    r"^blocks\.(\d+)\.attn2\.norm_k\.(.*)$": r"blocks.\1.cross_attn.norm_k.\2",
    r"^blocks\.(\d+)\.norm2\.(.*)$": r"blocks.\1.norm3.\2",
    r"^blocks\.(\d+)\.scale_shift_table$": r"blocks.\1.modulation",
    r"^blocks\.(\d+)\.ffn\.fc_in\.(.*)$": r"blocks.\1.ffn.0.\2",
    r"^blocks\.(\d+)\.ffn\.fc_out\.(.*)$": r"blocks.\1.ffn.2.\2",
    r"^blocks\.(\d+)\.ffn\.net\.0\.proj\.(.*)$": r"blocks.\1.ffn.0.\2",
    r"^blocks\.(\d+)\.ffn\.net\.2\.(.*)$": r"blocks.\1.ffn.2.\2",
}


def wan22_ti2v_5b_dit_state_dict_transform(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remap a diffusers Wan 2.2 TI2V-5B DiT state-dict to ``WanDiTNetwork`` keys.

    Applied automatically when :data:`PIPELINE_WAN22_TI2V_5B` loads the
    ``Wan-AI/Wan2.2-TI2V-5B-Diffusers/transformer`` checkpoint.
    """
    return remap_checkpoint_keys(state_dict, _WAN22_TI2V_5B_DIT_KEY_REMAP)


PIPELINE_WAN22_TI2V_5B = WanInferencePipelineConfig(
    name="wan22-ti2v-5b",
    enable_sync_and_profile=True,
    # Streaming I2V control encoder over the 5B VAE: AR step 0 encodes the
    # first frame into latent 0 with a one-hot stamp mask; later steps emit
    # a zero mask so the in-network ``stamp_image_latent`` blend is identity.
    encoder=WanI2VCtrlEncoderConfig(
        encoder=Wan22TI2V5BVAEEncoderConfig(),
    ),
    decoder=Wan22TI2V5BVAEDecoderConfig(),
    # No CLIP image branch: ``image_encoder=None`` also disables the matching
    # DiT cross-attention branch (``cross_attn_enable_img=False``).
    image_encoder=None,
    diffusion_model=DiffusionModelConfig(
        seed=42,
        transformer=Wan21TransformerConfig(
            network=WanDiTNetworkTI2V5BConfig(),
            checkpoint_path=WAN22_TI2V_5B_DIT_DIFFUSERS_PATH,
            state_dict_transform=wan22_ti2v_5b_dit_state_dict_transform,
            batch_shape=(),
            len_t=21,
            window_size_t=21,
            guidance_scale=5.0,
            # First-frame conditioning: re-inject the clean image latent each
            # step (stamp) and give frame-0 tokens ``t=0`` while the rest
            # denoise at the scheduler step.
            stamp_image_latent=True,
            ti2v_first_frame_per_token_timestep=True,
            # 5B injects the first frame via the stamp path, not channel-concat.
            concat_image_mask_to_latent=False,
        ),
        scheduler=FlowMatchUniPCSchedulerConfig(
            num_inference_steps=40,
            shift=5.0,
        ),
    ),
)
"""Wan 2.2 TI2V-5B inference pipeline (Wan-AI diffusers checkpoint).

One AR step covers the standard 81-frame / 640x1280 rollout
(``len_t == window_size_t == 21``). Base recipe for
``integrations/hy_worldplay``.
"""

WAN_CONFIGS: dict[str, WanInferencePipelineConfig] = {
    PIPELINE_WAN22_TI2V_5B.name: PIPELINE_WAN22_TI2V_5B,
}
"""All in-tree Wan pipeline configs, keyed by ``name``."""


__all__ = [
    "PIPELINE_WAN22_TI2V_5B",
    "WAN22_TI2V_5B_DIT_DIFFUSERS_PATH",
    "WAN_CONFIGS",
    "wan22_ti2v_5b_dit_state_dict_transform",
]
