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

Phase 2a deliverable: the Wan 2.2 TI2V 5B recipe. The pipeline reuses
the existing :class:`WanInferencePipeline` (no new module needed): the
TI2V mode is expressed as a configuration of the I2V control encoder
(now driven by the 5B 16x / 48ch / residual / patchify VAE) plus the
existing transformer's ``stamp_image_latent`` mask-inject path, with a
new ``ti2v_first_frame_per_token_timestep`` flag that flips the AR-0
scheduler timestep into a per-token tensor (``t=0`` at the first-frame
conditioning tokens, scheduler ``t`` elsewhere). The diffusers
checkpoints under
``Wan-AI/Wan2.2-TI2V-5B-Diffusers/{vae,transformer}`` load directly via
the :func:`wan22_ti2v_5b_dit_state_dict_transform` (DiT) and
:func:`wan22_ti2v_5b_vae_state_dict_transform` (VAE) remaps.

The recipe ships as importable config constants only; downstream
runners (``flashdreams-run`` slugs, plugin integrations such as
``hy_worldplay`` phase 2b) layer the I/O wrapper on top.
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
"""HF diffusers safetensors shard for the Wan 2.2 TI2V 5B DiT.

The 5B variant ships as a single safetensors file (no sharded index)
under the ``transformer/`` subfolder of the ``Wan-AI`` diffusers repo."""


# Diffusers ``WanTransformer3DModel`` -> bare ``WanDiTNetwork`` state-
# dict remap. The diffusers checkpoint stores the same submodules as
# ours under different paths: condition embedders, scale/shift table,
# attention projections (``attn1``/``attn2``), and FFN. The mapping is
# identical to FastVideo's Wan 2.2 14B remap because both 5B and 14B
# checkpoints inherit the same diffusers ``WanTransformer3DModel``
# layout -- only the layer counts and channel counts differ.
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
    """Remap a diffusers Wan 2.2 TI2V 5B DiT state-dict to ``WanDiTNetwork`` keys.

    Wan 2.2 5B and 14B share the upstream diffusers DiT layout, so this
    is structurally the same remap FastVideo's 14B MoE config uses --
    only the layer / channel counts of the downstream config differ.
    The remap is applied automatically when
    :data:`PIPELINE_WAN22_TI2V_5B` loads the upstream
    ``Wan-AI/Wan2.2-TI2V-5B-Diffusers/transformer`` checkpoint.
    """
    return remap_checkpoint_keys(state_dict, _WAN22_TI2V_5B_DIT_KEY_REMAP)


PIPELINE_WAN22_TI2V_5B = WanInferencePipelineConfig(
    name="wan22-ti2v-5b",
    enable_sync_and_profile=True,
    # The streaming I2V control encoder reuses the standard Wan
    # ``I2VCtrlEncoder``: AR step 0 encodes the first frame into latent
    # index 0 and stamps a one-hot mask, AR step >= 1 emits an all-
    # zero mask so the in-network ``stamp_image_latent`` blend
    # collapses to identity. Wrapped around the 5B 16x VAE encoder
    # (``Wan22TI2V5BVAEEncoderConfig``), this is the TI2V first-frame
    # seed pipeline -- no CLIP image branch required.
    encoder=WanI2VCtrlEncoderConfig(
        encoder=Wan22TI2V5BVAEEncoderConfig(),
    ),
    decoder=Wan22TI2V5BVAEDecoderConfig(),
    # Wan 2.2 TI2V 5B has no CLIP cross-attention branch; the first
    # frame is conditioned via the VAE latent seed + per-token t=0,
    # not via CLIP image features. Leaving ``image_encoder=None``
    # disables both the CLIP one-shot encoder and the matching DiT
    # cross-attention branch (see
    # ``WanDiTNetworkTI2V5BConfig.cross_attn_enable_img=False``).
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
            # The TI2V 5B recipe combines the existing mask-inject
            # stamp (clean image latent re-injected every denoising
            # step) with the new per-token timestep override (frame-0
            # tokens see ``t=0`` while the rest of the chunk denoises
            # at the scheduler step). Together they implement Wan 2.2
            # 5B's "VAE-seeded first-frame + per-token t=0" recipe.
            stamp_image_latent=True,
            ti2v_first_frame_per_token_timestep=True,
            # No channel-concat I2V layout: 5B's first frame is
            # injected via the stamp path, not by appending mask +
            # image-latent channels to the network input.
            concat_image_mask_to_latent=False,
        ),
        scheduler=FlowMatchUniPCSchedulerConfig(
            num_inference_steps=40,
            shift=5.0,
        ),
    ),
)
"""Wan 2.2 TI2V 5B inference pipeline (HF Wan-AI diffusers checkpoint).

A single AR step is sufficient for the standard 81-frame /
640x1280 TI2V rollout (``len_t == window_size_t == 21``), matching
upstream's defaults. The recipe is the prerequisite for
``integrations/hy_worldplay`` phase 2b, which layers HY-WorldPlay's
action + camera-trajectory + reconstituted-context-memory deltas on
top of this pipeline.
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
