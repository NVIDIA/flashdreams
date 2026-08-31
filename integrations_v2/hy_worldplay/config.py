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

"""Public HY-WorldPlay pipeline configuration."""

from __future__ import annotations

import copy
from typing import cast

from wan22.config import PIPELINE_WAN22_TI2V_5B

from flashdreams.infra.config import derive_config
from flashdreams.infra.diffusion.model import DiffusionModelConfig
from flashdreams.infra.diffusion.scheduler import (
    FlowMatchEulerDiscreteSchedulerConfig,
)
from flashdreams.recipes.wan.autoencoder.i2v import WanI2VCtrlEncoderConfig
from flashdreams.recipes.wan.transformer.impl.network import WanDiTNetworkConfig
from flashdreams.recipes.wan.transformer.wan21 import Wan21TransformerConfig
from hy_worldplay.impl._action import (
    HyWorldPlayWan21TransformerConfig,
    HyWorldPlayWanCtrlEncoderConfig,
    HyWorldPlayWanDiTNetworkConfig,
)
from hy_worldplay.impl._checkpoint import (
    HY_WORLDPLAY_DISTILLED_CKPT_PATH,
    hy_worldplay_distilled_state_dict_transform,
)
from hy_worldplay.impl.pipeline import HyWorldPlayPipelineConfig

_BASE_ENCODER = PIPELINE_WAN22_TI2V_5B.encoder
assert isinstance(_BASE_ENCODER, WanI2VCtrlEncoderConfig)

_BASE_TRANSFORMER = PIPELINE_WAN22_TI2V_5B.diffusion_model.transformer
assert isinstance(_BASE_TRANSFORMER, Wan21TransformerConfig)

_BASE_NETWORK = _BASE_TRANSFORMER.network
assert isinstance(_BASE_NETWORK, WanDiTNetworkConfig)


PIPELINE_HY_WORLDPLAY_WAN_I2V_5B = HyWorldPlayPipelineConfig(
    name="hy-worldplay-wan-i2v-5b",
    enable_sync_and_profile=PIPELINE_WAN22_TI2V_5B.enable_sync_and_profile,
    text_encoder=copy.deepcopy(PIPELINE_WAN22_TI2V_5B.text_encoder),
    image_encoder=copy.deepcopy(PIPELINE_WAN22_TI2V_5B.image_encoder),
    encoder=HyWorldPlayWanCtrlEncoderConfig(
        encoder=copy.deepcopy(_BASE_ENCODER.encoder),
    ),
    decoder=copy.deepcopy(PIPELINE_WAN22_TI2V_5B.decoder),
    diffusion_model=cast(
        DiffusionModelConfig,
        derive_config(
            PIPELINE_WAN22_TI2V_5B.diffusion_model,
            scheduler=FlowMatchEulerDiscreteSchedulerConfig(
                num_inference_steps=4,
                fixed_timesteps=(1000.0, 960.0, 888.8889, 727.2728, 0.0),
            ),
            transformer=HyWorldPlayWan21TransformerConfig(
                network=HyWorldPlayWanDiTNetworkConfig(
                    patch_size=_BASE_NETWORK.patch_size,
                    text_len=_BASE_NETWORK.text_len,
                    in_dim=_BASE_NETWORK.in_dim,
                    dim=_BASE_NETWORK.dim,
                    ffn_dim=_BASE_NETWORK.ffn_dim,
                    freq_dim=_BASE_NETWORK.freq_dim,
                    text_dim=_BASE_NETWORK.text_dim,
                    out_dim=_BASE_NETWORK.out_dim,
                    num_heads=_BASE_NETWORK.num_heads,
                    num_layers=_BASE_NETWORK.num_layers,
                    cross_attn_norm=_BASE_NETWORK.cross_attn_norm,
                    cross_attn_enable_img=_BASE_NETWORK.cross_attn_enable_img,
                    eps=_BASE_NETWORK.eps,
                    concat_padding_mask=_BASE_NETWORK.concat_padding_mask,
                    patch_embedding_type=_BASE_NETWORK.patch_embedding_type,
                    apply_rope_before_kvcache=(_BASE_NETWORK.apply_rope_before_kvcache),
                    use_prope_blocks=True,
                ),
                dtype=_BASE_TRANSFORMER.dtype,
                checkpoint_path=HY_WORLDPLAY_DISTILLED_CKPT_PATH,
                state_dict_transform=hy_worldplay_distilled_state_dict_transform,
                batch_shape=_BASE_TRANSFORMER.batch_shape,
                len_t=4,
                guidance_scale=1.0,
                window_size_t=4,
                sink_size_t=_BASE_TRANSFORMER.sink_size_t,
                h_extrapolation_ratio=_BASE_TRANSFORMER.h_extrapolation_ratio,
                w_extrapolation_ratio=_BASE_TRANSFORMER.w_extrapolation_ratio,
                compile_network=_BASE_TRANSFORMER.compile_network,
                use_cuda_graph=False,
                cuda_graph_warmup_iters=(_BASE_TRANSFORMER.cuda_graph_warmup_iters),
                stamp_image_latent=_BASE_TRANSFORMER.stamp_image_latent,
                concat_image_mask_to_latent=(
                    _BASE_TRANSFORMER.concat_image_mask_to_latent
                ),
                ti2v_first_frame_per_token_timestep=(
                    _BASE_TRANSFORMER.ti2v_first_frame_per_token_timestep
                ),
                first_frame_timestep_value=14.0,
            ),
        ),
    ),
)
"""HY-WorldPlay WAN-5B camera-controlled inference pipeline."""


HY_WORLDPLAY_CONFIGS: dict[str, HyWorldPlayPipelineConfig] = {
    PIPELINE_HY_WORLDPLAY_WAN_I2V_5B.name: PIPELINE_HY_WORLDPLAY_WAN_I2V_5B,
}
"""Public HY-WorldPlay pipeline configurations, keyed by slug."""


__all__ = [
    "HY_WORLDPLAY_CONFIGS",
    "PIPELINE_HY_WORLDPLAY_WAN_I2V_5B",
]
