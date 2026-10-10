# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public Wan integration surface for integration plugins."""

from flashdreams.recipes.wan.autoencoder.i2v import WanI2VCtrlEncoderConfig
from flashdreams.recipes.wan.autoencoder.vae import (
    AVAILABLE_WAN_VAE_CHECKPOINT_PATHS,
    WAN22_TI2V_5B_VAE_DIFFUSERS_PATH,
    WAN22_TI2V_5B_VAE_PATH,
    Wan22TI2V5BVAEDecoderConfig,
    Wan22TI2V5BVAEEncoderConfig,
    WanVAEDecoder,
    WanVAEDecoderConfig,
    WanVAEEncoder,
    WanVAEEncoderConfig,
    wan22_ti2v_5b_vae_state_dict_transform,
)
from flashdreams.recipes.wan.pipeline import (
    WanInferencePipeline,
    WanInferencePipelineCache,
    WanInferencePipelineConfig,
)
from flashdreams.recipes.wan.transformer.checkpoint import (
    wan_dit_state_dict_from_diffusers,
)
from flashdreams.recipes.wan.transformer.constants import NEGATIVE_PROMPT
from flashdreams.recipes.wan.transformer.impl.network import (
    WanDiTNetwork,
    WanDiTNetwork1pt3BConfig,
    WanDiTNetwork14BConfig,
    WanDiTNetworkConfig,
    WanDiTNetworkTI2V5BConfig,
)
from flashdreams.recipes.wan.transformer.wan21 import (
    Wan21Transformer,
    Wan21TransformerConfig,
)
from flashdreams.recipes.wan.transformer.wan22 import (
    Wan22Transformer,
    Wan22TransformerConfig,
)

__all__ = [
    "AVAILABLE_WAN_VAE_CHECKPOINT_PATHS",
    "NEGATIVE_PROMPT",
    "WAN22_TI2V_5B_VAE_DIFFUSERS_PATH",
    "WAN22_TI2V_5B_VAE_PATH",
    "Wan21Transformer",
    "Wan21TransformerConfig",
    "Wan22TI2V5BVAEDecoderConfig",
    "Wan22TI2V5BVAEEncoderConfig",
    "Wan22Transformer",
    "Wan22TransformerConfig",
    "WanDiTNetwork",
    "WanDiTNetwork1pt3BConfig",
    "WanDiTNetwork14BConfig",
    "WanDiTNetworkConfig",
    "WanDiTNetworkTI2V5BConfig",
    "WanI2VCtrlEncoderConfig",
    "WanInferencePipeline",
    "WanInferencePipelineCache",
    "WanInferencePipelineConfig",
    "WanVAEDecoder",
    "WanVAEDecoderConfig",
    "WanVAEEncoder",
    "WanVAEEncoderConfig",
    "wan22_ti2v_5b_vae_state_dict_transform",
    "wan_dit_state_dict_from_diffusers",
]
