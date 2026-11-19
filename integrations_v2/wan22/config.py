# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-rolled Wan pipeline configs.

The Wan 2.2 TI2V-5B recipe, shipped as importable config constants.
TI2V mode reuses :class:`WanInferencePipeline`: the I2V control encoder
(over the 5B VAE) seeds the first frame, and the transformer conditions
on it via ``stamp_image_latent`` and ``ti2v_first_frame_per_token_timestep``
(frame-0 tokens see ``t=0``, the rest denoise at the scheduler step). The
``Wan-AI/Wan2.2-TI2V-5B-Diffusers`` checkpoints load through the DiT and
VAE remap transforms. The package's application layers its I/O behavior on top.
"""

from __future__ import annotations

from pathlib import Path
import os
from flashdreams.infra.diffusion.model import DiffusionModelConfig
from flashdreams.infra.diffusion.scheduler import (
    FlowMatchUniPCSchedulerConfig,
)
from flashdreams.infra.encoder.text.umt5 import UMT5TextEncoderConfig
from flashdreams.recipes.wan.autoencoder.i2v import WanI2VCtrlEncoderConfig
from flashdreams.recipes.wan.autoencoder.vae import (
    Wan22TI2V5BVAEDecoderConfig,
    Wan22TI2V5BVAEEncoderConfig,
    wan22_ti2v_5b_vae_state_dict_transform,
)
from flashdreams.recipes.wan.pipeline import WanInferencePipelineConfig
from flashdreams.recipes.wan.transformer.impl.network import (
    WanDiTNetworkTI2V5BConfig,
)
from flashdreams.recipes.wan.transformer.wan21 import Wan21TransformerConfig
from wan22.impl.checkpoint import wan22_ti2v_5b_dit_state_dict_transform

DEFAULT_VIDEO_HEIGHT = 640
"""Default output height for the standard Wan 2.2 TI2V-5B rollout."""

DEFAULT_VIDEO_WIDTH = 1280
"""Default output width for the standard Wan 2.2 TI2V-5B rollout."""

DEFAULT_VIDEO_FPS = 16
"""Default presentation frame rate for the Wan 2.2 TI2V-5B demo."""

_LOCAL_WAN22_DIR = Path(os.environ.get("WAN22_MODEL_DIR", "/home/gjc/models/Wan2.2-TI2V-5B-Diffusers"))
if (_LOCAL_WAN22_DIR / "transformer/diffusion_pytorch_model.safetensors.index.json").is_file():
    WAN22_TI2V_5B_DIT_DIFFUSERS_PATH = str(
        _LOCAL_WAN22_DIR / "transformer/diffusion_pytorch_model.safetensors.index.json"
    )
    _WAN22_VAE_PATH = str(_LOCAL_WAN22_DIR / "vae/diffusion_pytorch_model.safetensors")
    _WAN22_TEXT_ENCODER = UMT5TextEncoderConfig(model_id_or_local_path=str(_LOCAL_WAN22_DIR))
    _WAN22_VAE_TRANSFORM = wan22_ti2v_5b_vae_state_dict_transform
else:
    WAN22_TI2V_5B_DIT_DIFFUSERS_PATH = (
        "https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers/resolve/main/"
        "transformer/diffusion_pytorch_model.safetensors.index.json"
    )
    _WAN22_VAE_PATH = "https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/resolve/main/Wan2.2_VAE.pth"
    _WAN22_TEXT_ENCODER = None
    _WAN22_VAE_TRANSFORM = None
"""HF sharded-checkpoint index for the Wan 2.2 TI2V-5B DiT."""


PIPELINE_WAN22_TI2V_5B = WanInferencePipelineConfig(
    name="wan22-ti2v-5b",
    # Streaming I2V control encoder over the 5B VAE: AR step 0 encodes the
    # first frame into latent 0 with a one-hot stamp mask; later steps emit
    # a zero mask so the in-network ``stamp_image_latent`` blend is identity.
    encoder=WanI2VCtrlEncoderConfig(
        encoder=Wan22TI2V5BVAEEncoderConfig(
            checkpoint_path=_WAN22_VAE_PATH,
            state_dict_transform=_WAN22_VAE_TRANSFORM,
        ),
    ),
    decoder=Wan22TI2V5BVAEDecoderConfig(
        checkpoint_path=_WAN22_VAE_PATH,
        state_dict_transform=_WAN22_VAE_TRANSFORM,
    ),
    text_encoder=_WAN22_TEXT_ENCODER,
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
            num_inference_steps=int(os.environ.get("WAN22_DENOISE_STEPS", "20")),
            shift=5.0,
        ),
    ),
)
"""Wan 2.2 TI2V-5B inference pipeline (Wan-AI diffusers checkpoint).

One AR step covers the standard 81-frame / 640x1280 rollout
(``len_t == window_size_t == 21``).
"""

WAN_CONFIGS: dict[str, WanInferencePipelineConfig] = {
    PIPELINE_WAN22_TI2V_5B.name: PIPELINE_WAN22_TI2V_5B,
}
"""All in-tree Wan pipeline configs, keyed by ``name``."""


__all__ = [
    "DEFAULT_VIDEO_FPS",
    "DEFAULT_VIDEO_HEIGHT",
    "DEFAULT_VIDEO_WIDTH",
    "PIPELINE_WAN22_TI2V_5B",
    "WAN22_TI2V_5B_DIT_DIFFUSERS_PATH",
    "WAN_CONFIGS",
]
