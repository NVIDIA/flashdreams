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

"""Pre-built pipeline configs for streaming Wan 2.1.

One module-level literal per shipped variant. Variants share the
same chassis (Wan 1.3B DiT, FlowMatch self-forcing scheduler) and
diverge on (a) checkpoint, (b) decoder, (c) ``len_t`` (chunkwise vs
framewise), (d) ``i2v`` (None vs I2VCtrl encoder).
"""

from __future__ import annotations

from typing import Any, TypedDict, cast

from torch import Tensor

from flashdreams.infra.config import derive_config
from flashdreams.infra.diffusion.model import DiffusionModelConfig
from flashdreams.infra.diffusion.scheduler.fm import FlowMatchSchedulerConfig
from flashdreams.recipes.taehv import TeahvVAEDecoderConfig
from flashdreams.recipes.wan.autoencoder.i2v import I2VCtrlEncoderConfig
from flashdreams.recipes.wan.autoencoder.vae import (
    AVAILABLE_WAN_VAE_CHECKPOINT_PATHS,
    WanVAEDecoderConfig,
    WanVAEEncoderConfig,
)
from flashdreams.recipes.wan.pipeline import WanInferencePipelineConfig
from flashdreams.recipes.wan.transformer.impl.network import WanDiTNetwork1pt3BConfig
from flashdreams.recipes.wan.transformer.wan21 import Wan21TransformerConfig


class _CausalForcingPaths(TypedDict):
    chunkwise: str
    framewise: str


class _AvailableCausalWan21Paths(TypedDict):
    self_forcing: str
    causal_forcing: _CausalForcingPaths


AVAILABLE_CAUSAL_WAN21_CHECKPOINT_PATHS: _AvailableCausalWan21Paths = {
    "self_forcing": "https://huggingface.co/gdhe17/Self-Forcing/blob/main/checkpoints/self_forcing_dmd.pt",
    "causal_forcing": {
        "chunkwise": "https://huggingface.co/zhuhz22/Causal-Forcing/blob/main/chunkwise/causal_forcing.pt",
        "framewise": "https://huggingface.co/zhuhz22/Causal-Forcing/blob/main/framewise/causal_forcing.pt",
    },
}

# Canonical pixel-space defaults; callers pass the matching latent
# (height, width) into :meth:`WanInferencePipeline.initialize_cache`.
DEFAULT_VIDEO_HEIGHT = 480
DEFAULT_VIDEO_WIDTH = 832
WAN_VAE_SPATIAL_COMPRESSION = 8


def _remap_self_or_causal_forcing_state_dict(
    state_dict: dict[str, Any],
) -> dict[str, Tensor]:
    """Strip Self-Forcing / Causal-Forcing wrapper prefixes from a state-dict.

    Drops the ``generator_ema`` / ``generator`` container, the ``model.`` /
    ``net.`` outer prefix, and the ``_fsdp_wrapped_module.`` inner prefix
    (framewise variant) so keys match a bare ``WanDiTNetwork``.
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


## Self-Forcing distilled checkpoint with the Wan VAE decoder, T2V chassis.
##
## ``len_t = 3`` is the chunkwise default. ``window_size_t = 21`` matches
## the upstream training crop. Self-Forcing 4-step distillation schedule
## ``[1000, 750, 500, 250]`` with ``shift=8.0``.
SELF_FORCING_T2V = WanInferencePipelineConfig(
    recipe_name="causal-wan21-self-forcing-t2v",
    encoder=None,
    decoder=WanVAEDecoderConfig(),
    diffusion_model=DiffusionModelConfig(
        seed=42,
        transformer=Wan21TransformerConfig(
            network=WanDiTNetwork1pt3BConfig(
                patch_embedding_type="conv3d",
            ),
            checkpoint_path=AVAILABLE_CAUSAL_WAN21_CHECKPOINT_PATHS["self_forcing"],
            state_dict_transform=_remap_self_or_causal_forcing_state_dict,
            batch_shape=(1,),
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

## Self-Forcing distilled checkpoint with the LightTAE (TAEHV) decoder,
## T2V chassis. Faster decoder; identical DiT.
SELF_FORCING_LIGHTTAE_T2V = cast(
    WanInferencePipelineConfig,
    derive_config(
        SELF_FORCING_T2V,
        recipe_name="causal-wan21-self-forcing-lighttae-t2v",
        decoder=TeahvVAEDecoderConfig(),
    ),
)

## Causal-Forcing chunkwise checkpoint with the Wan VAE decoder, T2V
## chassis. Same Wan 1.3B DiT, same scheduler shape; the schedule omits
## the explicit ``shift`` override (default).
CAUSAL_FORCING_CHUNKWISE_T2V = cast(
    WanInferencePipelineConfig,
    derive_config(
        SELF_FORCING_T2V,
        recipe_name="causal-wan21-causal-forcing-chunkwise-t2v",
        diffusion_model=dict(
            transformer=dict(
                checkpoint_path=AVAILABLE_CAUSAL_WAN21_CHECKPOINT_PATHS[
                    "causal_forcing"
                ]["chunkwise"],
            ),
            scheduler=dict(shift=5.0),
        ),
    ),
)

## Causal-Forcing framewise checkpoint, T2V chassis. ``len_t = 1``: one
## latent frame per chunk.
CAUSAL_FORCING_FRAMEWISE_T2V = cast(
    WanInferencePipelineConfig,
    derive_config(
        SELF_FORCING_T2V,
        recipe_name="causal-wan21-causal-forcing-framewise-t2v",
        diffusion_model=dict(
            transformer=dict(
                checkpoint_path=AVAILABLE_CAUSAL_WAN21_CHECKPOINT_PATHS[
                    "causal_forcing"
                ]["framewise"],
                len_t=1,
            ),
            scheduler=dict(shift=5.0),
        ),
    ),
)


## I2V variants: same chassis as the matching T2V variant, plus the I2V
## control encoder on the ``encoder`` slot. Framewise additionally flips
## ``stamp_image_latent`` so AR step 0 substitutes the image latent for
## the first temporal frame.
SELF_FORCING_I2V = cast(
    WanInferencePipelineConfig,
    derive_config(
        SELF_FORCING_T2V,
        recipe_name="causal-wan21-self-forcing-i2v",
        encoder=I2VCtrlEncoderConfig(
            encoder=WanVAEEncoderConfig(
                checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"],
            ),
        ),
    ),
)

SELF_FORCING_LIGHTTAE_I2V = cast(
    WanInferencePipelineConfig,
    derive_config(
        SELF_FORCING_LIGHTTAE_T2V,
        recipe_name="causal-wan21-self-forcing-lighttae-i2v",
        encoder=I2VCtrlEncoderConfig(
            encoder=WanVAEEncoderConfig(
                checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"],
            ),
        ),
    ),
)

CAUSAL_FORCING_CHUNKWISE_I2V = cast(
    WanInferencePipelineConfig,
    derive_config(
        CAUSAL_FORCING_CHUNKWISE_T2V,
        recipe_name="causal-wan21-causal-forcing-chunkwise-i2v",
        encoder=I2VCtrlEncoderConfig(
            encoder=WanVAEEncoderConfig(
                checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"],
            ),
        ),
    ),
)

CAUSAL_FORCING_FRAMEWISE_I2V = cast(
    WanInferencePipelineConfig,
    derive_config(
        CAUSAL_FORCING_FRAMEWISE_T2V,
        recipe_name="causal-wan21-causal-forcing-framewise-i2v",
        encoder=I2VCtrlEncoderConfig(
            encoder=WanVAEEncoderConfig(
                checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"],
            ),
        ),
        diffusion_model=dict(
            transformer=dict(stamp_image_latent=True),
        ),
    ),
)


CAUSAL_WAN21_CONFIGS: dict[str, WanInferencePipelineConfig] = {
    cfg.recipe_name: cfg
    for cfg in (
        SELF_FORCING_T2V,
        SELF_FORCING_LIGHTTAE_T2V,
        CAUSAL_FORCING_CHUNKWISE_T2V,
        CAUSAL_FORCING_FRAMEWISE_T2V,
        SELF_FORCING_I2V,
        SELF_FORCING_LIGHTTAE_I2V,
        CAUSAL_FORCING_CHUNKWISE_I2V,
        CAUSAL_FORCING_FRAMEWISE_I2V,
    )
}
"""All shipped streaming Wan 2.1 variants, keyed by ``recipe_name``."""
