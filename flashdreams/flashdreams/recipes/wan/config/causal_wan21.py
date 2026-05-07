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

"""Pre-built pipeline-config builders for streaming Wan 2.1."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypedDict

from torch import Tensor

from flashdreams.infra.config import InstantiateConfig
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


## Checkpoint remap


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


## Canonical Wan 2.1 streaming defaults

# Self-Forcing-style 4-step distillation schedule.
_DEFAULT_DENOISING_TIMESTEPS = [1000, 750, 500, 250]
_DEFAULT_NUM_TRAIN_TIMESTEPS = 1000

_DEFAULT_BATCH_SHAPE: tuple[int, ...] = (1,)
# Canonical pixel-space defaults; callers pass the matching latent
# (height, width) into :meth:`WanInferencePipeline.initialize_cache`.
DEFAULT_VIDEO_HEIGHT = 480
DEFAULT_VIDEO_WIDTH = 832
_DEFAULT_LEN_T_LATENT = 3  # framewise variant overrides to 1.
WAN_VAE_SPATIAL_COMPRESSION = 8


def _wan_vae_decoder_config() -> WanVAEDecoderConfig:
    """Wan VAE decoder config."""
    return WanVAEDecoderConfig()


def _taehv_vae_decoder_config() -> TeahvVAEDecoderConfig:
    """LightTAE (TAEHV) decoder config."""
    return TeahvVAEDecoderConfig()


def _scheduler_config(
    num_inference_steps: int = 4, shift: float = 5.0
) -> FlowMatchSchedulerConfig:
    """Self-Forcing flow-match scheduler defaults."""
    timesteps = _DEFAULT_DENOISING_TIMESTEPS[:num_inference_steps]
    return FlowMatchSchedulerConfig(
        num_inference_steps=num_inference_steps,
        denoising_timesteps=timesteps,
        warp_denoising_step=True,
        shift=shift,
        sigma_min=0.0,
        extra_one_step=True,
        num_train_timesteps=_DEFAULT_NUM_TRAIN_TIMESTEPS,
    )


def _transformer_config(
    *,
    checkpoint_path: str,
    compile_network: bool,
    len_t_latent: int = _DEFAULT_LEN_T_LATENT,
    stamp_image_latent: bool = False,
) -> Wan21TransformerConfig:
    """Wan 1.3B transformer defaults for causal/streaming inference."""
    return Wan21TransformerConfig(
        network=WanDiTNetwork1pt3BConfig(
            patch_embedding_type="conv3d",
        ),
        checkpoint_path=checkpoint_path,
        state_dict_transform=_remap_self_or_causal_forcing_state_dict,
        batch_shape=_DEFAULT_BATCH_SHAPE,
        len_t=len_t_latent,
        guidance_scale=1.0,
        window_size_t=21,
        sink_size_t=0,
        stamp_image_latent=stamp_image_latent,
        compile_network=compile_network,
    )


def _pipeline_encoder_config(*, i2v: bool) -> InstantiateConfig[Any] | None:
    """Per-AR-step encoder config: I2V control encoder, or ``None`` for T2V."""
    if not i2v:
        return None
    return I2VCtrlEncoderConfig(
        encoder=WanVAEEncoderConfig(
            checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"],
        ),
    )


## Builders


def build_self_forcing(
    *,
    compile_network: bool = True,
    seed: int = 42,
    i2v: bool = False,
    enable_sync_and_profile: bool = False,
) -> WanInferencePipelineConfig:
    """Self-Forcing distilled checkpoint with the Wan VAE decoder."""
    return WanInferencePipelineConfig(
        enable_sync_and_profile=enable_sync_and_profile,
        encoder=_pipeline_encoder_config(i2v=i2v),
        decoder=_wan_vae_decoder_config(),
        diffusion_model=DiffusionModelConfig(
            seed=seed,
            transformer=_transformer_config(
                checkpoint_path=AVAILABLE_CAUSAL_WAN21_CHECKPOINT_PATHS["self_forcing"],
                compile_network=compile_network,
            ),
            scheduler=_scheduler_config(num_inference_steps=4, shift=8.0),
        ),
    )


def build_self_forcing_lighttae(
    *,
    compile_network: bool = True,
    seed: int = 42,
    i2v: bool = False,
    enable_sync_and_profile: bool = False,
) -> WanInferencePipelineConfig:
    """Self-Forcing distilled checkpoint with the LightTAE (TAEHV) decoder."""
    return WanInferencePipelineConfig(
        enable_sync_and_profile=enable_sync_and_profile,
        encoder=_pipeline_encoder_config(i2v=i2v),
        decoder=_taehv_vae_decoder_config(),
        diffusion_model=DiffusionModelConfig(
            seed=seed,
            transformer=_transformer_config(
                checkpoint_path=AVAILABLE_CAUSAL_WAN21_CHECKPOINT_PATHS["self_forcing"],
                compile_network=compile_network,
            ),
            scheduler=_scheduler_config(num_inference_steps=4, shift=8.0),
        ),
    )


def build_causal_forcing_chunkwise(
    *,
    compile_network: bool = True,
    seed: int = 42,
    i2v: bool = False,
    enable_sync_and_profile: bool = False,
) -> WanInferencePipelineConfig:
    """Causal-Forcing chunkwise checkpoint with the Wan VAE decoder."""
    return WanInferencePipelineConfig(
        enable_sync_and_profile=enable_sync_and_profile,
        encoder=_pipeline_encoder_config(i2v=i2v),
        decoder=_wan_vae_decoder_config(),
        diffusion_model=DiffusionModelConfig(
            seed=seed,
            transformer=_transformer_config(
                checkpoint_path=AVAILABLE_CAUSAL_WAN21_CHECKPOINT_PATHS[
                    "causal_forcing"
                ]["chunkwise"],
                compile_network=compile_network,
            ),
            scheduler=_scheduler_config(num_inference_steps=4),
        ),
    )


def build_causal_forcing_framewise(
    *,
    compile_network: bool = True,
    seed: int = 42,
    i2v: bool = False,
    enable_sync_and_profile: bool = False,
) -> WanInferencePipelineConfig:
    """Causal-Forcing framewise checkpoint with the Wan VAE decoder."""
    return WanInferencePipelineConfig(
        enable_sync_and_profile=enable_sync_and_profile,
        encoder=_pipeline_encoder_config(i2v=i2v),
        decoder=_wan_vae_decoder_config(),
        diffusion_model=DiffusionModelConfig(
            seed=seed,
            transformer=_transformer_config(
                checkpoint_path=AVAILABLE_CAUSAL_WAN21_CHECKPOINT_PATHS[
                    "causal_forcing"
                ]["framewise"],
                compile_network=compile_network,
                # framewise: one latent frame per chunk; I2V replaces it with
                # the image latent at AR step 0.
                len_t_latent=1,
                stamp_image_latent=i2v,
            ),
            scheduler=_scheduler_config(num_inference_steps=4),
        ),
    )


CAUSAL_WAN21_CONFIG_BUILDERS: dict[str, Callable[..., WanInferencePipelineConfig]] = {
    "self_forcing": build_self_forcing,
    "self_forcing_lighttae": build_self_forcing_lighttae,
    "causal_forcing_chunkwise": build_causal_forcing_chunkwise,
    "causal_forcing_framewise": build_causal_forcing_framewise,
}
