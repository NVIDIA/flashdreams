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

"""Pipeline-config builders for streaming Lingbot World camera-control I2V.

Each builder takes only the runtime knobs the caller owns
(``torch.compile`` toggle, seed, profiling, streaming window) and
returns a fully constructed pipeline config. CP size is auto-detected
from ``torch.distributed.get_world_size()`` inside the transformer.
Shape knobs (batch / view / resolution / per-chunk latent T) are pinned
to canonical Lingbot defaults; callers that want different shapes
should construct the transformer config directly.
"""

from __future__ import annotations

from collections.abc import Callable

from flashdreams.infra.diffusion.model import DiffusionModelConfig
from flashdreams.infra.diffusion.scheduler.fm import FlowMatchSchedulerConfig
from flashdreams.recipes.alpadreams.encoder.pixel_shuffle import (
    PixelShuffleVAEEncoderConfig,
)
from flashdreams.recipes.lingbot_world.encoder.camctrl import (
    I2VCamCtrlEncoderConfig,
)
from flashdreams.recipes.lingbot_world.pipeline import (
    LingbotWorldInferencePipelineConfig,
)
from flashdreams.recipes.lingbot_world.transformer import (
    LingbotWorldTransformerConfig,
)
from flashdreams.recipes.lingbot_world.transformer.impl.network import (
    LingbotWorldDiTNetwork14BConfig,
)
from flashdreams.recipes.taehv import TeahvVAEDecoderConfig
from flashdreams.recipes.wan.autoencoder.i2v import I2VCtrlEncoderConfig
from flashdreams.recipes.wan.autoencoder.vae import (
    AVAILABLE_WAN_VAE_CHECKPOINT_PATHS,
    WanVAEDecoderConfig,
    WanVAEEncoderConfig,
)

AVAILABLE_LINGBOT_WORLD_CHECKPOINT_PATHS: dict[str, str] = {
    "LingBot-World-Fast": "https://huggingface.co/robbyant/lingbot-world-fast/blob/main/diffusion_pytorch_model.safetensors.index.json",
}


## Canonical Lingbot World streaming defaults

# Upstream Fast 4-step distilled schedule.
_DEFAULT_DENOISING_TIMESTEPS = [999, 978, 947, 825]
_DEFAULT_NUM_TRAIN_TIMESTEPS = 1000

_DEFAULT_BATCH_SHAPE: tuple[int, ...] = (1, 1)  # [B=1, V=1]
# Canonical pixel-space defaults; callers pass the matching latent
# (height, width) into :meth:`WanInferencePipeline.initialize_cache`.
DEFAULT_VIDEO_HEIGHT = 464
DEFAULT_VIDEO_WIDTH = 832
_DEFAULT_LEN_T_LATENT = 3
WAN_VAE_SPATIAL_COMPRESSION = 8


def _wan_vae_decoder_config() -> WanVAEDecoderConfig:
    """Streaming Wan VAE decoder (4x temporal, 8x spatial upsample)."""
    return WanVAEDecoderConfig()


def _taehv_vae_decoder_config() -> TeahvVAEDecoderConfig:
    """Tiny AutoEncoder (TAEHV) decoder — drop-in faster replacement for Wan VAE."""
    return TeahvVAEDecoderConfig()


def _scheduler_config(
    denoising_timesteps: list[int],
) -> FlowMatchSchedulerConfig:
    """Lingbot World flow-match scheduler.

    Takes the timestep list as a parameter so future variants that ship a
    different distilled schedule can plug it in without touching the rest
    of the builder. Both currently-shipped Lingbot World presets pass
    ``_DEFAULT_DENOISING_TIMESTEPS``.
    """
    return FlowMatchSchedulerConfig(
        num_inference_steps=len(denoising_timesteps),
        denoising_timesteps=denoising_timesteps,
        warp_denoising_step=False,
        shift=8.0,
        sigma_min=0.0,
        extra_one_step=True,
        num_train_timesteps=_DEFAULT_NUM_TRAIN_TIMESTEPS,
    )


def _transformer_config(
    *,
    checkpoint_path: str,
    compile_network: bool,
    window_size_t: int = 60,
    sink_size_t: int = 0,
) -> LingbotWorldTransformerConfig:
    """Lingbot World 14B transformer defaults for streaming inference."""
    return LingbotWorldTransformerConfig(
        network=LingbotWorldDiTNetwork14BConfig(
            patch_embedding_type="conv3d",
            control_type="cam",
            # 16 noise channels + 4-channel mask + 16-channel image latent
            # (channel-concat I2V layout). Must match the
            # ``concat_image_mask_to_latent=True`` setting below.
            in_dim=16 + 4 + 16,
        ),
        checkpoint_path=checkpoint_path,
        batch_shape=_DEFAULT_BATCH_SHAPE,
        len_t=_DEFAULT_LEN_T_LATENT,
        # CFG off by default to match the upstream Lingbot checkpoint.
        guidance_scale=1.0,
        # Streaming defaults.
        window_size_t=window_size_t,
        sink_size_t=sink_size_t,
        # I2V channel-concat (mask + first-frame latent), not stamping.
        stamp_image_latent=False,
        concat_image_mask_to_latent=True,
        compile_network=compile_network,
    )


def _pipeline_encoder_config() -> I2VCamCtrlEncoderConfig:
    """Composite per-AR-step encoder: Wan VAE I2V + Plücker PixelShuffle."""
    return I2VCamCtrlEncoderConfig(
        i2v=I2VCtrlEncoderConfig(
            encoder=WanVAEEncoderConfig(
                checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"],
            ),
        ),
        plucker=PixelShuffleVAEEncoderConfig(
            frame_selection_mode="last_frame",
        ),
    )


## Builders


def build_lingbot_world_fast(
    *,
    compile_network: bool = True,
    seed: int = 42,
    enable_sync_and_profile: bool = False,
    window_size_t: int = 60,
    sink_size_t: int = 0,
) -> LingbotWorldInferencePipelineConfig:
    """LingBot-World-Fast checkpoint, Wan VAE decoder, 4-step distilled schedule."""
    return LingbotWorldInferencePipelineConfig(
        enable_sync_and_profile=enable_sync_and_profile,
        encoder=_pipeline_encoder_config(),
        decoder=_wan_vae_decoder_config(),
        diffusion_model=DiffusionModelConfig(
            seed=seed,
            transformer=_transformer_config(
                checkpoint_path=AVAILABLE_LINGBOT_WORLD_CHECKPOINT_PATHS[
                    "LingBot-World-Fast"
                ],
                compile_network=compile_network,
                window_size_t=window_size_t,
                sink_size_t=sink_size_t,
            ),
            scheduler=_scheduler_config(_DEFAULT_DENOISING_TIMESTEPS),
        ),
    )


def build_lingbot_world_fast_flash(
    *,
    compile_network: bool = True,
    seed: int = 42,
    enable_sync_and_profile: bool = False,
    window_size_t: int = 15,
    sink_size_t: int = 3,
) -> LingbotWorldInferencePipelineConfig:
    """LingBot-World-Fast checkpoint, TAEHV decoder."""
    return LingbotWorldInferencePipelineConfig(
        enable_sync_and_profile=enable_sync_and_profile,
        encoder=_pipeline_encoder_config(),
        decoder=_taehv_vae_decoder_config(),
        diffusion_model=DiffusionModelConfig(
            seed=seed,
            transformer=_transformer_config(
                checkpoint_path=AVAILABLE_LINGBOT_WORLD_CHECKPOINT_PATHS[
                    "LingBot-World-Fast"
                ],
                compile_network=compile_network,
                window_size_t=window_size_t,
                sink_size_t=sink_size_t,
            ),
            scheduler=_scheduler_config(_DEFAULT_DENOISING_TIMESTEPS),
        ),
    )


LINGBOT_WORLD_CONFIG_BUILDERS: dict[
    str, Callable[..., LingbotWorldInferencePipelineConfig]
] = {
    "LingBot-World-Fast": build_lingbot_world_fast,
    "LingBot-World-Fast-Flash": build_lingbot_world_fast_flash,
}
