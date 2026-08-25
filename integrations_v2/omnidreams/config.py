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

"""User-facing configs for Omnidreams.

Hosts the pre-built :class:`OmnidreamsPipelineConfig` literals and the
model-owned hooks consumed by the reusable scene-driving application.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import torch
from omnidreams.impl.encoder.pixel_shuffle import (
    PixelShuffleVAEEncoderConfig,
)
from omnidreams.impl.pipeline import (
    OmnidreamsPipelineConfig,
)
from omnidreams.impl.transformer import CosmosTransformerConfig
from omnidreams.impl.transformer.network import (
    CosmosDiTNetworkConfig,
)
from omnidreams.impl.vae_native import (
    OmnidreamsWanVAEEncoderConfig as WanVAEEncoderConfig,
)

from flashdreams.infra.config import derive_config
from flashdreams.infra.diffusion.model import DiffusionModelConfig
from flashdreams.infra.diffusion.scheduler.fm import (
    FlowMatchSchedulerConfig,
)
from flashdreams.infra.diffusion.scheduler.fm_unipc import (
    FlowMatchUniPCSchedulerConfig,
)
from flashdreams.infra.encoder.text.cosmos_reason1 import (
    CosmosReason1TextEncoderConfig,
)
from flashdreams.recipes.taehv import (
    AVAILABLE_TAEHV_CHECKPOINT_PATHS,
    TeahvVAEDecoderConfig,
)
from flashdreams.recipes.wan.autoencoder.vae import (
    AVAILABLE_WAN_VAE_CHECKPOINT_PATHS,
    WanVAEDecoderConfig,
)
from omnidreams.apps.interactive_drive.manifest import (
    load_world_model_manifest,
    resolve_world_model_manifest_path,
)
from omnidreams.apps.interactive_drive.model import FlashdreamsWorldModelSession
from omnidreams.impl.scenes import hf_hub_download_scene
from hdmap2v import (
    Hdmap2VApplicationDefaults,
    Hdmap2VApplicationHooks,
)
from hdmap2v.interactive_drive.backends.base import RenderBackend
from hdmap2v.interactive_drive.backends.raster import RasterRenderBackend
from hdmap2v.interactive_drive.backends.world_model import (
    WorldModelRenderBackend,
)
from hdmap2v.interactive_drive.config import AppConfig

AVAILABLE_OMNIDREAMS_CHECKPOINT_PATHS: dict[str, str] = {
    "1view-vae-chunk2": (
        "https://huggingface.co/nvidia/omni-dreams-models/resolve/main/"
        "single_view/2b_res720p_30fps_i2v_hdmap_distilled.pt"
    ),
    # internal-only checkpoints must be provided at runtime.
    "1view-pshuffle-chunk4": "MISSING",
    "1view-vae-chunk3": "MISSING",
    "4view-pshuffle-chunk4": "MISSING",
    "4view-vae-chunk4": "MISSING",
    "1view-diffusion-forcing-chunk2": "MISSING",
    "1view-bidirectional-chunk48": "MISSING",
}
"""Checkpoint paths for the Omnidreams pipeline."""

_LIGHTVAE_FP8_STATE_ENV = "OMNIDREAMS_LIGHTVAE_FP8_STATE_PATH"


def _lightvae_fp8_state_path() -> str | None:
    return os.environ.get(_LIGHTVAE_FP8_STATE_ENV)


SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE = OmnidreamsPipelineConfig(
    name="omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae",
    text_encoder=CosmosReason1TextEncoderConfig(),
    image_encoder=WanVAEEncoderConfig(
        checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["lightvae"],
    ),
    enable_sync_and_profile=True,
    encoder=WanVAEEncoderConfig(
        checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["lightvae"],
    ),
    decoder=TeahvVAEDecoderConfig(
        checkpoint_path=AVAILABLE_TAEHV_CHECKPOINT_PATHS["lighttae"],
    ),
    diffusion_model=DiffusionModelConfig(
        seed=42,
        context_noise=128,
        transformer=CosmosTransformerConfig(
            network=CosmosDiTNetworkConfig(
                # 16 channels: Wan-VAE HDMap branch.
                additional_concat_ch=16,
                enable_cross_view_attn=False,
                cp_method="ring",
            ),
            checkpoint_path=AVAILABLE_OMNIDREAMS_CHECKPOINT_PATHS["1view-vae-chunk2"],
            batch_shape=(1,),
            num_views=1,
            len_t=2,
            h_extrapolation_ratio=3.0,
            w_extrapolation_ratio=3.0,
            window_size_t=6,
            sink_size_t=0,
            compile_network=True,
            use_cuda_graph=True,
            skip_finalize_kv_cache=False,
            guidance_scale=1.0,
        ),
        scheduler=FlowMatchSchedulerConfig(
            num_inference_steps=2,
            denoising_timesteps=[1000, 450],
            warp_denoising_step=True,
            shift=5.0,
            sigma_min=0.0,
            extra_one_step=True,
            num_train_timesteps=1000,
        ),
    ),
)
"""Base single-view chunk2 chassis: light Wan VAE HDMap encoder + LightTAE.

Self-Forcing distilled: 2-step flow-match, ``len_t=2``, ``window_size_t=6``,
CFG off. Every chunk2 variant derives from this and flips a few fields.
"""

SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE_PERF = cast(
    OmnidreamsPipelineConfig,
    derive_config(
        SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE,
        name="omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf",
        image_encoder=dict(use_compile=True, use_cuda_graph=True),
        encoder=dict(use_compile=True, use_cuda_graph=True),
        decoder=dict(use_compile=True, use_cuda_graph=True),
    ),
)  # ty:ignore[redundant-cast]
"""Performance-tuned variant: enable ``use_compile`` / ``use_cuda_graph``
on the image encoder, the per-AR-step encoder, and the decoder."""

SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE_NATIVE_PERF = cast(
    OmnidreamsPipelineConfig,
    derive_config(
        SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE_PERF,
        name="omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-native-perf",
        image_encoder=dict(
            dtype=torch.float16,
            use_compile=False,
            use_cuda_graph=False,
            native_vae_acceleration="required",
            native_vae_backend="fp8",
            native_vae_fp8_state_path=_lightvae_fp8_state_path(),
        ),
        encoder=dict(
            dtype=torch.float16,
            use_compile=False,
            use_cuda_graph=False,
            native_vae_acceleration="required",
            native_vae_backend="fp8",
            native_vae_fp8_state_path=_lightvae_fp8_state_path(),
        ),
    ),
)  # ty:ignore[redundant-cast]
"""Native VAE perf variant: LightVAE FP8 encoder with PyTorch LightTAE decoder.

Set ``OMNIDREAMS_LIGHTVAE_FP8_STATE_PATH`` before setup to provide the
calibrated FP8 LightVAE state required by the native encoder.
"""

SV_2STEPS_CHUNK2_LOC6_VAE_VAE = cast(
    OmnidreamsPipelineConfig,
    derive_config(
        SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE,
        name="omnidreams-sv-2steps-chunk2-loc6-vae-vae",
        image_encoder=dict(checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"]),
        encoder=dict(checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"]),
        decoder=WanVAEDecoderConfig(
            checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"],
            use_compile=False,
            use_cuda_graph=True,
        ),
    ),
)  # ty:ignore[redundant-cast]
"""Single-view, chunk2, full Wan VAE for both HDMap encoding and decoding."""

SV_2STEPS_CHUNK3_LOC6_VAE_VAE = cast(
    OmnidreamsPipelineConfig,
    derive_config(
        SV_2STEPS_CHUNK2_LOC6_VAE_VAE,
        name="omnidreams-sv-2steps-chunk3-loc6-vae-vae",
        diffusion_model=dict(
            transformer=dict(
                checkpoint_path=AVAILABLE_OMNIDREAMS_CHECKPOINT_PATHS[
                    "1view-vae-chunk3"
                ],
                len_t=3,
            ),
        ),
    ),
)  # ty:ignore[redundant-cast]
"""Single-view, chunk3, full Wan VAE for both HDMap encoding and decoding.

Same chassis as ``SV_2STEPS_CHUNK2_LOC6_VAE_VAE`` but with ``len_t=3``
and the matching chunk3 checkpoint.
"""

SV_2STEPS_CHUNK4_LOC8_PSHUFFLE_LIGHTTAE = cast(
    OmnidreamsPipelineConfig,
    derive_config(
        SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE,
        name="omnidreams-sv-2steps-chunk4-loc8-pshuffle-lighttae",
        image_encoder=dict(checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"]),
        encoder=PixelShuffleVAEEncoderConfig(),
        diffusion_model=dict(
            transformer=dict(
                network=dict(additional_concat_ch=192),
                checkpoint_path=AVAILABLE_OMNIDREAMS_CHECKPOINT_PATHS[
                    "1view-pshuffle-chunk4"
                ],
                len_t=4,
                window_size_t=8,
            ),
        ),
    ),
)  # ty:ignore[redundant-cast]
"""Single-view chunk4: PixelShuffle HDMap encoder + LightTAE decoder.

Diverges from the chunk2 base: ``additional_concat_ch=192``, ``len_t=4``,
``window_size_t=8``, the chunk4 checkpoint, the PixelShuffle per-AR-step
encoder, and ``image_encoder`` on the standard "vae" checkpoint.
"""

MV_2STEPS_CHUNK4_LOC8_PSHUFFLE_LIGHTTAE = cast(
    OmnidreamsPipelineConfig,
    derive_config(
        SV_2STEPS_CHUNK4_LOC8_PSHUFFLE_LIGHTTAE,
        name="omnidreams-mv-2steps-chunk4-loc8-pshuffle-lighttae",
        diffusion_model=dict(
            transformer=dict(
                network=dict(enable_cross_view_attn=True),
                checkpoint_path=AVAILABLE_OMNIDREAMS_CHECKPOINT_PATHS[
                    "4view-pshuffle-chunk4"
                ],
                num_views=4,
            ),
        ),
    ),
)  # ty:ignore[redundant-cast]
"""4-view, chunk4, PixelShuffle HDMap encoder + LightTAE decoder."""


SV_35STEPS_CHUNK2_LOC24_COSMOS2_2B_RES720P_30FPS_HDMAP_VAE_MADS1M = OmnidreamsPipelineConfig(
    name="omnidreams-sv-35steps-chunk2-loc24-cosmos2-2b-res720p-30fps-hdmap-vae-mads1m",
    text_encoder=CosmosReason1TextEncoderConfig(),
    image_encoder=WanVAEEncoderConfig(
        checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"],
    ),
    enable_sync_and_profile=True,
    encoder=WanVAEEncoderConfig(
        checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"],
    ),
    decoder=WanVAEDecoderConfig(
        checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"],
    ),
    diffusion_model=DiffusionModelConfig(
        seed=1,
        context_noise=128,
        transformer=CosmosTransformerConfig(
            network=CosmosDiTNetworkConfig(
                additional_concat_ch=16,
                enable_cross_view_attn=False,
                cp_method="ring",
            ),
            checkpoint_path=AVAILABLE_OMNIDREAMS_CHECKPOINT_PATHS[
                "1view-diffusion-forcing-chunk2"
            ],
            batch_shape=(1,),
            num_views=1,
            len_t=2,
            h_extrapolation_ratio=3.0,
            w_extrapolation_ratio=3.0,
            window_size_t=24,
            sink_size_t=0,
            compile_network=True,
            use_cuda_graph=True,
            skip_finalize_kv_cache=False,
            guidance_scale=3.0,
        ),
        scheduler=FlowMatchUniPCSchedulerConfig(
            num_inference_steps=35,
            shift=5.0,
        ),
    ),
)
"""Teacher: omnidreams diffusion-forcing causal AR (2B / 720p / chunk2 UniPC).

``state_t=24`` = 12 chunk2 latent blocks (93 decoded Wan frames). CFG on
(``guidance_scale=3.0``); 35-step UniPC (``shift=5.0``).
"""

SV_35STEPS_CHUNK48_LOC48_COSMOS2_2B_RES720P_30FPS_HDMAP_VAE_MADS1M = cast(
    OmnidreamsPipelineConfig,
    derive_config(
        SV_35STEPS_CHUNK2_LOC24_COSMOS2_2B_RES720P_30FPS_HDMAP_VAE_MADS1M,
        name="omnidreams-sv-35steps-chunk48-loc48-cosmos2-2b-res720p-30fps-hdmap-vae-mads1m",
        diffusion_model=dict(
            seed=1,
            context_noise=0,
            transformer=dict(
                checkpoint_path=AVAILABLE_OMNIDREAMS_CHECKPOINT_PATHS[
                    "1view-bidirectional-chunk48"
                ],
                len_t=48,
                window_size_t=48,
                skip_finalize_kv_cache=True,
            ),
        ),
    ),
)  # ty:ignore[redundant-cast]
"""Teacher: omnidreams bidirectional (single-view / 2B / 720p / chunk48 UniPC).

``len_t == window_size_t == 48`` -> one AR step for the whole video;
``skip_finalize_kv_cache=True`` (no KV cache to advance after one rollout).
"""


# Experiments: ablations on the chunk2 perf chassis. ``noise*`` variants vary
# the terminal denoising timestep (``noise350`` -> ``[1000, 350]``) to study
# the skip-finalize-kv-cache ablation.

EXPERIMENT1_BASELINE = cast(
    OmnidreamsPipelineConfig,
    derive_config(
        SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE_PERF,
        name="omnidreams-experiment1-baseline",
    ),
)  # ty:ignore[redundant-cast]

EXPERIMENT1_SKIP_FINALIZE_KV_CACHE = cast(
    OmnidreamsPipelineConfig,
    derive_config(
        SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE_PERF,
        name="omnidreams-experiment1-skip-finalize-kv-cache",
        diffusion_model=dict(
            transformer=dict(skip_finalize_kv_cache=True),
        ),
    ),
)  # ty:ignore[redundant-cast]

EXPERIMENT1_SKIP_FINALIZE_KV_CACHE_NOISE350 = cast(
    OmnidreamsPipelineConfig,
    derive_config(
        EXPERIMENT1_SKIP_FINALIZE_KV_CACHE,
        name="omnidreams-experiment1-skip-finalize-kv-cache-noise350",
        diffusion_model=dict(
            scheduler=dict(denoising_timesteps=[1000, 350]),
        ),
    ),
)  # ty:ignore[redundant-cast]

EXPERIMENT1_SKIP_FINALIZE_KV_CACHE_NOISE250 = cast(
    OmnidreamsPipelineConfig,
    derive_config(
        EXPERIMENT1_SKIP_FINALIZE_KV_CACHE,
        name="omnidreams-experiment1-skip-finalize-kv-cache-noise250",
        diffusion_model=dict(
            scheduler=dict(denoising_timesteps=[1000, 250]),
        ),
    ),
)  # ty:ignore[redundant-cast]

EXPERIMENT1_SKIP_FINALIZE_KV_CACHE_NOISE150 = cast(
    OmnidreamsPipelineConfig,
    derive_config(
        EXPERIMENT1_SKIP_FINALIZE_KV_CACHE,
        name="omnidreams-experiment1-skip-finalize-kv-cache-noise150",
        diffusion_model=dict(
            scheduler=dict(denoising_timesteps=[1000, 150]),
        ),
    ),
)  # ty:ignore[redundant-cast]

EXPERIMENT1_SKIP_FINALIZE_KV_CACHE_NOISE100 = cast(
    OmnidreamsPipelineConfig,
    derive_config(
        EXPERIMENT1_SKIP_FINALIZE_KV_CACHE,
        name="omnidreams-experiment1-skip-finalize-kv-cache-noise100",
        diffusion_model=dict(
            scheduler=dict(denoising_timesteps=[1000, 100]),
        ),
    ),
)  # ty:ignore[redundant-cast]


OMNIDREAMS_CONFIGS: dict[str, OmnidreamsPipelineConfig] = {
    cfg.name: cfg
    for cfg in (
        SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE,
        SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE_PERF,
        SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE_NATIVE_PERF,
        SV_2STEPS_CHUNK2_LOC6_VAE_VAE,
        SV_2STEPS_CHUNK3_LOC6_VAE_VAE,
        SV_2STEPS_CHUNK4_LOC8_PSHUFFLE_LIGHTTAE,
        MV_2STEPS_CHUNK4_LOC8_PSHUFFLE_LIGHTTAE,
        SV_35STEPS_CHUNK2_LOC24_COSMOS2_2B_RES720P_30FPS_HDMAP_VAE_MADS1M,
        SV_35STEPS_CHUNK48_LOC48_COSMOS2_2B_RES720P_30FPS_HDMAP_VAE_MADS1M,
        EXPERIMENT1_BASELINE,
        EXPERIMENT1_SKIP_FINALIZE_KV_CACHE,
        EXPERIMENT1_SKIP_FINALIZE_KV_CACHE_NOISE350,
        EXPERIMENT1_SKIP_FINALIZE_KV_CACHE_NOISE250,
        EXPERIMENT1_SKIP_FINALIZE_KV_CACHE_NOISE150,
        EXPERIMENT1_SKIP_FINALIZE_KV_CACHE_NOISE100,
    )
}
"""All shipped Omnidreams variants, keyed by ``name``."""


## Scene-driving application binding

DEFAULT_PROMPT_1V = (
    "Driving scene from a front-facing car camera. Urban environment with roads, "
    "vehicles, pedestrians, traffic signs, and buildings. Clear visibility, "
    "realistic lighting, photorealistic quality. High resolution dashcam footage "
    "of city driving."
)
"""Default prompt for single-view OmniDreams variants."""

DEFAULT_PROMPT_4V = (
    "Wide-angle urban street scene from a low, dashboard-level viewpoint. "
    "A straight two-lane road with a faded center line and curbside parking on "
    "both sides. Parked sedans and SUVs in neutral colors line the curbs. On the "
    "right, a white stucco mid-rise building with blue fabric awnings, rectangular "
    "windows, and small storefronts at street level. On the left, a low commercial "
    "strip with dark trim, glass fronts, signage, and shaded sidewalks. Mature green "
    "trees punctuate both sides. Clear blue sky with sparse soft clouds. Bright midday "
    "sunlight, natural colors, realistic materials, crisp shadows, clean asphalt texture."
)
"""Default prompt for multi-view OmniDreams variants."""


def _create_omnidreams_backend(config: AppConfig) -> RenderBackend:
    """Create the raster or OmniDreams world-model backend."""
    if config.backend == "raster":
        return RasterRenderBackend(config.chunk, config.raster, bev=config.bev)
    if config.manifest_path is None:
        raise ValueError("OmniDreams requires a world-model manifest.")
    return WorldModelRenderBackend(
        manifest=load_world_model_manifest(config.manifest_path),
        chunk=config.chunk,
        raster=config.raster,
        bev=config.bev,
        offload_text_encoder=config.world_model_offload_text_encoder,
        postprocess=config.postprocess,
        session_factory=FlashdreamsWorldModelSession,
    )


OMNIDREAMS_APPLICATION_DEFAULTS = Hdmap2VApplicationDefaults(
    title="OmniDreams",
    slug="hdmap2v",
    backend="world_model",
    total_blocks=60,
)
"""Model-owned defaults for the OmniDreams scene-driving application."""

OMNIDREAMS_APPLICATION_HOOKS = Hdmap2VApplicationHooks(
    backend_factory=_create_omnidreams_backend,
    manifest_loader=load_world_model_manifest,
    manifest_resolver=resolve_world_model_manifest_path,
)
"""Model-owned hooks for the OmniDreams scene-driving application."""

DEFAULT_INTERACTIVE_DRIVE_SCENE_UUID = "0d404ff7-2b66-498c-b047-1ed8cded60d4"
"""Default scene used by the interactive-drive application."""


def resolve_default_interactive_drive_scene() -> Path:
    """Download and return the default interactive-drive scene."""
    return hf_hub_download_scene(DEFAULT_INTERACTIVE_DRIVE_SCENE_UUID)
