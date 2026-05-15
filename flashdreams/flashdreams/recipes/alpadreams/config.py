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

"""User-facing configs for Alpadreams.

Hosts both the pre-built :class:`AlpadreamsPipelineConfig` literals
and the per-slug :class:`AlpadreamsRunnerConfig` literals that drive
``flashdreams-run``. Per-variant runner literals are projected from
``ALPADREAMS_CONFIGS`` by :func:`_build_alpadreams_runners` so adding
a new pipeline only requires registering a CLI description below. The
runner-config literals self-register with
:mod:`flashdreams.configs.registry` at import time.
"""

from __future__ import annotations

from typing import cast

from flashdreams.configs.registry import register_runner
from flashdreams.core.io.internal import use_internal_storage
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
from flashdreams.infra.runner import RunnerConfig
from flashdreams.recipes.alpadreams.encoder.pixel_shuffle import (
    PixelShuffleVAEEncoderConfig,
)
from flashdreams.recipes.alpadreams.hf import omni_dreams_hf_url
from flashdreams.recipes.alpadreams.pipeline import (
    AlpadreamsPipelineConfig,
)
from flashdreams.recipes.alpadreams.runner import AlpadreamsRunnerConfig
from flashdreams.recipes.alpadreams.transformer import CosmosTransformerConfig
from flashdreams.recipes.alpadreams.transformer.impl.network import (
    CosmosDiTNetworkConfig,
)
from flashdreams.recipes.taehv import (
    AVAILABLE_TAEHV_CHECKPOINT_PATHS,
    TeahvVAEDecoderConfig,
)
from flashdreams.recipes.wan.autoencoder.vae import (
    AVAILABLE_WAN_VAE_CHECKPOINT_PATHS,
    WanVAEDecoderConfig,
    WanVAEEncoderConfig,
)

_INTERNAL_ALPADREAMS_CHECKPOINT_PATHS: dict[str, str] = {
    "1view-pshuffle-chunk4": "s3://flashdreams/assets/checkpoints/alpadreams/16N@cosmos_v2_2b_SF_res720p_30fps_i2v_hdmap_chunk4_pixel_shuffle_resume.pt",
    "1view-vae-chunk2": "s3://flashdreams/assets/checkpoints/alpadreams/32n_cosmos_v2_2b_SF_res720p_30fps_i2v_hdmap_chunk2_vae_encode_189f_loc6_sft_urban_stationary_mixed_gcp_student_resume.pt",
    "1view-vae-chunk3": "s3://flashdreams/assets/checkpoints/alpadreams/32n_cosmos_v2_2b_SF_res720p_30fps_i2v_hdmap_chunk3_vae_encode_loc6_gcp.pt",
    "4view-pshuffle-chunk4": "s3://flashdreams/assets/checkpoints/alpadreams/32n_cosmos_v2_2b_SF_4view_res720p_fps30_chunk4_i2v_hdmap_pixel_shuffle_loc8st2_gcp.pt",
    "4view-vae-chunk4": "s3://flashdreams/assets/checkpoints/alpadreams/32n_cosmos_v2_2b_SF_4view_res720p_fps30_chunk4_i2v_hdmap_vae_encoding_loc8st2_gcp.pt",
    "1view-diffusion-forcing-chunk2": "s3://flashdreams/assets/checkpoints/alpadreams/16N@causal_cosmos2_2B_res720p_30fps_hdmap_hdmap_pretrained_chunk2_vae_mads1m_1080p@20260225100739_000010600.pt",
    "1view-bidirectional-chunk48": "s3://flashdreams/assets/checkpoints/alpadreams/32N@teacher_cosmos2_2B_res720p_30fps_hdmap_vae_mads1m_189frames_1080p@20260309090017_000005000.pt",
}

# HF mirrors override the s3 URLs above for slugs that have been mirrored.
# Unmirrored slugs fall through to s3 so recipe configs still import; mirror
# new slugs here as they land on HF.
_PUBLIC_ALPADREAMS_CHECKPOINT_PATHS: dict[str, str] = {
    "1view-vae-chunk2": omni_dreams_hf_url(
        "omni-dreams-models",
        "resolve/main/single_view/2b_res720p_30fps_i2v_hdmap_distilled.pt",
    ),
}

AVAILABLE_ALPADREAMS_CHECKPOINT_PATHS: dict[str, str] = (
    dict(_INTERNAL_ALPADREAMS_CHECKPOINT_PATHS)
    if use_internal_storage()
    else {
        **_INTERNAL_ALPADREAMS_CHECKPOINT_PATHS,
        **_PUBLIC_ALPADREAMS_CHECKPOINT_PATHS,
    }
)
"""Resolved at module import; set ``FLASHDREAMS_INTERNAL_STORAGE`` first."""

SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE = AlpadreamsPipelineConfig(
    recipe_name="alpadreams-sv-2steps-chunk2-loc6-lightvae-lighttae",
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
            ),
            checkpoint_path=AVAILABLE_ALPADREAMS_CHECKPOINT_PATHS["1view-vae-chunk2"],
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
"""Base: single-view, chunk2, light Wan VAE HDMap encoder + LightTAE decoder.

The reference Self-Forcing distilled chassis: 2-step flow-match
scheduler, ``len_t=2``, ``window_size_t=6``, CFG off, no
``skip_finalize_kv_cache``. Every chunk2 variant derives from this
one and flips a small set of fields.
"""

SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE_PERF = cast(
    AlpadreamsPipelineConfig,
    derive_config(
        SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE,
        recipe_name="alpadreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf",
        image_encoder=dict(use_compile=True, use_cuda_graph=True),
        encoder=dict(use_compile=True, use_cuda_graph=True),
        decoder=dict(use_compile=True, use_cuda_graph=True),
    ),
)  # ty:ignore[redundant-cast]
"""Performance-tuned variant: enable ``use_compile`` / ``use_cuda_graph``
on the image encoder, the per-AR-step encoder, and the decoder."""

SV_2STEPS_CHUNK2_LOC6_VAE_VAE = cast(
    AlpadreamsPipelineConfig,
    derive_config(
        SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE,
        recipe_name="alpadreams-sv-2steps-chunk2-loc6-vae-vae",
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
    AlpadreamsPipelineConfig,
    derive_config(
        SV_2STEPS_CHUNK2_LOC6_VAE_VAE,
        recipe_name="alpadreams-sv-2steps-chunk3-loc6-vae-vae",
        diffusion_model=dict(
            transformer=dict(
                checkpoint_path=AVAILABLE_ALPADREAMS_CHECKPOINT_PATHS[
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
    AlpadreamsPipelineConfig,
    derive_config(
        SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE,
        recipe_name="alpadreams-sv-2steps-chunk4-loc8-pshuffle-lighttae",
        image_encoder=dict(checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"]),
        encoder=PixelShuffleVAEEncoderConfig(),
        diffusion_model=dict(
            transformer=dict(
                network=dict(additional_concat_ch=192),
                checkpoint_path=AVAILABLE_ALPADREAMS_CHECKPOINT_PATHS[
                    "1view-pshuffle-chunk4"
                ],
                len_t=4,
                window_size_t=8,
            ),
        ),
    ),
)  # ty:ignore[redundant-cast]
"""Single-view, chunk4, PixelShuffle HDMap encoder + LightTAE decoder.

Diverges from the chunk2 base on (a) ``additional_concat_ch=192`` for
the PixelShuffle branch, (b) ``len_t=4``, (c) ``window_size_t=8``,
(d) the chunk4 checkpoint, and (e) the per-AR-step encoder is the
:class:`PixelShuffleVAEEncoderConfig` instead of a Wan VAE encoder.
``image_encoder`` reverts to the standard "vae" checkpoint.
"""

MV_2STEPS_CHUNK4_LOC8_PSHUFFLE_LIGHTTAE = cast(
    AlpadreamsPipelineConfig,
    derive_config(
        SV_2STEPS_CHUNK4_LOC8_PSHUFFLE_LIGHTTAE,
        recipe_name="alpadreams-mv-2steps-chunk4-loc8-pshuffle-lighttae",
        diffusion_model=dict(
            transformer=dict(
                network=dict(enable_cross_view_attn=True),
                checkpoint_path=AVAILABLE_ALPADREAMS_CHECKPOINT_PATHS[
                    "4view-pshuffle-chunk4"
                ],
                num_views=4,
            ),
        ),
    ),
)  # ty:ignore[redundant-cast]
"""4-view, chunk4, PixelShuffle HDMap encoder + LightTAE decoder."""


SV_35STEPS_CHUNK2_LOC24_COSMOS2_2B_RES720P_30FPS_HDMAP_VAE_MADS1M = AlpadreamsPipelineConfig(
    recipe_name="alpadreams-sv-35steps-chunk2-loc24-cosmos2-2b-res720p-30fps-hdmap-vae-mads1m",
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
            ),
            checkpoint_path=AVAILABLE_ALPADREAMS_CHECKPOINT_PATHS[
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
"""Teacher: alpadreams diffusion-forcing causal AR (2B / 720p / chunk2 UniPC).

``state_t=24``: 12 chunk2 latent blocks, or 93 decoded frames with
the Wan decoder. CFG on (``guidance_scale=3.0``); 35-step UniPC
scheduler (``shift=5.0``).
"""

SV_35STEPS_CHUNK48_LOC48_COSMOS2_2B_RES720P_30FPS_HDMAP_VAE_MADS1M = cast(
    AlpadreamsPipelineConfig,
    derive_config(
        SV_35STEPS_CHUNK2_LOC24_COSMOS2_2B_RES720P_30FPS_HDMAP_VAE_MADS1M,
        recipe_name="alpadreams-sv-35steps-chunk48-loc48-cosmos2-2b-res720p-30fps-hdmap-vae-mads1m",
        diffusion_model=dict(
            seed=1,
            context_noise=0,
            transformer=dict(
                checkpoint_path=AVAILABLE_ALPADREAMS_CHECKPOINT_PATHS[
                    "1view-bidirectional-chunk48"
                ],
                len_t=48,
                window_size_t=48,
                skip_finalize_kv_cache=True,
            ),
        ),
    ),
)  # ty:ignore[redundant-cast]
"""Teacher: alpadreams bidirectional (single-view / 2B / 720p / chunk48 UniPC).

``len_t == window_size_t == 48`` -> single-AR-step rollout for the
whole 48-chunk video. ``skip_finalize_kv_cache=True`` because the
bidirectional teacher doesn't need to advance the KV cache after the
one rollout it ever does.
"""


## Experiments: ablations on top of the chunk2 perf chassis
#
# ``experiment1_baseline`` re-publishes the perf config under a stable
# experiment slug (same fields). The ``noise*`` variants vary the
# terminal denoising timestep (``[1000, T2]``) to study the
# skip-KV-cache-finalize ablation; the field name reflects the second
# timestep (``noise350`` -> ``[1000, 350]``).

EXPERIMENT1_BASELINE = cast(
    AlpadreamsPipelineConfig,
    derive_config(
        SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE_PERF,
        recipe_name="alpadreams-experiment1-baseline",
    ),
)  # ty:ignore[redundant-cast]

EXPERIMENT1_SKIP_FINALIZE_KV_CACHE = cast(
    AlpadreamsPipelineConfig,
    derive_config(
        SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE_PERF,
        recipe_name="alpadreams-experiment1-skip-finalize-kv-cache",
        diffusion_model=dict(
            transformer=dict(skip_finalize_kv_cache=True),
        ),
    ),
)  # ty:ignore[redundant-cast]

EXPERIMENT1_SKIP_FINALIZE_KV_CACHE_NOISE350 = cast(
    AlpadreamsPipelineConfig,
    derive_config(
        EXPERIMENT1_SKIP_FINALIZE_KV_CACHE,
        recipe_name="alpadreams-experiment1-skip-finalize-kv-cache-noise350",
        diffusion_model=dict(
            scheduler=dict(denoising_timesteps=[1000, 350]),
        ),
    ),
)  # ty:ignore[redundant-cast]

EXPERIMENT1_SKIP_FINALIZE_KV_CACHE_NOISE250 = cast(
    AlpadreamsPipelineConfig,
    derive_config(
        EXPERIMENT1_SKIP_FINALIZE_KV_CACHE,
        recipe_name="alpadreams-experiment1-skip-finalize-kv-cache-noise250",
        diffusion_model=dict(
            scheduler=dict(denoising_timesteps=[1000, 250]),
        ),
    ),
)  # ty:ignore[redundant-cast]

EXPERIMENT1_SKIP_FINALIZE_KV_CACHE_NOISE150 = cast(
    AlpadreamsPipelineConfig,
    derive_config(
        EXPERIMENT1_SKIP_FINALIZE_KV_CACHE,
        recipe_name="alpadreams-experiment1-skip-finalize-kv-cache-noise150",
        diffusion_model=dict(
            scheduler=dict(denoising_timesteps=[1000, 150]),
        ),
    ),
)  # ty:ignore[redundant-cast]

EXPERIMENT1_SKIP_FINALIZE_KV_CACHE_NOISE100 = cast(
    AlpadreamsPipelineConfig,
    derive_config(
        EXPERIMENT1_SKIP_FINALIZE_KV_CACHE,
        recipe_name="alpadreams-experiment1-skip-finalize-kv-cache-noise100",
        diffusion_model=dict(
            scheduler=dict(denoising_timesteps=[1000, 100]),
        ),
    ),
)  # ty:ignore[redundant-cast]


ALPADREAMS_CONFIGS: dict[str, AlpadreamsPipelineConfig] = {
    cfg.recipe_name: cfg
    for cfg in (
        SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE,
        SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE_PERF,
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
"""All shipped Alpadreams variants, keyed by ``recipe_name``."""


## Per-variant runner-config literals (slug == ``recipe_name``).

_DEFAULT_PROMPT_1V = (
    "Driving scene from a front-facing car camera. Urban environment with roads, "
    "vehicles, pedestrians, traffic signs, and buildings. Clear visibility, "
    "realistic lighting, photorealistic quality. High resolution dashcam footage "
    "of city driving."
)
_DEFAULT_PROMPT_4V = (
    "Wide-angle urban street scene from a low, dashboard-level viewpoint. "
    "A straight two-lane road with a faded center line and curbside parking on "
    "both sides. Parked sedans and SUVs in neutral colors line the curbs. On the "
    "right, a white stucco mid-rise building with blue fabric awnings, rectangular "
    "windows, and small storefronts at street level. On the left, a low commercial "
    "strip with dark trim, glass fronts, signage, and shaded sidewalks. Mature green "
    "trees punctuate both sides. Clear blue sky with sparse soft clouds. Bright midday "
    "sunlight, natural colors, realistic materials, crisp shadows, clean asphalt texture."
)

_ALPADREAMS_DESCRIPTIONS: dict[str, str] = {
    "alpadreams-sv-2steps-chunk2-loc6-lightvae-lighttae": (
        "Single-view 2-step distilled chunk2 (LightVAE + LightTAE)."
    ),
    "alpadreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf": (
        "Single-view chunk2 perf preset (compile + CUDA graphs across all stages)."
    ),
    "alpadreams-sv-2steps-chunk2-loc6-vae-vae": (
        "Single-view chunk2 with the full Wan VAE on encoder + decoder."
    ),
    "alpadreams-sv-2steps-chunk3-loc6-vae-vae": (
        "Single-view chunk3 (len_t=3) with the full Wan VAE."
    ),
    "alpadreams-sv-2steps-chunk4-loc8-pshuffle-lighttae": (
        "Single-view chunk4 with the PixelShuffle HDMap encoder + LightTAE."
    ),
    "alpadreams-mv-2steps-chunk4-loc8-pshuffle-lighttae": (
        "4-camera multi-view chunk4 (PixelShuffle HDMap + LightTAE)."
    ),
    "alpadreams-sv-35steps-chunk2-loc24-cosmos2-2b-res720p-30fps-hdmap-vae-mads1m": (
        "Teacher: single-view 35-step UniPC chunk2 (Cosmos2 2B, 720p, CFG=3.0)."
    ),
    "alpadreams-sv-35steps-chunk48-loc48-cosmos2-2b-res720p-30fps-hdmap-vae-mads1m": (
        "Teacher: single-view 35-step bidirectional chunk48 (one rollout, 720p)."
    ),
    "alpadreams-experiment1-baseline": (
        "Experiment-1 baseline (re-publishes the chunk2 perf chassis)."
    ),
    "alpadreams-experiment1-skip-finalize-kv-cache": (
        "Experiment-1: skip-finalize-kv-cache ablation."
    ),
    "alpadreams-experiment1-skip-finalize-kv-cache-noise350": (
        "Experiment-1: skip-finalize + denoising_timesteps=[1000, 350]."
    ),
    "alpadreams-experiment1-skip-finalize-kv-cache-noise250": (
        "Experiment-1: skip-finalize + denoising_timesteps=[1000, 250]."
    ),
    "alpadreams-experiment1-skip-finalize-kv-cache-noise150": (
        "Experiment-1: skip-finalize + denoising_timesteps=[1000, 150]."
    ),
    "alpadreams-experiment1-skip-finalize-kv-cache-noise100": (
        "Experiment-1: skip-finalize + denoising_timesteps=[1000, 100]."
    ),
}
"""Per-variant CLI descriptions, keyed by ``recipe_name``."""


def _build_alpadreams_runners() -> dict[str, RunnerConfig]:
    """Project ``ALPADREAMS_CONFIGS`` into per-variant runner literals."""
    runners: dict[str, RunnerConfig] = {}
    for name, pipeline_cfg in ALPADREAMS_CONFIGS.items():
        transformer_cfg = pipeline_cfg.diffusion_model.transformer
        assert isinstance(transformer_cfg, CosmosTransformerConfig)
        prompt = (
            _DEFAULT_PROMPT_4V if transformer_cfg.num_views == 4 else _DEFAULT_PROMPT_1V
        )
        assert name in _ALPADREAMS_DESCRIPTIONS, (
            f"missing CLI description for alpadreams slug {name!r}; "
            "add an entry to ``_ALPADREAMS_DESCRIPTIONS``."
        )
        runners[name] = AlpadreamsRunnerConfig(
            runner_name=name,
            description=_ALPADREAMS_DESCRIPTIONS[name],
            pipeline=pipeline_cfg,
            prompt=prompt,
        )
    return runners


ALPADREAMS_RUNNERS: dict[str, RunnerConfig] = _build_alpadreams_runners()
"""All shipped Alpadreams runners (single- and multi-view variants),
keyed by ``runner_name``."""

for _name, _cfg in ALPADREAMS_RUNNERS.items():
    register_runner(_name, _cfg, source="builtin")
