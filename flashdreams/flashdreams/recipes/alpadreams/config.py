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

"""Pipeline-config builders for Alpadreams."""

from __future__ import annotations

from collections.abc import Callable

from flashdreams.infra.diffusion.model import DiffusionModelConfig
from flashdreams.infra.diffusion.scheduler.fm import (
    FlowMatchSchedulerConfig,
)
from flashdreams.infra.diffusion.scheduler.fm_unipc import (
    FlowMatchUniPCSchedulerConfig,
)
from flashdreams.infra.encoder.text.cosmos_qwen import (
    CosmosReason1TextEncoderConfig,
)
from flashdreams.recipes.alpadreams.encoder.pixel_shuffle import (
    PixelShuffleVAEEncoderConfig,
)
from flashdreams.recipes.alpadreams.pipeline import (
    AlpadreamsPipelineConfig,
)
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

AVAILABLE_ALPADREAMS_CHECKPOINT_PATHS: dict[str, str] = {
    "1view-pshuffle-chunk4": "s3://flashdreams/assets/checkpoints/alpadreams/16N@cosmos_v2_2b_SF_res720p_30fps_i2v_hdmap_chunk4_pixel_shuffle_resume.pt",
    "1view-vae-chunk2": "s3://flashdreams/assets/checkpoints/alpadreams/32n_cosmos_v2_2b_SF_res720p_30fps_i2v_hdmap_chunk2_vae_encode_189f_loc6_sft_urban_stationary_mixed_gcp_student_resume.pt",
    "1view-vae-chunk3": "s3://flashdreams/assets/checkpoints/alpadreams/32n_cosmos_v2_2b_SF_res720p_30fps_i2v_hdmap_chunk3_vae_encode_loc6_gcp.pt",
    "4view-pshuffle-chunk4": "s3://flashdreams/assets/checkpoints/alpadreams/32n_cosmos_v2_2b_SF_4view_res720p_fps30_chunk4_i2v_hdmap_pixel_shuffle_loc8st2_gcp.pt",
    "4view-vae-chunk4": "s3://flashdreams/assets/checkpoints/alpadreams/32n_cosmos_v2_2b_SF_4view_res720p_fps30_chunk4_i2v_hdmap_vae_encoding_loc8st2_gcp.pt",
    "1view-diffusion-forcing-chunk2": "s3://flashdreams/assets/checkpoints/alpadreams/16N@causal_cosmos2_2B_res720p_30fps_hdmap_hdmap_pretrained_chunk2_vae_mads1m_1080p@20260225100739_000010600.pt",
    "1view-bidirectional-chunk48": "s3://flashdreams/assets/checkpoints/alpadreams/32N@teacher_cosmos2_2B_res720p_30fps_hdmap_vae_mads1m_189frames_1080p@20260309090017_000005000.pt",
}


# ---------------------------------------------------------------------------
# Canonical alpadreams 720p defaults (60 latent rows / 80 latent cols at 720p
# after the 8x VAE spatial compression).
# ---------------------------------------------------------------------------

_DEFAULT_BATCH_SHAPE: tuple[int, ...] = (1,)
_DEFAULT_VIDEO_HEIGHT = 704
_DEFAULT_VIDEO_WIDTH = 1280
_WAN_VAE_SPATIAL_COMPRESSION = 8
_DEFAULT_DENOISING_TIMESTEPS = [1000, 450]
_DEFAULT_NUM_TRAIN_TIMESTEPS = 1000


def _wan_vae_decoder_config(
    *, use_compile: bool = False, use_cuda_graph: bool = True
) -> WanVAEDecoderConfig:
    return WanVAEDecoderConfig(
        checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"],
        use_compile=use_compile,
        use_cuda_graph=use_cuda_graph,
    )


def _teahv_vae_decoder_config(
    *, use_compile: bool = False, use_cuda_graph: bool = True
) -> TeahvVAEDecoderConfig:
    return TeahvVAEDecoderConfig(
        checkpoint_path=AVAILABLE_TAEHV_CHECKPOINT_PATHS["lighttae"],
        use_compile=use_compile,
        use_cuda_graph=use_cuda_graph,
    )


def _scheduler_config(
    denoising_timesteps: list[int] = _DEFAULT_DENOISING_TIMESTEPS,
) -> FlowMatchSchedulerConfig:
    """Alpadreams 2-step Self-Forcing flow-match scheduler defaults."""
    return FlowMatchSchedulerConfig(
        num_inference_steps=len(denoising_timesteps),
        denoising_timesteps=denoising_timesteps,
        warp_denoising_step=True,
        shift=5.0,
        sigma_min=0.0,
        extra_one_step=True,
        num_train_timesteps=_DEFAULT_NUM_TRAIN_TIMESTEPS,
    )


def _transformer_config(
    *,
    checkpoint_path: str,
    cp_size: int,
    num_views: int,
    len_t_latent: int,
    window_size_t: int,
    encode_with_pixel_shuffle: bool,
    guidance_scale: float = 1.0,
    skip_finalize_kv_cache: bool = False,
    compile_network: bool = True,
    use_cuda_graph: bool = True,
) -> CosmosTransformerConfig:
    return CosmosTransformerConfig(
        network=CosmosDiTNetworkConfig(),
        checkpoint_path=checkpoint_path,
        batch_shape=_DEFAULT_BATCH_SHAPE,
        num_views=num_views,
        height=_DEFAULT_VIDEO_HEIGHT // _WAN_VAE_SPATIAL_COMPRESSION,
        width=_DEFAULT_VIDEO_WIDTH // _WAN_VAE_SPATIAL_COMPRESSION,
        len_t=len_t_latent,
        cp_size=cp_size,
        enable_hdmap_condition=True,
        encode_with_pixel_shuffle=encode_with_pixel_shuffle,
        h_extrapolation_ratio=3.0,
        w_extrapolation_ratio=3.0,
        window_size_t=window_size_t,
        sink_size_t=0,
        compile_network=compile_network,
        use_cuda_graph=use_cuda_graph,
        skip_finalize_kv_cache=skip_finalize_kv_cache,
        guidance_scale=guidance_scale,
    )


def _wan_vae_encoder(
    *,
    checkpoint_name: str = "vae",
    use_compile: bool = False,
    use_cuda_graph: bool = True,
) -> WanVAEEncoderConfig:
    return WanVAEEncoderConfig(
        checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS[checkpoint_name],
        use_compile=use_compile,
        use_cuda_graph=use_cuda_graph,
    )


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def build_sv_2steps_chunk2_loc6_lightvae_lighttae(
    *,
    cp_size: int = 1,
    compile_network: bool = True,
    seed: int = 42,
) -> AlpadreamsPipelineConfig:
    """Single-view, chunk2, light Wan VAE HDMap encoder + LightTAE decoder."""
    return AlpadreamsPipelineConfig(
        text_encoder=CosmosReason1TextEncoderConfig(),
        image_encoder=_wan_vae_encoder(checkpoint_name="lightvae"),
        enable_sync_and_profile=True,
        encoder=_wan_vae_encoder(checkpoint_name="lightvae"),
        decoder=_teahv_vae_decoder_config(),
        diffusion_model=DiffusionModelConfig(
            seed=seed,
            context_noise=128,
            transformer=_transformer_config(
                checkpoint_path=AVAILABLE_ALPADREAMS_CHECKPOINT_PATHS[
                    "1view-vae-chunk2"
                ],
                cp_size=cp_size,
                compile_network=compile_network,
                num_views=1,
                len_t_latent=2,
                window_size_t=6,
                encode_with_pixel_shuffle=False,
            ),
            scheduler=_scheduler_config(),
        ),
    )


# Performance optimized version of the above config
def build_sv_2steps_chunk2_loc6_lightvae_lighttae_perf(
    *,
    cp_size: int = 1,
    compile_network: bool = True,
    seed: int = 42,
    skip_finalize_kv_cache: bool = False,
    denoising_timesteps: list[int] = _DEFAULT_DENOISING_TIMESTEPS,
) -> AlpadreamsPipelineConfig:
    """Single-view, chunk2, light Wan VAE HDMap encoder + LightTAE decoder."""
    return AlpadreamsPipelineConfig(
        text_encoder=CosmosReason1TextEncoderConfig(),
        image_encoder=_wan_vae_encoder(
            checkpoint_name="lightvae", use_compile=True, use_cuda_graph=True
        ),
        enable_sync_and_profile=True,
        encoder=_wan_vae_encoder(
            checkpoint_name="lightvae", use_compile=True, use_cuda_graph=True
        ),
        decoder=_teahv_vae_decoder_config(use_compile=True, use_cuda_graph=True),
        diffusion_model=DiffusionModelConfig(
            seed=seed,
            context_noise=128,
            transformer=_transformer_config(
                checkpoint_path=AVAILABLE_ALPADREAMS_CHECKPOINT_PATHS[
                    "1view-vae-chunk2"
                ],
                cp_size=cp_size,
                compile_network=compile_network,
                num_views=1,
                len_t_latent=2,
                window_size_t=6,
                encode_with_pixel_shuffle=False,
                skip_finalize_kv_cache=skip_finalize_kv_cache,
            ),
            scheduler=_scheduler_config(denoising_timesteps=denoising_timesteps),
        ),
    )


def build_sv_2steps_chunk2_loc6_vae_vae(
    *,
    cp_size: int = 1,
    compile_network: bool = True,
    seed: int = 42,
) -> AlpadreamsPipelineConfig:
    """Single-view, chunk2, full Wan VAE for both HDMap encoding and decoding."""
    return AlpadreamsPipelineConfig(
        text_encoder=CosmosReason1TextEncoderConfig(),
        image_encoder=_wan_vae_encoder(),
        enable_sync_and_profile=True,
        encoder=_wan_vae_encoder(),
        decoder=_wan_vae_decoder_config(),
        diffusion_model=DiffusionModelConfig(
            seed=seed,
            context_noise=128,
            transformer=_transformer_config(
                checkpoint_path=AVAILABLE_ALPADREAMS_CHECKPOINT_PATHS[
                    "1view-vae-chunk2"
                ],
                cp_size=cp_size,
                compile_network=compile_network,
                num_views=1,
                len_t_latent=2,
                window_size_t=6,
                encode_with_pixel_shuffle=False,
            ),
            scheduler=_scheduler_config(),
        ),
    )


def build_sv_2steps_chunk3_loc6_vae_vae(
    *,
    cp_size: int = 1,
    compile_network: bool = True,
    seed: int = 42,
) -> AlpadreamsPipelineConfig:
    """Single-view, chunk3, full Wan VAE for both HDMap encoding and decoding."""
    return AlpadreamsPipelineConfig(
        text_encoder=CosmosReason1TextEncoderConfig(),
        image_encoder=_wan_vae_encoder(),
        enable_sync_and_profile=True,
        encoder=_wan_vae_encoder(),
        decoder=_wan_vae_decoder_config(),
        diffusion_model=DiffusionModelConfig(
            seed=seed,
            context_noise=128,
            transformer=_transformer_config(
                checkpoint_path=AVAILABLE_ALPADREAMS_CHECKPOINT_PATHS[
                    "1view-vae-chunk3"
                ],
                cp_size=cp_size,
                compile_network=compile_network,
                num_views=1,
                len_t_latent=3,
                window_size_t=6,
                encode_with_pixel_shuffle=False,
            ),
            scheduler=_scheduler_config(),
        ),
    )


def build_sv_2steps_chunk4_loc8_pshuffle_lighttae(
    *,
    cp_size: int = 1,
    compile_network: bool = True,
    seed: int = 42,
) -> AlpadreamsPipelineConfig:
    """Single-view, chunk4, PixelShuffle HDMap encoder + LightTAE decoder."""
    return AlpadreamsPipelineConfig(
        text_encoder=CosmosReason1TextEncoderConfig(),
        image_encoder=_wan_vae_encoder(),
        enable_sync_and_profile=True,
        encoder=PixelShuffleVAEEncoderConfig(),
        decoder=_teahv_vae_decoder_config(),
        diffusion_model=DiffusionModelConfig(
            seed=seed,
            context_noise=128,
            transformer=_transformer_config(
                checkpoint_path=AVAILABLE_ALPADREAMS_CHECKPOINT_PATHS[
                    "1view-pshuffle-chunk4"
                ],
                cp_size=cp_size,
                compile_network=compile_network,
                num_views=1,
                len_t_latent=4,
                window_size_t=8,
                encode_with_pixel_shuffle=True,
            ),
            scheduler=_scheduler_config(),
        ),
    )


def build_mv_2steps_chunk4_loc8_pshuffle_lighttae(
    *,
    cp_size: int = 1,
    compile_network: bool = True,
    seed: int = 42,
) -> AlpadreamsPipelineConfig:
    """4-view, chunk4, PixelShuffle HDMap encoder + LightTAE decoder."""
    return AlpadreamsPipelineConfig(
        text_encoder=CosmosReason1TextEncoderConfig(),
        image_encoder=_wan_vae_encoder(),
        enable_sync_and_profile=True,
        encoder=PixelShuffleVAEEncoderConfig(),
        decoder=_teahv_vae_decoder_config(),
        diffusion_model=DiffusionModelConfig(
            seed=seed,
            context_noise=128,
            transformer=_transformer_config(
                checkpoint_path=AVAILABLE_ALPADREAMS_CHECKPOINT_PATHS[
                    "4view-pshuffle-chunk4"
                ],
                cp_size=cp_size,
                compile_network=compile_network,
                num_views=4,
                len_t_latent=4,
                window_size_t=8,
                encode_with_pixel_shuffle=True,
            ),
            scheduler=_scheduler_config(),
        ),
    )


# experiments1
def experiment1_baseline(
    *,
    cp_size: int = 1,
    compile_network: bool = True,
    seed: int = 42,
) -> AlpadreamsPipelineConfig:
    denoising_timesteps = [1000, 450]
    return build_sv_2steps_chunk2_loc6_lightvae_lighttae_perf(
        cp_size=cp_size,
        compile_network=compile_network,
        seed=seed,
        skip_finalize_kv_cache=False,
        denoising_timesteps=denoising_timesteps,
    )


def experiment1_skip_finalize_kv_cache(
    *,
    cp_size: int = 1,
    compile_network: bool = True,
    seed: int = 42,
) -> AlpadreamsPipelineConfig:
    denoising_timesteps = [1000, 450]
    return build_sv_2steps_chunk2_loc6_lightvae_lighttae_perf(
        cp_size=cp_size,
        compile_network=compile_network,
        seed=seed,
        skip_finalize_kv_cache=True,
        denoising_timesteps=denoising_timesteps,
    )


def experiment1_skip_finalize_kv_cache_noise350(
    *,
    cp_size: int = 1,
    compile_network: bool = True,
    seed: int = 42,
) -> AlpadreamsPipelineConfig:
    denoising_timesteps = [1000, 350]
    return build_sv_2steps_chunk2_loc6_lightvae_lighttae_perf(
        cp_size=cp_size,
        compile_network=compile_network,
        seed=seed,
        skip_finalize_kv_cache=True,
        denoising_timesteps=denoising_timesteps,
    )


def experiment1_skip_finalize_kv_cache_noise250(
    *,
    cp_size: int = 1,
    compile_network: bool = True,
    seed: int = 42,
) -> AlpadreamsPipelineConfig:
    denoising_timesteps = [1000, 250]
    return build_sv_2steps_chunk2_loc6_lightvae_lighttae_perf(
        cp_size=cp_size,
        compile_network=compile_network,
        seed=seed,
        skip_finalize_kv_cache=True,
        denoising_timesteps=denoising_timesteps,
    )


def experiment1_skip_finalize_kv_cache_noise150(
    *,
    cp_size: int = 1,
    compile_network: bool = True,
    seed: int = 42,
) -> AlpadreamsPipelineConfig:
    denoising_timesteps = [1000, 150]
    return build_sv_2steps_chunk2_loc6_lightvae_lighttae_perf(
        cp_size=cp_size,
        compile_network=compile_network,
        seed=seed,
        skip_finalize_kv_cache=True,
        denoising_timesteps=denoising_timesteps,
    )


def experiment1_skip_finalize_kv_cache_noise100(
    *,
    cp_size: int = 1,
    compile_network: bool = True,
    seed: int = 42,
) -> AlpadreamsPipelineConfig:
    denoising_timesteps = [1000, 100]
    return build_sv_2steps_chunk2_loc6_lightvae_lighttae_perf(
        cp_size=cp_size,
        compile_network=compile_network,
        seed=seed,
        skip_finalize_kv_cache=True,
        denoising_timesteps=denoising_timesteps,
    )


# ---------------------------------------------------------------------------
# Alpadreams diffusion forcing (causal AR 2B / 720p / chunk2 UniPC)
# ---------------------------------------------------------------------------


def build_sv_35steps_chunk2_loc24_cosmos2_2B_res720p_30fps_hdmap_vae_mads1m(
    *,
    cp_size: int = 1,
    compile_network: bool = True,
    use_cuda_graph: bool = True,
    seed: int = 1,
) -> AlpadreamsPipelineConfig:
    """Build the alpadreams diffusion-forcing causal AR pipeline.

    The I4 config uses ``state_t=24``: 12 chunk2 latent blocks, or 93 decoded
    frames with the Wan decoder.
    """
    return AlpadreamsPipelineConfig(
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
            seed=seed,
            context_noise=128,
            transformer=_transformer_config(
                checkpoint_path=AVAILABLE_ALPADREAMS_CHECKPOINT_PATHS[
                    "1view-diffusion-forcing-chunk2"
                ],
                cp_size=cp_size,
                compile_network=compile_network,
                num_views=1,
                len_t_latent=2,
                window_size_t=24,
                encode_with_pixel_shuffle=False,
                skip_finalize_kv_cache=False,
                use_cuda_graph=use_cuda_graph,
                guidance_scale=3.0,
            ),
            scheduler=FlowMatchUniPCSchedulerConfig(
                num_inference_steps=35,
                shift=5.0,
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Alpadreams bidirectional (single-view / 2B / 720p / chunk48 / UniPC)
# ---------------------------------------------------------------------------


def build_sv_35steps_chunk48_loc48_cosmos2_2B_res720p_30fps_hdmap_vae_mads1m(
    *,
    cp_size: int = 1,
    compile_network: bool = True,
    use_cuda_graph: bool = True,
    seed: int = 1,
    num_chunks: int = 48,
) -> AlpadreamsPipelineConfig:
    """Single-view, bidirectional Cosmos2 2B / 720p / chunk48 pipeline.

    ``num_chunks`` is the transformer's latent temporal length
    (``len_t``) for the single generated block. The public pixel-space
    frame count is derived later by the pipeline's decoder-aware
    ``get_num_frames`` helper. The underlying checkpoint was trained for
    48 chunks; entrypoints may choose a smaller value for their runtime
    memory budget.

    The transformer checkpoint path is baked into the recipe; override
    in this builder if you need a different one.
    """
    decoder_config = WanVAEDecoderConfig(
        checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"],
    )
    assert num_chunks >= 1, f"num_chunks must be positive, got {num_chunks}."
    return AlpadreamsPipelineConfig(
        text_encoder=CosmosReason1TextEncoderConfig(),
        image_encoder=WanVAEEncoderConfig(
            checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"],
        ),
        enable_sync_and_profile=True,
        encoder=WanVAEEncoderConfig(
            checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"],
        ),
        decoder=decoder_config,
        diffusion_model=DiffusionModelConfig(
            seed=seed,
            transformer=_transformer_config(
                checkpoint_path=AVAILABLE_ALPADREAMS_CHECKPOINT_PATHS[
                    "1view-bidirectional-chunk48"
                ],
                cp_size=cp_size,
                compile_network=compile_network,
                num_views=1,
                len_t_latent=num_chunks,
                window_size_t=num_chunks,
                encode_with_pixel_shuffle=False,
                skip_finalize_kv_cache=True,
                use_cuda_graph=use_cuda_graph,
                guidance_scale=3.0,
            ),
            scheduler=FlowMatchUniPCSchedulerConfig(
                num_inference_steps=35,
                shift=5.0,
            ),
        ),
    )


ALPADREAMS_CONFIG_BUILDERS: dict[str, Callable[..., AlpadreamsPipelineConfig]] = {
    "sv_2steps_chunk2_loc6_lightvae_lighttae": build_sv_2steps_chunk2_loc6_lightvae_lighttae,
    "sv_2steps_chunk2_loc6_lightvae_lighttae_perf": build_sv_2steps_chunk2_loc6_lightvae_lighttae_perf,
    "sv_2steps_chunk2_loc6_vae_vae": build_sv_2steps_chunk2_loc6_vae_vae,
    "sv_2steps_chunk3_loc6_vae_vae": build_sv_2steps_chunk3_loc6_vae_vae,
    "sv_2steps_chunk4_loc8_pshuffle_lighttae": build_sv_2steps_chunk4_loc8_pshuffle_lighttae,
    "mv_2steps_chunk4_loc8_pshuffle_lighttae": build_mv_2steps_chunk4_loc8_pshuffle_lighttae,
    # teachers
    "sv_35steps_chunk2_loc24_cosmos2_2B_res720p_30fps_hdmap_vae_mads1m": build_sv_35steps_chunk2_loc24_cosmos2_2B_res720p_30fps_hdmap_vae_mads1m,
    "sv_35steps_chunk48_loc48_cosmos2_2B_res720p_30fps_hdmap_vae_mads1m": build_sv_35steps_chunk48_loc48_cosmos2_2B_res720p_30fps_hdmap_vae_mads1m,
    # experiments
    "experiment1_baseline": experiment1_baseline,
    "experiment1_skip_finalize_kv_cache": experiment1_skip_finalize_kv_cache,
    "experiment1_skip_finalize_kv_cache_noise350": experiment1_skip_finalize_kv_cache_noise350,
    "experiment1_skip_finalize_kv_cache_noise250": experiment1_skip_finalize_kv_cache_noise250,
    "experiment1_skip_finalize_kv_cache_noise150": experiment1_skip_finalize_kv_cache_noise150,  # rec
    "experiment1_skip_finalize_kv_cache_noise100": experiment1_skip_finalize_kv_cache_noise100,  # rec
}
