"""Pre-built :class:`AlpadreamsPipelineConfig` builders."""

from __future__ import annotations

from collections.abc import Callable

from flashdreams.infra.diffusion.model import DiffusionModelConfig
from flashdreams.infra.diffusion.scheduler.fm import (
    FlowMatchSchedulerConfig,
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
from flashdreams.recipes.alpadreams.transformer import (
    CosmosTransformerConfig,
)
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


AVAILABLE_ALPADREAMS_CHECKPOINT_PATHS: dict[str, dict[str, str | dict[str, str]]] = {
    "single_view": {
        "pixel_shuffle": "s3://flashdreams/assets/checkpoints/alpadreams/16N@cosmos_v2_2b_SF_res720p_30fps_i2v_hdmap_chunk4_pixel_shuffle_resume.pt",
        "vae_encoding": {
            "chunk2": "s3://flashdreams/assets/checkpoints/alpadreams/32n_cosmos_v2_2b_SF_res720p_30fps_i2v_hdmap_chunk2_vae_encode_189f_loc6_sft_urban_stationary_mixed_gcp_student_resume.pt",
            "chunk3": "s3://flashdreams/assets/checkpoints/alpadreams/32n_cosmos_v2_2b_SF_res720p_30fps_i2v_hdmap_chunk3_vae_encode_loc6_gcp.pt",
        },
    },
    "4views": {
        "pixel_shuffle": "s3://flashdreams/assets/checkpoints/alpadreams/32n_cosmos_v2_2b_SF_4view_res720p_fps30_chunk4_i2v_hdmap_pixel_shuffle_loc8st2_gcp.pt",
        "vae_encoding": "s3://flashdreams/assets/checkpoints/alpadreams/32n_cosmos_v2_2b_SF_4view_res720p_fps30_chunk4_i2v_hdmap_vae_encoding_loc8st2_gcp.pt",
    },
}


# ---------------------------------------------------------------------------
# Canonical alpadreams 720p defaults (60 latent rows / 80 latent cols at 720p
# after the 8x VAE spatial compression).
# ---------------------------------------------------------------------------

_DEFAULT_BATCH_SHAPE: tuple[int, ...] = (1,)
_DEFAULT_VIDEO_HEIGHT = 720
_DEFAULT_VIDEO_WIDTH = 1280
_WAN_VAE_SPATIAL_COMPRESSION = 8
_DEFAULT_DENOISING_TIMESTEPS = [1000, 450]
_DEFAULT_NUM_TRAIN_TIMESTEPS = 1000


def _wan_vae_decoder_config() -> WanVAEDecoderConfig:
    return WanVAEDecoderConfig(
        checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"],
    )


def _teahv_vae_decoder_config() -> TeahvVAEDecoderConfig:
    return TeahvVAEDecoderConfig(
        checkpoint_path=AVAILABLE_TAEHV_CHECKPOINT_PATHS["lighttae"],
    )


def _scheduler_config() -> FlowMatchSchedulerConfig:
    """Alpadreams 2-step Self-Forcing flow-match scheduler defaults."""
    return FlowMatchSchedulerConfig(
        num_inference_steps=len(_DEFAULT_DENOISING_TIMESTEPS),
        denoising_timesteps=_DEFAULT_DENOISING_TIMESTEPS,
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
    compile_network: bool,
    num_views: int,
    len_t_latent: int,
    window_size_t: int,
    encode_with_pixel_shuffle: bool,
    kv_drop_t: int = 1,
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
        kv_drop_t=kv_drop_t,
        compile_network=compile_network,
    )


def _wan_vae_encoder(*, checkpoint_name: str = "vae") -> WanVAEEncoderConfig:
    return WanVAEEncoderConfig(
        checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS[checkpoint_name],
    )


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def build_sv_2steps_chunk2_loc6_lightvae_lighttae(
    *,
    cp_size: int = 1,
    compile_network: bool = True,
    seed: int = 42,
    # FIXME: this recipe uses a stateful Wan VAE HDMap encoder, which the
    # ``kv_drop_t > 0`` first-cut does not support (see
    # ``AlpadreamsPipeline.__init__`` for the gate). Default to the
    # legacy non-overlapping rollout here so this recipe keeps working;
    # ``kv_drop_t > 0`` only defaults to 1 on the PixelShuffle chunk4
    # recipes where it is safe.
    kv_drop_t: int = 0,
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
                checkpoint_path=AVAILABLE_ALPADREAMS_CHECKPOINT_PATHS["single_view"][
                    "vae_encoding"
                ]["chunk2"],  # type: ignore[index]
                cp_size=cp_size,
                compile_network=compile_network,
                num_views=1,
                len_t_latent=2,
                window_size_t=6,
                encode_with_pixel_shuffle=False,
                kv_drop_t=kv_drop_t,
            ),
            scheduler=_scheduler_config(),
        ),
    )


def build_sv_2steps_chunk2_loc6_vae_vae(
    *,
    cp_size: int = 1,
    compile_network: bool = True,
    seed: int = 42,
    # See note in ``build_sv_2steps_chunk2_loc6_lightvae_lighttae``: this
    # recipe also uses a stateful Wan VAE HDMap encoder, so the
    # ``kv_drop_t > 0`` path is gated and we default to legacy rollout.
    kv_drop_t: int = 0,
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
                checkpoint_path=AVAILABLE_ALPADREAMS_CHECKPOINT_PATHS["single_view"][
                    "vae_encoding"
                ]["chunk2"],  # type: ignore[index]
                cp_size=cp_size,
                compile_network=compile_network,
                num_views=1,
                len_t_latent=2,
                window_size_t=6,
                encode_with_pixel_shuffle=False,
                kv_drop_t=kv_drop_t,
            ),
            scheduler=_scheduler_config(),
        ),
    )


def build_sv_2steps_chunk3_loc6_vae_vae(
    *,
    cp_size: int = 1,
    compile_network: bool = True,
    seed: int = 42,
    # See note in ``build_sv_2steps_chunk2_loc6_lightvae_lighttae``: this
    # recipe also uses a stateful Wan VAE HDMap encoder, so the
    # ``kv_drop_t > 0`` path is gated and we default to legacy rollout.
    kv_drop_t: int = 0,
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
                checkpoint_path=AVAILABLE_ALPADREAMS_CHECKPOINT_PATHS["single_view"][
                    "vae_encoding"
                ]["chunk3"],  # type: ignore[index]
                cp_size=cp_size,
                compile_network=compile_network,
                num_views=1,
                len_t_latent=3,
                window_size_t=6,
                encode_with_pixel_shuffle=False,
                kv_drop_t=kv_drop_t,
            ),
            scheduler=_scheduler_config(),
        ),
    )


def build_sv_2steps_chunk4_loc8_pshuffle_lighttae(
    *,
    cp_size: int = 1,
    compile_network: bool = True,
    seed: int = 42,
    kv_drop_t: int = 1,
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
                checkpoint_path=AVAILABLE_ALPADREAMS_CHECKPOINT_PATHS["single_view"][
                    "pixel_shuffle"
                ],  # type: ignore[index]
                cp_size=cp_size,
                compile_network=compile_network,
                num_views=1,
                len_t_latent=4,
                window_size_t=8,
                encode_with_pixel_shuffle=True,
                kv_drop_t=kv_drop_t,
            ),
            scheduler=_scheduler_config(),
        ),
    )


def build_mv_2steps_chunk4_loc8_pshuffle_lighttae(
    *,
    cp_size: int = 1,
    compile_network: bool = True,
    seed: int = 42,
    kv_drop_t: int = 1,
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
                checkpoint_path=AVAILABLE_ALPADREAMS_CHECKPOINT_PATHS["4views"][
                    "pixel_shuffle"
                ],  # type: ignore[arg-type]
                cp_size=cp_size,
                compile_network=compile_network,
                num_views=4,
                len_t_latent=4,
                window_size_t=8,
                encode_with_pixel_shuffle=True,
                kv_drop_t=kv_drop_t,
            ),
            scheduler=_scheduler_config(),
        ),
    )


ALPADREAMS_CONFIG_BUILDERS: dict[str, Callable[..., AlpadreamsPipelineConfig]] = {
    "sv_2steps_chunk2_loc6_lightvae_lighttae": build_sv_2steps_chunk2_loc6_lightvae_lighttae,
    "sv_2steps_chunk2_loc6_vae_vae": build_sv_2steps_chunk2_loc6_vae_vae,
    "sv_2steps_chunk3_loc6_vae_vae": build_sv_2steps_chunk3_loc6_vae_vae,
    "sv_2steps_chunk4_loc8_pshuffle_lighttae": build_sv_2steps_chunk4_loc8_pshuffle_lighttae,
    "mv_2steps_chunk4_loc8_pshuffle_lighttae": build_mv_2steps_chunk4_loc8_pshuffle_lighttae,
}
