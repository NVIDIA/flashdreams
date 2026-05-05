# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-built pipeline-config builders for MemRoPE Wan 2.1."""

from __future__ import annotations

from collections.abc import Callable

from flashdreams.recipes.wan.config.causal_wan21 import (
    AVAILABLE_CAUSAL_WAN21_CHECKPOINT_PATHS,
    _DEFAULT_BATCH_SHAPE,
    _DEFAULT_VIDEO_HEIGHT,
    _DEFAULT_VIDEO_WIDTH,
    _remap_self_or_causal_forcing_state_dict,
    _scheduler_config,
    _wan_vae_decoder_config,
    _WAN_VAE_SPATIAL_COMPRESSION,
)
from flashdreams.recipes.wan.memrope_diffusion import MemRoPEDiffusionModelConfig
from flashdreams.recipes.wan.pipeline import WanInferencePipelineConfig
from flashdreams.recipes.wan.transformer.impl.memrope_network import (
    MemRoPEWanDiTNetwork1pt3BConfig,
)
from flashdreams.recipes.wan.transformer.memrope_wan21 import (
    MemRoPEWan21TransformerConfig,
)


def _memrope_transformer_config(
    *,
    checkpoint_path: str,
    cp_size: int,
    compile_network: bool,
    sink_size_t: int,
    memory_size_t: int,
    recent_size_t: int,
    len_t_latent: int = 3,
    ema_alpha_long: float = 0.01,
    ema_alpha_short: float = 0.1,
) -> MemRoPEWan21TransformerConfig:
    """MemRoPE Wan 1.3B transformer defaults for T2V streaming inference."""
    return MemRoPEWan21TransformerConfig(
        network=MemRoPEWanDiTNetwork1pt3BConfig(
            patch_embedding_type="conv3d",
        ),
        checkpoint_path=checkpoint_path,
        state_dict_transform=_remap_self_or_causal_forcing_state_dict,
        batch_shape=_DEFAULT_BATCH_SHAPE,
        height=_DEFAULT_VIDEO_HEIGHT // _WAN_VAE_SPATIAL_COMPRESSION,
        width=_DEFAULT_VIDEO_WIDTH // _WAN_VAE_SPATIAL_COMPRESSION,
        len_t=len_t_latent,
        cp_size=cp_size,
        guidance_scale=1.0,
        window_size_t=memory_size_t + recent_size_t + len_t_latent,
        sink_size_t=sink_size_t,
        memory_size_t=memory_size_t,
        recent_size_t=recent_size_t,
        ema_alpha_long=ema_alpha_long,
        ema_alpha_short=ema_alpha_short,
        compile_network=compile_network,
    )


def build_self_forcing_memrope_s3m2r13(
    *,
    cp_size: int = 1,
    compile_network: bool = True,
    seed: int = 0,
    i2v: bool = False,
    enable_sync_and_profile: bool = False,
) -> WanInferencePipelineConfig:
    """Self-Forcing checkpoint with MemRoPE s3m2r13 attention."""
    assert not i2v, "MemRoPE Wan 2.1 config currently supports T2V only"
    return WanInferencePipelineConfig(
        enable_sync_and_profile=enable_sync_and_profile,
        encoder=None,
        decoder=_wan_vae_decoder_config(),
        diffusion_model=MemRoPEDiffusionModelConfig(
            seed=seed,
            transformer=_memrope_transformer_config(
                checkpoint_path=AVAILABLE_CAUSAL_WAN21_CHECKPOINT_PATHS[
                    "self_forcing"
                ],
                cp_size=cp_size,
                compile_network=compile_network,
                sink_size_t=3,
                memory_size_t=2,
                recent_size_t=13,
            ),
            scheduler=_scheduler_config(num_inference_steps=4),
        ),
    )


def build_self_forcing_memrope_s3m2r4(
    *,
    cp_size: int = 1,
    compile_network: bool = True,
    seed: int = 0,
    i2v: bool = False,
    enable_sync_and_profile: bool = False,
) -> WanInferencePipelineConfig:
    """Self-Forcing checkpoint with official-style MemRoPE s3m2r4 attention."""
    assert not i2v, "MemRoPE Wan 2.1 config currently supports T2V only"
    return WanInferencePipelineConfig(
        enable_sync_and_profile=enable_sync_and_profile,
        encoder=None,
        decoder=_wan_vae_decoder_config(),
        diffusion_model=MemRoPEDiffusionModelConfig(
            seed=seed,
            transformer=_memrope_transformer_config(
                checkpoint_path=AVAILABLE_CAUSAL_WAN21_CHECKPOINT_PATHS[
                    "self_forcing"
                ],
                cp_size=cp_size,
                compile_network=compile_network,
                sink_size_t=3,
                memory_size_t=2,
                recent_size_t=4,
            ),
            scheduler=_scheduler_config(num_inference_steps=4),
        ),
    )


def build_self_forcing_memrope_s3m0r15(
    *,
    cp_size: int = 1,
    compile_network: bool = True,
    seed: int = 0,
    i2v: bool = False,
    enable_sync_and_profile: bool = False,
) -> WanInferencePipelineConfig:
    """Self-Forcing checkpoint with MemRoPE sink-only s3m0r15 attention."""
    assert not i2v, "MemRoPE Wan 2.1 config currently supports T2V only"
    return WanInferencePipelineConfig(
        enable_sync_and_profile=enable_sync_and_profile,
        encoder=None,
        decoder=_wan_vae_decoder_config(),
        diffusion_model=MemRoPEDiffusionModelConfig(
            seed=seed,
            transformer=_memrope_transformer_config(
                checkpoint_path=AVAILABLE_CAUSAL_WAN21_CHECKPOINT_PATHS[
                    "self_forcing"
                ],
                cp_size=cp_size,
                compile_network=compile_network,
                sink_size_t=3,
                memory_size_t=0,
                recent_size_t=15,
            ),
            scheduler=_scheduler_config(num_inference_steps=4),
        ),
    )


def build_self_forcing_memrope_s3m0r6(
    *,
    cp_size: int = 1,
    compile_network: bool = True,
    seed: int = 0,
    i2v: bool = False,
    enable_sync_and_profile: bool = False,
) -> WanInferencePipelineConfig:
    """Self-Forcing checkpoint with MemRoPE sink-only s3m0r6 attention."""
    assert not i2v, "MemRoPE Wan 2.1 config currently supports T2V only"
    return WanInferencePipelineConfig(
        enable_sync_and_profile=enable_sync_and_profile,
        encoder=None,
        decoder=_wan_vae_decoder_config(),
        diffusion_model=MemRoPEDiffusionModelConfig(
            seed=seed,
            transformer=_memrope_transformer_config(
                checkpoint_path=AVAILABLE_CAUSAL_WAN21_CHECKPOINT_PATHS[
                    "self_forcing"
                ],
                cp_size=cp_size,
                compile_network=compile_network,
                sink_size_t=3,
                memory_size_t=0,
                recent_size_t=6,
            ),
            scheduler=_scheduler_config(num_inference_steps=4),
        ),
    )


def build_self_forcing_memrope_s0m2r16(
    *,
    cp_size: int = 1,
    compile_network: bool = True,
    seed: int = 0,
    i2v: bool = False,
    enable_sync_and_profile: bool = False,
) -> WanInferencePipelineConfig:
    """Self-Forcing checkpoint with MemRoPE no-sink s0m2r16 attention."""
    assert not i2v, "MemRoPE Wan 2.1 config currently supports T2V only"
    return WanInferencePipelineConfig(
        enable_sync_and_profile=enable_sync_and_profile,
        encoder=None,
        decoder=_wan_vae_decoder_config(),
        diffusion_model=MemRoPEDiffusionModelConfig(
            seed=seed,
            transformer=_memrope_transformer_config(
                checkpoint_path=AVAILABLE_CAUSAL_WAN21_CHECKPOINT_PATHS[
                    "self_forcing"
                ],
                cp_size=cp_size,
                compile_network=compile_network,
                sink_size_t=0,
                memory_size_t=2,
                recent_size_t=16,
            ),
            scheduler=_scheduler_config(num_inference_steps=4),
        ),
    )


def build_self_forcing_memrope_s0m2r7(
    *,
    cp_size: int = 1,
    compile_network: bool = True,
    seed: int = 0,
    i2v: bool = False,
    enable_sync_and_profile: bool = False,
) -> WanInferencePipelineConfig:
    """Self-Forcing checkpoint with MemRoPE no-sink s0m2r7 attention."""
    assert not i2v, "MemRoPE Wan 2.1 config currently supports T2V only"
    return WanInferencePipelineConfig(
        enable_sync_and_profile=enable_sync_and_profile,
        encoder=None,
        decoder=_wan_vae_decoder_config(),
        diffusion_model=MemRoPEDiffusionModelConfig(
            seed=seed,
            transformer=_memrope_transformer_config(
                checkpoint_path=AVAILABLE_CAUSAL_WAN21_CHECKPOINT_PATHS[
                    "self_forcing"
                ],
                cp_size=cp_size,
                compile_network=compile_network,
                sink_size_t=0,
                memory_size_t=2,
                recent_size_t=7,
            ),
            scheduler=_scheduler_config(num_inference_steps=4),
        ),
    )


build_self_forcing_memrope = build_self_forcing_memrope_s3m2r13


MEMROPE_WAN21_CONFIG_BUILDERS: dict[str, Callable[..., WanInferencePipelineConfig]] = {
    "self_forcing_memrope": build_self_forcing_memrope,
    "self_forcing_memrope_s3m2r13": build_self_forcing_memrope_s3m2r13,
    "self_forcing_memrope_s3m0r15": build_self_forcing_memrope_s3m0r15,
    "self_forcing_memrope_s3m0r6": build_self_forcing_memrope_s3m0r6,
    "self_forcing_memrope_s0m2r16": build_self_forcing_memrope_s0m2r16,
    "self_forcing_memrope_s0m2r7": build_self_forcing_memrope_s0m2r7,
    "self_forcing_memrope_s3m2r4": build_self_forcing_memrope_s3m2r4,
}
