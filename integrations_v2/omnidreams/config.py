# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public OmniDreams world-model pipeline configurations."""

from __future__ import annotations

from typing import cast

import torch
from omnidreams.impl.pipeline import OmnidreamsPipelineConfig
from omnidreams.impl.transformer import CosmosTransformerConfig
from omnidreams.impl.transformer.modules import AttentionBackend
from omnidreams.impl.transformer.network import CosmosDiTNetworkConfig
from omnidreams.impl.vae_native import (
    OmnidreamsWanVAEEncoderConfig as WanVAEEncoderConfig,
)

from flashdreams.accelerated.multi_head_attention.optimized import (
    OptimizedImplConfig,
    QKVFusionOption,
    QuantizationOption,
    SDPABackend,
)
from flashdreams.infra.config import derive_config
from flashdreams.infra.diffusion.model import DiffusionModelConfig
from flashdreams.infra.diffusion.scheduler.fm import FlowMatchSchedulerConfig
from flashdreams.infra.encoder.text.cosmos_reason1 import (
    CosmosReason1TextEncoderConfig,
)
from flashdreams.recipes.taehv import (
    AVAILABLE_TAEHV_CHECKPOINT_PATHS,
    TeahvVAEDecoderConfig,
)
from flashdreams.recipes.wan.autoencoder.vae import AVAILABLE_WAN_VAE_CHECKPOINT_PATHS

AVAILABLE_OMNIDREAMS_CHECKPOINT_PATHS: dict[str, str] = {
    "1view-vae-chunk2": (
        "https://huggingface.co/nvidia/omni-dreams-models/resolve/main/"
        "single_view/2b_res720p_30fps_i2v_hdmap_distilled.pt"
    ),
}
"""Checkpoint paths required by the public OmniDreams configs."""


OMNIDREAMS_PIPELINE_CONFIG = OmnidreamsPipelineConfig(
    name="omnidreams",
    text_encoder=CosmosReason1TextEncoderConfig(),
    image_encoder=WanVAEEncoderConfig(
        checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["lightvae"],
        use_compile=False,
        use_cuda_graph=True,
    ),
    enable_sync_and_profile=True,
    encoder=WanVAEEncoderConfig(
        checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["lightvae"],
        use_compile=False,
        use_cuda_graph=True,
    ),
    decoder=TeahvVAEDecoderConfig(
        checkpoint_path=AVAILABLE_TAEHV_CHECKPOINT_PATHS["lighttae"],
        use_compile=False,
        use_cuda_graph=True,
    ),
    diffusion_model=DiffusionModelConfig(
        seed=42,
        context_noise=128,
        transformer=CosmosTransformerConfig(
            network=CosmosDiTNetworkConfig(
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
            compile_network=False,
            use_cuda_graph=True,
            skip_finalize_kv_cache=False,
            guidance_scale=1.0,
        ),
        scheduler=FlowMatchSchedulerConfig(
            num_inference_steps=2,
            denoising_timesteps=[1000, 500],
            warp_denoising_step=True,
            shift=5.0,
            sigma_min=0.0,
            extra_one_step=True,
            num_train_timesteps=1000,
        ),
    ),
)
"""Regular OmniDreams world-model pipeline configuration."""


OMNIDREAMS_OPTIMIZED_GB300_PIPELINE_CONFIG = cast(
    OmnidreamsPipelineConfig,
    derive_config(
        OMNIDREAMS_PIPELINE_CONFIG,
        name="omnidreams-optimized-gb300",
        diffusion_model=dict(
            transformer=dict(
                skip_finalize_kv_cache=True,
                network=dict(
                    self_attention_backend=AttentionBackend.OPTIMIZED,
                    cross_attention_backend=AttentionBackend.OPTIMIZED,
                    self_attn_optimized_impl_config=OptimizedImplConfig(
                        qkv_fusion_option=QKVFusionOption.FULL,
                        sdpa_backend=SDPABackend.CUDNN,
                        use_tma=False,
                        quantization=QuantizationOption(
                            projection=None,
                            quantized_sdpa=True,
                        ),
                    ),
                    cross_attn_optimized_impl_config=OptimizedImplConfig(
                        qkv_fusion_option=QKVFusionOption.FUSE_KV,
                        sdpa_backend=SDPABackend.FA2,
                        use_tma=False,
                        quantization=QuantizationOption(
                            projection=None,
                            quantized_sdpa=False,
                        ),
                    ),
                ),
            ),
        ),
    ),
)  # ty:ignore[redundant-cast]
"""GB300 config using the benchmark-selected optimized DiT attention policy."""


OMNIDREAMS_OPTIMIZED_RTX_PRO_6000_PIPELINE_CONFIG = cast(
    OmnidreamsPipelineConfig,
    derive_config(
        OMNIDREAMS_PIPELINE_CONFIG,
        name="omnidreams-optimized-rtx-pro-6000",
        diffusion_model=dict(
            transformer=dict(
                skip_finalize_kv_cache=True,
                network=dict(
                    self_attention_backend=AttentionBackend.OPTIMIZED,
                    cross_attention_backend=AttentionBackend.OMNIDREAMS,
                    self_attn_optimized_impl_config=OptimizedImplConfig(
                        qkv_fusion_option=QKVFusionOption.FULL,
                        sdpa_backend=SDPABackend.FA2,
                        use_tma=True,
                        quantization=QuantizationOption(
                            projection=torch.float8_e4m3fn,
                            quantized_sdpa=True,
                        ),
                    ),
                ),
            ),
        ),
    ),
)  # ty:ignore[redundant-cast]
"""RTX PRO 6000 config using the benchmark-selected optimized DiT policy."""


OMNIDREAMS_PERF_PIPELINE_CONFIG = cast(
    OmnidreamsPipelineConfig,
    derive_config(
        OMNIDREAMS_PIPELINE_CONFIG,
        name="omnidreams-perf",
        diffusion_model=dict(
            transformer=dict(
                compile_network=False,
                skip_finalize_kv_cache=True,
                native_dit_acceleration="required",
                native_dit_backend="fp8_kvcache_cudnn",
                native_dit_attention_backend="cudnn",
            ),
            scheduler=dict(denoising_timesteps=[1000, 100]),
        ),
    ),
)  # ty:ignore[redundant-cast]
"""Performance-tuned OmniDreams world-model pipeline configuration."""


OMNIDREAMS_FAST_PERF_PIPELINE_CONFIG = cast(
    OmnidreamsPipelineConfig,
    derive_config(
        OMNIDREAMS_PERF_PIPELINE_CONFIG,
        name="omnidreams-fast-perf",
        diffusion_model=dict(seed=None),
        image_encoder=dict(
            dtype=torch.float16,
            use_compile=False,
            use_cuda_graph=False,
            native_vae_acceleration="required",
            native_vae_backend="fp8",
            native_vae_fp8_auto_export=True,
        ),
        encoder=dict(
            dtype=torch.float16,
            use_compile=False,
            use_cuda_graph=False,
            native_vae_acceleration="required",
            native_vae_backend="fp8",
            native_vae_fp8_auto_export=True,
        ),
    ),
)  # ty:ignore[redundant-cast]
"""Fast config that uses native FP8 LightVAE with cached calibration."""


OMNIDREAMS_CONFIGS: dict[str, OmnidreamsPipelineConfig] = {
    config.name: config
    for config in (
        OMNIDREAMS_PIPELINE_CONFIG,
        OMNIDREAMS_OPTIMIZED_GB300_PIPELINE_CONFIG,
        OMNIDREAMS_OPTIMIZED_RTX_PRO_6000_PIPELINE_CONFIG,
        OMNIDREAMS_PERF_PIPELINE_CONFIG,
        OMNIDREAMS_FAST_PERF_PIPELINE_CONFIG,
    )
}
"""The public OmniDreams pipeline configurations, keyed by slug."""
