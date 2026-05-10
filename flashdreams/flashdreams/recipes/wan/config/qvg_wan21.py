# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""QVG-specific pipeline-config builders for streaming Wan 2.1."""

from __future__ import annotations

from collections.abc import Callable

from flashdreams.core.attention.kv_compress import KVCompressionConfig
from flashdreams.infra.diffusion.model import DiffusionModelConfig
from flashdreams.recipes.wan.config.causal_wan21 import (
    AVAILABLE_CAUSAL_WAN21_CHECKPOINT_PATHS,
    _scheduler_config,
    _transformer_config,
    _wan_vae_decoder_config,
)
from flashdreams.recipes.wan.pipeline import WanInferencePipelineConfig


def _qvg_kv_compression_config(*, quant_type: str) -> KVCompressionConfig:
    """QVG defaults matching the open-source Self-Forcing scripts."""
    return KVCompressionConfig(
        backend="qvg",
        schedule={"compress_every_n_chunks": 8},
        protected_recent_chunks=0,
        protected_sink_tokens=0,
        backend_config={
            "quant_type": quant_type,
            "cache_num_k_centroids": 256,
            "cache_num_v_centroids": 256,
            "kmeans_max_iters": 2,
            "quant_block_size": 64,
            "num_prq_stages": 1,
            "scale_dtype": "float8_e4m3fn",
            "kmeans_init": "random",
            "store_prerope_keys": True,
            "kernel_impl": "official_triton",
            "preserve_rng": False,
        },
    )


def build_self_forcing_qvg_int2(
    *,
    cp_size: int = 1,
    compile_network: bool = True,
    seed: int = 42,
    i2v: bool = False,
    enable_sync_and_profile: bool = False,
) -> WanInferencePipelineConfig:
    """Self-Forcing distilled checkpoint with QVG INT2 KV compression."""
    assert not i2v, "QVG v1 config currently supports T2V only"
    assert cp_size == 1, "QVG v1 config currently supports single-GPU only"
    return WanInferencePipelineConfig(
        enable_sync_and_profile=enable_sync_and_profile,
        encoder=None,
        decoder=_wan_vae_decoder_config(),
        diffusion_model=DiffusionModelConfig(
            seed=seed,
            _noise_in_unpatchified_shape=True,
            transformer=_transformer_config(
                checkpoint_path=AVAILABLE_CAUSAL_WAN21_CHECKPOINT_PATHS[
                    "self_forcing"
                ],
                compile_network=compile_network,
                kv_compression=_qvg_kv_compression_config(
                    quant_type="triton-nstages-kmeans-int2"
                ),
            ),
            scheduler=_scheduler_config(num_inference_steps=4, shift=8.0),
        ),
    )


def build_self_forcing_qvg_int4(
    *,
    cp_size: int = 1,
    compile_network: bool = True,
    seed: int = 42,
    i2v: bool = False,
    enable_sync_and_profile: bool = False,
) -> WanInferencePipelineConfig:
    """Self-Forcing distilled checkpoint with QVG INT4 KV compression."""
    assert not i2v, "QVG v1 config currently supports T2V only"
    assert cp_size == 1, "QVG v1 config currently supports single-GPU only"
    return WanInferencePipelineConfig(
        enable_sync_and_profile=enable_sync_and_profile,
        encoder=None,
        decoder=_wan_vae_decoder_config(),
        diffusion_model=DiffusionModelConfig(
            seed=seed,
            _noise_in_unpatchified_shape=True,
            transformer=_transformer_config(
                checkpoint_path=AVAILABLE_CAUSAL_WAN21_CHECKPOINT_PATHS[
                    "self_forcing"
                ],
                compile_network=compile_network,
                kv_compression=_qvg_kv_compression_config(
                    quant_type="triton-nstages-kmeans-int4"
                ),
            ),
            scheduler=_scheduler_config(num_inference_steps=4, shift=8.0),
        ),
    )


QVG_WAN21_CONFIG_BUILDERS: dict[str, Callable[..., WanInferencePipelineConfig]] = {
    "self_forcing_qvg_int2": build_self_forcing_qvg_int2,
    "self_forcing_qvg_int4": build_self_forcing_qvg_int4,
}
