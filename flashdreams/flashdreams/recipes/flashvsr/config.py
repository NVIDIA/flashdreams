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

"""Pipeline-config builders for FlashVSR.

Mirrors :mod:`flashdreams.recipes.alpadreams.config`: a constants block
listing the canonical FlashVSR-v1.1 weight locations, private sub-config
helpers (``_scheduler_config``, ``_transformer_config``, etc.), and one
``build_*`` function per supported pipeline configuration.

The default builder is :func:`build_flashvsr_v1_1`, which composes the
``flashvsr_tiny_long`` checkpoint into a streaming VSR pipeline with the
1-step flow-match scheduler that FlashVSR was distilled against.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import torch

from flashdreams.infra.diffusion.model import DiffusionModelConfig
from flashdreams.infra.diffusion.scheduler.fm import FlowMatchSchedulerConfig
from flashdreams.recipes.flashvsr.corrector import ColorCorrectorImplementation
from flashdreams.recipes.flashvsr.decoder import FlashVSRDecoderConfig
from flashdreams.recipes.flashvsr.encoder import FlashVSREncoderConfig
from flashdreams.recipes.flashvsr.pipeline import FlashVSRPipelineConfig
from flashdreams.recipes.flashvsr.transformer import FlashVSRTransformerConfig
from flashdreams.recipes.flashvsr.transformer.network import FlashVSRDiTNetworkConfig

__all__ = [
    "AVAILABLE_FLASHVSR_DIT_CHECKPOINT_PATHS",
    "AVAILABLE_FLASHVSR_PROJECTOR_PATHS",
    "AVAILABLE_FLASHVSR_TCDECODER_PATHS",
    "AVAILABLE_FLASHVSR_PROMPT_PATHS",
    "FLASHVSR_CONFIG_BUILDERS",
    "build_flashvsr_v1_1",
]


# ---------------------------------------------------------------------------
# Canonical FlashVSR-v1.1 weight locations
#
# Pulled directly from upstream (HuggingFace + raw GitHub) on first use via
# ``flashdreams.core.checkpoint.load.load_checkpoint``: HF URLs go through
# the HF cache, the GitHub raw prompt blob gets cached under
# ``<FLASHDREAMS_CACHE_DIR>/http_checkpoints/``. Swap to ``s3://...`` once
# the assets land in the shared bucket.
# Mirrors ``AVAILABLE_*_CHECKPOINT_PATHS`` in ``alpadreams/config.py``.
# ---------------------------------------------------------------------------

_flashvsr_base = lambda repo: f"https://huggingface.co/JunhaoZhuang/{repo}/resolve/main"
_flashvsr_prompt_path = "https://raw.githubusercontent.com/OpenImagingLab/FlashVSR/main/examples/WanVSR/prompt_tensor/posi_prompt.pth"

AVAILABLE_FLASHVSR_CHECKPOINT_PATHS: dict[str, dict[str, str]] = {
    "v1.1-tiny-long": {
        "encoder": f"{_flashvsr_base('FlashVSR-v1.1')}/LQ_proj_in.ckpt",
        "decoder": f"{_flashvsr_base('FlashVSR-v1.1')}/TCDecoder.ckpt",
        "dit": f"{_flashvsr_base('FlashVSR-v1.1')}/diffusion_pytorch_model_streaming_dmd.safetensors",
        "prompt": _flashvsr_prompt_path,
    },
}


# ---------------------------------------------------------------------------
# Sub-config helpers
# ---------------------------------------------------------------------------


def _scheduler_config() -> FlowMatchSchedulerConfig:
    """1-step flow-match scheduler matching FlashVSR's distilled training.

    With ``num_inference_steps=1`` and ``denoising_timesteps=[1000]`` at
    ``shift=8`` the scheduler reduces to ``clean = noisy - sigma * flow``
    with ``sigma(t=1000) = 8/(1+(8-1)*1) = 1``, i.e. ``clean = noisy - flow``.
    This matches the legacy upsampler's ``cur_latents - noise_pred`` step.
    """
    return FlowMatchSchedulerConfig(
        num_inference_steps=1,
        denoising_timesteps=[1000],
        warp_denoising_step=True,
        shift=8.0,
        sigma_min=0.0,
        extra_one_step=True,
        num_train_timesteps=1000,
    )


def _transformer_config(
    *,
    target_H: int,
    target_W: int,
    sparse_ratio: float,
    kv_ratio: int,
    local_range: int,
    dit_checkpoint_path: str,
    compile_network: bool,
    dtype: torch.dtype,
) -> FlashVSRTransformerConfig:
    """Build the FlashVSR transformer config for a given target resolution.

    The legacy upsampler scaled ``topk_ratio`` linearly with the inverse
    pixel-count ratio (``sparse_ratio * 768*1280 / (target_H*target_W)``);
    we preserve that to keep the absolute top-k budget constant across
    resolutions.
    """
    return FlashVSRTransformerConfig(
        network=FlashVSRDiTNetworkConfig(),  # ``flashvsr_tiny_long`` defaults
        dtype=dtype,
        checkpoint_path=dit_checkpoint_path,
        batch_shape=(1,),
        height=target_H // 8,
        width=target_W // 8,
        len_t=2,
        cp_size=1,
        guidance_scale=1.0,
        topk_ratio=sparse_ratio * 768 * 1280 / (target_H * target_W),
        kv_ratio=kv_ratio,
        local_range=local_range,
        compile_network=compile_network,
    )


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def build_flashvsr_v1_1(
    *,
    input_H: int,
    input_W: int,
    scale: Literal[2, 4] = 2,
    sparse_ratio: float = 2.0,
    kv_ratio: int = 3,
    local_range: int = 11,
    compile_network: bool = False,
    compile_decoder: bool = False,
    compile_encoder: bool = False,
    color_corrector_implementation: ColorCorrectorImplementation = "cuda",
    enable_sync_and_profile: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
) -> FlashVSRPipelineConfig:
    """Default FlashVSR-v1.1 streaming VSR pipeline.

    Args:
        input_H, input_W: LR input pixel dims; both must be divisible by
            ``128 / scale`` (DiT post-patchify needs 8-window).
        scale: Output is ``input * scale``.
        sparse_ratio: Block-sparse attention budget; ``2.0`` stable,
            ``1.5`` faster.
        kv_ratio: Prior chunks kept in streaming self-attn KV; buffer
            holds ``kv_ratio + 1`` at attention time.
        local_range: Local-block window radius for the topk draft mask.
        compile_network / compile_decoder / compile_encoder: Per-component
            ``torch.compile`` switches.
        color_corrector_implementation: ``"cuda"`` (hand-rolled AdaIN) or
            ``"torch"`` (wavelet + AdaIN reference).
        enable_sync_and_profile: Per-AR-step CUDA-event profiling; adds
            one ``cuda.synchronize()`` per step.
        dtype: Compute dtype. ``bfloat16`` matches FlashVSR-tiny weights.
        seed: Diffusion-model initial-noise RNG seed.

    Weights pulled from HuggingFace via
    :data:`AVAILABLE_FLASHVSR_CHECKPOINT_PATHS`.
    """
    target_H = input_H * scale
    target_W = input_W * scale
    checkpoint_path = AVAILABLE_FLASHVSR_CHECKPOINT_PATHS["v1.1-tiny-long"]
    return FlashVSRPipelineConfig(
        prompt_path=checkpoint_path["prompt"],
        enable_sync_and_profile=enable_sync_and_profile,
        encoder=FlashVSREncoderConfig(
            input_H=input_H,
            input_W=input_W,
            scale=scale,
            projector_checkpoint_path=checkpoint_path["encoder"],
            use_compile=compile_encoder,
            use_cuda_graph=True,
            dtype=dtype,
        ),
        decoder=FlashVSRDecoderConfig(
            tcdecoder_checkpoint_path=checkpoint_path["decoder"],
            use_compile=compile_decoder,
            use_cuda_graph=True,
            color_corrector_implementation=color_corrector_implementation,
            dtype=dtype,
        ),
        diffusion_model=DiffusionModelConfig(
            seed=seed,
            context_noise=0,  # FlashVSR doesn't re-noise between AR steps.
            transformer=_transformer_config(
                target_H=target_H,
                target_W=target_W,
                sparse_ratio=sparse_ratio,
                kv_ratio=kv_ratio,
                local_range=local_range,
                dit_checkpoint_path=checkpoint_path["dit"],
                compile_network=compile_network,
                dtype=dtype,
            ),
            scheduler=_scheduler_config(),
        ),
    )


# Mirrors ``ALPADREAMS_CONFIG_BUILDERS`` in ``alpadreams/config.py``: a
# named registry so callers (``run_flashvsr.py``, downstream notebooks)
# can pick a builder by string. Only one entry today; add more as
# variants land.
FLASHVSR_CONFIG_BUILDERS: dict[str, Callable[..., FlashVSRPipelineConfig]] = {
    "v1.1": build_flashvsr_v1_1,
}
