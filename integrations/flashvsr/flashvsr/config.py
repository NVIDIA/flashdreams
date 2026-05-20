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
from flashdreams.infra.runner import RunnerConfig
from flashvsr.corrector import ColorCorrectorImplementation
from flashvsr.decoder import FlashVSRDecoderConfig
from flashvsr.encoder import FlashVSREncoderConfig
from flashvsr.pipeline import FlashVSRPipelineConfig
from flashvsr.runner import FlashVSRRunnerConfig
from flashvsr.transformer import FlashVSRTransformerConfig
from flashvsr.transformer.network import FlashVSRDiTNetworkConfig

__all__ = [
    "AVAILABLE_FLASHVSR_CHECKPOINT_PATHS",
    "FLASHVSR_CONFIG_BUILDERS",
    "PIPELINE_FLASHVSR_V1_1_FULL_ATTN",
    "PIPELINE_FLASHVSR_V1_1_SPARSE_1_5",
    "PIPELINE_FLASHVSR_V1_1_SPARSE_2_0",
    "RUNNER_CONFIGS",
    "RUNNER_FLASHVSR_V1_1_FULL_ATTN",
    "RUNNER_FLASHVSR_V1_1_SPARSE_1_5",
    "RUNNER_FLASHVSR_V1_1_SPARSE_2_0",
    "build_flashvsr_v1_1",
]


## FlashVSR-v1.1 weight locations

_flashvsr_base = lambda repo: f"https://huggingface.co/JunhaoZhuang/{repo}/resolve/main"
"""Hugging Face base URL for a FlashVSR repo under ``JunhaoZhuang/`` (the
upstream authors' org). Per-file checkpoint URLs append the filename."""

_flashvsr_prompt_path = "https://raw.githubusercontent.com/OpenImagingLab/FlashVSR/main/examples/WanVSR/prompt_tensor/posi_prompt.pth"
"""Raw GitHub URL for FlashVSR's frozen ``posi_prompt.pth`` UMT5
embedding (``[1, 512, 4096]``). Atomically downloaded into
``<FLASHDREAMS_CACHE_DIR>/flashvsr/`` on first use via
:func:`flashvsr.pipeline._load_prompt_tensor`."""

AVAILABLE_FLASHVSR_CHECKPOINT_PATHS: dict[str, dict[str, str]] = {
    "v1.1-tiny-long": {
        "encoder": f"{_flashvsr_base('FlashVSR-v1.1')}/LQ_proj_in.ckpt",
        "decoder": f"{_flashvsr_base('FlashVSR-v1.1')}/TCDecoder.ckpt",
        "dit": f"{_flashvsr_base('FlashVSR-v1.1')}/diffusion_pytorch_model_streaming_dmd.safetensors",
        "prompt": _flashvsr_prompt_path,
    },
}
"""Per-variant ``{component -> URL}`` map for FlashVSR checkpoints. HF
URLs flow through ``hf_hub_download``; the GitHub raw prompt URL flows
through :func:`flashdreams.core.io.download.download_to_cache` (the
prompt is a precomputed UMT5 tensor, not a checkpoint). Mirrors
``alpadreams.config.AVAILABLE_ALPADREAMS_CHECKPOINT_PATHS``."""


## Sub-config helpers


def _scheduler_config() -> FlowMatchSchedulerConfig:
    """1-step flow-match scheduler matching FlashVSR's distilled training.

    ``num_inference_steps=1`` + ``denoising_timesteps=[1000]`` + ``shift=8``
    gives ``sigma(t=1000) = 1``, so the step reduces to
    ``clean = noisy - flow`` (legacy ``cur_latents - noise_pred``).
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
    use_cuda_graph: bool,
    dtype: torch.dtype,
    attention_mode: Literal["sparse", "full"],
) -> FlashVSRTransformerConfig:
    """FlashVSR transformer config at a given target resolution.

    ``topk_ratio = sparse_ratio * 768*1280 / (target_H*target_W)`` keeps
    the absolute top-k budget constant across resolutions (legacy
    upsampler convention).

    Per-rollout latent ``(target_H // 8, target_W // 8)`` is fed to
    :meth:`Wan21Transformer.initialize_autoregressive_cache` by
    :meth:`FlashVSRPipeline.initialize_cache`, not baked in here.
    """
    return FlashVSRTransformerConfig(
        # ``flashvsr_tiny_long`` defaults, with the requested attention backend.
        network=FlashVSRDiTNetworkConfig(attention_mode=attention_mode),
        dtype=dtype,
        checkpoint_path=dit_checkpoint_path,
        batch_shape=(1,),
        len_t=2,
        guidance_scale=1.0,
        topk_ratio=sparse_ratio * 768 * 1280 / (target_H * target_W),
        kv_ratio=kv_ratio,
        local_range=local_range,
        attention_mode=attention_mode,
        compile_network=compile_network,
        use_cuda_graph=use_cuda_graph,
    )


## Builders


def build_flashvsr_v1_1(
    *,
    input_H: int,
    input_W: int,
    scale: Literal[2, 4] = 2,
    sparse_ratio: float = 2.0,
    kv_ratio: int = 3,
    local_range: int = 11,
    compile_network: bool = False,
    use_cuda_graph: bool = False,
    color_corrector_implementation: ColorCorrectorImplementation = "cuda",
    enable_sync_and_profile: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
    recipe_name: str = "flashvsr-v1.1",
    attention_mode: Literal["sparse", "full"] = "sparse",
) -> FlashVSRPipelineConfig:
    """Default FlashVSR-v1.1 streaming VSR pipeline.

    Args:
        input_H, input_W: LR input pixel dims. ``input * scale`` must
            be divisible by 128 (DiT 8-window x 16x pixel-shuffle).
        scale: Output is ``input * scale``.
        sparse_ratio: Block-sparse attention budget. ``2.0`` stable,
            ``1.5`` faster.
        kv_ratio: Prior chunks kept in streaming self-attn KV; buffer
            holds ``kv_ratio + 1`` at attention time.
        local_range: Local-block window radius for the top-k draft mask.
        attention_mode: ``"sparse"`` preserves FlashVSR's legacy block-sparse
            self-attention. ``"full"`` uses dense Wan self-attention and is the
            supported path for multi-GPU context parallelism.
        compile_network: Single ``torch.compile`` switch for DiT +
            encoder projector + decoder. Maps to
            :attr:`FlashVSRTransformerConfig.compile_network` plus the
            encoder / decoder ``use_compile`` knobs.
        use_cuda_graph: Capture the steady-state DiT call into a CUDA
            graph and replay it (Phase 2 of
            ``internal/upsampler/PERF_NOTES.md``). Needs
            ``compile_network=True``. Encoder / decoder cudagraphs are
            hard-coded on inside this builder. Defaults off; flip on
            per-resolution once proven stable.
        color_corrector_implementation: ``"cuda"`` (hand-rolled AdaIN)
            or ``"torch"`` (wavelet + AdaIN reference).
        enable_sync_and_profile: Per-AR-step CUDA-event profiling; adds
            one ``cuda.synchronize()`` per step.
        dtype: Compute dtype. ``bfloat16`` matches FlashVSR-tiny weights.
        seed: Diffusion-model initial-noise RNG seed.
        recipe_name: Slug for the returned pipeline (mirrored into
            ``runner_name`` by callers). Override per variant when
            shipping multiple presets from this builder.

    Weights pulled from :data:`AVAILABLE_FLASHVSR_CHECKPOINT_PATHS`.
    """
    # Post-crop ``target_H/target_W`` follow the encoder's
    # bicubic-then-128-multiple-crop rule (see
    # :class:`flashvsr.encoder.FlashVSREncoderConfig`); ``topk_ratio``
    # has to match the **post-crop** dims to mirror upstream's
    # ``sparse_ratio * 768*1280 / (th*tw)`` formula. For 128-aligned
    # inputs (704x1280, 384x640, ...) this is a no-op floor; for
    # smaller inputs (< 128 / scale on either axis) the builder
    # asserts here so we don't silently divide by zero in the
    # ``topk_ratio`` formula below. The encoder re-asserts the same
    # invariant at ``setup()`` time for direct EncoderConfig users.
    target_H = ((input_H * scale) // 128) * 128
    target_W = ((input_W * scale) // 128) * 128
    assert target_H > 0 and target_W > 0, (
        f"input_H * scale = {input_H * scale} and input_W * scale = "
        f"{input_W * scale} must both be at least 128; got "
        f"input_H={input_H}, input_W={input_W}, scale={scale}."
    )
    checkpoint_path = AVAILABLE_FLASHVSR_CHECKPOINT_PATHS["v1.1-tiny-long"]
    return FlashVSRPipelineConfig(
        recipe_name=recipe_name,
        prompt_path=checkpoint_path["prompt"],
        enable_sync_and_profile=enable_sync_and_profile,
        encoder=FlashVSREncoderConfig(
            input_H=input_H,
            input_W=input_W,
            scale=scale,
            projector_checkpoint_path=checkpoint_path["encoder"],
            use_compile=compile_network,
            use_cuda_graph=True,
            dtype=dtype,
        ),
        decoder=FlashVSRDecoderConfig(
            tcdecoder_checkpoint_path=checkpoint_path["decoder"],
            use_compile=compile_network,
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
                use_cuda_graph=use_cuda_graph,
                dtype=dtype,
                attention_mode=attention_mode,
            ),
            scheduler=_scheduler_config(),
        ),
    )


FLASHVSR_CONFIG_BUILDERS: dict[str, Callable[..., FlashVSRPipelineConfig]] = {
    "v1.1": build_flashvsr_v1_1,
}
"""Slug-keyed builder registry (mirrors ``ALPADREAMS_CONFIG_BUILDERS``).
String-keyed entry point for callers that pick a builder by name."""


## Shipped runner literals

# ``flashdreams-run`` discovers these via the ``flashdreams.runner_configs``
# entry-point group in ``pyproject.toml``. Pipeline ``input_H`` /
# ``input_W`` are placeholders (the runner overrides them with the input
# video's native ``(H, W)`` via ``derive_config`` before ``setup()`` and
# also re-derives ``transformer.topk_ratio`` from the per-video post-crop
# target); the literals therefore exist to supply every non-resolution
# knob plus a valid scaffold for programmatic ``PIPELINE_*.setup()``
# callers. 704x1280 is 128-aligned at ``scale=2`` (-> 1408x2560,
# no encoder crop) and matches Alpadreams' native output.


def _build_sparse_ratio_variant(
    sparse_ratio: float, preset_label: str
) -> tuple[FlashVSRPipelineConfig, FlashVSRRunnerConfig]:
    """(Pipeline, runner) pair for one ``sparse_ratio`` preset.

    Each variant rebuilds via :func:`build_flashvsr_v1_1` rather than
    ``derive_config``-ing siblings, so ``sparse_ratio`` bakes correctly
    into ``transformer.topk_ratio`` inside the builder. ``derive_config``
    sees only the post-bake ``topk_ratio`` and would force us to
    duplicate the ``sparse_ratio * 768*1280 / area`` formula here.
    """
    pipeline = build_flashvsr_v1_1(
        recipe_name=f"flashvsr-v1.1-sparse-ratio-{sparse_ratio}",
        input_H=704,
        input_W=1280,
        scale=2,
        sparse_ratio=sparse_ratio,
        compile_network=True,
        use_cuda_graph=True,
        enable_sync_and_profile=True,
    )
    runner = FlashVSRRunnerConfig(
        runner_name=pipeline.recipe_name,
        description=(
            "FlashVSR-v1.1 streaming video super-resolution "
            f"(2x; sparse_ratio={sparse_ratio} {preset_label} preset; "
            "pipeline dims track the input video)."
        ),
        pipeline=pipeline,
        sparse_ratio=sparse_ratio,
    )
    return pipeline, runner


PIPELINE_FLASHVSR_V1_1_SPARSE_2_0, RUNNER_FLASHVSR_V1_1_SPARSE_2_0 = (
    _build_sparse_ratio_variant(sparse_ratio=2.0, preset_label="stable")
)
PIPELINE_FLASHVSR_V1_1_SPARSE_1_5, RUNNER_FLASHVSR_V1_1_SPARSE_1_5 = (
    _build_sparse_ratio_variant(sparse_ratio=1.5, preset_label="faster")
)

PIPELINE_FLASHVSR_V1_1_FULL_ATTN = build_flashvsr_v1_1(
    recipe_name="flashvsr-v1.1-full-attn",
    input_H=704,
    input_W=1280,
    scale=2,
    attention_mode="full",
    compile_network=True,
    use_cuda_graph=True,
    enable_sync_and_profile=True,
)
RUNNER_FLASHVSR_V1_1_FULL_ATTN = FlashVSRRunnerConfig(
    runner_name=PIPELINE_FLASHVSR_V1_1_FULL_ATTN.recipe_name,
    description=(
        "FlashVSR-v1.1 streaming video super-resolution "
        "(2x; dense full attention; supports multi-GPU context parallelism; "
        "pipeline dims track the input video)."
    ),
    pipeline=PIPELINE_FLASHVSR_V1_1_FULL_ATTN,
)

RUNNER_CONFIGS: dict[str, RunnerConfig] = {
    cfg.runner_name: cfg
    for cfg in (
        RUNNER_FLASHVSR_V1_1_SPARSE_2_0,
        RUNNER_FLASHVSR_V1_1_SPARSE_1_5,
        RUNNER_FLASHVSR_V1_1_FULL_ATTN,
    )
}
"""``{runner_name -> RunnerConfig}`` entry-point map exported under
``flashdreams.runner_configs`` for ``flashdreams-run`` discovery."""
