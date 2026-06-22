# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pipeline and runner config literals for LTX-Video full integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

from flashdreams.infra.config import derive_config
from flashdreams.infra.diffusion.model import DiffusionModelConfig
from flashdreams.infra.diffusion.scheduler.fm import FlowMatchSchedulerConfig
from flashdreams.infra.pipeline import StreamInferencePipelineConfig
from flashdreams.infra.runner import RunnerConfig
from flashdreams.recipes.template.transformer import TemplateTransformerConfig
from flashdreams.recipes.template.transformer.network import TemplateDiTConfig

from ltx_video.pipeline import LTXVideoStreamingPipeline
from ltx_video.runner import LTXVideoT2VRunnerConfig

CHECKPOINT = "Lightricks/LTX-Video"

_PLACEHOLDER_DIFFUSION = DiffusionModelConfig(
    seed=42,
    transformer=TemplateTransformerConfig(
        network=TemplateDiTConfig(
            in_channels=32,
            context_channels=16,
            model_channels=128,
            num_heads=2,
        ),
        patch_size=(2, 2, 2),
        len_t=8,
        window_size_t=8,
        sink_size_t=0,
        guidance_scale=1.0,
    ),
    scheduler=FlowMatchSchedulerConfig(
        num_inference_steps=2,
        denoising_timesteps=[1000, 500],
        warp_denoising_step=True,
        shift=5.0,
        num_train_timesteps=1000,
    ),
)


@dataclass(kw_only=True)
class LTXVideoPipelineConfig(StreamInferencePipelineConfig):
    """Config for :class:`LTXVideoStreamingPipeline`."""

    _target: type[LTXVideoStreamingPipeline] = field(
        default_factory=lambda: LTXVideoStreamingPipeline
    )

    name: str = "ltx-video-t2v-2b"
    checkpoint: str = CHECKPOINT
    device: str = "cuda"
    chunk_frames: int = 25
    chunk_overlap: int = 1
    num_inference_steps: int = 50
    guidance_scale: float = 3.0
    kv_cache: bool = False
    kv_window_size: int | None = None
    compile: bool = False
    cuda_graphs: bool = False
    use_taehv: bool = False
    flash_attention: bool = True
    manual_denoise: bool = False
    diffusion_model: DiffusionModelConfig = field(
        default_factory=lambda: _PLACEHOLDER_DIFFUSION
    )
    encoder: None = None
    decoder: None = None


PIPELINE_LTX_T2V_2B = LTXVideoPipelineConfig(
    name="ltx-video-t2v-2b",
    kv_cache=False,
    compile=False,
    cuda_graphs=False,
    flash_attention=True,
)

PIPELINE_LTX_T2V_2B_OPTIMIZED = cast(
    LTXVideoPipelineConfig,
    derive_config(
        PIPELINE_LTX_T2V_2B,
        name="ltx-video-t2v-2b-optimized",
        kv_cache=True,
        compile=True,
        cuda_graphs=True,
        manual_denoise=True,
        num_inference_steps=50,
    ),
)

PIPELINE_LTX_T2V_2B_TAEHV = cast(
    LTXVideoPipelineConfig,
    derive_config(
        PIPELINE_LTX_T2V_2B_OPTIMIZED,
        name="ltx-video-t2v-2b-taehv",
        use_taehv=True,
    ),
)

RUNNER_LTX_T2V_2B = LTXVideoT2VRunnerConfig(
    runner_name=PIPELINE_LTX_T2V_2B.name,
    description=(
        "LTX-Video 2B T2V streaming (causal VAE chunk decode, ~2s time-to-first-frame)."
    ),
    pipeline=PIPELINE_LTX_T2V_2B,
)

RUNNER_LTX_T2V_2B_OPTIMIZED = LTXVideoT2VRunnerConfig(
    runner_name=PIPELINE_LTX_T2V_2B_OPTIMIZED.name,
    description=(
        "LTX-Video 2B T2V full stack: manual denoise + KV-cache + "
        "torch.compile + CUDA graphs + FlashAttention."
    ),
    pipeline=PIPELINE_LTX_T2V_2B_OPTIMIZED,
)

RUNNER_LTX_T2V_2B_TAEHV = LTXVideoT2VRunnerConfig(
    runner_name=PIPELINE_LTX_T2V_2B_TAEHV.name,
    description="LTX-Video 2B T2V optimized + TAEHV fast decoder.",
    pipeline=PIPELINE_LTX_T2V_2B_TAEHV,
)

RUNNER_CONFIGS: dict[str, RunnerConfig] = {
    cfg.runner_name: cfg
    for cfg in (
        RUNNER_LTX_T2V_2B,
        RUNNER_LTX_T2V_2B_OPTIMIZED,
        RUNNER_LTX_T2V_2B_TAEHV,
    )
}
