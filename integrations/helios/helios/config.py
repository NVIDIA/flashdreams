# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pipeline and runner config literals for Helios integration."""

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
from helios.pipeline import HeliosStreamingPipeline
from helios.runner import HeliosT2VRunnerConfig

CHECKPOINT_DISTILLED = "BestWishYsh/Helios-Distilled"
CHECKPOINT_BASE = "BestWishYsh/Helios-Base"

_PLACEHOLDER_DIFFUSION = DiffusionModelConfig(
    seed=42,
    transformer=TemplateTransformerConfig(
        network=TemplateDiTConfig(
            in_channels=16,
            context_channels=16,
            model_channels=128,
            num_heads=2,
        ),
        patch_size=(1, 2, 2),
        len_t=9,
        window_size_t=9,
        sink_size_t=0,
        guidance_scale=1.0,
    ),
    scheduler=FlowMatchSchedulerConfig(
        num_inference_steps=6,
        denoising_timesteps=[1000, 800, 600, 400, 200, 0],
        warp_denoising_step=True,
        shift=5.0,
        num_train_timesteps=1000,
    ),
)


@dataclass(kw_only=True)
class HeliosPipelineConfig(StreamInferencePipelineConfig):
    """Config for :class:`HeliosStreamingPipeline`."""

    _target: type[HeliosStreamingPipeline] = field(
        default_factory=lambda: HeliosStreamingPipeline
    )

    name: str = "helios-distilled-t2v-14b"
    checkpoint: str = CHECKPOINT_DISTILLED
    device: str = "cuda"
    pyramid_steps: list[int] = field(default_factory=lambda: [2, 2, 2])
    guidance_scale: float = 1.0
    amplify_first_chunk: bool = True
    history_len: int = 8
    compile: bool = False
    warmup_discard_chunks: int = 0
    flash_attention: bool = True
    enable_parallelism: bool = False
    cp_backend: str = "ulysses"
    group_offload: bool = False
    diffusion_model: DiffusionModelConfig = field(
        default_factory=lambda: _PLACEHOLDER_DIFFUSION
    )
    encoder: None = None
    decoder: None = None


PIPELINE_HELIOS_DISTILLED_T2V_14B = HeliosPipelineConfig(
    name="helios-distilled-t2v-14b",
    checkpoint=CHECKPOINT_DISTILLED,
    pyramid_steps=[2, 2, 2],
    guidance_scale=1.0,
    compile=False,
)

PIPELINE_HELIOS_BASE_T2V_14B = cast(
    HeliosPipelineConfig,
    derive_config(
        PIPELINE_HELIOS_DISTILLED_T2V_14B,
        name="helios-base-t2v-14b",
        checkpoint=CHECKPOINT_BASE,
        pyramid_steps=[20, 20, 20],
        guidance_scale=5.0,
    ),
)

PIPELINE_HELIOS_DISTILLED_T2V_14B_2GPU = cast(
    HeliosPipelineConfig,
    derive_config(
        PIPELINE_HELIOS_DISTILLED_T2V_14B,
        name="helios-distilled-t2v-14b-2gpu",
        enable_parallelism=True,
        cp_backend="ulysses",
    ),
)

PIPELINE_HELIOS_DISTILLED_T2V_14B_OPTIMIZED = cast(
    HeliosPipelineConfig,
    derive_config(
        PIPELINE_HELIOS_DISTILLED_T2V_14B,
        name="helios-distilled-t2v-14b-optimized",
        # Helios pyramid DiT hits Inductor/sympy errors under torch.compile; Panel C
        # uses a discarded warmup chunk to prime FlashAttention/cuDNN instead.
        compile=False,
        warmup_discard_chunks=1,
    ),
)

RUNNER_HELIOS_DISTILLED_T2V_14B = HeliosT2VRunnerConfig(
    runner_name=PIPELINE_HELIOS_DISTILLED_T2V_14B.name,
    description=(
        "Helios-Distilled 14B T2V streaming (33-frame chunks, pyramid [2,2,2], ~19 FPS)."
    ),
    pipeline=PIPELINE_HELIOS_DISTILLED_T2V_14B,
)

RUNNER_HELIOS_BASE_T2V_14B = HeliosT2VRunnerConfig(
    runner_name=PIPELINE_HELIOS_BASE_T2V_14B.name,
    description="Helios-Base 14B T2V streaming (50-step pyramid, highest quality).",
    pipeline=PIPELINE_HELIOS_BASE_T2V_14B,
)

RUNNER_HELIOS_DISTILLED_T2V_14B_2GPU = HeliosT2VRunnerConfig(
    runner_name=PIPELINE_HELIOS_DISTILLED_T2V_14B_2GPU.name,
    description="Helios-Distilled 14B T2V with Ulysses context parallelism (2+ GPUs).",
    pipeline=PIPELINE_HELIOS_DISTILLED_T2V_14B_2GPU,
)

RUNNER_CONFIGS: dict[str, RunnerConfig] = {
    cfg.runner_name: cfg
    for cfg in (
        RUNNER_HELIOS_DISTILLED_T2V_14B,
        RUNNER_HELIOS_BASE_T2V_14B,
        RUNNER_HELIOS_DISTILLED_T2V_14B_2GPU,
    )
}
