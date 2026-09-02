# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registered MiniMax H3 workflow and runner configs."""

from __future__ import annotations

from flashdreams.infra.runner import RunnerConfig
from minimax_h3.model import MiniMaxH3DiffusionModelConfig
from minimax_h3.pipeline import MiniMaxH3PipelineConfig
from minimax_h3.runner import (
    MiniMaxH3FL2VARunnerConfig,
    MiniMaxH3Ref2VARunnerConfig,
    MiniMaxH3T2VARunnerConfig,
)
from minimax_h3.transformer import (
    H3_REF_TRANSFORMER_CHECKPOINT,
    MiniMaxH3TransformerConfig,
)

PIPELINE_MINIMAX_H3_T2VA = MiniMaxH3PipelineConfig(
    name="minimax-h3-t2va",
    workflow="t2va",
)
PIPELINE_MINIMAX_H3_FL2VA = MiniMaxH3PipelineConfig(
    name="minimax-h3-fl2va",
    workflow="fl2va",
)
PIPELINE_MINIMAX_H3_REF2VA = MiniMaxH3PipelineConfig(
    name="minimax-h3-ref2va",
    workflow="ref2va",
    diffusion_model=MiniMaxH3DiffusionModelConfig(
        transformer=MiniMaxH3TransformerConfig(
            checkpoint_path=H3_REF_TRANSFORMER_CHECKPOINT,
            device="cuda",
            execution_device="cuda",
            sequential_cpu_offload=False,
        )
    ),
)

RUNNER_MINIMAX_H3_T2VA = MiniMaxH3T2VARunnerConfig(
    runner_name=PIPELINE_MINIMAX_H3_T2VA.name,
    description="MiniMax H3 prompt-to-video generation with low-host-RAM staging.",
    pipeline=PIPELINE_MINIMAX_H3_T2VA,
)
RUNNER_MINIMAX_H3_FL2VA = MiniMaxH3FL2VARunnerConfig(
    runner_name=PIPELINE_MINIMAX_H3_FL2VA.name,
    description=(
        "MiniMax H3 first-frame, last-frame, or dual-keyframe video generation."
    ),
    pipeline=PIPELINE_MINIMAX_H3_FL2VA,
)
RUNNER_MINIMAX_H3_REF2VA = MiniMaxH3Ref2VARunnerConfig(
    runner_name=PIPELINE_MINIMAX_H3_REF2VA.name,
    description="MiniMax H3 ordered image, video, and audio reference generation.",
    pipeline=PIPELINE_MINIMAX_H3_REF2VA,
)

RUNNER_CONFIGS: dict[str, RunnerConfig] = {
    config.runner_name: config
    for config in (
        RUNNER_MINIMAX_H3_T2VA,
        RUNNER_MINIMAX_H3_FL2VA,
        RUNNER_MINIMAX_H3_REF2VA,
    )
}
