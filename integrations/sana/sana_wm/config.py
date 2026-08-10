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

"""Static configs for the SANA-WM runner."""

from __future__ import annotations

from flashdreams.infra.pipeline import StreamInferencePipelineConfig
from flashdreams.infra.runner import RunnerConfig
from sana_wm.conditioning import (
    SanaWMConditioningEncoderConfig,
    SanaWMStreamingConditioningEncoderConfig,
)
from sana_wm.constants import DEFAULT_STREAMING_DENOISING_STEP_LIST
from sana_wm.decoder import SanaWMStreamingVideoDecoderConfig, SanaWMVideoDecoderConfig
from sana_wm.diffusion import SanaWMDiffusionModelConfig
from sana_wm.runner import SanaWMRunnerConfig, SanaWMStreamingRunnerConfig
from sana_wm.scheduler import SanaWMLTXEulerSchedulerConfig
from sana_wm.transformer import (
    SanaWMStreamingTransformerConfig,
    SanaWMTransformerConfig,
)

PIPELINE_SANA_WM_BIDIRECTIONAL = StreamInferencePipelineConfig(
    name="sana-wm-bidirectional",
    encoder=SanaWMConditioningEncoderConfig(),
    diffusion_model=SanaWMDiffusionModelConfig(
        transformer=SanaWMTransformerConfig(),
        scheduler=SanaWMLTXEulerSchedulerConfig(),
        seed=42,
    ),
    decoder=SanaWMVideoDecoderConfig(),
)
"""FlashDreams SANA-WM pipeline."""

PIPELINE_SANA_WM_STREAMING = StreamInferencePipelineConfig(
    name="sana-wm-streaming",
    encoder=SanaWMStreamingConditioningEncoderConfig(),
    diffusion_model=SanaWMDiffusionModelConfig(
        transformer=SanaWMStreamingTransformerConfig(),
        scheduler=SanaWMLTXEulerSchedulerConfig(
            num_inference_steps=len(DEFAULT_STREAMING_DENOISING_STEP_LIST) - 1,
            shift=8.0,
            denoising_step_list=DEFAULT_STREAMING_DENOISING_STEP_LIST,
        ),
        seed=42,
    ),
    decoder=SanaWMStreamingVideoDecoderConfig(),
)
"""FlashDreams SANA-WM streaming pipeline."""

RUNNER_SANA_WM_BIDIRECTIONAL = SanaWMRunnerConfig(
    runner_name=PIPELINE_SANA_WM_BIDIRECTIONAL.name,
    description="SANA-WM bidirectional I2V runner (Stage-1 DiT + LTX-2 refiner).",
    pipeline=PIPELINE_SANA_WM_BIDIRECTIONAL,
)
"""SANA-WM runner config."""

RUNNER_SANA_WM_STREAMING = SanaWMStreamingRunnerConfig(
    runner_name=PIPELINE_SANA_WM_STREAMING.name,
    description=(
        "SANA-WM streaming I2V runner (chunk-causal Stage-1 + streaming "
        "LTX-2 refiner/VAE path)."
    ),
    pipeline=PIPELINE_SANA_WM_STREAMING,
)
"""SANA-WM streaming runner config."""

RUNNER_CONFIGS: dict[str, RunnerConfig] = {
    cfg.runner_name: cfg
    for cfg in (
        RUNNER_SANA_WM_BIDIRECTIONAL,
        RUNNER_SANA_WM_STREAMING,
    )
}
"""SANA-WM runner configs keyed by ``runner_name``."""

__all__ = [
    "PIPELINE_SANA_WM_BIDIRECTIONAL",
    "PIPELINE_SANA_WM_STREAMING",
    "RUNNER_CONFIGS",
    "RUNNER_SANA_WM_BIDIRECTIONAL",
    "RUNNER_SANA_WM_STREAMING",
]
