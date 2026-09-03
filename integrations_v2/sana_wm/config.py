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

"""Public SANA-WM pipeline configurations."""

from __future__ import annotations

from flashdreams.infra.pipeline import StreamInferencePipelineConfig
from sana_wm.impl.conditioning import (
    SanaWMConditioningEncoderConfig,
    SanaWMStreamingConditioningEncoderConfig,
)
from sana_wm.impl.constants import DEFAULT_STREAMING_DENOISING_STEP_LIST
from sana_wm.impl.decoder import (
    SanaWMStreamingVideoDecoderConfig,
    SanaWMVideoDecoderConfig,
)
from sana_wm.impl.diffusion import SanaWMDiffusionModelConfig
from sana_wm.impl.scheduler import SanaWMLTXEulerSchedulerConfig
from sana_wm.impl.transformer import (
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

SANA_WM_CONFIGS: dict[str, StreamInferencePipelineConfig] = {
    cfg.name: cfg
    for cfg in (PIPELINE_SANA_WM_BIDIRECTIONAL, PIPELINE_SANA_WM_STREAMING)
}
"""Public SANA-WM pipeline configurations, keyed by slug."""

__all__ = [
    "PIPELINE_SANA_WM_BIDIRECTIONAL",
    "PIPELINE_SANA_WM_STREAMING",
    "SANA_WM_CONFIGS",
]
