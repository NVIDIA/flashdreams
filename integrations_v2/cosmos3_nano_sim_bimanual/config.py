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

"""Pipeline configuration for Cosmos Interactive hand generation."""

from flashdreams.infra.diffusion.model import DiffusionModelConfig
from flashdreams.infra.diffusion.scheduler import FlowMatchSchedulerConfig
from flashdreams.infra.pipeline import StreamInferencePipelineConfig

from . import action_format
from .decoder import HandDecoderConfig
from .encoder import HandActionEncoderConfig
from .pipeline import (
    HandStreamInferencePipelineConfig,
)
from .transformer import HandTransformerConfig

DEFAULT_VIDEO_WIDTH = action_format.OUTPUT_WIDTH
"""Exact content width emitted by the hand decoder."""

DEFAULT_VIDEO_HEIGHT = action_format.OUTPUT_HEIGHT
"""Exact content height emitted by the hand decoder."""

DENOISING_TIMESTEPS = {
    4: (1000, 750, 500, 250),
}
"""GA four-step SDE schedule; shift 5 yields sigmas 1, 15/16, 5/6, 5/8."""

PIPELINE_COSMOS3_NANO_SIM_BIMANUAL = HandStreamInferencePipelineConfig(
    name="cosmos3-nano-sim-bimanual",
    encoder=HandActionEncoderConfig(),
    diffusion_model=DiffusionModelConfig(
        seed=0,
        transformer=HandTransformerConfig(),
        scheduler=FlowMatchSchedulerConfig(
            num_inference_steps=4,
            denoising_timesteps=list(DENOISING_TIMESTEPS[4]),
            shift=5.0,
        ),
    ),
    decoder=HandDecoderConfig(),
)
"""Checkpoint-equivalent baseline with no compilation or cache-policy changes."""

COSMOS3_NANO_SIM_BIMANUAL_CONFIGS: dict[str, StreamInferencePipelineConfig] = {
    PIPELINE_COSMOS3_NANO_SIM_BIMANUAL.name: PIPELINE_COSMOS3_NANO_SIM_BIMANUAL,
}
"""Cosmos Interactive hand variants keyed by stable pipeline slug."""

__all__ = [
    "COSMOS3_NANO_SIM_BIMANUAL_CONFIGS",
    "DENOISING_TIMESTEPS",
    "DEFAULT_VIDEO_HEIGHT",
    "DEFAULT_VIDEO_WIDTH",
    "PIPELINE_COSMOS3_NANO_SIM_BIMANUAL",
]
