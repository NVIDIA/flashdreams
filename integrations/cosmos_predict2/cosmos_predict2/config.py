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

"""Configs for non-streaming Cosmos-Predict2 T2V."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from flashdreams.runtime.video_runner import (
    ImageConditionedVideoRunnerConfig,
    VideoRunnerConfig,
)

DEFAULT_PROMPT = (
    "A high-definition video captures the precision of robotic welding in an industrial setting. "
    "The first frame showcases a robotic arm, equipped with a welding torch, positioned over a "
    "large metal structure. The welding process is in full swing, with bright sparks and intense "
    "light illuminating the scene, creating a vivid display of blue and white hues. A significant "
    "amount of smoke billows around the welding area, partially obscuring the view but emphasizing "
    "the heat and activity. The background reveals parts of the workshop environment, including a "
    "ventilation system and various pieces of machinery, indicating a busy and functional industrial "
    "workspace. As the video progresses, the robotic arm maintains its steady position, continuing "
    "the welding process and moving to its left. The welding torch consistently emits sparks and light, "
    "and the smoke continues to rise, diffusing slightly as it moves upward. The metal surface beneath "
    "the torch shows ongoing signs of heating and melting. The scene retains its industrial ambiance, "
    "with the welding sparks and smoke dominating the visual field, underscoring the ongoing nature of "
    "the welding operation."
)
"""Default demo prompt used when no ``--prompt`` is supplied."""

DEFAULT_I2V_IMAGE_URL = "https://media.githubusercontent.com/media/nvidia-cosmos/cosmos-predict2.5/refs/heads/main/assets/base/robot_welding.jpg"


@dataclass(kw_only=True)
class Cosmos2T2VRunnerConfig(VideoRunnerConfig):
    """Runner config for the Cosmos-Predict2 T2V variant."""

    prompt: str | Path = DEFAULT_PROMPT
    pixel_height: int = 720
    pixel_width: int = 1280
    fps: int = 16


@dataclass(kw_only=True)
class Cosmos2I2VRunnerConfig(ImageConditionedVideoRunnerConfig, Cosmos2T2VRunnerConfig):
    """Runner config for the Cosmos-Predict2 I2V variant."""

    image_path: str | Path = DEFAULT_I2V_IMAGE_URL
    image_cache_subdir = "cosmos_predict2"


from flashdreams.infra.diffusion.model import DiffusionModelConfig
from flashdreams.infra.diffusion.scheduler import (
    FlowMatchUniPCSchedulerConfig,
)
from flashdreams.infra.runner import RunnerConfig
from flashdreams.recipes.cosmos.pipeline import CosmosInferencePipelineConfig
from flashdreams.recipes.cosmos.transformer import CosmosTransformerConfig
from flashdreams.recipes.cosmos.transformer.impl.network import (
    CosmosDiTNetworkConfig,
    state_dict_transform,
)
from flashdreams.recipes.wan import WanVAEDecoderConfig, WanVAEEncoderConfig

CHECKPOINT_PATH_POST_TRAINED_2B = (
    "https://huggingface.co/nvidia/Cosmos-Predict2.5-2B/blob/main/base/post-trained/"
    "81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt"
)
"""Cosmos-Predict 2.5 2B post-trained EMA checkpoint."""

PIPELINE_COSMOS2_T2V_2B_720P = CosmosInferencePipelineConfig(
    name="cosmos2-t2v-2b-720p",
    enable_sync_and_profile=True,
    encoder=None,
    decoder=WanVAEDecoderConfig(),
    diffusion_model=DiffusionModelConfig(
        seed=42,
        transformer=CosmosTransformerConfig(
            network=CosmosDiTNetworkConfig(cp_method="ring"),
            checkpoint_path=CHECKPOINT_PATH_POST_TRAINED_2B,
            state_dict_transform=state_dict_transform,
            batch_shape=(),
            len_t=24,
            window_size_t=24,
            # Official code uses formula with 7.0: cond + guidance * (cond - uncond)
            # Equivalent to our formula with 8.0: uncond + guidance * (cond - uncond)
            guidance_scale=8.0,
            compile_network=True,
            use_cuda_graph=False,
        ),
        scheduler=FlowMatchUniPCSchedulerConfig(
            num_inference_steps=35,
            shift=5.0,
            use_kerras_sigma=True,
            enable_tqdm=True,
        ),
    ),
)
RUNNER_COSMOS2_T2V_2B_720P = Cosmos2T2VRunnerConfig(
    runner_name=PIPELINE_COSMOS2_T2V_2B_720P.name,
    description="Cosmos-Predict2 2B T2V at 720p (single AR step, prompt-only).",
    pipeline=PIPELINE_COSMOS2_T2V_2B_720P,
)


PIPELINE_COSMOS2_I2V_2B_720P = CosmosInferencePipelineConfig(
    name="cosmos2-i2v-2b-720p",
    enable_sync_and_profile=True,
    encoder=None,
    decoder=WanVAEDecoderConfig(),
    diffusion_model=DiffusionModelConfig(
        seed=42,
        transformer=CosmosTransformerConfig(
            network=CosmosDiTNetworkConfig(cp_method="ring"),
            checkpoint_path=CHECKPOINT_PATH_POST_TRAINED_2B,
            state_dict_transform=state_dict_transform,
            batch_shape=(),
            len_t=24,
            window_size_t=24,
            guidance_scale=8.0,
            compile_network=True,
            use_cuda_graph=False,
            conditional_frame_timestep=0.1,
        ),
        scheduler=FlowMatchUniPCSchedulerConfig(
            num_inference_steps=35,
            shift=5.0,
            use_kerras_sigma=True,
            enable_tqdm=True,
        ),
    ),
    image_encoder=WanVAEEncoderConfig(),
)
RUNNER_COSMOS2_I2V_2B_720P = Cosmos2I2VRunnerConfig(
    runner_name=PIPELINE_COSMOS2_I2V_2B_720P.name,
    description="Cosmos-Predict2 2B I2V at 720p (single AR step, prompt + first-frame image).",
    pipeline=PIPELINE_COSMOS2_I2V_2B_720P,
)


RUNNER_CONFIGS: dict[str, RunnerConfig] = {
    cfg.runner_name: cfg
    for cfg in (RUNNER_COSMOS2_T2V_2B_720P, RUNNER_COSMOS2_I2V_2B_720P)
}
