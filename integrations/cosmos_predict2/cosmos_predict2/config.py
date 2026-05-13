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

from cosmos_predict2.runner import Cosmos2T2VRunnerConfig
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
from flashdreams.recipes.wan import WanVAEDecoderConfig

CHECKPOINT_PATH_T2V_2B = (
    "https://huggingface.co/nvidia/Cosmos-Predict2.5-2B/blob/main/base/post-trained/"
    "81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt"
)
"""Cosmos-Predict 2.5 2B post-trained EMA checkpoint."""

PIPELINE_COSMOS2_T2V_2B_720P = CosmosInferencePipelineConfig(
    recipe_name="cosmos2-t2v-2b-720p",
    enable_sync_and_profile=True,
    encoder=None,
    decoder=WanVAEDecoderConfig(),
    diffusion_model=DiffusionModelConfig(
        seed=42,
        transformer=CosmosTransformerConfig(
            network=CosmosDiTNetworkConfig(),
            checkpoint_path=CHECKPOINT_PATH_T2V_2B,
            state_dict_transform=state_dict_transform,
            batch_shape=(),
            len_t=24,
            window_size_t=24,
            guidance_scale=7.0,
            compile_network=False,
            use_cuda_graph=False,
        ),
        scheduler=FlowMatchUniPCSchedulerConfig(
            num_inference_steps=35,
            shift=5.0,
            use_kerras_sigma=True,
        ),
    ),
)
RUNNER_COSMOS2_T2V_2B_720P = Cosmos2T2VRunnerConfig(
    runner_name=PIPELINE_COSMOS2_T2V_2B_720P.recipe_name,
    description="Cosmos-Predict2 2B T2V at 720p (single AR step, prompt-only).",
    pipeline=PIPELINE_COSMOS2_T2V_2B_720P,
)


RUNNER_CONFIGS: dict[str, RunnerConfig] = {
    cfg.runner_name: cfg for cfg in (RUNNER_COSMOS2_T2V_2B_720P,)
}
