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

"""Cosmos3 Nano Sim Bimanual defaults for the HandDreams V2 application."""

from cosmos3_nano_sim_bimanual.config import (
    DENOISING_TIMESTEPS,
    PIPELINE_COSMOS3_NANO_SIM_BIMANUAL,
)
from cosmos3_nano_sim_bimanual.fetch_checkpoint import DEFAULT_CHECKPOINT
from flashdreams.infra.config import derive_config

DEFAULT_HAND_PIPELINE_CONFIG = derive_config(
    PIPELINE_COSMOS3_NANO_SIM_BIMANUAL,
    decoder={
        "backend": "taehv2super",
        "device": "cuda:0",
    },
    diffusion_model={
        "seed": 0,
        "scheduler": {
            "num_inference_steps": 4,
            "denoising_timesteps": list(DENOISING_TIMESTEPS[4]),
        },
        "transformer": {
            "checkpoint": DEFAULT_CHECKPOINT,
            "device": "cuda:0",
            "chunk_size": 2,
            "window_frames": 64,
            "cache_mode": "block",
            "cache_refresh_mode": "chunk",
            "overlap_cache_update": True,
            "compile_denoise": True,
            "compile_denoise_mode": "default",
            "valid_length_cache_attention": True,
            "cuda_graph_denoise": True,
            "cuda_graph_warmup_iters": 2,
        },
    },
)
"""Interactive demo's latency-oriented preset, distinct from the model baseline."""

__all__ = ["DEFAULT_HAND_PIPELINE_CONFIG"]
