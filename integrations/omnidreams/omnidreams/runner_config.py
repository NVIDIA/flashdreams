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

"""Legacy ``flashdreams-run`` configuration for OmniDreams pipelines."""

from flashdreams.infra.runner import RunnerConfig
from omnidreams.config import (
    EXPERIMENT1_BASELINE,
    EXPERIMENT1_SKIP_FINALIZE_KV_CACHE,
    EXPERIMENT1_SKIP_FINALIZE_KV_CACHE_NOISE100,
    EXPERIMENT1_SKIP_FINALIZE_KV_CACHE_NOISE150,
    EXPERIMENT1_SKIP_FINALIZE_KV_CACHE_NOISE250,
    EXPERIMENT1_SKIP_FINALIZE_KV_CACHE_NOISE350,
    MV_2STEPS_CHUNK4_LOC8_PSHUFFLE_LIGHTTAE,
    SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE,
    SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE_NATIVE_PERF,
    SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE_PERF,
    SV_2STEPS_CHUNK2_LOC6_VAE_VAE,
    SV_2STEPS_CHUNK3_LOC6_VAE_VAE,
    SV_2STEPS_CHUNK4_LOC8_PSHUFFLE_LIGHTTAE,
    SV_35STEPS_CHUNK2_LOC24_COSMOS2_2B_RES720P_30FPS_HDMAP_VAE_MADS1M,
    SV_35STEPS_CHUNK48_LOC48_COSMOS2_2B_RES720P_30FPS_HDMAP_VAE_MADS1M,
)
from omnidreams.runner import OmnidreamsRunnerConfig

## Per-variant runner-config literals (slug == ``name``).

_DEFAULT_PROMPT_1V = (
    "Driving scene from a front-facing car camera. Urban environment with roads, "
    "vehicles, pedestrians, traffic signs, and buildings. Clear visibility, "
    "realistic lighting, photorealistic quality. High resolution dashcam footage "
    "of city driving."
)
_DEFAULT_PROMPT_4V = (
    "Wide-angle urban street scene from a low, dashboard-level viewpoint. "
    "A straight two-lane road with a faded center line and curbside parking on "
    "both sides. Parked sedans and SUVs in neutral colors line the curbs. On the "
    "right, a white stucco mid-rise building with blue fabric awnings, rectangular "
    "windows, and small storefronts at street level. On the left, a low commercial "
    "strip with dark trim, glass fronts, signage, and shaded sidewalks. Mature green "
    "trees punctuate both sides. Clear blue sky with sparse soft clouds. Bright midday "
    "sunlight, natural colors, realistic materials, crisp shadows, clean asphalt texture."
)

RUNNER_SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE = OmnidreamsRunnerConfig(
    runner_name=SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE.name,
    description="Single-view 2-step distilled chunk2 (LightVAE + LightTAE).",
    pipeline=SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE,
    prompt=_DEFAULT_PROMPT_1V,
)

RUNNER_SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE_PERF = OmnidreamsRunnerConfig(
    runner_name=SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE_PERF.name,
    description=(
        "Single-view chunk2 perf preset (compile + CUDA graphs across all stages)."
    ),
    pipeline=SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE_PERF,
    prompt=_DEFAULT_PROMPT_1V,
)

RUNNER_SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE_NATIVE_PERF = OmnidreamsRunnerConfig(
    runner_name=SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE_NATIVE_PERF.name,
    description=(
        "Single-view chunk2 native VAE perf preset "
        "(LightVAE FP8 encoder + PyTorch LightTAE decoder)."
    ),
    pipeline=SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE_NATIVE_PERF,
    prompt=_DEFAULT_PROMPT_1V,
)

RUNNER_SV_2STEPS_CHUNK2_LOC6_VAE_VAE = OmnidreamsRunnerConfig(
    runner_name=SV_2STEPS_CHUNK2_LOC6_VAE_VAE.name,
    description="Single-view chunk2 with the full Wan VAE on encoder + decoder.",
    pipeline=SV_2STEPS_CHUNK2_LOC6_VAE_VAE,
    prompt=_DEFAULT_PROMPT_1V,
)

RUNNER_SV_2STEPS_CHUNK3_LOC6_VAE_VAE = OmnidreamsRunnerConfig(
    runner_name=SV_2STEPS_CHUNK3_LOC6_VAE_VAE.name,
    description="Single-view chunk3 (len_t=3) with the full Wan VAE.",
    pipeline=SV_2STEPS_CHUNK3_LOC6_VAE_VAE,
    prompt=_DEFAULT_PROMPT_1V,
)

RUNNER_SV_2STEPS_CHUNK4_LOC8_PSHUFFLE_LIGHTTAE = OmnidreamsRunnerConfig(
    runner_name=SV_2STEPS_CHUNK4_LOC8_PSHUFFLE_LIGHTTAE.name,
    description="Single-view chunk4 with the PixelShuffle HDMap encoder + LightTAE.",
    pipeline=SV_2STEPS_CHUNK4_LOC8_PSHUFFLE_LIGHTTAE,
    prompt=_DEFAULT_PROMPT_1V,
)

RUNNER_MV_2STEPS_CHUNK4_LOC8_PSHUFFLE_LIGHTTAE = OmnidreamsRunnerConfig(
    runner_name=MV_2STEPS_CHUNK4_LOC8_PSHUFFLE_LIGHTTAE.name,
    description="4-camera multi-view chunk4 (PixelShuffle HDMap + LightTAE).",
    pipeline=MV_2STEPS_CHUNK4_LOC8_PSHUFFLE_LIGHTTAE,
    prompt=_DEFAULT_PROMPT_4V,
)

RUNNER_SV_35STEPS_CHUNK2_LOC24_COSMOS2_2B_RES720P_30FPS_HDMAP_VAE_MADS1M = OmnidreamsRunnerConfig(
    runner_name=SV_35STEPS_CHUNK2_LOC24_COSMOS2_2B_RES720P_30FPS_HDMAP_VAE_MADS1M.name,
    description=(
        "Teacher: single-view 35-step UniPC chunk2 (Cosmos2 2B, 720p, CFG=3.0)."
    ),
    pipeline=SV_35STEPS_CHUNK2_LOC24_COSMOS2_2B_RES720P_30FPS_HDMAP_VAE_MADS1M,
    prompt=_DEFAULT_PROMPT_1V,
)

RUNNER_SV_35STEPS_CHUNK48_LOC48_COSMOS2_2B_RES720P_30FPS_HDMAP_VAE_MADS1M = OmnidreamsRunnerConfig(
    runner_name=SV_35STEPS_CHUNK48_LOC48_COSMOS2_2B_RES720P_30FPS_HDMAP_VAE_MADS1M.name,
    description=(
        "Teacher: single-view 35-step bidirectional chunk48 (one rollout, 720p)."
    ),
    pipeline=SV_35STEPS_CHUNK48_LOC48_COSMOS2_2B_RES720P_30FPS_HDMAP_VAE_MADS1M,
    prompt=_DEFAULT_PROMPT_1V,
)

RUNNER_EXPERIMENT1_BASELINE = OmnidreamsRunnerConfig(
    runner_name=EXPERIMENT1_BASELINE.name,
    description="Experiment-1 baseline (re-publishes the chunk2 perf chassis).",
    pipeline=EXPERIMENT1_BASELINE,
    prompt=_DEFAULT_PROMPT_1V,
)

RUNNER_EXPERIMENT1_SKIP_FINALIZE_KV_CACHE = OmnidreamsRunnerConfig(
    runner_name=EXPERIMENT1_SKIP_FINALIZE_KV_CACHE.name,
    description="Experiment-1: skip-finalize-kv-cache ablation.",
    pipeline=EXPERIMENT1_SKIP_FINALIZE_KV_CACHE,
    prompt=_DEFAULT_PROMPT_1V,
)

RUNNER_EXPERIMENT1_SKIP_FINALIZE_KV_CACHE_NOISE350 = OmnidreamsRunnerConfig(
    runner_name=EXPERIMENT1_SKIP_FINALIZE_KV_CACHE_NOISE350.name,
    description="Experiment-1: skip-finalize + denoising_timesteps=[1000, 350].",
    pipeline=EXPERIMENT1_SKIP_FINALIZE_KV_CACHE_NOISE350,
    prompt=_DEFAULT_PROMPT_1V,
)

RUNNER_EXPERIMENT1_SKIP_FINALIZE_KV_CACHE_NOISE250 = OmnidreamsRunnerConfig(
    runner_name=EXPERIMENT1_SKIP_FINALIZE_KV_CACHE_NOISE250.name,
    description="Experiment-1: skip-finalize + denoising_timesteps=[1000, 250].",
    pipeline=EXPERIMENT1_SKIP_FINALIZE_KV_CACHE_NOISE250,
    prompt=_DEFAULT_PROMPT_1V,
)

RUNNER_EXPERIMENT1_SKIP_FINALIZE_KV_CACHE_NOISE150 = OmnidreamsRunnerConfig(
    runner_name=EXPERIMENT1_SKIP_FINALIZE_KV_CACHE_NOISE150.name,
    description="Experiment-1: skip-finalize + denoising_timesteps=[1000, 150].",
    pipeline=EXPERIMENT1_SKIP_FINALIZE_KV_CACHE_NOISE150,
    prompt=_DEFAULT_PROMPT_1V,
)

RUNNER_EXPERIMENT1_SKIP_FINALIZE_KV_CACHE_NOISE100 = OmnidreamsRunnerConfig(
    runner_name=EXPERIMENT1_SKIP_FINALIZE_KV_CACHE_NOISE100.name,
    description="Experiment-1: skip-finalize + denoising_timesteps=[1000, 100].",
    pipeline=EXPERIMENT1_SKIP_FINALIZE_KV_CACHE_NOISE100,
    prompt=_DEFAULT_PROMPT_1V,
)


OMNIDREAMS_RUNNERS: dict[str, RunnerConfig] = {
    cfg.runner_name: cfg
    for cfg in (
        RUNNER_SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE,
        RUNNER_SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE_PERF,
    )
}
"""All shipped Omnidreams runners (single- and multi-view variants),
keyed by ``runner_name``."""

__all__ = [
    "OMNIDREAMS_RUNNERS",
    "RUNNER_EXPERIMENT1_BASELINE",
    "RUNNER_EXPERIMENT1_SKIP_FINALIZE_KV_CACHE",
    "RUNNER_EXPERIMENT1_SKIP_FINALIZE_KV_CACHE_NOISE100",
    "RUNNER_EXPERIMENT1_SKIP_FINALIZE_KV_CACHE_NOISE150",
    "RUNNER_EXPERIMENT1_SKIP_FINALIZE_KV_CACHE_NOISE250",
    "RUNNER_EXPERIMENT1_SKIP_FINALIZE_KV_CACHE_NOISE350",
    "RUNNER_MV_2STEPS_CHUNK4_LOC8_PSHUFFLE_LIGHTTAE",
    "RUNNER_SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE",
    "RUNNER_SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE_NATIVE_PERF",
    "RUNNER_SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE_PERF",
    "RUNNER_SV_2STEPS_CHUNK2_LOC6_VAE_VAE",
    "RUNNER_SV_2STEPS_CHUNK3_LOC6_VAE_VAE",
    "RUNNER_SV_2STEPS_CHUNK4_LOC8_PSHUFFLE_LIGHTTAE",
    "RUNNER_SV_35STEPS_CHUNK2_LOC24_COSMOS2_2B_RES720P_30FPS_HDMAP_VAE_MADS1M",
    "RUNNER_SV_35STEPS_CHUNK48_LOC48_COSMOS2_2B_RES720P_30FPS_HDMAP_VAE_MADS1M",
]
