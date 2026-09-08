# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static FlashDreams configuration for the published Waypoint 1.5 checkpoint."""

from __future__ import annotations

from flashdreams.infra.diffusion.model import DiffusionModelConfig
from flashdreams.infra.pipeline import StreamInferencePipelineConfig
from waypoint.impl.decoder import WaypointTAEHVDecoderConfig
from waypoint.impl.encoder import WaypointControlEncoderConfig
from waypoint.impl.pipeline import WaypointInferencePipelineConfig
from waypoint.impl.scheduler import WaypointEulerSchedulerConfig
from waypoint.impl.spec import WAYPOINT_1_5
from waypoint.impl.transformer import WaypointTransformerConfig

WAYPOINT_1_5_CHECKPOINT = (
    "https://huggingface.co/Overworld/Waypoint-1.5-1B/resolve/main/model.safetensors"
)
"""Published raw Waypoint 1.5 DiT checkpoint."""

PIPELINE_WAYPOINT_1_5 = WaypointInferencePipelineConfig(
    name="waypoint-1.5-1b",
    diffusion_model=DiffusionModelConfig(
        transformer=WaypointTransformerConfig(checkpoint_path=WAYPOINT_1_5_CHECKPOINT),
        scheduler=WaypointEulerSchedulerConfig(
            num_inference_steps=WAYPOINT_1_5.num_denoising_steps,
            num_train_timesteps=1,
            fixed_timesteps=WAYPOINT_1_5.scheduler_sigmas,
        ),
        context_noise=0,
    ),
    encoder=WaypointControlEncoderConfig(),
    decoder=WaypointTAEHVDecoderConfig(
        use_cuda_graph=False,
        use_compile=False,
    ),
)
"""Waypoint 1.5 DiT, fixed four-step Euler schedule, and matching TAEHV decoder."""

WAYPOINT_CONFIGS: dict[str, StreamInferencePipelineConfig] = {
    PIPELINE_WAYPOINT_1_5.name: PIPELINE_WAYPOINT_1_5,
}
"""Waypoint pipeline variants keyed by stable slug."""
