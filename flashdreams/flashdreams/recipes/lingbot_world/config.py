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

"""User-facing configs for streaming Lingbot World camera-control I2V.

Hosts both the pre-built :class:`LingbotWorldInferencePipelineConfig`
literals and the per-slug :class:`LingbotWorldRunnerConfig` literals
that drive ``flashdreams-run``. CP size is auto-detected from
``torch.distributed.get_world_size()`` inside the transformer; shape
knobs (batch / view / resolution / per-chunk latent T) are pinned to
canonical Lingbot defaults. The runner-config literals self-register
with :mod:`flashdreams.configs.registry` at import time.
"""

from __future__ import annotations

from typing import cast

from flashdreams.configs.registry import register_runner
from flashdreams.infra.config import derive_config
from flashdreams.infra.diffusion.model import DiffusionModelConfig
from flashdreams.infra.diffusion.scheduler.fm import FlowMatchSchedulerConfig
from flashdreams.infra.runner import RunnerConfig
from flashdreams.recipes.alpadreams.encoder.pixel_shuffle import (
    PixelShuffleVAEEncoderConfig,
)
from flashdreams.recipes.lingbot_world.encoder.camctrl import (
    I2VCamCtrlEncoderConfig,
)
from flashdreams.recipes.lingbot_world.pipeline import (
    LingbotWorldInferencePipelineConfig,
)
from flashdreams.recipes.lingbot_world.runner import LingbotWorldRunnerConfig
from flashdreams.recipes.lingbot_world.transformer import (
    LingbotWorldTransformerConfig,
)
from flashdreams.recipes.lingbot_world.transformer.impl.network import (
    LingbotWorldDiTNetwork14BConfig,
)
from flashdreams.recipes.taehv import TeahvVAEDecoderConfig
from flashdreams.recipes.wan.autoencoder.i2v import I2VCtrlEncoderConfig
from flashdreams.recipes.wan.autoencoder.vae import (
    AVAILABLE_WAN_VAE_CHECKPOINT_PATHS,
    WanVAEDecoderConfig,
    WanVAEEncoderConfig,
)

AVAILABLE_LINGBOT_WORLD_CHECKPOINT_PATHS: dict[str, str] = {
    "LingBot-World-Fast": "https://huggingface.co/robbyant/lingbot-world-fast/blob/main/diffusion_pytorch_model.safetensors.index.json",
}

DEFAULT_VIDEO_HEIGHT = 464
"""Canonical pixel-space height; callers pass the matching latent
``(height, width)`` into :meth:`WanInferencePipeline.initialize_cache`."""

DEFAULT_VIDEO_WIDTH = 832
"""Canonical pixel-space width."""

WAN_VAE_SPATIAL_COMPRESSION = 8
"""Pixel-side / latent-side ratio of the Wan VAE."""


LINGBOT_WORLD_FAST = LingbotWorldInferencePipelineConfig(
    recipe_name="lingbot-world-fast",
    enable_sync_and_profile=True,
    encoder=I2VCamCtrlEncoderConfig(
        i2v=I2VCtrlEncoderConfig(
            encoder=WanVAEEncoderConfig(
                checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"],
            ),
        ),
        plucker=PixelShuffleVAEEncoderConfig(
            frame_selection_mode="last_frame",
        ),
    ),
    decoder=WanVAEDecoderConfig(),
    diffusion_model=DiffusionModelConfig(
        seed=42,
        transformer=LingbotWorldTransformerConfig(
            network=LingbotWorldDiTNetwork14BConfig(
                patch_embedding_type="conv3d",
                control_type="cam",
                in_dim=16 + 4 + 16,
            ),
            checkpoint_path=AVAILABLE_LINGBOT_WORLD_CHECKPOINT_PATHS[
                "LingBot-World-Fast"
            ],
            batch_shape=(1, 1),
            len_t=3,
            guidance_scale=1.0,
            window_size_t=60,
            sink_size_t=0,
            stamp_image_latent=False,
            concat_image_mask_to_latent=True,
            compile_network=True,
        ),
        scheduler=FlowMatchSchedulerConfig(
            num_inference_steps=4,
            denoising_timesteps=[999, 978, 947, 825],
            warp_denoising_step=False,
            shift=8.0,
            sigma_min=0.0,
            extra_one_step=True,
            num_train_timesteps=1000,
        ),
    ),
)
"""LingBot-World-Fast: streaming camera-control I2V chassis.

Wan 2.1 14B with the camera-control block (LingbotWorldDiTNetwork14B).
Composite per-AR-step encoder = Wan VAE I2V + Plücker PixelShuffle.
Wan VAE decoder, 4-step distilled flow-match schedule
``[999, 978, 947, 825]``. ``window_size_t=60`` / ``sink_size_t=0``
are the upstream Lingbot Fast defaults.

``in_dim = 16 + 4 + 16``: 16 noise channels + 4-channel mask +
16-channel image latent (channel-concat I2V layout). Must match
``concat_image_mask_to_latent=True``.
"""

LINGBOT_WORLD_FAST_FLASH = cast(
    LingbotWorldInferencePipelineConfig,
    derive_config(
        LINGBOT_WORLD_FAST,
        recipe_name="lingbot-world-fast-flash",
        decoder=TeahvVAEDecoderConfig(),
        diffusion_model=dict(
            transformer=dict(
                window_size_t=15,
                sink_size_t=3,
            ),
        ),
    ),
)
"""LingBot-World-Fast-Flash: TAEHV decoder + tighter streaming window.

``window_size_t=15`` / ``sink_size_t=3`` and the LightTAE (TAEHV)
decoder for the lowest-latency streaming preset.
"""


LINGBOT_WORLD_CONFIGS: dict[str, LingbotWorldInferencePipelineConfig] = {
    cfg.recipe_name: cfg
    for cfg in (
        LINGBOT_WORLD_FAST,
        LINGBOT_WORLD_FAST_FLASH,
    )
}
"""All shipped Lingbot-World variants, keyed by ``recipe_name``."""


## Per-variant runner-config literals (slug == ``recipe_name``).

_LINGBOT_WORLD_DESCRIPTIONS: dict[str, str] = {
    "lingbot-world-fast": (
        "Lingbot World Fast streaming camera-control I2V (Wan VAE decoder)."
    ),
    "lingbot-world-fast-flash": (
        "Lingbot World Fast-Flash (LightTAE decoder, tighter streaming window)."
    ),
}
"""Per-variant CLI descriptions, keyed by ``recipe_name``."""

LINGBOT_WORLD_RUNNERS: dict[str, RunnerConfig] = {
    name: LingbotWorldRunnerConfig(
        runner_name=name,
        description=_LINGBOT_WORLD_DESCRIPTIONS[name],
        pipeline=cfg,
    )
    for name, cfg in LINGBOT_WORLD_CONFIGS.items()
}
"""All shipped Lingbot-World runners, keyed by ``runner_name``."""

for _name, _cfg in LINGBOT_WORLD_RUNNERS.items():
    register_runner(_name, _cfg, source="builtin")
