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

"""Configs for the LingBot-World streaming camera-control I2V model."""

from __future__ import annotations

from pathlib import Path

import torch

from flashdreams.infra.config import derive_config
from flashdreams.infra.diffusion.model import DiffusionModelConfig
from flashdreams.infra.diffusion.scheduler.fm import FlowMatchSchedulerConfig
from flashdreams.infra.runner_io import load_first_frame_tensor
from flashdreams.recipes.taehv import TeahvVAEDecoderConfig
from flashdreams.recipes.wan.autoencoder.vae import (
    WanVAEDecoderConfig,
    WanVAEEncoderConfig,
)
from action2v import (
    Action2VApplicationDefaults,
    Action2VApplicationHooks,
    Action2VInputPaths,
)
from action2v.controls import CameraPoseIntegrator, KeyboardResampler
from lingbot.impl.encoder.camctrl import (
    CamCtrlInput,
    I2VCamCtrlEncoderConfig,
    LingbotI2VCtrlEncoderConfig,
)
from lingbot.impl.example_data import ensure_example_data_downloaded
from lingbot.impl.input_mapping import load_camera_trace
from lingbot.impl.pipeline import LingbotWorldInferencePipelineConfig
from lingbot.impl.transformer import LingbotWorldTransformerConfig
from lingbot.impl.transformer.network import LingbotWorldDiTNetwork14BConfig

LINGBOT_WORLD_V1_CHECKPOINT_PATH = (
    "https://huggingface.co/robbyant/lingbot-world-fast/blob/main/"
    "diffusion_pytorch_model.safetensors.index.json"
)
"""LingBot-World v1 transformer checkpoint index."""

LINGBOT_WORLD_V2_CHECKPOINT_PATH = (
    "https://huggingface.co/robbyant/lingbot-world-v2-14b-causal-fast/blob/main/"
    "transformers/diffusion_pytorch_model.safetensors.index.json"
)
"""LingBot-World v2 transformer checkpoint index."""

CHECKPOINT_PATH = LINGBOT_WORLD_V1_CHECKPOINT_PATH
"""Backward-compatible alias for the LingBot-World v1 checkpoint."""


# Official LingBot-World-Fast pipeline config.
PIPELINE_LINGBOT_WORLD_FAST = LingbotWorldInferencePipelineConfig(
    name="lingbot-world-fast",
    enable_sync_and_profile=True,
    encoder=I2VCamCtrlEncoderConfig(
        i2v=LingbotI2VCtrlEncoderConfig(
            encoder=WanVAEEncoderConfig(),
        ),
    ),
    decoder=WanVAEDecoderConfig(),
    diffusion_model=DiffusionModelConfig(
        seed=42,
        transformer=LingbotWorldTransformerConfig(
            network=LingbotWorldDiTNetwork14BConfig(
                patch_embedding_type="conv3d",
                control_type="cam",
                cp_method="ulysses",
                # 16 noise channels + 4-channel mask + 16-channel image latent
                # (channel-concat I2V layout). Must match the
                # ``concat_image_mask_to_latent=True`` setting below.
                in_dim=16 + 4 + 16,
            ),
            checkpoint_path=LINGBOT_WORLD_V1_CHECKPOINT_PATH,
            stream_checkpoint=True,
            # Single-rollout layout: tensors flow through the stack as
            # ``[T, C, H, W]`` (or ``[T, ...]``) with no leading batch/view dim.
            batch_shape=(),
            # Latent frames the transformer consumes per AR chunk.
            len_t=3,
            # CFG off by default to match the upstream Lingbot checkpoint.
            guidance_scale=1.0,
            window_size_t=63,
            sink_size_t=0,
            # I2V channel-concat (mask + first-frame latent), not stamping.
            stamp_image_latent=False,
            concat_image_mask_to_latent=True,
            compile_network=True,
        ),
        scheduler=FlowMatchSchedulerConfig(
            # Upstream Fast 4-step distilled schedule (matches the
            # LingBot-World-Fast checkpoint).
            num_inference_steps=4,
            denoising_timesteps=[1000, 1000 - 179, 1000 - 358, 1000 - 679],
            warp_denoising_step=True,
            shift=10.0,
            sigma_max=0.999,
            sigma_min=0.0,
            extra_one_step=True,
            num_train_timesteps=1000,
            timestep_dtype=torch.int64,
        ),
    ),
)
# Faster interactive variant for persistent streaming:
# - LightTAE (TAEHV) decoder.
# - Tighter streaming window: ``window_size_t=15`` (down from 63).
# - Static sink: ``sink_size_t=3`` to keep early-frame anchors.
PIPELINE_LINGBOT_WORLD_FAST_TAEHV_WINDOW15_SINK3 = derive_config(
    PIPELINE_LINGBOT_WORLD_FAST,
    name="lingbot-world-fast-taehv-window15-sink3",
    decoder=TeahvVAEDecoderConfig(),
    diffusion_model=dict(
        transformer=dict(
            window_size_t=15,
            sink_size_t=3,
        ),
    ),
)
# LingBot-World v2 uses the same architecture and runtime as v1. The
# transformer checkpoint is the only model-level substitution; it inherits
# the bounded checkpoint loader from the v1 base config.
PIPELINE_LINGBOT_WORLD_V2_14B_CAUSAL_FAST = derive_config(
    PIPELINE_LINGBOT_WORLD_FAST,
    name="lingbot-world-v2-14b-causal-fast",
    diffusion_model=dict(
        transformer=dict(checkpoint_path=LINGBOT_WORLD_V2_CHECKPOINT_PATH),
    ),
)
PIPELINE_LINGBOT_WORLD_V2_14B_CAUSAL_FAST_TAEHV_WINDOW15_SINK3 = derive_config(
    PIPELINE_LINGBOT_WORLD_V2_14B_CAUSAL_FAST,
    name="lingbot-world-v2-14b-causal-fast-taehv-window15-sink3",
    decoder=TeahvVAEDecoderConfig(),
    diffusion_model=dict(
        transformer=dict(
            window_size_t=15,
            sink_size_t=3,
        ),
    ),
)
PIPELINE_CONFIGS: dict[str, LingbotWorldInferencePipelineConfig] = {
    cfg.name: cfg
    for cfg in (
        PIPELINE_LINGBOT_WORLD_FAST,
        PIPELINE_LINGBOT_WORLD_FAST_TAEHV_WINDOW15_SINK3,
        PIPELINE_LINGBOT_WORLD_V2_14B_CAUSAL_FAST,
        PIPELINE_LINGBOT_WORLD_V2_14B_CAUSAL_FAST_TAEHV_WINDOW15_SINK3,
    )
}
"""All shipped LingBot-World pipeline configs, keyed by ``name``."""

DEFAULT_LINGBOT_PRESET = PIPELINE_LINGBOT_WORLD_FAST_TAEHV_WINDOW15_SINK3.name
"""Pipeline preset used by the interactive camera-control application."""

DEFAULT_LINGBOT_PROMPT = (
    "A photorealistic driving scene viewed from a moving camera."
)
"""Fallback prompt for interactive runs without a bundled prompt asset."""


def _create_keyboard_resampler(frames_per_second: int) -> KeyboardResampler:
    """Create a keyboard resampler for the requested generation rate."""
    return KeyboardResampler(fps=frames_per_second)


def _load_lingbot_actions(
    *,
    action_path: Path,
    calibration_path: Path | None,
    total_blocks: int,
    **kwargs: object,
) -> object:
    """Load a LingBot camera trace through the generic Action2V contract."""
    del total_blocks
    if calibration_path is None:
        raise ValueError("LingBot Action2V requires --calibration-path.")
    return load_camera_trace(
        camera_poses_path=action_path,
        camera_intrinsics_path=calibration_path,
        **kwargs,
    )


def _load_lingbot_example(*, is_rank_zero: bool, example_idx: int) -> Action2VInputPaths:
    """Resolve the LingBot example assets as model-neutral Action2V paths."""
    example_dir = ensure_example_data_downloaded(
        is_rank_zero=is_rank_zero,
        example_idx=example_idx,
    )
    return Action2VInputPaths(
        image_path=example_dir / "image.jpg",
        action_path=example_dir / "poses.npy",
        calibration_path=example_dir / "intrinsics.npy",
        prompt_path=example_dir / "prompt.txt",
    )


LINGBOT_APPLICATION_DEFAULTS = Action2VApplicationDefaults(
    title="LingBot-World Action2V",
    slug="action2v-lingbot",
    preset_id=DEFAULT_LINGBOT_PRESET,
    prompt=DEFAULT_LINGBOT_PROMPT,
    frames_per_second=16,
    pixel_width=832,
    pixel_height=464,
    total_blocks=20,
)
"""Model-owned defaults for the LingBot camera-control application."""

LINGBOT_APPLICATION_HOOKS = Action2VApplicationHooks(
    pipeline_configs=PIPELINE_CONFIGS,
    image_loader=load_first_frame_tensor,
    action_loader=_load_lingbot_actions,
    example_loader=_load_lingbot_example,
    keyboard_factory=_create_keyboard_resampler,
    camera_factory=CameraPoseIntegrator,
    control_factory=CamCtrlInput,
)
"""Model-owned hooks for the LingBot camera-control application."""
