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

"""Pre-rolled Wan pipeline configs.

The Wan 2.2 TI2V-5B recipe, shipped as importable config constants.
TI2V mode reuses :class:`WanInferencePipeline`: the I2V control encoder
(over the 5B VAE) seeds the first frame, and the transformer conditions
on it via ``stamp_image_latent`` and ``ti2v_first_frame_per_token_timestep``
(frame-0 tokens see ``t=0``, the rest denoise at the scheduler step). The
``Wan-AI/Wan2.2-TI2V-5B-Diffusers`` checkpoints load through the DiT and
VAE remap transforms. The package's application and downstream runners
(e.g. ``hy_worldplay``) layer their I/O behavior on top.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch
from t2v import (
    T2VApplicationDefaults,
    T2VIntegrationHooks,
    T2VModelState,
    TextInputSpec,
)

from flashdreams.core.checkpoint.remap import remap_checkpoint_keys
from flashdreams.infra.diffusion.model import DiffusionModelConfig
from flashdreams.infra.diffusion.scheduler import (
    FlowMatchUniPCSchedulerConfig,
)
from flashdreams.infra.runner_io import load_first_frame_tensor
from flashdreams.recipes.wan.autoencoder.i2v import WanI2VCtrlEncoderConfig
from flashdreams.recipes.wan.autoencoder.vae import (
    Wan22TI2V5BVAEDecoderConfig,
    Wan22TI2V5BVAEEncoderConfig,
)
from flashdreams.recipes.wan.pipeline import WanInferencePipelineConfig
from flashdreams.recipes.wan.transformer.impl.network import (
    WanDiTNetworkTI2V5BConfig,
)
from flashdreams.recipes.wan.transformer.wan21 import Wan21TransformerConfig

DEFAULT_VIDEO_HEIGHT = 640
"""Default output height for the standard Wan 2.2 TI2V-5B rollout."""

DEFAULT_VIDEO_WIDTH = 1280
"""Default output width for the standard Wan 2.2 TI2V-5B rollout."""

DEFAULT_VIDEO_FPS = 16
"""Default presentation frame rate for the Wan 2.2 TI2V-5B demo."""

WAN22_TI2V_5B_DIT_DIFFUSERS_PATH = (
    "https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers/resolve/main/"
    "transformer/diffusion_pytorch_model.safetensors.index.json"
)
"""HF sharded-checkpoint index for the Wan 2.2 TI2V-5B DiT."""


# Diffusers ``WanTransformer3DModel`` -> ``WanDiTNetwork`` key remap:
# condition embedders, scale/shift table, attention projections
# (``attn1``/``attn2``), and FFN.
_WAN22_TI2V_5B_DIT_KEY_REMAP: dict[str, str] = {
    r"^condition_embedder\.text_embedder\.linear_1\.(.*)$": r"text_embedding.0.\1",
    r"^condition_embedder\.text_embedder\.linear_2\.(.*)$": r"text_embedding.2.\1",
    r"^condition_embedder\.time_embedder\.linear_1\.(.*)$": r"time_embedding.0.\1",
    r"^condition_embedder\.time_embedder\.linear_2\.(.*)$": r"time_embedding.2.\1",
    r"^condition_embedder\.time_proj\.(.*)$": r"time_projection.1.\1",
    r"^scale_shift_table$": r"head.modulation",
    r"^proj_out\.(.*)$": r"head.head.\1",
    r"^blocks\.(\d+)\.attn1\.to_q\.(.*)$": r"blocks.\1.self_attn.q.\2",
    r"^blocks\.(\d+)\.attn1\.to_k\.(.*)$": r"blocks.\1.self_attn.k.\2",
    r"^blocks\.(\d+)\.attn1\.to_v\.(.*)$": r"blocks.\1.self_attn.v.\2",
    r"^blocks\.(\d+)\.attn1\.to_out\.0\.(.*)$": r"blocks.\1.self_attn.o.\2",
    r"^blocks\.(\d+)\.attn2\.to_q\.(.*)$": r"blocks.\1.cross_attn.q.\2",
    r"^blocks\.(\d+)\.attn2\.to_k\.(.*)$": r"blocks.\1.cross_attn.k.\2",
    r"^blocks\.(\d+)\.attn2\.to_v\.(.*)$": r"blocks.\1.cross_attn.v.\2",
    r"^blocks\.(\d+)\.attn2\.to_out\.0\.(.*)$": r"blocks.\1.cross_attn.o.\2",
    r"^blocks\.(\d+)\.attn1\.norm_q\.(.*)$": r"blocks.\1.self_attn.norm_q.\2",
    r"^blocks\.(\d+)\.attn1\.norm_k\.(.*)$": r"blocks.\1.self_attn.norm_k.\2",
    r"^blocks\.(\d+)\.attn2\.norm_q\.(.*)$": r"blocks.\1.cross_attn.norm_q.\2",
    r"^blocks\.(\d+)\.attn2\.norm_k\.(.*)$": r"blocks.\1.cross_attn.norm_k.\2",
    r"^blocks\.(\d+)\.norm2\.(.*)$": r"blocks.\1.norm3.\2",
    r"^blocks\.(\d+)\.scale_shift_table$": r"blocks.\1.modulation",
    r"^blocks\.(\d+)\.ffn\.fc_in\.(.*)$": r"blocks.\1.ffn.0.\2",
    r"^blocks\.(\d+)\.ffn\.fc_out\.(.*)$": r"blocks.\1.ffn.2.\2",
    r"^blocks\.(\d+)\.ffn\.net\.0\.proj\.(.*)$": r"blocks.\1.ffn.0.\2",
    r"^blocks\.(\d+)\.ffn\.net\.2\.(.*)$": r"blocks.\1.ffn.2.\2",
}


def wan22_ti2v_5b_dit_state_dict_transform(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remap a diffusers Wan 2.2 TI2V-5B DiT state-dict to ``WanDiTNetwork`` keys.

    Applied automatically when :data:`PIPELINE_WAN22_TI2V_5B` loads the
    ``Wan-AI/Wan2.2-TI2V-5B-Diffusers/transformer`` checkpoint.
    """
    return remap_checkpoint_keys(state_dict, _WAN22_TI2V_5B_DIT_KEY_REMAP)


PIPELINE_WAN22_TI2V_5B = WanInferencePipelineConfig(
    name="wan22-ti2v-5b",
    enable_sync_and_profile=True,
    # Streaming I2V control encoder over the 5B VAE: AR step 0 encodes the
    # first frame into latent 0 with a one-hot stamp mask; later steps emit
    # a zero mask so the in-network ``stamp_image_latent`` blend is identity.
    encoder=WanI2VCtrlEncoderConfig(
        encoder=Wan22TI2V5BVAEEncoderConfig(),
    ),
    decoder=Wan22TI2V5BVAEDecoderConfig(),
    # No CLIP image branch: ``image_encoder=None`` also disables the matching
    # DiT cross-attention branch (``cross_attn_enable_img=False``).
    image_encoder=None,
    diffusion_model=DiffusionModelConfig(
        seed=42,
        transformer=Wan21TransformerConfig(
            network=WanDiTNetworkTI2V5BConfig(),
            checkpoint_path=WAN22_TI2V_5B_DIT_DIFFUSERS_PATH,
            state_dict_transform=wan22_ti2v_5b_dit_state_dict_transform,
            batch_shape=(),
            len_t=21,
            window_size_t=21,
            guidance_scale=5.0,
            # First-frame conditioning: re-inject the clean image latent each
            # step (stamp) and give frame-0 tokens ``t=0`` while the rest
            # denoise at the scheduler step.
            stamp_image_latent=True,
            ti2v_first_frame_per_token_timestep=True,
            # 5B injects the first frame via the stamp path, not channel-concat.
            concat_image_mask_to_latent=False,
        ),
        scheduler=FlowMatchUniPCSchedulerConfig(
            num_inference_steps=40,
            shift=5.0,
        ),
    ),
)
"""Wan 2.2 TI2V-5B inference pipeline (Wan-AI diffusers checkpoint).

One AR step covers the standard 81-frame / 640x1280 rollout
(``len_t == window_size_t == 21``). Base recipe for
``integrations_v2/hy_worldplay``.
"""

TI2V_DEFAULT_PROMPT = "A cinematic scene continuing naturally from the first frame."
"""Default prompt exposed to the shared TI2V application."""

TI2V_IMAGE_PATH_FIELD = "image_path"
"""Text-field key used for the TI2V first-frame path."""


def _configure_ti2v_argument_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--image-path",
        type=Path,
        required=True,
        help="Initial image path shown in the SlangPy UI.",
    )


def _ti2v_text_values_from_arguments(args: argparse.Namespace) -> Mapping[str, str]:
    return {TI2V_IMAGE_PATH_FIELD: str(args.image_path)}


def _validate_ti2v_text_values(values: Mapping[str, str]) -> None:
    path = Path(values[TI2V_IMAGE_PATH_FIELD])
    if not path.is_file():
        raise FileNotFoundError(f"First-frame image does not exist: {path}")


def _validate_ti2v_total_blocks(total_blocks: int) -> None:
    if total_blocks != 1:
        raise ValueError("Wan 2.2 TI2V generates exactly one block.")


def _ti2v_cache_initializer(
    image_loader: Callable[..., torch.Tensor],
) -> Callable[[T2VModelState], Any]:
    def initialize_cache(state: T2VModelState) -> Any:
        image_path = Path(state.text_values[TI2V_IMAGE_PATH_FIELD])
        if not image_path.is_file():
            raise FileNotFoundError(f"First-frame image does not exist: {image_path}")
        device = torch.device(getattr(state.pipeline, "device", "cuda"))
        first_frame = image_loader(
            image_path,
            pixel_height=state.session_desc.video_height,
            pixel_width=state.session_desc.video_width,
            device=device,
            dtype=torch.bfloat16,
        )
        return state.pipeline.initialize_cache(text=[state.prompt], image=first_frame)

    return initialize_cache


def create_ti2v_application_hooks(
    image_loader: Callable[..., torch.Tensor] = load_first_frame_tensor,
) -> T2VIntegrationHooks:
    """Create Wan 2.2 hooks for the shared TI2V application.

    Args:
        image_loader: First-frame tensor loader.

    Returns:
        Model-specific hooks consumed by the shared application.
    """
    return T2VIntegrationHooks(
        configure_argument_parser=_configure_ti2v_argument_parser,
        text_values_from_arguments=_ti2v_text_values_from_arguments,
        validate_text_values=_validate_ti2v_text_values,
        validate_total_blocks=_validate_ti2v_total_blocks,
        initialize_cache=_ti2v_cache_initializer(image_loader),
    )


TI2V_APPLICATION_DEFAULTS = T2VApplicationDefaults(
    pipeline_config=PIPELINE_WAN22_TI2V_5B,
    total_blocks=1,
    pixel_height=DEFAULT_VIDEO_HEIGHT,
    pixel_width=DEFAULT_VIDEO_WIDTH,
    prompt=TI2V_DEFAULT_PROMPT,
    fps=DEFAULT_VIDEO_FPS,
    text_inputs=(
        TextInputSpec(
            name=TI2V_IMAGE_PATH_FIELD,
            label="First frame",
            required=True,
        ),
    ),
)
"""Defaults exposed to the shared TI2V application."""

TI2V_APPLICATION_HOOKS = create_ti2v_application_hooks()
"""Wan 2.2 hooks exposed to the shared TI2V application."""

WAN_CONFIGS: dict[str, WanInferencePipelineConfig] = {
    PIPELINE_WAN22_TI2V_5B.name: PIPELINE_WAN22_TI2V_5B,
}
"""All in-tree Wan pipeline configs, keyed by ``name``."""


__all__ = [
    "DEFAULT_VIDEO_FPS",
    "DEFAULT_VIDEO_HEIGHT",
    "DEFAULT_VIDEO_WIDTH",
    "PIPELINE_WAN22_TI2V_5B",
    "TI2V_APPLICATION_DEFAULTS",
    "TI2V_APPLICATION_HOOKS",
    "WAN22_TI2V_5B_DIT_DIFFUSERS_PATH",
    "WAN_CONFIGS",
    "create_ti2v_application_hooks",
    "wan22_ti2v_5b_dit_state_dict_transform",
]
