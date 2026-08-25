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

"""Static pipeline config for the HY-WorldPlay WAN-5B I2V integration."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from wan22.config import PIPELINE_WAN22_TI2V_5B

from action2v import (
    Action2VApplicationDefaults,
    Action2VApplicationHooks,
    Action2VInputPaths,
)
from action2v.controls import CameraPoseIntegrator, KeyboardResampler
from flashdreams.core.io.disk import default_flashdreams_cache_dir
from flashdreams.core.io.download import download_to_cache
from flashdreams.infra.runner_io import load_first_frame_tensor
from flashdreams.infra.diffusion.scheduler import (
    FlowMatchEulerDiscreteSchedulerConfig,
)
from flashdreams.recipes.wan.autoencoder.i2v import WanI2VCtrlEncoderConfig
from flashdreams.recipes.wan.pipeline import WanInferencePipelineConfig
from flashdreams.recipes.wan.transformer.impl.network import WanDiTNetworkConfig
from flashdreams.recipes.wan.transformer.wan21 import Wan21TransformerConfig
from hy_worldplay.impl._action import (
    HyWorldPlayWan21TransformerConfig,
    HyWorldPlayWanCtrlEncoderConfig,
    HyWorldPlayWanDiTNetworkConfig,
)
from hy_worldplay.impl._checkpoint import (
    HY_WORLDPLAY_DISTILLED_CKPT_PATH,
    hy_worldplay_distilled_state_dict_transform,
)
from hy_worldplay.impl._pose import parse_pose_data

__all__ = [
    "HY_WORLDPLAY_APPLICATION_DEFAULTS",
    "HY_WORLDPLAY_APPLICATION_HOOKS",
    "PIPELINE_HY_WORLDPLAY_WAN_I2V_5B",
]

_DEFAULT_PROMPT = (
    "First-person view walking around ancient Athens, with Greek architecture "
    "and marble structures"
)
_EXAMPLE_DATA_BASE_URL = (
    "https://raw.githubusercontent.com/Tencent-Hunyuan/HY-WorldPlay/main/assets"
)
_EXAMPLE_DATA_DIR = default_flashdreams_cache_dir() / "example_data/hy_worldplay"
_EXAMPLE_IMAGE_FILENAME = "test.png"
_EXAMPLE_ACTION_FILENAME = "test_forward_32_latents.json"


def _build_hy_worldplay_pipeline() -> WanInferencePipelineConfig:
    """Deep-copy the Wan 2.2 TI2V-5B recipe and layer the full HY-WorldPlay stack on top.

    Swaps in the distilled 4-step Euler scheduler, the action / camera
    HY encoder, and the HY transformer + DiT network (PRoPE blocks
    enabled). The transformer is copied field-by-field rather than
    derived: :func:`derive_config` can't change the dataclass type, and
    an explicit copy fails loudly if a base field is added.
    """
    pipeline = copy.deepcopy(PIPELINE_WAN22_TI2V_5B)
    pipeline.name = "hy-worldplay-wan-i2v-5b"

    # Distilled WAN-5B fixed-timestep schedule (base recipe stays on UniPC).
    pipeline.diffusion_model.scheduler = FlowMatchEulerDiscreteSchedulerConfig(
        num_inference_steps=4,
        fixed_timesteps=(1000.0, 960.0, 888.8889, 727.2728, 0.0),
    )

    assert isinstance(pipeline.encoder, WanI2VCtrlEncoderConfig)
    pipeline.encoder = HyWorldPlayWanCtrlEncoderConfig(
        encoder=pipeline.encoder.encoder,
    )

    base_t = pipeline.diffusion_model.transformer
    # Narrow to the concrete config so the subclass-only attributes
    # copied below resolve.
    assert isinstance(base_t, Wan21TransformerConfig)
    base_n = base_t.network
    assert isinstance(base_n, WanDiTNetworkConfig)
    pipeline.diffusion_model.transformer = HyWorldPlayWan21TransformerConfig(
        network=HyWorldPlayWanDiTNetworkConfig(
            patch_size=base_n.patch_size,
            text_len=base_n.text_len,
            in_dim=base_n.in_dim,
            dim=base_n.dim,
            ffn_dim=base_n.ffn_dim,
            freq_dim=base_n.freq_dim,
            text_dim=base_n.text_dim,
            out_dim=base_n.out_dim,
            num_heads=base_n.num_heads,
            num_layers=base_n.num_layers,
            cross_attn_norm=base_n.cross_attn_norm,
            cross_attn_enable_img=base_n.cross_attn_enable_img,
            eps=base_n.eps,
            concat_padding_mask=base_n.concat_padding_mask,
            patch_embedding_type=base_n.patch_embedding_type,
            apply_rope_before_kvcache=base_n.apply_rope_before_kvcache,
            use_prope_blocks=True,
        ),
        dtype=base_t.dtype,
        # Inference loads HY-WorldPlay's distilled WAN-5B weights by default;
        # ``--ckpt-path`` overrides with a local ``model.pt``.
        checkpoint_path=HY_WORLDPLAY_DISTILLED_CKPT_PATH,
        state_dict_transform=hy_worldplay_distilled_state_dict_transform,
        batch_shape=base_t.batch_shape,
        # 4-latent AR chunks (not the base recipe's 21); sets total
        # frame counts and RoPE positions.
        len_t=4,
        # CFG is baked into the distilled checkpoint; ``1.0`` skips the
        # uncond branch.
        guidance_scale=1.0,
        # Match the rolling KV window to a single chunk.
        window_size_t=4,
        sink_size_t=base_t.sink_size_t,
        h_extrapolation_ratio=base_t.h_extrapolation_ratio,
        w_extrapolation_ratio=base_t.w_extrapolation_ratio,
        compile_network=base_t.compile_network,
        # CUDA-graph capture is unsafe on the HY-WorldPlay memory-prefill
        # path. The ``CUDAGraphWrapper`` captures pointers into the KV
        # cache, but HY re-runs ``prefill_memory_kv_cache`` every chunk: it
        # resets+repopulates each PRoPE block's memory KV from a *different*
        # FOV-selected frame set (``select_mem_frames_wan``), reallocating
        # the underlying storage. A graph captured on one chunk then replays
        # against another chunk's stale/freed memory-KV slots, decoding to a
        # "shatter" of speckle corruption (deterministic across seeds and
        # prompts; only the captured/replayed chunks are hit, so it presents
        # as one or two garbled chunks mid-rollout). Disabling capture keeps
        # ``compile_network`` (Inductor) -- still ~4x faster diffuse than
        # vendor -- without the unsafe replay. See ``HY_DEBUG_DISABLE_CUDA_GRAPH``.
        use_cuda_graph=False,
        cuda_graph_warmup_iters=base_t.cuda_graph_warmup_iters,
        stamp_image_latent=base_t.stamp_image_latent,
        concat_image_mask_to_latent=base_t.concat_image_mask_to_latent,
        ti2v_first_frame_per_token_timestep=(
            base_t.ti2v_first_frame_per_token_timestep
        ),
        # First-frame context runs at the stabilisation sigma 14, which
        # the distilled checkpoint's AdaLN table is fitted to.
        first_frame_timestep_value=14.0,
    )
    return pipeline


PIPELINE_HY_WORLDPLAY_WAN_I2V_5B = _build_hy_worldplay_pipeline()
"""Wan 2.2 TI2V-5B + HY-WorldPlay distilled stack: HY encoder /
transformer / network with PRoPE blocks and the 4-step Euler schedule.
Production HY-WorldPlay WAN-5B configuration."""


@dataclass(frozen=True, slots=True)
class _HyWorldPlayActionTrace:
    """Parsed action data consumed by the generic Action2V application."""

    poses: Tensor
    intrinsics: Tensor
    world_scale: float
    viewmats: Tensor
    camera_matrices: Tensor
    labels: Tensor


def _create_keyboard_resampler(frames_per_second: int) -> KeyboardResampler:
    """Create the model-neutral keyboard input resampler."""
    return KeyboardResampler(fps=frames_per_second)


def _load_hy_worldplay_actions(
    *,
    action_path: Path,
    calibration_path: Path | None,
    total_blocks: int,
    **_: object,
) -> _HyWorldPlayActionTrace:
    """Parse a HY-WorldPlay pose file into the Action2V trace contract."""
    del calibration_path
    viewmats, camera_matrices, labels = parse_pose_data(
        action_path,
        total_blocks * 4,
    )
    poses = torch.linalg.inv(viewmats).to(torch.float32)
    intrinsics = torch.stack(
        (
            camera_matrices[:, 0, 0],
            camera_matrices[:, 1, 1],
            camera_matrices[:, 0, 2],
            camera_matrices[:, 1, 2],
        ),
        dim=-1,
    ).to(torch.float32)
    return _HyWorldPlayActionTrace(
        poses=poses,
        intrinsics=intrinsics,
        world_scale=1.0,
        viewmats=viewmats,
        camera_matrices=camera_matrices,
        labels=labels,
    )


def _initialize_hy_worldplay_rollout(
    *,
    pipeline: object,
    trace: _HyWorldPlayActionTrace,
    **_: object,
) -> None:
    """Bind parsed action and camera tensors to the HY-WorldPlay encoder."""
    from hy_worldplay.impl._action import HyWorldPlayWanCtrlEncoder

    encoder = getattr(pipeline, "encoder", None)
    if not isinstance(encoder, HyWorldPlayWanCtrlEncoder):
        raise TypeError("HY-WorldPlay Action2V requires HyWorldPlayWanCtrlEncoder.")
    dtype = next(pipeline.parameters()).dtype
    encoder.set_action_labels(trace.labels.unsqueeze(0))
    encoder.set_camera_data(
        trace.viewmats.to(dtype=dtype).unsqueeze(0),
        trace.camera_matrices.to(dtype=dtype).unsqueeze(0),
    )


def _no_per_step_control(**_: object) -> None:
    """Return no per-step payload because HY state is bound on its encoder."""
    return None


def _load_hy_worldplay_example(
    *,
    is_rank_zero: bool,
    example_idx: int,
) -> Action2VInputPaths:
    """Download the upstream HY-WorldPlay example assets on the primary rank."""
    del example_idx
    if is_rank_zero:
        download_to_cache(
            f"{_EXAMPLE_DATA_BASE_URL}/img/{_EXAMPLE_IMAGE_FILENAME}",
            cache_dir=_EXAMPLE_DATA_DIR,
            filename=_EXAMPLE_IMAGE_FILENAME,
        )
        download_to_cache(
            f"{_EXAMPLE_DATA_BASE_URL}/pose/{_EXAMPLE_ACTION_FILENAME}",
            cache_dir=_EXAMPLE_DATA_DIR,
            filename=_EXAMPLE_ACTION_FILENAME,
        )
    return Action2VInputPaths(
        image_path=_EXAMPLE_DATA_DIR / _EXAMPLE_IMAGE_FILENAME,
        action_path=_EXAMPLE_DATA_DIR / _EXAMPLE_ACTION_FILENAME,
    )


HY_WORLDPLAY_APPLICATION_DEFAULTS = Action2VApplicationDefaults(
    title="HY-WorldPlay Action2V",
    slug="action2v-hy-worldplay",
    preset_id=PIPELINE_HY_WORLDPLAY_WAN_I2V_5B.name,
    prompt=_DEFAULT_PROMPT,
    frames_per_second=16,
    pixel_width=1280,
    pixel_height=704,
    total_blocks=4,
)
"""Model-owned defaults for the HY-WorldPlay Action2V binding."""

HY_WORLDPLAY_APPLICATION_HOOKS = Action2VApplicationHooks(
    pipeline_configs={
        PIPELINE_HY_WORLDPLAY_WAN_I2V_5B.name: PIPELINE_HY_WORLDPLAY_WAN_I2V_5B
    },
    image_loader=load_first_frame_tensor,
    action_loader=_load_hy_worldplay_actions,
    example_loader=_load_hy_worldplay_example,
    keyboard_factory=_create_keyboard_resampler,
    camera_factory=CameraPoseIntegrator,
    control_factory=_no_per_step_control,
    rollout_initializer=_initialize_hy_worldplay_rollout,
)
"""Model-owned hooks for the HY-WorldPlay Action2V binding."""


