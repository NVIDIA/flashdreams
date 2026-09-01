# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HY-WorldPlay binding for the reusable Cam2V application."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from cam2v import (
    Cam2VApplication,
    Cam2VApplicationDefaults,
    CameraControlInput,
)
from torch import Tensor

from flashdreams.api_v2.application import IApplication
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout
from hy_worldplay.config import PIPELINE_HY_WORLDPLAY_WAN_I2V_5B
from hy_worldplay.impl._action import HyWorldPlayWanCtrlEncoder
from hy_worldplay.impl._memory import generate_points_in_sphere
from hy_worldplay.impl._pose import parse_pose_data
from hy_worldplay.impl.conditioning import (
    resolve_hy_worldplay_conditioning,
)
from hy_worldplay.impl.pipeline import HyWorldPlayPipelineConfig

_CACHE_STATE_ATTRIBUTE = "_hy_worldplay_cam2v_state"
_INSTALL_HINT = (
    "Install the HY-WorldPlay integration: pip install flashdreams-hy-worldplay."
)


@dataclass(kw_only=True)
class _CameraHistory:
    """Per-rollout latent-rate camera history retained on the pipeline cache."""

    poses: list[Tensor] = field(default_factory=list)
    """Camera-to-world chunks in latent-frame order."""

    intrinsics: list[Tensor] = field(default_factory=list)
    """Pixel-space intrinsic chunks in latent-frame order."""

    memory_configured: bool = False
    """Whether the encoder's reconstituted-context selector is armed."""


def _latent_camera_chunk(
    camera_input: CameraControlInput,
    *,
    autoregressive_index: int,
    temporal_compression_ratio: int,
) -> tuple[Tensor, Tensor]:
    poses = camera_input.poses.detach().to(dtype=torch.float32, device="cpu")
    intrinsics = camera_input.intrinsics.detach().to(
        dtype=torch.float32,
        device="cpu",
    )
    if poses.ndim != 3 or poses.shape[-2:] != (4, 4):
        raise ValueError(
            f"HY-WorldPlay Cam2V poses must have shape [T, 4, 4], got "
            f"{tuple(poses.shape)}."
        )
    if intrinsics.shape != (poses.shape[0], 4):
        raise ValueError(
            "HY-WorldPlay Cam2V intrinsics must have shape [T, 4], got "
            f"{tuple(intrinsics.shape)}."
        )
    if camera_input.world_scale <= 0:
        raise ValueError("HY-WorldPlay Cam2V world_scale must be > 0.")

    start = 0 if autoregressive_index == 0 else temporal_compression_ratio - 1
    indices = torch.arange(start, poses.shape[0], temporal_compression_ratio)
    poses = poses[indices].clone()
    poses[:, :3, 3] /= camera_input.world_scale
    return poses, intrinsics[indices]


def _pose_mapping(poses: Tensor, intrinsics: Tensor) -> dict[str, dict[str, list]]:
    pose_data: dict[str, dict[str, list]] = {}
    for index, (pose, intrinsic) in enumerate(zip(poses, intrinsics, strict=True)):
        fx, fy, cx, cy = intrinsic.tolist()
        if cx == 0 or cy == 0:
            raise ValueError(
                "HY-WorldPlay Cam2V intrinsics require non-zero principal points."
            )
        pose_data[str(index)] = {
            "extrinsic": pose.tolist(),
            "K": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        }
    return pose_data


def _configure_memory(
    encoder: HyWorldPlayWanCtrlEncoder,
    *,
    config: HyWorldPlayPipelineConfig,
    device: torch.device,
) -> None:
    generator = torch.Generator(device=device).manual_seed(config.memory_seed)
    encoder.set_memory_config(
        points_local=generate_points_in_sphere(
            config.memory_points_count,
            config.memory_points_radius,
            generator=generator,
            device=device,
        ),
        context_window_length=config.context_window_length,
        memory_frames=config.memory_frames,
        temporal_context_size=config.temporal_context_size,
        pred_latent_size=config.memory_pred_latent_size,
        fov_h_deg=config.memory_fov_h_deg,
        fov_v_deg=config.memory_fov_v_deg,
        device=device,
    )


def generate_hy_worldplay_step(
    pipeline: Any,
    autoregressive_index: int,
    cache: Any,
    camera_input: CameraControlInput,
) -> Tensor:
    """Bind live camera history and generate one HY-WorldPlay step."""
    encoder = pipeline.encoder
    if not isinstance(encoder, HyWorldPlayWanCtrlEncoder):
        raise TypeError(
            "HY-WorldPlay Cam2V requires HyWorldPlayWanCtrlEncoder, got "
            f"{type(encoder).__name__}."
        )
    pipeline_config = getattr(pipeline, "config", None)
    if not isinstance(pipeline_config, HyWorldPlayPipelineConfig):
        raise TypeError(
            "HY-WorldPlay Cam2V requires HyWorldPlayPipelineConfig, got "
            f"{type(pipeline_config).__name__}."
        )

    history = getattr(cache, _CACHE_STATE_ATTRIBUTE, None)
    if autoregressive_index == 0:
        history = _CameraHistory()
        setattr(cache, _CACHE_STATE_ATTRIBUTE, history)
    if not isinstance(history, _CameraHistory):
        raise RuntimeError("HY-WorldPlay Cam2V rollout must start at AR step 0.")

    poses, intrinsics = _latent_camera_chunk(
        camera_input,
        autoregressive_index=autoregressive_index,
        temporal_compression_ratio=encoder.temporal_compression_ratio,
    )
    history.poses.append(poses)
    history.intrinsics.append(intrinsics)
    rollout_poses = torch.cat(history.poses)
    rollout_intrinsics = torch.cat(history.intrinsics)
    viewmats, Ks, actions = parse_pose_data(
        _pose_mapping(rollout_poses, rollout_intrinsics),
        rollout_poses.shape[0],
    )

    parameter = next(pipeline.parameters())
    encoder.set_action_labels(actions)
    encoder.set_camera_data(
        viewmats.to(dtype=parameter.dtype).unsqueeze(0),
        Ks.to(dtype=parameter.dtype).unsqueeze(0),
    )
    if not history.memory_configured:
        _configure_memory(
            encoder,
            config=pipeline_config,
            device=parameter.device,
        )
        history.memory_configured = True
    return pipeline.generate(autoregressive_index, cache)


HY_WORLDPLAY_CAM2V_DEFAULTS = Cam2VApplicationDefaults(
    pipeline_config=PIPELINE_HY_WORLDPLAY_WAN_I2V_5B,
    input_resolver=resolve_hy_worldplay_conditioning,
    total_blocks=20,
    pixel_width=1280,
    pixel_height=704,
    first_frame_dtype=torch.bfloat16,
    first_frame_interpolation="lanczos4",
    generate_step=generate_hy_worldplay_step,
    fps=16,
    log_model_timing=True,
    output_layout=VideoTensorLayout.tchw,
    install_hint=_INSTALL_HINT,
    input_defaults={
        "world_scale": 2.5,
        "example_data": False,
        "example_idx": 0,
    },
)
"""HY-WorldPlay defaults for the reusable Cam2V application."""


class HyWorldPlayCam2VApplication(Cam2VApplication):
    """HY-WorldPlay specialization of the shared Cam2V application."""

    def __init__(self) -> None:
        super().__init__(defaults=HY_WORLDPLAY_CAM2V_DEFAULTS)


def create_app() -> IApplication:
    """Return a HY-WorldPlay camera-to-video application."""
    return HyWorldPlayCam2VApplication()


__all__ = [
    "HyWorldPlayCam2VApplication",
    "create_app",
    "generate_hy_worldplay_step",
]
