# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SANA-WM binding for the shared camera-to-video application."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from cam2v import (
    Cam2VApplication,
    Cam2VApplicationDefaults,
    CameraControlInput,
)
from torch import Tensor

from flashdreams.api_v2.application import IApplication
from flashdreams.infra.config import derive_config
from flashdreams.infra.decoder import StreamingVideoDecoder
from flashdreams.infra.pipeline import StreamInferencePipeline
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout
from sana_wm.config import PIPELINE_SANA_WM_STREAMING
from sana_wm.impl.conditioning import (
    SanaWMStreamingI2VConditioningRequest,
    resolve_sana_wm_conditioning,
)
from sana_wm.impl.constants import DEFAULT_FPS
from sana_wm.impl.controls import SanaWMCameraPoseIntegrator
from sana_wm.impl.decoder import SanaWMDecodedVideo
from sana_wm.impl.scheduler import SanaWMLTXEulerSchedulerConfig
from sana_wm.impl.transformer import SanaWMStreamingTransformerConfig

_CACHE_STATE_ATTRIBUTE = "_sana_wm_cam2v_state"
_INSTALL_HINT = "Install the SANA-WM integration: pip install flashdreams-sana-wm."


@dataclass(kw_only=True)
class _SanaCam2VState:
    """Per-rollout SANA-WM state retained on the pipeline cache."""

    image: Tensor
    """Normalized first-frame tensor consumed by the VAE encoder."""

    prompt: str
    """Text condition shared by every generated block."""

    fps: int
    """Output frame rate."""

    steps: int
    """Streaming Stage-1 denoising steps."""

    flow_shift: float
    """Streaming scheduler flow shift."""

    seed: int
    """Stage-1 random seed."""

    num_frame_per_block: int
    """Latent frames generated in each active block."""

    poses: list[np.ndarray] = field(default_factory=list)
    """Accumulated pixel-rate camera-to-world chunks."""

    intrinsics: list[np.ndarray] = field(default_factory=list)
    """Accumulated pixel-rate calibration chunks."""


class SanaWMCam2VPipeline(StreamInferencePipeline):
    """Adapt SANA-WM's cache and output cadence to the Cam2V contract."""

    def initialize_cache(self, *, text: list[str], image: Tensor) -> Any:
        """Initialize one live rollout from the shared Cam2V static inputs."""
        if len(text) != 1 or not text[0].strip():
            raise ValueError("SANA-WM Cam2V requires exactly one non-empty prompt.")
        if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
            raise ValueError(
                "SANA-WM Cam2V first-frame tensors must have shape [1, 3, H, W]."
            )
        transformer = self.config.diffusion_model.transformer
        scheduler = self.config.diffusion_model.scheduler
        if not isinstance(transformer, SanaWMStreamingTransformerConfig):
            raise TypeError("SANA-WM Cam2V requires the streaming transformer config.")
        if not isinstance(scheduler, SanaWMLTXEulerSchedulerConfig):
            raise TypeError("SANA-WM Cam2V requires the LTX Euler scheduler config.")
        seed = self.config.diffusion_model.seed
        if seed is None:
            raise ValueError("SANA-WM Cam2V requires a fixed diffusion seed.")

        prompt = text[0].strip()
        fps = SANA_WM_CAM2V_DEFAULTS.fps
        cache = super().initialize_cache(decoder_context={"prompt": prompt, "fps": fps})
        setattr(
            cache,
            _CACHE_STATE_ATTRIBUTE,
            _SanaCam2VState(
                image=image,
                prompt=prompt,
                fps=fps,
                steps=int(scheduler.num_inference_steps),
                flow_shift=float(scheduler.shift),
                seed=seed,
                num_frame_per_block=int(transformer.num_frame_per_block),
            ),
        )
        return cache

    def get_num_output_frames(self, autoregressive_index: int) -> int:
        """Return newly decoded frames for one SANA-WM streaming block."""
        transformer = self.config.diffusion_model.transformer
        if not isinstance(transformer, SanaWMStreamingTransformerConfig):
            raise TypeError("SANA-WM Cam2V requires the streaming transformer config.")
        if not isinstance(self.decoder, StreamingVideoDecoder):
            raise TypeError("SANA-WM Cam2V requires the streaming video decoder.")
        latent_frames = int(transformer.num_frame_per_block)
        if autoregressive_index == 0:
            latent_frames += 1
        return int(
            self.decoder.get_output_temporal_size(
                autoregressive_index,
                latent_frames,
            )
        )


def generate_sana_wm_step(
    pipeline: Any,
    autoregressive_index: int,
    cache: Any,
    camera_input: CameraControlInput,
) -> Tensor:
    """Append live camera controls and generate one SANA-WM block."""
    if not isinstance(pipeline, SanaWMCam2VPipeline):
        raise TypeError("SANA-WM Cam2V requires SanaWMCam2VPipeline.")
    state = getattr(cache, _CACHE_STATE_ATTRIBUTE, None)
    if not isinstance(state, _SanaCam2VState):
        raise RuntimeError("SANA-WM Cam2V cache was not initialized.")
    if autoregressive_index != len(state.poses):
        raise ValueError("SANA-WM Cam2V steps must be generated in order.")
    if camera_input.world_scale <= 0:
        raise ValueError("SANA-WM Cam2V world_scale must be > 0.")

    poses = camera_input.poses.detach().to(device="cpu", dtype=torch.float32).numpy()
    intrinsics = (
        camera_input.intrinsics.detach().to(device="cpu", dtype=torch.float32).numpy()
    )
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("SANA-WM Cam2V poses must have shape [T, 4, 4].")
    if intrinsics.shape != (poses.shape[0], 4):
        raise ValueError("SANA-WM Cam2V intrinsics must have shape [T, 4].")
    poses = poses.copy()
    poses[:, :3, 3] /= camera_input.world_scale
    if autoregressive_index == 0:
        poses = np.concatenate([np.eye(4, dtype=np.float32)[None], poses])
        intrinsics = np.concatenate([intrinsics[:1], intrinsics])
    state.poses.append(poses)
    state.intrinsics.append(intrinsics)
    rollout_poses = np.concatenate(state.poses)
    rollout_intrinsics = np.concatenate(state.intrinsics)

    decoded = pipeline.generate(
        autoregressive_index,
        cache,
        input=SanaWMStreamingI2VConditioningRequest(
            image=state.image,
            prompt=state.prompt,
            poses_c2w=rollout_poses,
            intrinsics_vec4=rollout_intrinsics,
            num_frames=len(rollout_poses),
            fps=state.fps,
            steps=state.steps,
            cfg_scale=1.0,
            flow_shift=state.flow_shift,
            seed=state.seed,
            num_frame_per_block=state.num_frame_per_block,
        ),
    )
    if not isinstance(decoded, SanaWMDecodedVideo):
        raise TypeError(
            f"SANA-WM Cam2V expected SanaWMDecodedVideo, got {type(decoded).__name__}."
        )
    frames = torch.from_numpy(np.ascontiguousarray(decoded.video_hwc)).to(
        device=pipeline.device,
        dtype=torch.float32,
    )
    return (frames.permute(0, 3, 1, 2) / 127.5 - 1.0).contiguous()


PIPELINE_SANA_WM_STREAMING_CAM2V = derive_config(
    PIPELINE_SANA_WM_STREAMING,
    _target=SanaWMCam2VPipeline,
)
"""SANA-WM streaming pipeline exposing the shared Cam2V contract."""

SANA_WM_CAM2V_DEFAULTS = Cam2VApplicationDefaults(
    pipeline_config=PIPELINE_SANA_WM_STREAMING_CAM2V,
    input_resolver=resolve_sana_wm_conditioning,
    total_blocks=10,
    pixel_width=1280,
    pixel_height=704,
    first_frame_dtype=torch.bfloat16,
    first_frame_interpolation="lanczos4",
    generate_step=generate_sana_wm_step,
    pose_integrator_factory=SanaWMCameraPoseIntegrator,
    fps=DEFAULT_FPS,
    log_model_timing=True,
    output_layout=VideoTensorLayout.tchw,
    install_hint=_INSTALL_HINT,
    input_defaults={
        "world_scale": 1.0,
        "example_data": False,
        "example_idx": 0,
    },
)
"""SANA-WM defaults for the reusable Cam2V application."""


class SanaWMCam2VApplication(Cam2VApplication):
    """SANA-WM specialization of the shared Cam2V application."""

    def __init__(self) -> None:
        super().__init__(defaults=SANA_WM_CAM2V_DEFAULTS)

    def _validate_frame_size(self, session_desc: Any, pipeline: Any) -> None:
        """Reject output sizes outside SANA-WM's fixed trained resolution."""
        super()._validate_frame_size(session_desc, pipeline)
        expected = (
            SANA_WM_CAM2V_DEFAULTS.pixel_width,
            SANA_WM_CAM2V_DEFAULTS.pixel_height,
        )
        actual = (session_desc.video_width, session_desc.video_height)
        if actual != expected:
            raise ValueError(
                f"SANA-WM Cam2V requires {expected[0]}x{expected[1]}, got "
                f"{actual[0]}x{actual[1]}."
            )


def create_app() -> IApplication:
    """Return the SANA-WM streaming camera-to-video application."""
    return SanaWMCam2VApplication()


__all__ = [
    "SanaWMCam2VApplication",
    "create_app",
    "generate_sana_wm_step",
]
