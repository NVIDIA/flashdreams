# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration-owned defaults and resolved inputs for camera-to-video apps."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch

from flashdreams.runtime_v2.session_desc import BackpressureMode, PresentationMode
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

Cam2VInputResolver = Callable[[Mapping[str, Any]], "Cam2VConditioning"]
"""Resolve application arguments into one session's camera conditioning."""


@dataclass(frozen=True, kw_only=True, slots=True)
class Cam2VConditioning:
    """Static camera-to-video conditioning resolved before a session starts."""

    prompt: str
    """Text condition used to initialize the model cache."""

    first_frame_path: Path
    """Image used to initialize the camera-to-video rollout."""

    base_intrinsics: torch.Tensor
    """Pixel-space camera intrinsics ``[fx, fy, cx, cy]``."""

    world_scale: float
    """Scale applied to camera translations by the model's camera encoder."""

    def __post_init__(self) -> None:
        first_frame_path = Path(self.first_frame_path)
        intrinsics = torch.as_tensor(self.base_intrinsics, dtype=torch.float32)
        if intrinsics.numel() != 4:
            raise ValueError(
                "Cam2VConditioning.base_intrinsics must contain four values."
            )
        if self.world_scale < 0:
            raise ValueError("Cam2VConditioning.world_scale must be >= 0.")
        object.__setattr__(self, "prompt", self.prompt.strip())
        object.__setattr__(self, "first_frame_path", first_frame_path)
        object.__setattr__(self, "base_intrinsics", intrinsics.reshape(1, 4).clone())


@dataclass(frozen=True, kw_only=True, slots=True)
class Cam2VApplicationDefaults:
    """Defaults one model integration contributes to the shared Cam2V app."""

    pipeline_config: Any
    """Model pipeline configuration owned by the integration."""

    input_resolver: Cam2VInputResolver
    """Integration hook that resolves paths and camera calibration."""

    total_blocks: int
    """Default number of autoregressive blocks in one rollout."""

    pixel_width: int
    """Default generated frame width."""

    pixel_height: int
    """Default generated frame height."""

    device: str = "cuda"
    """Device on which the application constructs the shared pipeline."""

    fps: int = 16
    """Generated-video frame rate and model-generation-thread pacing limit."""

    output_layout: VideoTensorLayout = VideoTensorLayout.tchw
    """Tensor layout emitted by the model pipeline."""

    backpressure_mode: BackpressureMode = BackpressureMode.BLOCK
    """Preserve every generated model frame in presentation order."""

    presentation_mode: PresentationMode = PresentationMode.ONLY_PRESENT_NEW
    """Render and transmit the overlay once for each selected model frame."""

    ui_fps: int = 60
    """Rate at which the io-thread reads inputs and runs the UI loop."""

    log_every_blocks: int = 1
    """Default interval between steady-state timing log records."""

    warmup_blocks: int = 5
    """Leading blocks excluded from steady-state FPS."""

    install_hint: str = ""
    """Optional dependency hint included in first-frame loading failures."""

    input_defaults: Mapping[str, Any] = field(default_factory=dict)
    """Integration-owned default prompt, asset paths, and example selection."""

    def __post_init__(self) -> None:
        if self.total_blocks <= 0:
            raise ValueError("Cam2VApplicationDefaults.total_blocks must be > 0.")
        if self.pixel_width <= 0 or self.pixel_height <= 0:
            raise ValueError("Cam2VApplicationDefaults dimensions must be > 0.")
        if self.fps <= 0 or self.ui_fps <= 0:
            raise ValueError("Cam2VApplicationDefaults frame rates must be > 0.")
        if self.log_every_blocks <= 0:
            raise ValueError("Cam2VApplicationDefaults.log_every_blocks must be > 0.")
        if self.warmup_blocks < 0:
            raise ValueError("Cam2VApplicationDefaults.warmup_blocks must be >= 0.")
        object.__setattr__(
            self,
            "input_defaults",
            MappingProxyType(dict(self.input_defaults)),
        )

    @classmethod
    def from_runner_config(
        cls,
        runner_config: Any,
        *,
        input_resolver: Cam2VInputResolver,
        total_blocks: int | None = None,
        install_hint: str = "",
    ) -> "Cam2VApplicationDefaults":
        """Read shared application defaults from an integration runner config."""
        required = ["pipeline", "pixel_height", "pixel_width"]
        if total_blocks is None:
            required.append("total_blocks")
        missing = [name for name in required if not hasattr(runner_config, name)]
        if missing:
            raise TypeError(
                f"Runner config {type(runner_config).__name__} is missing "
                f"camera-to-video application defaults: {missing}."
            )
        input_names = (
            "prompt",
            "prompt_path",
            "image_path",
            "pose_path",
            "intrinsic_path",
            "world_scale",
            "example_data",
            "example_idx",
        )
        return cls(
            pipeline_config=runner_config.pipeline,
            input_resolver=input_resolver,
            total_blocks=(
                int(runner_config.total_blocks)
                if total_blocks is None
                else int(total_blocks)
            ),
            pixel_width=int(runner_config.pixel_width),
            pixel_height=int(runner_config.pixel_height),
            device=str(getattr(runner_config, "device", "cuda")),
            fps=int(getattr(runner_config, "fps", 16)),
            output_layout=_output_layout(runner_config),
            install_hint=install_hint,
            input_defaults={
                name: getattr(runner_config, name)
                for name in input_names
                if hasattr(runner_config, name)
            },
        )


def _output_layout(runner_config: Any) -> VideoTensorLayout:
    """Return a runner's output layout using the v2 enum spelling."""
    value = getattr(runner_config, "postprocess_output_layout", None)
    if value is None:
        return VideoTensorLayout.tchw
    if isinstance(value, VideoTensorLayout):
        return value
    return VideoTensorLayout(str(value))


__all__ = [
    "Cam2VApplicationDefaults",
    "Cam2VConditioning",
    "Cam2VInputResolver",
]
