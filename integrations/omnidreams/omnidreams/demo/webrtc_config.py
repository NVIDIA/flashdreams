# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared OmniDreams WebRTC runtime configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omnidreams.scenes import SCENE_VARIANT_DEFAULT

from flashdreams.serving.webrtc.encoders import EncoderBackend

from .runtime import PipelineFactory
from .spec import DEFAULT_OMNIDREAMS_WEBRTC_SCENE_UUID


@dataclass(frozen=True, slots=True)
class OmnidreamsWebRTCModelRuntimeConfig:
    """Configuration for one scene-driven OmniDreams WebRTC runtime."""

    pipeline_config_name: str
    """User-facing name of the selected OmniDreams pipeline."""

    pipeline_config: Any
    """Resolved single-view OmniDreams pipeline configuration."""

    scene_dir: Path | None = None
    """Local scene root; ``None`` downloads the selected Hugging Face scene."""

    scene_uuid: str | None = DEFAULT_OMNIDREAMS_WEBRTC_SCENE_UUID
    """Scene UUID used for remote lookup or local archive selection."""

    scene_variant: str = SCENE_VARIANT_DEFAULT
    """Weather variant selected from the scene assets."""

    seed: int | None = 42
    """Per-rollout seed; ``None`` selects fresh entropy for every session."""

    device: str = "cuda:0"
    """Device used for rendering and model inference."""

    video_height: int = 704
    """Generated video height in pixels."""

    video_width: int = 1280
    """Generated video width in pixels."""

    fps: int = 30
    """Input sampling and output playback frame rate."""

    camera_name: str = "camera_front_wide_120fov"
    """Scene camera controlled by browser keyboard input."""

    move_speed_per_s: float = 6.0
    """Forward and reverse translation speed in scene units per second."""

    rotate_speed_rad_per_s: float = math.radians(35.0)
    """Left and right rotation speed in radians per second."""

    warmup_chunks: int = 10
    """Number of synthetic chunks generated before accepting sessions."""

    warmup_timeout_s: float = 600.0
    """Maximum duration for WebRTC loopback warmup."""

    debug_serve_hdmaps: bool = False
    """Stream rendered conditioning frames without running video generation."""

    encoder_backend: EncoderBackend = "auto"
    """WebRTC video encoder selection policy."""

    encoder_bitrate_bps: int = 6_000_000
    """Target WebRTC video bitrate in bits per second."""

    encoder_gop: int = 30
    """WebRTC video encoder group-of-pictures length."""

    pipeline_factory: PipelineFactory | None = None
    """Optional test/runtime override for constructing the shared pipeline."""


__all__ = ["OmnidreamsWebRTCModelRuntimeConfig"]
