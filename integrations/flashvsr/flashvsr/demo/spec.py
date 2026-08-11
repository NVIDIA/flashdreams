# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed video-source specifications for FlashVSR demos."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
from loguru import logger

from flashdreams.infra.runner_io import (
    read_video_fps,
    read_video_rgb,
    resolve_input_path,
    rgb_video_to_normalized_tensor,
)
from flashvsr.runtime import TailPolicy

DEFAULT_FLASHVSR_INPUT_URL = (
    "https://raw.githubusercontent.com/OpenImagingLab/FlashVSR/main/"
    "examples/WanVSR/inputs/example1.mp4"
)
FLASHVSR_INPUT_CACHE_DIR = (
    Path(os.path.expanduser(os.getenv("FLASHDREAMS_CACHE_DIR", "~/.cache/flashdreams")))
    / "flashvsr"
)

CropRegion = Literal["none", "bottom_half", "top_half"]


@dataclass(frozen=True, kw_only=True, slots=True)
class FlashVSRVideoScenario:
    """User-facing source-video and chunking options."""

    input_path: str | Path | None = DEFAULT_FLASHVSR_INPUT_URL
    """Optional server-side source; a missing value requires a WebRTC upload."""

    chunk_size: Literal[8, 16] = 16
    fps: float | None = None
    crop_region: CropRegion = "none"
    tail_policy: TailPolicy = "drop"
    loop_input: bool = False

    def __post_init__(self) -> None:
        if self.chunk_size not in {8, 16}:
            raise ValueError("FlashVSR scenario chunk_size must be 8 or 16.")
        if self.fps is not None and self.fps <= 0:
            raise ValueError("FlashVSR scenario fps must be > 0 when provided.")
        if self.crop_region not in {"none", "bottom_half", "top_half"}:
            raise ValueError(
                "FlashVSR crop_region must be 'none', 'bottom_half', or 'top_half'."
            )
        if self.tail_policy not in {"drop", "pad"}:
            raise ValueError("FlashVSR tail_policy must be 'drop' or 'pad'.")


@dataclass(frozen=True, slots=True, eq=False)
class PreparedFlashVSRVideo:
    """Decoded CPU source video plus model/output shape facts."""

    scenario: FlashVSRVideoScenario
    resolved_path: Path
    video: torch.Tensor
    input_height: int
    input_width: int
    target_height: int
    target_width: int
    fps: float

    @property
    def total_frames(self) -> int:
        """Return the number of decoded source frames."""
        return int(self.video.shape[2])


def resolve_video_scenario(value: Any) -> FlashVSRVideoScenario:
    """Normalize a public demo scenario into a FlashVSR video scenario."""
    if value is None:
        return FlashVSRVideoScenario()
    if isinstance(value, FlashVSRVideoScenario):
        return value
    if isinstance(value, str | Path):
        return FlashVSRVideoScenario(input_path=value)
    if not isinstance(value, Mapping):
        raise TypeError(
            "FlashVSR scenario must be a path, mapping, FlashVSRVideoScenario, or None."
        )
    return FlashVSRVideoScenario(
        input_path=value.get(
            "input_path", value.get("input", DEFAULT_FLASHVSR_INPUT_URL)
        ),
        chunk_size=cast(Literal[8, 16], int(value.get("chunk_size", 16))),
        fps=(None if value.get("fps") is None else float(value["fps"])),
        crop_region=cast(CropRegion, str(value.get("crop_region", "none"))),
        tail_policy=cast(TailPolicy, str(value.get("tail_policy", "drop"))),
        loop_input=bool(value.get("loop_input", False)),
    )


def prepare_video_source(
    scenario: FlashVSRVideoScenario,
    *,
    scale: int,
) -> PreparedFlashVSRVideo:
    """Resolve, decode, crop, and normalize one low-resolution input video."""
    if scenario.input_path is None:
        raise ValueError(
            "No FlashVSR input video is configured. Upload an MP4 in the "
            "WebRTC UI or launch with --input."
        )
    resolved_path = resolve_input_path(
        scenario.input_path,
        cache_dir=FLASHVSR_INPUT_CACHE_DIR,
    )
    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"FlashVSR input video does not exist: {scenario.input_path!r}."
        )
    logger.info("Reading FlashVSR demo input {}.", resolved_path)
    video_rgb = read_video_rgb(resolved_path)
    if video_rgb.ndim != 4 or video_rgb.shape[-1] != 3:
        raise ValueError(
            "FlashVSR input decoder must produce [T,H,W,3] RGB frames, "
            f"got {tuple(video_rgb.shape)}."
        )
    if video_rgb.shape[0] <= 0:
        raise ValueError("FlashVSR input video contains no frames.")
    if scenario.crop_region != "none":
        height = int(video_rgb.shape[1])
        half = height // 2
        if half <= 0:
            raise ValueError("FlashVSR input is too short to crop vertically.")
        if scenario.crop_region == "bottom_half":
            video_rgb = video_rgb[:, height - half :, :, :]
        else:
            video_rgb = video_rgb[:, :half, :, :]

    _, height, width, _ = video_rgb.shape
    target_height = (height * scale // 128) * 128
    target_width = (width * scale // 128) * 128
    if target_height <= 0 or target_width <= 0:
        raise ValueError(
            "FlashVSR input is too small after cropping: "
            f"input={height}x{width}, scale={scale}; each scaled axis must be >= 128."
        )
    fps = scenario.fps
    if fps is None:
        try:
            fps = float(read_video_fps(resolved_path))
        except Exception:
            logger.warning("Could not read input fps; using 30 fps.")
            fps = 30.0
    video = (
        rgb_video_to_normalized_tensor(
            video_rgb,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        .permute(1, 0, 2, 3)
        .unsqueeze(0)
    )
    cold_frames = 5 if scenario.chunk_size == 8 else 13
    if (
        not scenario.loop_input
        and scenario.tail_policy == "drop"
        and video.shape[2] < cold_frames
    ):
        raise ValueError(
            f"FlashVSR input has {video.shape[2]} frames; chunk_size="
            f"{scenario.chunk_size} needs at least {cold_frames}."
        )
    return PreparedFlashVSRVideo(
        scenario=scenario,
        resolved_path=resolved_path,
        video=video.contiguous(),
        input_height=height,
        input_width=width,
        target_height=target_height,
        target_width=target_width,
        fps=float(fps),
    )


__all__ = [
    "CropRegion",
    "DEFAULT_FLASHVSR_INPUT_URL",
    "FLASHVSR_INPUT_CACHE_DIR",
    "FlashVSRVideoScenario",
    "PreparedFlashVSRVideo",
    "prepare_video_source",
    "resolve_video_scenario",
]
