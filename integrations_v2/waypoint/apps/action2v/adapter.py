# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Waypoint binding for the shared Action2V application."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import cast

import torch
from action2v import Action2VApplication, Action2VApplicationDefaults
from torch import Tensor

from flashdreams.api_v2.application import IApplication
from flashdreams.core.io.disk import default_flashdreams_cache_dir
from flashdreams.core.io.download import download_to_cache
from flashdreams.infra.config import derive_config
from flashdreams.runtime_v2.session_desc import SessionDesc
from waypoint.config import PIPELINE_WAYPOINT_1_5
from waypoint.impl.input_mapping import WaypointActionMapper
from waypoint.impl.model_session import WaypointModelSession
from waypoint.impl.pipeline import WaypointInferencePipeline

_OUTPUT_WIDTH = 1024
_OUTPUT_HEIGHT = 512
_PLAYBACK_FPS = 60
_DEFAULT_FIRST_FRAME_FILENAME = "crystal_desert_blade.jpg"
_DEFAULT_FIRST_FRAME_URL = (
    "https://raw.githubusercontent.com/Overworldai/Biome/14343a6/seeds/"
    + _DEFAULT_FIRST_FRAME_FILENAME
)

PipelineFactory = Callable[[int, torch.device, bool], WaypointInferencePipeline]
SeedLoader = Callable[[Path], Tensor]


def _seed_loader(path: Path, session_desc: SessionDesc) -> Tensor:
    del session_desc
    return load_seed_display_frames(path)


def _action_mapper(session_desc: SessionDesc, sensitivity: float):
    return WaypointActionMapper(
        video_width=session_desc.video_width,
        video_height=session_desc.video_height,
        mouse_sensitivity=sensitivity,
    )


def _model_session_builder(
    pipeline: object,
    pipeline_lock: threading.Lock,
    session_desc: SessionDesc,
    seed_frames: Tensor,
    seed: int,
) -> WaypointModelSession:
    return WaypointModelSession(
        pipeline=cast(WaypointInferencePipeline, pipeline),
        pipeline_lock=pipeline_lock,
        session_desc=session_desc,
        seed_frames=seed_frames,
        seed=seed,
    )


WAYPOINT_ACTION2V_DEFAULTS = Action2VApplicationDefaults(
    slug="action2v-waypoint-1-5-1b",
    pipeline_factory=lambda seed, device, profile: _create_pipeline(
        seed, device, profile
    ),
    seed_loader=_seed_loader,
    action_mapper_factory=_action_mapper,
    model_session_builder=_model_session_builder,
    pixel_width=_OUTPUT_WIDTH,
    pixel_height=_OUTPUT_HEIGHT,
    fps=_PLAYBACK_FPS,
    default_first_frame_resolver=lambda: _download_default_first_frame(),
    metadata={
        "model": "waypoint-1.5-1b",
        "frames_per_action": 4,
        "internal_resolution": "1024x512",
    },
)
"""Waypoint hooks and native output defaults for the shared Action2V app."""


class WaypointApplication(Action2VApplication):
    """Waypoint specialization of the shared Action2V application."""

    def __init__(
        self,
        *,
        pipeline_factory: PipelineFactory | None = None,
        seed_loader: SeedLoader | None = None,
    ) -> None:
        """Create a lazy Waypoint application with optional test seams."""
        resolved_seed_loader = seed_loader or load_seed_display_frames
        defaults = replace(
            WAYPOINT_ACTION2V_DEFAULTS,
            pipeline_factory=pipeline_factory or _create_pipeline,
            seed_loader=lambda path, desc: resolved_seed_loader(path),
        )
        super().__init__(defaults=defaults)


def load_seed_display_frames(path: Path) -> Tensor:
    """Load one RGB/RGBA image as four normalized 1024x512 TCHW seed frames.

    Args:
        path: Image used to establish the initial world state.

    Returns:
        Four identical float32 RGB frames in the ``[-1, 1]`` range.

    Raises:
        FileNotFoundError: The image path does not exist.
        ValueError: Pillow cannot decode an RGB or RGBA image.
    """
    if not path.is_file():
        raise FileNotFoundError(f"first-frame image does not exist: {path}")

    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as image:
            image.load()
            if image.mode not in {"RGB", "RGBA"}:
                raise ValueError(
                    f"first-frame image must be RGB or RGBA, got mode {image.mode}"
                )
            image = image.convert("RGB").resize(
                (_OUTPUT_WIDTH, _OUTPUT_HEIGHT),
                resample=Image.Resampling.BILINEAR,
            )
            pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    except UnidentifiedImageError as error:
        raise ValueError(f"first-frame image is not decodable: {path}") from error

    frame = pixels.view(_OUTPUT_HEIGHT, _OUTPUT_WIDTH, 3).permute(2, 0, 1)
    return frame.unsqueeze(0).repeat(4, 1, 1, 1).float().div(127.5).sub(1.0)


def _create_pipeline(
    seed: int, device: torch.device, profile: bool
) -> WaypointInferencePipeline:
    config = derive_config(
        PIPELINE_WAYPOINT_1_5,
        diffusion_model={"seed": seed},
        enable_sync_and_profile=profile,
    )
    return cast(
        WaypointInferencePipeline,
        config.setup().to(device=device).eval(),
    )


def _download_default_first_frame() -> Path:
    return download_to_cache(
        _DEFAULT_FIRST_FRAME_URL,
        cache_dir=default_flashdreams_cache_dir() / "default_inputs" / "waypoint",
        filename=_DEFAULT_FIRST_FRAME_FILENAME,
        validator=load_seed_display_frames,
    )


def create_app() -> IApplication:
    """Return a new lazy Waypoint Action2V application."""
    return WaypointApplication()


__all__ = [
    "WAYPOINT_ACTION2V_DEFAULTS",
    "WaypointApplication",
    "create_app",
    "load_seed_display_frames",
]
