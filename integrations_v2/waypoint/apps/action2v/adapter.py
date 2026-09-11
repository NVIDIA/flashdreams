# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Waypoint binding for the shared Action2V application."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

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

_OUTPUT_WIDTH = 1024
_OUTPUT_HEIGHT = 512
_PLAYBACK_FPS = 60
_DEFAULT_FIRST_FRAME_FILENAME = "crystal_desert_blade.jpg"
_DEFAULT_FIRST_FRAME_URL = (
    "https://raw.githubusercontent.com/Overworldai/Biome/14343a6/seeds/"
    + _DEFAULT_FIRST_FRAME_FILENAME
)

SeedLoader = Callable[[Path], Tensor]


def _resolve_first_frame(values: Mapping[str, Any]) -> Path:
    image_path = values.get("image_path")
    if image_path is not None:
        return Path(image_path)
    if values.get("example_data"):
        return _download_example_image()
    raise ValueError("Waypoint Action2V requires --image-path or --example-data.")


def _seed_loader(path: Path, session_desc: SessionDesc) -> Tensor:
    del session_desc
    return load_seed_display_frames(path)


def _action_mapper(session_desc: SessionDesc, sensitivity: float):
    return WaypointActionMapper(
        video_width=session_desc.video_width,
        video_height=session_desc.video_height,
        mouse_sensitivity=sensitivity,
    )


WAYPOINT_ACTION2V_DEFAULTS = Action2VApplicationDefaults(
    slug="action2v-waypoint-1-5-1b",
    pipeline_config=derive_config(
        PIPELINE_WAYPOINT_1_5,
        diffusion_model={"seed": 42},
    ),
    input_resolver=_resolve_first_frame,
    seed_loader=_seed_loader,
    action_mapper_factory=_action_mapper,
    total_blocks=10_000,
    pixel_width=_OUTPUT_WIDTH,
    pixel_height=_OUTPUT_HEIGHT,
    fps=_PLAYBACK_FPS,
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
        pipeline_config: Any | None = None,
        seed_loader: SeedLoader | None = None,
    ) -> None:
        """Create a lazy Waypoint application with optional test seams."""
        resolved_seed_loader = seed_loader or load_seed_display_frames
        defaults = replace(
            WAYPOINT_ACTION2V_DEFAULTS,
            pipeline_config=(
                pipeline_config
                if pipeline_config is not None
                else WAYPOINT_ACTION2V_DEFAULTS.pipeline_config
            ),
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
        raise FileNotFoundError(f"image does not exist: {path}")

    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as image:
            image.load()
            if image.mode not in {"RGB", "RGBA"}:
                raise ValueError(f"image must be RGB or RGBA, got mode {image.mode}")
            image = image.convert("RGB").resize(
                (_OUTPUT_WIDTH, _OUTPUT_HEIGHT),
                resample=Image.Resampling.BILINEAR,
            )
            pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    except UnidentifiedImageError as error:
        raise ValueError(f"image is not decodable: {path}") from error

    frame = pixels.view(_OUTPUT_HEIGHT, _OUTPUT_WIDTH, 3).permute(2, 0, 1)
    return frame.unsqueeze(0).repeat(4, 1, 1, 1).float().div(127.5).sub(1.0)


def _download_example_image() -> Path:
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
