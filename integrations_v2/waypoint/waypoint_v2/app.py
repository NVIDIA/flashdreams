# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Waypoint 1.5 V2 application."""

from __future__ import annotations

import argparse
import secrets
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from loguru import logger
from torch import Tensor

from flashdreams.api_v2.application import IApplication
from flashdreams.core.io.disk import default_flashdreams_cache_dir
from flashdreams.core.io.download import download_to_cache
from flashdreams.infra.config import derive_config
from flashdreams.runtime_v2.session_desc import (
    BackpressureMode,
    PresentationMode,
    SessionDesc,
)
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout
from waypoint import WaypointControl, load_controls_from_file
from waypoint.config import PIPELINE_WAYPOINT_1_5
from waypoint.pipeline import WaypointInferencePipeline

from waypoint_v2.session import WaypointSession

_OUTPUT_WIDTH = 1280
_OUTPUT_HEIGHT = 720
_PLAYBACK_FPS = 60
_EXAMPLE_IMAGE_FILENAME = "crystal_desert_blade.jpg"
_EXAMPLE_IMAGE_URL = (
    "https://raw.githubusercontent.com/Overworldai/Biome/14343a6/seeds/"
    + _EXAMPLE_IMAGE_FILENAME
)
_EXAMPLE_CONTROLS = Path(__file__).parent / "assets" / "example_controls.json"

PipelineFactory = Callable[[int, torch.device, bool], WaypointInferencePipeline]
SeedLoader = Callable[[Path], Tensor]
ExampleResolver = Callable[[], Path]


@dataclass(frozen=True, slots=True)
class _ApplicationConfig:
    seed_image: Path | None
    controls: tuple[WaypointControl, ...] | None
    seed: int
    device: torch.device
    profile: bool
    mouse_sensitivity: float


class WaypointApplication(IApplication):
    """Load one Waypoint model and create isolated image-established sessions."""

    def __init__(
        self,
        *,
        pipeline_factory: PipelineFactory | None = None,
        seed_loader: SeedLoader | None = None,
        example_resolver: ExampleResolver | None = None,
    ) -> None:
        """Create a lazy Waypoint application.

        Args:
            pipeline_factory: Test seam replacing real checkpoint construction.
            seed_loader: Test seam replacing image decode and normalization.
            example_resolver: Test seam replacing the example image download.
        """
        self._pipeline_factory = pipeline_factory or _create_pipeline
        self._seed_loader = seed_loader or load_seed_display_frames
        self._example_resolver = example_resolver or _download_example_image
        self._config: _ApplicationConfig | None = None
        self._pipeline: WaypointInferencePipeline | None = None
        self._pipeline_lock = threading.Lock()

    def init(self, commandline_args: Sequence[str]) -> None:
        """Parse seed, control mode, device, and deterministic rollout settings.

        Args:
            commandline_args: Application-specific command-line arguments.

        Raises:
            ValueError: Inputs do not describe a valid file or live rollout.
        """
        parser = argparse.ArgumentParser(
            prog="waypoint-1.5-1b",
            description="Run Waypoint 1.5 from an image using file or live controls.",
        )
        parser.add_argument("--seed-image", type=Path)
        parser.add_argument("--example-data", action="store_true")
        parser.add_argument("--controls-file", type=Path)
        parser.add_argument(
            "--actions",
            type=int,
            help="Use this many actions from a controls file.",
        )
        parser.add_argument("--seed", type=int)
        parser.add_argument("--device", default="cuda")
        parser.add_argument("--profile", action="store_true")
        parser.add_argument("--mouse-sensitivity", type=float, default=1.0)
        args = parser.parse_args(list(commandline_args))

        if args.seed_image is None and not args.example_data:
            raise ValueError("pass --seed-image or --example-data")
        if args.seed is not None and args.seed < 0:
            raise ValueError(f"--seed must be non-negative, got {args.seed}")
        if (
            not torch.isfinite(torch.tensor(args.mouse_sensitivity)).item()
            or args.mouse_sensitivity < 0
        ):
            raise ValueError("--mouse-sensitivity must be finite and non-negative")

        controls_path = args.controls_file
        if controls_path is None and args.example_data:
            controls_path = _EXAMPLE_CONTROLS
        controls = (
            load_controls_from_file(controls_path)
            if controls_path is not None
            else None
        )
        if args.actions is not None:
            if controls is None:
                raise ValueError("--actions requires --controls-file or --example-data")
            if not 1 <= args.actions <= len(controls):
                raise ValueError(
                    f"--actions must be in [1, {len(controls)}], got {args.actions}"
                )
            controls = controls[: args.actions]

        seed = secrets.randbits(63) if args.seed is None else args.seed
        if args.seed is None:
            logger.info(f"[waypoint-1.5-1b] generated seed {seed}")
        self._config = _ApplicationConfig(
            seed_image=args.seed_image,
            controls=controls,
            seed=seed,
            device=torch.device(args.device),
            profile=args.profile,
            mouse_sensitivity=args.mouse_sensitivity,
        )

    def session_desc(self) -> SessionDesc:
        """Return Waypoint's fixed 720p TCHW presentation contract."""
        return SessionDesc(
            output_layout=VideoTensorLayout.tchw,
            backpressure_mode=BackpressureMode.BLOCK,
            presentation_mode=PresentationMode.ONLY_PRESENT_NEW,
            frames_per_second_for_ui=_PLAYBACK_FPS,
            frames_per_second_for_step=_PLAYBACK_FPS,
            video_width=_OUTPUT_WIDTH,
            video_height=_OUTPUT_HEIGHT,
            metadata={
                "model": "waypoint-1.5-1b",
                "frames_per_action": 4,
                "internal_resolution": "1024x512",
            },
        )

    def create_session(self, session_desc: SessionDesc) -> WaypointSession:
        """Resolve cheap inputs, load the shared model once, and create a session.

        Args:
            session_desc: Runtime-requested presentation settings.

        Returns:
            An uninitialized session with independent cache, controls, and RNG.

        Raises:
            RuntimeError: The application has not been initialized.
            ValueError: Waypoint cannot honor the requested layout or dimensions.
        """
        config = self._require_config()
        _validate_session_desc(session_desc)
        seed_image = config.seed_image or self._example_resolver()
        seed_frames = self._seed_loader(seed_image)
        pipeline = self._ensure_pipeline(config)
        return WaypointSession(
            pipeline=pipeline,
            pipeline_lock=self._pipeline_lock,
            session_desc=session_desc,
            seed_frames=seed_frames,
            seed=config.seed,
            controls=config.controls,
            mouse_sensitivity=config.mouse_sensitivity,
        )

    def close(self) -> None:
        """Release the application-owned model reference."""
        self._pipeline = None

    def _require_config(self) -> _ApplicationConfig:
        if self._config is None:
            raise RuntimeError("WaypointApplication.init() must run first")
        return self._config

    def _ensure_pipeline(self, config: _ApplicationConfig) -> WaypointInferencePipeline:
        if self._pipeline is None:
            self._pipeline = self._pipeline_factory(
                config.seed, config.device, config.profile
            )
        return self._pipeline


def load_seed_display_frames(path: Path) -> Tensor:
    """Load one RGB/RGBA image as four normalized 720p TCHW seed frames.

    Args:
        path: Image used to establish the initial world state.

    Returns:
        Four identical float32 RGB frames in the ``[-1, 1]`` range.

    Raises:
        FileNotFoundError: The image path does not exist.
        ValueError: Pillow cannot decode an RGB or RGBA image.
    """
    if not path.is_file():
        raise FileNotFoundError(f"seed image does not exist: {path}")

    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as image:
            image.load()
            if image.mode not in {"RGB", "RGBA"}:
                raise ValueError(
                    f"seed image must be RGB or RGBA, got mode {image.mode}"
                )
            image = image.convert("RGB").resize(
                (_OUTPUT_WIDTH, _OUTPUT_HEIGHT),
                resample=Image.Resampling.BILINEAR,
            )
            pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    except UnidentifiedImageError as error:
        raise ValueError(f"seed image is not decodable: {path}") from error

    frame = pixels.view(_OUTPUT_HEIGHT, _OUTPUT_WIDTH, 3).permute(2, 0, 1)
    return frame.unsqueeze(0).repeat(4, 1, 1, 1).float().div(127.5).sub(1.0)


def _validate_session_desc(session_desc: SessionDesc) -> None:
    if session_desc.output_layout is not VideoTensorLayout.tchw:
        raise ValueError(
            "Waypoint only produces tchw output, got "
            f"{session_desc.output_layout.value}."
        )
    size = (session_desc.video_width, session_desc.video_height)
    expected = (_OUTPUT_WIDTH, _OUTPUT_HEIGHT)
    if size != expected:
        raise ValueError(
            f"Waypoint presentation size must be {expected[0]}x{expected[1]}, "
            f"got {size[0]}x{size[1]}."
        )


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


def _download_example_image() -> Path:
    return download_to_cache(
        _EXAMPLE_IMAGE_URL,
        cache_dir=default_flashdreams_cache_dir() / "example_data" / "waypoint",
        filename=_EXAMPLE_IMAGE_FILENAME,
        validator=load_seed_display_frames,
    )


def create_app() -> IApplication:
    """Return a new lazy Waypoint 1.5 V2 application."""
    return WaypointApplication()


__all__ = ["WaypointApplication", "create_app", "load_seed_display_frames"]
