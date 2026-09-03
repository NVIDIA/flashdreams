# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable action-to-video application on the FlashDreams v2 API."""

from __future__ import annotations

import argparse
import math
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from torch import Tensor

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.session import ISession
from flashdreams.infra.config import derive_config
from flashdreams.runtime_v2.session_desc import (
    BackpressureMode,
    PresentationMode,
    SessionDesc,
)
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

from .session import (
    Action2VSession,
    ActionMapper,
    _normalize_reset_key,
)

Action2VInputResolver = Callable[[Mapping[str, Any]], Path]
"""Resolve application arguments into one session's first frame."""

Action2VSeedLoader = Callable[[Path, SessionDesc], Tensor]
"""Load seed display frames for one resolved session description."""

Action2VActionMapperFactory = Callable[[SessionDesc, float], ActionMapper]
"""Create a live snapshot mapper for one session's dimensions."""

_DEFAULT_RESET_KEY = "T"
"""Key that requests an in-place session reset."""


@dataclass(frozen=True, kw_only=True, slots=True)
class Action2VApplicationDefaults:
    """Model integration hooks and defaults for the shared Action2V app."""

    slug: str
    """Application name used in help and log messages."""

    pipeline_config: Any
    """Model pipeline configuration owned by the integration."""

    input_resolver: Action2VInputResolver
    """Integration hook that resolves the first-frame path."""

    seed_loader: Action2VSeedLoader
    """Integration hook that loads and normalizes initial display frames."""

    action_mapper_factory: Action2VActionMapperFactory
    """Integration hook that maps live model-neutral snapshots."""

    total_blocks: int
    """Default number of autoregressive blocks in one rollout."""

    pixel_width: int
    """Native generated frame width."""

    pixel_height: int
    """Native generated frame height."""

    fps: int
    """Playback rate and initial model-loop pacing limit."""

    device: str = "cuda"
    """Device on which the application constructs the shared pipeline."""

    output_layout: VideoTensorLayout = VideoTensorLayout.tchw
    """Tensor layout emitted by the model pipeline."""

    backpressure_mode: BackpressureMode = BackpressureMode.BLOCK
    """Preserve every generated model frame in presentation order."""

    presentation_mode: PresentationMode = PresentationMode.ON_DEMAND
    """Present each generated frame exactly once by default."""

    ui_fps: int | None = None
    """Input/UI polling rate; ``None`` uses ``fps``."""

    input_defaults: Mapping[str, Any] = field(default_factory=dict)
    """Integration-owned defaults for image and example-data selection."""

    metadata: Mapping[str, Any] = field(default_factory=dict)
    """Application metadata copied into each session description."""

    def __post_init__(self) -> None:
        if not self.slug:
            raise ValueError("Action2VApplicationDefaults.slug must not be empty.")
        if self.total_blocks <= 0:
            raise ValueError("Action2VApplicationDefaults.total_blocks must be > 0.")
        if self.pixel_width <= 0 or self.pixel_height <= 0:
            raise ValueError("Action2VApplicationDefaults dimensions must be > 0.")
        if self.fps <= 0 or (self.ui_fps is not None and self.ui_fps <= 0):
            raise ValueError("Action2VApplicationDefaults frame rates must be > 0.")
        object.__setattr__(
            self,
            "input_defaults",
            MappingProxyType(dict(self.input_defaults)),
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


class Action2VApplication(IApplication):
    """Reusable first-frame and live-input shell for action-to-video models."""

    def __init__(self, *, defaults: Action2VApplicationDefaults) -> None:
        self.defaults = defaults
        self._pipeline_config = defaults.pipeline_config
        self._device = defaults.device
        self._total_blocks = defaults.total_blocks
        self._use_ui = True
        self._mouse_sensitivity = 1.0
        self._reset_key = _DEFAULT_RESET_KEY
        self._input_values: dict[str, Any] | None = None
        self._pipeline: Any | None = None
        self._pipeline_lock = threading.Lock()

    @property
    def pipeline_config(self) -> Any:
        """Return the model configuration after command-line overrides."""
        return self._pipeline_config

    def init(self, commandline_args: Sequence[str]) -> None:
        """Parse shared action-to-video inputs without loading the model."""
        parser = argparse.ArgumentParser(
            prog=f"flashdreams-run-v2 {self.defaults.slug} --",
            description="Generate video from an image and keyboard/mouse actions.",
        )
        input_defaults = self.defaults.input_defaults
        parser.add_argument(
            "--image-path",
            type=Path,
            default=input_defaults.get("image_path"),
            help="Image that establishes the initial world state.",
        )
        parser.add_argument(
            "--example-data",
            action=argparse.BooleanOptionalAction,
            default=bool(input_defaults.get("example_data", False)),
            help="Use the integration's packaged or downloadable example image.",
        )
        parser.add_argument(
            "--device",
            default=self.defaults.device,
            help="Model device. Default: %(default)s.",
        )
        parser.add_argument(
            "--total-blocks",
            type=int,
            default=self.defaults.total_blocks,
            help="Autoregressive chunks generated per rollout. Default: %(default)s.",
        )
        parser.add_argument(
            "--ui",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Capture the pointer for interactive action controls.",
        )
        parser.add_argument(
            "--mouse-sensitivity",
            type=float,
            default=1.0,
            help="Non-negative multiplier for pointer motion. Default: %(default)s.",
        )
        parser.add_argument(
            "--reset-key",
            type=_normalize_reset_key,
            default=_DEFAULT_RESET_KEY,
            help="ASCII letter that resets the session. Default: %(default)s.",
        )
        parser.add_argument("--seed", type=int, default=None)
        args = parser.parse_args(list(commandline_args))

        if args.total_blocks <= 0:
            raise ValueError("--total-blocks must be > 0.")
        if not math.isfinite(args.mouse_sensitivity) or args.mouse_sensitivity < 0:
            raise ValueError("--mouse-sensitivity must be finite and non-negative")

        self._pipeline_config = self.defaults.pipeline_config
        if args.seed is not None:
            self._pipeline_config = derive_config(
                self._pipeline_config,
                diffusion_model={"seed": args.seed},
            )
        self._device = args.device
        self._total_blocks = args.total_blocks
        self._use_ui = args.ui
        self._mouse_sensitivity = args.mouse_sensitivity
        self._reset_key = args.reset_key
        self._input_values = {
            "image_path": args.image_path,
            "example_data": args.example_data,
        }

    def session_desc(self) -> SessionDesc:
        """Return the model's native output shape and interactive rates."""
        return SessionDesc(
            output_layout=self.defaults.output_layout,
            backpressure_mode=self.defaults.backpressure_mode,
            presentation_mode=self.defaults.presentation_mode,
            frames_per_second_for_ui=self.defaults.ui_fps or self.defaults.fps,
            frames_per_second_for_step=self.defaults.fps,
            video_width=self.defaults.pixel_width,
            video_height=self.defaults.pixel_height,
            metadata=dict(self.defaults.metadata),
        )

    def create_session(self, session_desc: SessionDesc) -> ISession:
        """Load shared modules lazily and create one isolated action rollout."""
        input_values = self._input_values
        if input_values is None:
            raise RuntimeError(
                f"{type(self).__name__}.init() must run before create_session()."
            )
        self._validate_session_desc(session_desc)
        first_frame = self.defaults.input_resolver(input_values)
        seed_frames = self.defaults.seed_loader(first_frame, session_desc)
        seed = self._pipeline_config.diffusion_model.seed
        if seed is None:
            raise ValueError("Action2V pipeline config must set diffusion_model.seed.")
        pipeline = self._ensure_pipeline()
        return Action2VSession(
            pipeline=pipeline,
            pipeline_lock=self._pipeline_lock,
            session_desc=session_desc,
            seed_frames=seed_frames,
            seed=seed,
            action_mapper=self.defaults.action_mapper_factory(
                session_desc, self._mouse_sensitivity
            ),
            total_blocks=self._total_blocks,
            use_ui=self._use_ui,
            reset_key=self._reset_key,
        )

    def close(self) -> None:
        """Release the application-owned pipeline after all sessions stop."""
        pipeline = self._pipeline
        self._pipeline = None
        self._input_values = None
        close = getattr(pipeline, "close", None)
        if callable(close):
            close()

    def _ensure_pipeline(self) -> Any:
        if self._pipeline is None:
            self._pipeline = self._pipeline_config.setup().to(self._device).eval()
        return self._pipeline

    def _validate_session_desc(self, session_desc: SessionDesc) -> None:
        if session_desc.output_layout is not self.defaults.output_layout:
            raise ValueError(
                "This action-to-video model only produces "
                f"{self.defaults.output_layout.value} output, got "
                f"{session_desc.output_layout.value}."
            )
        expected = (self.defaults.pixel_width, self.defaults.pixel_height)
        actual = (session_desc.video_width, session_desc.video_height)
        if actual != expected:
            raise ValueError(
                f"Action2V presentation size must be {expected[0]}x{expected[1]}, "
                f"got {actual[0]}x{actual[1]}."
            )


__all__ = ["Action2VApplication", "Action2VApplicationDefaults"]
