# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable action-to-video application on the FlashDreams v2 API."""

from __future__ import annotations

import argparse
import math
import secrets
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
from loguru import logger
from torch import Tensor

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.session import ISession
from flashdreams.runtime_v2.session_desc import (
    BackpressureMode,
    PresentationMode,
    SessionDesc,
)
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

from .session import (
    Action2VModelSession,
    Action2VSession,
    ActionMapper,
)

Action2VPipelineFactory = Callable[[int, torch.device, bool], Any]
"""Load one application-owned model pipeline."""

Action2VSeedLoader = Callable[[Path, SessionDesc], Tensor]
"""Load seed display frames for one resolved session description."""

Action2VActionMapperFactory = Callable[[SessionDesc, float], ActionMapper]
"""Create a live snapshot mapper for one session's dimensions."""

Action2VModelSessionBuilder = Callable[
    [Any, threading.Lock, SessionDesc, Tensor, int], Action2VModelSession
]
"""Build integration-owned rollout state over the shared model pipeline."""


@dataclass(frozen=True, kw_only=True, slots=True)
class Action2VApplicationDefaults:
    """Model integration hooks and defaults for the shared Action2V app."""

    slug: str
    """Application name used in help and log messages."""

    pipeline_factory: Action2VPipelineFactory
    """Integration hook that loads the shared model pipeline."""

    seed_loader: Action2VSeedLoader
    """Integration hook that loads and normalizes initial display frames."""

    action_mapper_factory: Action2VActionMapperFactory
    """Integration hook that maps live model-neutral snapshots."""

    model_session_builder: Action2VModelSessionBuilder
    """Integration hook that owns cache, RNG, and generation semantics."""

    pixel_width: int
    """Native generated frame width."""

    pixel_height: int
    """Native generated frame height."""

    fps: int
    """Playback rate and initial model-loop pacing limit."""

    default_first_frame_resolver: Callable[[], Path] | None = None
    """Resolve an integration-provided first frame lazily when omitted."""

    device: str = "cuda"
    """Device on which the application constructs the shared pipeline."""

    output_layout: VideoTensorLayout = VideoTensorLayout.tchw
    """Tensor layout emitted by model sessions."""

    backpressure_mode: BackpressureMode = BackpressureMode.BLOCK
    """Preserve every generated model frame in presentation order."""

    presentation_mode: PresentationMode = PresentationMode.ON_DEMAND
    """Present each generated frame exactly once by default."""

    ui_fps: int | None = None
    """Input/UI polling rate; ``None`` uses :attr:`fps`."""

    metadata: Mapping[str, Any] = field(default_factory=dict)
    """Application metadata copied into each session description."""

    def __post_init__(self) -> None:
        if not self.slug:
            raise ValueError("Action2VApplicationDefaults.slug must not be empty.")
        if self.pixel_width <= 0 or self.pixel_height <= 0:
            raise ValueError("Action2VApplicationDefaults dimensions must be > 0.")
        if self.fps <= 0 or (self.ui_fps is not None and self.ui_fps <= 0):
            raise ValueError("Action2VApplicationDefaults frame rates must be > 0.")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class _ApplicationConfig:
    first_frame: Path | None
    seed: int
    device: torch.device
    profile: bool
    mouse_sensitivity: float


class Action2VApplication(IApplication):
    """Reusable first-frame and live-input shell for action-to-video models."""

    def __init__(self, *, defaults: Action2VApplicationDefaults) -> None:
        self.defaults = defaults
        self._config: _ApplicationConfig | None = None
        self._pipeline: Any | None = None
        self._pipeline_lock = threading.Lock()

    def init(self, commandline_args: Sequence[str]) -> None:
        """Parse first-frame, device, and deterministic RNG settings."""
        parser = argparse.ArgumentParser(
            prog=f"flashdreams-run-v2 {self.defaults.slug} --",
            description="Generate video from an image and keyboard/mouse actions.",
        )
        parser.add_argument(
            "--first-frame",
            type=Path,
            required=self.defaults.default_first_frame_resolver is None,
            help="Image that establishes the initial world state.",
        )
        parser.add_argument(
            "--seed",
            type=int,
            help="Non-negative model RNG seed; generated randomly when omitted.",
        )
        parser.add_argument(
            "--device",
            default=self.defaults.device,
            help="Model device. Default: %(default)s.",
        )
        parser.add_argument(
            "--profile",
            action="store_true",
            help="Enable integration pipeline profiling.",
        )
        parser.add_argument(
            "--mouse-sensitivity",
            type=float,
            default=1.0,
            help="Non-negative multiplier for pointer motion. Default: %(default)s.",
        )
        self._configure_argument_parser(parser)
        args = parser.parse_args(list(commandline_args))

        if args.seed is not None and args.seed < 0:
            raise ValueError(f"--seed must be non-negative, got {args.seed}")
        if not math.isfinite(args.mouse_sensitivity) or args.mouse_sensitivity < 0:
            raise ValueError("--mouse-sensitivity must be finite and non-negative")
        self._validate_arguments(args)

        seed = secrets.randbits(63) if args.seed is None else args.seed
        if args.seed is None:
            logger.info("[{}] generated seed {}", self.defaults.slug, seed)
        self._config = _ApplicationConfig(
            first_frame=args.first_frame,
            seed=seed,
            device=torch.device(args.device),
            profile=args.profile,
            mouse_sensitivity=args.mouse_sensitivity,
        )
        self._apply_parsed_arguments(args)

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
        config = self._require_config()
        self._validate_session_desc(session_desc)
        first_frame = config.first_frame
        if first_frame is None:
            resolver = self.defaults.default_first_frame_resolver
            assert resolver is not None
            first_frame = resolver()
        seed_frames = self.defaults.seed_loader(first_frame, session_desc)
        pipeline = self._ensure_pipeline(config)
        return Action2VSession(
            model_session_factory=lambda: self.defaults.model_session_builder(
                pipeline,
                self._pipeline_lock,
                session_desc,
                seed_frames,
                config.seed,
            ),
            session_desc=session_desc,
            action_mapper=self.defaults.action_mapper_factory(
                session_desc, config.mouse_sensitivity
            ),
        )

    def close(self) -> None:
        """Release the application-owned pipeline after all sessions stop."""
        pipeline = self._pipeline
        self._pipeline = None
        self._config = None
        close = getattr(pipeline, "close", None)
        if callable(close):
            close()

    def _configure_argument_parser(self, parser: argparse.ArgumentParser) -> None:
        """Add integration-specific application arguments to ``parser``."""

    def _validate_arguments(self, args: argparse.Namespace) -> None:
        """Reject invalid integration-specific arguments."""

    def _apply_parsed_arguments(self, args: argparse.Namespace) -> None:
        """Retain integration-specific arguments after shared validation."""

    def _require_config(self) -> _ApplicationConfig:
        if self._config is None:
            raise RuntimeError(f"{type(self).__name__}.init() must run first")
        return self._config

    def _ensure_pipeline(self, config: _ApplicationConfig) -> Any:
        if self._pipeline is None:
            self._pipeline = self.defaults.pipeline_factory(
                config.seed, config.device, config.profile
            )
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
