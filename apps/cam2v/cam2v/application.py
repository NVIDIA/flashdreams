# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable camera-to-video application on the FlashDreams v2 API."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.session import ISession
from flashdreams.infra.config import derive_config
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

from .defaults import Cam2VApplicationDefaults, Cam2VConditioning
from .session import Cam2VSession, Cam2VSessionConfig


class Cam2VApplication(IApplication):
    """Reusable interactive camera-to-video application.

    The shared class owns command-line parsing, pipeline lifetime, session
    validation, and model-generation-loop construction. A concrete model
    integration contributes a pipeline config and an input resolver through
    :class:`Cam2VApplicationDefaults`.
    """

    def __init__(self, *, defaults: Cam2VApplicationDefaults) -> None:
        self.defaults = defaults
        self._pipeline_config = defaults.pipeline_config
        self._device = defaults.device
        self._total_blocks = defaults.total_blocks
        self._warmup_blocks = defaults.warmup_blocks
        self._use_ui = True
        self._input_values: dict[str, Any] | None = None
        self._pipeline: Any | None = None

    @property
    def pipeline_config(self) -> Any:
        """Return the model configuration after command-line overrides."""
        return self._pipeline_config

    def init(self, commandline_args: Sequence[str]) -> None:
        """Parse shared camera-to-video inputs without loading the model."""
        parser = argparse.ArgumentParser(
            prog="flashdreams-run-v2 CAM2V_SLUG --",
            description="Generate video from a first frame and keyboard camera input.",
        )
        input_defaults = self.defaults.input_defaults
        parser.add_argument("--prompt", default=input_defaults.get("prompt", ""))
        parser.add_argument(
            "--prompt-path",
            type=Path,
            default=input_defaults.get("prompt_path"),
        )
        parser.add_argument(
            "--image-path",
            type=Path,
            default=input_defaults.get("image_path"),
        )
        parser.add_argument(
            "--pose-path",
            type=Path,
            default=input_defaults.get("pose_path"),
        )
        parser.add_argument(
            "--intrinsic-path",
            type=Path,
            default=input_defaults.get("intrinsic_path"),
        )
        parser.add_argument(
            "--world-scale",
            type=float,
            default=input_defaults.get("world_scale"),
        )
        parser.add_argument(
            "--example-data",
            action=argparse.BooleanOptionalAction,
            default=bool(input_defaults.get("example_data", False)),
        )
        parser.add_argument(
            "--example-idx",
            type=int,
            default=int(input_defaults.get("example_idx", 0)),
        )
        parser.add_argument(
            "--device",
            default=self.defaults.device,
            help="Device used by the shared model. Default: %(default)s.",
        )
        parser.add_argument(
            "--total-blocks",
            type=int,
            default=self.defaults.total_blocks,
            help="Autoregressive chunks generated per rollout. Default: %(default)s.",
        )
        parser.add_argument(
            "--warmup-blocks",
            type=int,
            default=self.defaults.warmup_blocks,
            help="Leading chunks excluded from steady-state FPS.",
        )
        parser.add_argument(
            "--ui",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Render the shared camera controls and timing overlay.",
        )
        parser.add_argument(
            "--compile",
            action=argparse.BooleanOptionalAction,
            default=None,
        )
        parser.add_argument(
            "--sync-and-profile",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Synchronize CUDA and log per-stage pipeline timings.",
        )
        parser.add_argument("--seed", type=int, default=None)
        self._configure_argument_parser(parser)
        args = parser.parse_args(list(commandline_args))

        self._pipeline_config = self.defaults.pipeline_config
        self._validate_arguments(args)
        self._apply_parsed_arguments(args)
        if args.compile is not None:
            self._pipeline_config = derive_config(
                self._pipeline_config,
                diffusion_model={"transformer": {"compile_network": args.compile}},
            )
        if args.sync_and_profile is not None:
            self._pipeline_config = derive_config(
                self._pipeline_config,
                enable_sync_and_profile=args.sync_and_profile,
            )
        if args.seed is not None:
            self._pipeline_config = derive_config(
                self._pipeline_config,
                diffusion_model={"seed": args.seed},
            )
        self._device = args.device
        self._total_blocks = args.total_blocks
        self._warmup_blocks = args.warmup_blocks
        self._use_ui = args.ui
        self._input_values = {
            "prompt": args.prompt,
            "prompt_path": args.prompt_path,
            "image_path": args.image_path,
            "pose_path": args.pose_path,
            "intrinsic_path": args.intrinsic_path,
            "world_scale": args.world_scale,
            "example_data": args.example_data,
            "example_idx": args.example_idx,
            "total_blocks": args.total_blocks,
        }

    def session_desc(self) -> SessionDesc:
        """Return the model's default output shape and interactive rates."""
        return SessionDesc(
            output_layout=self.defaults.output_layout,
            backpressure_mode=self.defaults.backpressure_mode,
            presentation_mode=self.defaults.presentation_mode,
            frames_per_second_for_ui=self.defaults.ui_fps,
            frames_per_second_for_step=self.defaults.fps,
            video_width=self.defaults.pixel_width,
            video_height=self.defaults.pixel_height,
        )

    def create_session(self, session_desc: SessionDesc) -> ISession:
        """Create an isolated rollout after lazily loading the shared pipeline."""
        input_values = self._input_values
        if input_values is None:
            raise RuntimeError(
                f"{type(self).__name__}.init() must run before create_session()."
            )
        self._validate_layout(session_desc)
        resolved_values = {
            **input_values,
            "pixel_height": session_desc.video_height,
            "pixel_width": session_desc.video_width,
            "fps": session_desc.frames_per_second_for_step,
        }
        conditioning = self.defaults.input_resolver(resolved_values)
        if not isinstance(conditioning, Cam2VConditioning):
            raise TypeError(
                "Cam2VApplicationDefaults.input_resolver must return Cam2VConditioning."
            )

        pipeline = self._pipeline
        if pipeline is None:
            pipeline = self._pipeline_config.setup().to(self._device).eval()
            self._pipeline = pipeline
        self._validate_frame_size(session_desc, pipeline)
        return Cam2VSession(
            pipeline=pipeline,
            session_desc=session_desc,
            config=Cam2VSessionConfig(
                conditioning=conditioning,
                total_blocks=self._total_blocks,
                device=torch.device(self._device),
                first_frame_dtype=self.defaults.first_frame_dtype,
                first_frame_interpolation=self.defaults.first_frame_interpolation,
                generate_step=self.defaults.generate_step,
                pose_integrator_factory=self.defaults.pose_integrator_factory,
                warmup_blocks=self._warmup_blocks,
                log_model_timing=self.defaults.log_model_timing,
                install_hint=self.defaults.install_hint,
            ),
            use_ui=self._use_ui,
        )

    def close(self) -> None:
        """Release the application-owned pipeline after all sessions stop."""
        pipeline = self._pipeline
        self._pipeline = None
        self._input_values = None
        close = getattr(pipeline, "close", None)
        if callable(close):
            close()

    def _configure_argument_parser(self, parser: argparse.ArgumentParser) -> None:
        """Add integration-specific application arguments to ``parser``."""

    def _apply_parsed_arguments(self, args: argparse.Namespace) -> None:
        """Retain integration-specific arguments after shared validation."""

    def _validate_arguments(self, args: argparse.Namespace) -> None:
        """Reject invalid rollout and timing settings."""
        if args.total_blocks <= 0:
            raise ValueError("--total-blocks must be > 0.")
        if args.warmup_blocks < 0:
            raise ValueError("--warmup-blocks must be >= 0.")
        if args.world_scale is not None and args.world_scale < 0:
            raise ValueError("--world-scale must be >= 0 when set.")

    def _validate_layout(self, session_desc: SessionDesc) -> None:
        """Reject output layouts that differ from the model's declared layout."""
        if session_desc.output_layout is not self.defaults.output_layout:
            raise ValueError(
                "This camera-to-video model only produces "
                f"{self.defaults.output_layout.value} output, got "
                f"{session_desc.output_layout.value}."
            )
        if self._use_ui and session_desc.output_layout is not VideoTensorLayout.tchw:
            raise ValueError("The Cam2V SlangPy UI overlay requires tchw output.")

    def _validate_frame_size(self, session_desc: SessionDesc, pipeline: Any) -> None:
        """Reject frame dimensions that cannot map to integral latents."""
        decoder = getattr(pipeline, "decoder", None)
        ratio = getattr(decoder, "spatial_compression_ratio", None)
        if not isinstance(ratio, int) or ratio <= 0:
            raise TypeError(
                "Cam2V requires a decoder with a positive integer "
                "spatial_compression_ratio."
            )
        if session_desc.video_width % ratio or session_desc.video_height % ratio:
            raise ValueError(
                f"Frame dimensions must be multiples of {ratio}, got "
                f"{session_desc.video_width}x{session_desc.video_height}."
            )


__all__ = ["Cam2VApplication"]
