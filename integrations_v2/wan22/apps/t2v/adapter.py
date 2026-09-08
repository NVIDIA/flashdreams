# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wan 2.2 text-and-image-to-video application adapter."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Any

import torch
from t2v import T2VApplication, T2VApplicationDefaults, T2VSession

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.session import ISession
from flashdreams.infra.runner_io import load_first_frame_tensor
from flashdreams.runtime_v2.session_desc import SessionDesc
from wan22.config import (
    DEFAULT_VIDEO_FPS,
    DEFAULT_VIDEO_HEIGHT,
    DEFAULT_VIDEO_WIDTH,
    PIPELINE_WAN22_TI2V_5B,
)

WAN22_T2V_DEFAULTS = T2VApplicationDefaults(
    pipeline_config=PIPELINE_WAN22_TI2V_5B,
    total_blocks=1,
    pixel_height=DEFAULT_VIDEO_HEIGHT,
    pixel_width=DEFAULT_VIDEO_WIDTH,
    fps=DEFAULT_VIDEO_FPS,
)


class Wan22TI2VSession(T2VSession):
    """Wan 2.2 session conditioned on one first-frame image."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._first_frame_path: Path | None = None

    def set_first_frame_path(self, first_frame_path: Path) -> None:
        """Set the first-frame path before session initialization.

        Args:
            first_frame_path: Image used to seed the generated clip.

        Raises:
            RuntimeError: The model loop is already registered.
        """
        if self._registered_model_loop is not None:
            raise RuntimeError(
                "Cannot change the first frame after session initialization."
            )
        self._first_frame_path = first_frame_path

    def init(self) -> None:
        """Load the first frame and initialize the image-conditioned cache."""
        first_frame_path = self._first_frame_path
        if first_frame_path is None:
            raise RuntimeError(
                "Set a first-frame path before initializing the Wan 2.2 session."
            )
        self._image = load_first_frame_tensor(
            first_frame_path,
            pixel_height=self._session_desc.video_height,
            pixel_width=self._session_desc.video_width,
            device=torch.device(self._pipeline.device),
            dtype=torch.bfloat16,
        )
        super().init()


class Wan22TI2VApplication(T2VApplication):
    """Wan 2.2 single-block text-and-image-to-video application."""

    session_type = Wan22TI2VSession

    def __init__(self, pipeline_config: Any | None = None) -> None:
        """Configure the application from the Wan 2.2 pipeline literal.

        Args:
            pipeline_config: Optional test or deployment pipeline override.
        """
        defaults = WAN22_T2V_DEFAULTS
        if pipeline_config is not None:
            defaults = dataclasses.replace(
                defaults,
                pipeline_config=pipeline_config,
            )
        super().__init__(defaults=defaults)
        self._first_frame_path: Path | None = None

    def _configure_argument_parser(self, parser: argparse.ArgumentParser) -> None:
        """Add the required first-frame argument.

        Args:
            parser: Shared T2V argument parser.
        """
        parser.add_argument("--image-path", type=Path, required=True)

    def _apply_parsed_arguments(self, args: argparse.Namespace) -> None:
        """Validate and retain the first-frame path.

        Args:
            args: Parsed shared and integration-specific arguments.

        Raises:
            FileNotFoundError: The first-frame image does not exist.
        """
        first_frame_path = args.image_path
        if not first_frame_path.is_file():
            raise FileNotFoundError(
                f"Wan 2.2 first-frame image does not exist: {first_frame_path}"
            )
        self._first_frame_path = first_frame_path

    def _validate_total_blocks(self, total_blocks: int) -> None:
        """Reject multi-block requests unsupported by bidirectional Wan 2.2.

        Args:
            total_blocks: Requested autoregressive block count.

        Raises:
            ValueError: ``total_blocks`` is not exactly one.
        """
        super()._validate_total_blocks(total_blocks)
        if total_blocks != 1:
            raise ValueError(
                "Wan 2.2 TI2V generates its whole clip in one block; "
                f"--total-blocks must be 1, got {total_blocks}."
            )

    def create_session(self, session_desc: SessionDesc) -> ISession:
        """Create a session with its static first-frame path.

        Args:
            session_desc: Accepted output geometry and presentation settings.

        Returns:
            Uninitialized Wan 2.2 session.

        Raises:
            RuntimeError: :meth:`init` has not retained a first frame.
        """
        first_frame_path = self._first_frame_path
        if first_frame_path is None:
            raise RuntimeError(
                "Wan22TI2VApplication.init() must run before create_session()."
            )
        session = super().create_session(session_desc)
        if not isinstance(session, Wan22TI2VSession):
            raise TypeError("Wan 2.2 application created an unexpected session type.")
        session.set_first_frame_path(first_frame_path)
        return session


def create_app() -> IApplication:
    """Create the Wan 2.2 text-and-image-to-video application."""
    return Wan22TI2VApplication()


__all__ = ["Wan22TI2VApplication", "Wan22TI2VSession", "create_app"]
