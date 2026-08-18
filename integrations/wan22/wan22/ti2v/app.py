# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wan 2.2 text-and-image-to-video application factory."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from t2v import (
    T2VApplication,
    T2VApplicationDefaults,
    T2VApplicationSession,
)

from flashdreams.demo import (
    CanonicalInputSchema,
    IFlashDreamsApplication,
    IFlashDreamsApplicationSession,
)
from flashdreams.infra.runner_io import load_first_frame_tensor
from wan22.config import (
    DEFAULT_VIDEO_FPS,
    DEFAULT_VIDEO_HEIGHT,
    DEFAULT_VIDEO_WIDTH,
    PIPELINE_WAN22_TI2V_5B,
)


class Wan22TI2VApplicationSession(T2VApplicationSession):
    """Cache-isolated Wan 2.2 session conditioned on one first frame."""

    def __init__(self, *, config: Any, pipeline: Any) -> None:
        super().__init__(config=config, pipeline=pipeline)
        self._first_frame_path: Path | None = None

    def set_first_frame_path(self, first_frame_path: Path) -> None:
        """Set the first-frame path before model cache initialization."""
        if self._cache is not None:
            raise RuntimeError(
                "Cannot change the first frame after session initialization."
            )
        self._first_frame_path = first_frame_path

    def _initialize_cache(self, pipeline: Any) -> Any:
        first_frame_path = self._first_frame_path
        if first_frame_path is None:
            raise RuntimeError(
                "Set a first-frame path before initializing the Wan 2.2 session."
            )
        if self._prompt is None:
            raise RuntimeError("Set a prompt before initializing the Wan 2.2 session.")

        first_frame = load_first_frame_tensor(
            first_frame_path,
            pixel_height=self.config.pixel_height,
            pixel_width=self.config.pixel_width,
            device=torch.device(pipeline.device),
            dtype=torch.bfloat16,
        )
        return pipeline.initialize_cache(
            text=[self._prompt],
            image=first_frame,
        )


class Wan22TI2VApplication(T2VApplication):
    """Wan 2.2 single-block text-and-image-to-video application."""

    session_type = Wan22TI2VApplicationSession

    def __init__(self) -> None:
        super().__init__(
            defaults=T2VApplicationDefaults(
                pipeline_config=PIPELINE_WAN22_TI2V_5B,
                total_blocks=1,
                pixel_height=DEFAULT_VIDEO_HEIGHT,
                pixel_width=DEFAULT_VIDEO_WIDTH,
                fps=DEFAULT_VIDEO_FPS,
            )
        )
        self._first_frame_path: Path | None = None

    @property
    def input_schema(self) -> CanonicalInputSchema:
        """Declare static first-frame conditioning with no live controls."""
        return CanonicalInputSchema(
            description="first-frame-conditioned text-and-image-to-video"
        )

    def _configure_argument_parser(self, parser: argparse.ArgumentParser) -> None:
        """Add the required first-frame argument."""
        parser.add_argument("--image-path", type=Path, required=True)

    def _apply_parsed_arguments(self, args: argparse.Namespace) -> None:
        """Validate and retain the first-frame path."""
        first_frame_path = args.image_path
        if not first_frame_path.is_file():
            raise FileNotFoundError(
                f"Wan 2.2 first-frame image does not exist: {first_frame_path}"
            )
        self._first_frame_path = first_frame_path

    def _validate_total_blocks(self, total_blocks: int) -> None:
        """Reject multi-block requests unsupported by bidirectional Wan 2.2."""
        super()._validate_total_blocks(total_blocks)
        if total_blocks > 1:
            raise ValueError(
                "Wan 2.2 TI2V supports exactly one autoregressive block; "
                "--total-blocks must be 1."
            )

    def create_session(self) -> IFlashDreamsApplicationSession:
        """Create a Wan 2.2 session with its static first frame."""
        first_frame_path = self._first_frame_path
        if first_frame_path is None:
            raise RuntimeError(
                "Wan22TI2VApplication.init() must run before create_session()."
            )
        session = super().create_session()
        if not isinstance(session, Wan22TI2VApplicationSession):
            raise TypeError("Wan 2.2 application created an unexpected session type.")
        session.set_first_frame_path(first_frame_path)
        return session


def create_app() -> IFlashDreamsApplication:
    """Create the Wan 2.2 text-and-image-to-video application."""
    return Wan22TI2VApplication()


__all__ = [
    "Wan22TI2VApplication",
    "Wan22TI2VApplicationSession",
    "create_app",
]
