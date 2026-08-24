# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Description of the session a runtime asks an application for."""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from flashdreams.runtime_v2.video_tensor import VideoTensorLayout


class PresentationMode(Enum):
    """How model frames wait for the UI."""

    BLOCK = "block"
    """Wait when the presentation queue is full."""

    DROP_OLDEST = "drop_oldest"
    """Drop the oldest queued model step."""

    LOSSLESS = "lossless"
    """Show every model frame once and in order."""


@dataclass(frozen=True, kw_only=True, slots=True)
class SessionDesc:
    """Description of a session, passed to create one and to open a window on it.

    The runtime fills this in to ask an application for a session, and the
    session reports back what it resolved to. The same description then
    configures the client window through ``OutputSink.open``.
    """

    output_layout: VideoTensorLayout = VideoTensorLayout.tchw
    """Declared tensor layout for generated video results."""

    presentation_mode: PresentationMode = PresentationMode.BLOCK
    """How model frames wait for the UI."""

    frames_per_second_for_ui: int = 60
    """Rate to read input and present finished results at, in frames per second."""

    frames_per_second_for_step: int = 30
    """Maximum model-loop iterations per second."""

    video_width: int = 1280
    """Output video width in pixels."""

    video_height: int = 720
    """Output video height in pixels."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Extra values a runtime and an application agree on. Nothing here reads it."""

    def __post_init__(self) -> None:
        if not isinstance(self.presentation_mode, PresentationMode):
            raise TypeError("SessionDesc.presentation_mode must be a PresentationMode.")
        if (
            not math.isfinite(self.frames_per_second_for_ui)
            or self.frames_per_second_for_ui <= 0
        ):
            raise ValueError(
                "SessionDesc.frames_per_second_for_ui must be > 0 when set."
            )
        if (
            not math.isfinite(self.frames_per_second_for_step)
            or self.frames_per_second_for_step <= 0
        ):
            raise ValueError(
                "SessionDesc.frames_per_second_for_step must be > 0 when set."
            )
        if self.video_width <= 0:
            raise ValueError("SessionDesc.video_width must be > 0 when set.")
        if self.video_height <= 0:
            raise ValueError("SessionDesc.video_height must be > 0 when set.")


__all__ = ["PresentationMode", "SessionDesc"]
