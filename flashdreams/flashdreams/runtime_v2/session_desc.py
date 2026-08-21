# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Description of the session a runtime asks an application for."""

import math
from dataclasses import dataclass, field, replace
from typing import Any

from flashdreams.runtime_v2.video_tensor import VideoTensorLayout


@dataclass(frozen=True, kw_only=True, slots=True)
class SessionDesc:
    """Description of a session, passed to create one and to open a window on it.

    The runtime fills this in to ask an application for a session, and the
    session reports back what it resolved to. The same description then
    configures the client window through ``OutputSink.open``.
    """

    output_layout: VideoTensorLayout = VideoTensorLayout.tchw
    """Declared tensor layout for generated video results."""

    frames_per_second_for_ui: int = 60
    """Rate to read input and present finished results at, in frames per second."""

    frames_per_second_for_step: int = 30
    """Rate to generate at, in frames per second. Nothing paces by it yet."""

    video_width: int = 1280
    """Output video width in pixels."""

    video_height: int = 720
    """Output video height in pixels."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Extra values a runtime and an application agree on. Nothing here reads it."""

    def __post_init__(self) -> None:
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


@dataclass(frozen=True, kw_only=True, slots=True)
class SessionDescRequest:
    """The parts of a session description a caller explicitly requests.

    Leaving a field unset preserves the application's initialized default. The
    runtime resolves this request before it asks the application to create a
    session.
    """

    output_layout: VideoTensorLayout | None = None
    frames_per_second_for_ui: int | None = None
    frames_per_second_for_step: int | None = None
    video_width: int | None = None
    video_height: int | None = None
    metadata: dict[str, Any] | None = None

    def resolve(self, default: SessionDesc) -> SessionDesc:
        """Return ``default`` with every explicitly requested field replaced."""
        return replace(
            default,
            output_layout=(
                default.output_layout
                if self.output_layout is None
                else self.output_layout
            ),
            frames_per_second_for_ui=(
                default.frames_per_second_for_ui
                if self.frames_per_second_for_ui is None
                else self.frames_per_second_for_ui
            ),
            frames_per_second_for_step=(
                default.frames_per_second_for_step
                if self.frames_per_second_for_step is None
                else self.frames_per_second_for_step
            ),
            video_width=(
                default.video_width if self.video_width is None else self.video_width
            ),
            video_height=(
                default.video_height
                if self.video_height is None
                else self.video_height
            ),
            metadata=default.metadata if self.metadata is None else self.metadata,
        )
