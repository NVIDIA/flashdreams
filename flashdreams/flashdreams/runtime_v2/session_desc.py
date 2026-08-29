# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Description of the session a runtime asks an application for."""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from flashdreams.runtime_v2.video_tensor import VideoTensorLayout


class BackpressureMode(Enum):
    """What the model thread does when its presentation chunk queue is full."""

    BLOCK = "block"
    """Wait when the presentation queue is full."""

    DROP_OLDEST = "drop_oldest"
    """Drop the oldest queued model chunk."""


class PresentationMode(Enum):
    """What the UI loop does when no new model frame is ready."""

    ON_DEMAND = "on_demand"
    """Present only when there is a new model frame and it is selected."""

    CONTINUOUS = "continuous"
    """Present every UI tick, reusing the newest model frame when necessary."""


@dataclass(frozen=True, kw_only=True, slots=True)
class SessionDesc:
    """Description of a session, passed to create one and to open a window on it.

    The runtime fills this in to ask an application for a session, and the
    session reports back what it resolved to. The same description then
    configures the client window through ``OutputSink.open``.
    """

    output_layout: VideoTensorLayout = VideoTensorLayout.tchw
    """Declared tensor layout for generated video results."""

    backpressure_mode: BackpressureMode = BackpressureMode.BLOCK
    """What the model thread does when its output queue is full."""

    presentation_mode: PresentationMode = PresentationMode.CONTINUOUS
    """What the UI loop does when no new model frame is ready."""

    frames_per_second_for_ui: int = 30
    """Rate to poll input and tick the UI, in frames per second."""

    frames_per_second_for_step: int = 30
    """Initial video rate and maximum model-loop iterations per second."""

    video_width: int = 1280
    """Output video width in pixels."""

    video_height: int = 720
    """Output video height in pixels."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Extra values a runtime and an application agree on. Nothing here reads it."""

    def __post_init__(self) -> None:
        if not isinstance(self.backpressure_mode, BackpressureMode):
            raise TypeError("SessionDesc.backpressure_mode must be a BackpressureMode.")
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


__all__ = ["BackpressureMode", "PresentationMode", "SessionDesc"]
