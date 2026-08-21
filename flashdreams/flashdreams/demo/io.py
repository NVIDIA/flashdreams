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

"""Transport-neutral application output and factory contracts."""

from __future__ import annotations

import math
from abc import abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from flashdreams.infra.results import StepResult
from flashdreams.runtime._utils import freeze_mapping
from flashdreams.runtime.inputs import CanonicalInputSchema, CanonicalInputWindow
from flashdreams.runtime.output import OutputArtifact


@dataclass(frozen=True, kw_only=True, slots=True)
class SessionInfo:
    """Sink-facing metadata known after application session initialization."""

    output_layout: str | None = None
    """Declared tensor layout for generated video results."""

    steady_output_frame_count: int | None = None
    """Expected frame count for steady-state output chunks."""

    frames_per_second: float | None = None
    """Presentation frame rate, when the application produces timed media."""

    video_width: int | None = None
    """Output video width in pixels, when known."""

    video_height: int | None = None
    """Output video height in pixels, when known."""

    metadata: Mapping[str, object] = field(default_factory=dict)
    """Additional immutable application metadata for sink setup."""

    def __post_init__(self) -> None:
        if self.output_layout is not None and not self.output_layout.strip():
            raise ValueError("SessionInfo.output_layout must be non-empty when set.")
        if (
            self.steady_output_frame_count is not None
            and self.steady_output_frame_count < 0
        ):
            raise ValueError(
                "SessionInfo.steady_output_frame_count must be >= 0 when set."
            )
        if self.frames_per_second is not None and (
            not math.isfinite(self.frames_per_second) or self.frames_per_second <= 0
        ):
            raise ValueError("SessionInfo.frames_per_second must be > 0 when set.")
        if self.video_width is not None and self.video_width <= 0:
            raise ValueError("SessionInfo.video_width must be > 0 when set.")
        if self.video_height is not None and self.video_height <= 0:
            raise ValueError("SessionInfo.video_height must be > 0 when set.")
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))


@dataclass(frozen=True, kw_only=True, slots=True)
class OutputDecision:
    """Flow-control decision returned after one output write."""

    should_stop: bool = False
    """Whether the application should stop generating this session."""

    dropped: bool = False
    """Whether the sink dropped the submitted output chunk."""

    drop_policy: Literal["none", "drop_newest", "drop_oldest"] = "none"
    """Queue policy responsible for a dropped chunk."""

    backpressure_s: float = 0.0
    """Pacing delay for a realtime driver to account for before the next step."""

    metadata: Mapping[str, object] = field(default_factory=dict)
    """Sink-specific immutable delivery metadata."""

    def __post_init__(self) -> None:
        if self.drop_policy not in {"none", "drop_newest", "drop_oldest"}:
            raise ValueError(f"Unsupported drop_policy={self.drop_policy!r}.")
        if not math.isfinite(self.backpressure_s) or self.backpressure_s < 0:
            raise ValueError("OutputDecision.backpressure_s must be finite and >= 0.")
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))


@runtime_checkable
class InputHandler(Protocol):
    """Provide time-windowed canonical input state to the host."""

    @abstractmethod
    def open(
        self,
        session_info: SessionInfo,
    ) -> None:
        """Prepare input resources for an application session."""
        ...

    @abstractmethod
    def current_inputs(self) -> CanonicalInputWindow:
        """Return the latest canonical values and their session time window."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release input resources."""
        ...


@runtime_checkable
class OutputSink(Protocol):
    """Consume canonical generated results for one application session."""

    produces_artifacts: bool
    """Whether closing the sink can produce persistent artifacts."""

    @abstractmethod
    def open(self, session_info: SessionInfo) -> None:
        """Prepare output resources for a session."""
        ...

    @abstractmethod
    def begin_generation(self, generation: int) -> None:
        """Start a generation and discard stale live output when required."""
        ...

    @abstractmethod
    def write(self, result: StepResult) -> OutputDecision:
        """Consume one result and return flow-control state."""
        ...

    @abstractmethod
    def close(self) -> Sequence[OutputArtifact]:
        """Finalize output resources and return persistent artifacts."""
        ...


@runtime_checkable
class IOFactory(Protocol):
    """Create isolated application input handling and output delivery."""

    @abstractmethod
    def create_input_handler(self, input_schema: CanonicalInputSchema) -> InputHandler:
        """Create a handler for the application-declared canonical inputs."""
        ...

    @abstractmethod
    def create_output_sink(self) -> OutputSink:
        """Create the output sink for one application run."""
        ...


__all__ = [
    "IOFactory",
    "InputHandler",
    "OutputDecision",
    "OutputSink",
    "SessionInfo",
]
