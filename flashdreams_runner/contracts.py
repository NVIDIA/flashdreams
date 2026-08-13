# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Application, runtime, session, and I/O mode contracts."""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flashdreams.infra.postprocess import VideoTensorLayout
from flashdreams.runtime import InferenceInput, OutputArtifact, StepRequest, StepResult

if TYPE_CHECKING:
    from flashdreams.runtime.demo import SessionInfo


@dataclass(frozen=True, kw_only=True, slots=True)
class AppConfig:
    """Application output configuration consumed by runner-owned I/O modes."""

    model_id: str
    """Stable application or model identity."""

    fps: int | float
    """Output video frame rate."""

    output_layout: VideoTensorLayout
    """Layout of video tensors returned by application sessions."""

    video_width: int
    """Output video width in pixels."""

    video_height: int
    """Output video height in pixels."""

    default_steps: int | None = None
    """Default finite-mode iteration count; ``None`` requires a mode override."""

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("AppConfig.model_id must be non-empty.")
        if float(self.fps) <= 0:
            raise ValueError("AppConfig.fps must be > 0.")
        if self.video_width <= 0 or self.video_height <= 0:
            raise ValueError("AppConfig video dimensions must be > 0.")
        if self.default_steps is not None and self.default_steps <= 0:
            raise ValueError("AppConfig.default_steps must be > 0 when set.")


@dataclass(kw_only=True, slots=True)
class ApplicationArguments:
    """Command-line request passed to an application's runtime factory."""

    mode: str
    """Selected runner I/O mode."""

    parser: argparse.ArgumentParser
    """Parser containing runner and selected-mode options."""

    argv: Sequence[str]
    """Arguments remaining after application and mode selection."""

    _options: argparse.Namespace | None = field(default=None, init=False, repr=False)

    def parse_args(self) -> argparse.Namespace:
        """Parse runner, mode, and application options exactly once."""
        if self._options is None:
            self._options = self.parser.parse_args(self.argv)
        return self._options

    @property
    def options(self) -> argparse.Namespace:
        """Return options parsed by the application runtime factory."""
        if self._options is None:
            raise RuntimeError(
                "Application create_runtime() must call arguments.parse_args()."
            )
        return self._options


class InputHandler(Protocol):
    """Supply initial and per-iteration inputs for one runner-owned session."""

    def open(self) -> None:
        """Prepare input resources for a session."""
        ...

    def initial_input(self) -> InferenceInput:
        """Return the input used to construct the application session."""
        ...

    def read(self) -> InferenceInput | None:
        """Return the next iteration input, or ``None`` to stop the loop."""
        ...

    def close(self) -> None:
        """Release input resources."""
        ...


class OutputHandler(Protocol):
    """Present or persist outputs from one runner-owned session."""

    def open(self, config: AppConfig) -> None:
        """Prepare output resources for a session."""
        ...

    def write(self, result: StepResult) -> None:
        """Consume one generated application output."""
        ...

    def close(self) -> Sequence[OutputArtifact]:
        """Finalize output resources and return persistent artifacts."""
        ...


class Runtime(ABC):
    """Application-owned model weights and process-wide inference state.

    The runner creates one runtime for the process, initializes it with the
    selected I/O mode, and creates one or more isolated sessions from it.
    """

    @property
    @abstractmethod
    def config(self) -> AppConfig:
        """Return application configuration required by runner-owned modes."""

    @abstractmethod
    def initialize(self, *, device: str, io_handler: "IOHandler") -> None:
        """Perform one-time initialization for the selected device and mode."""

    @abstractmethod
    def create_session(self, initial_input: InferenceInput | None = None) -> "Session":
        """Create an isolated application session."""

    @abstractmethod
    def destroy(self) -> None:
        """Release model weights and process-wide resources."""

    # These aliases let shared FlashDreams serving code consume the application
    # ABI directly while the runner-facing contract stays create/generate/destroy.
    def start_session(self, inputs: InferenceInput) -> "Session":
        """Create a session through the shared inference-runtime API."""
        return self.create_session(inputs)

    def close(self) -> None:
        """Destroy the runtime through the shared inference-runtime API."""
        self.destroy()

    def peek_input_fps(self) -> float:
        """Return the input clock rate used by realtime presentation."""
        return float(self.config.fps)


class Session(ABC):
    """Application-owned state and generation logic for one user session."""

    @property
    @abstractmethod
    def step_index(self) -> int:
        """Return the index of the next generation iteration."""

    @property
    def steady_output_frame_count(self) -> int | None:
        """Return the steady output chunk size when the application knows it."""
        return None

    @abstractmethod
    def generate(self, inputs: InferenceInput) -> StepResult:
        """Run one application main-loop iteration."""

    @abstractmethod
    def destroy(self) -> None:
        """Release per-session state."""

    # Shared serving uses the inference-session spelling of this same ABI.
    def next_step_request(self) -> StepRequest | None:
        """Describe the next iteration, or stop a finite shared session."""
        metadata: dict[str, int] = {}
        if self.steady_output_frame_count is not None:
            metadata["steady_output_frame_count"] = self.steady_output_frame_count
        return StepRequest(step_index=self.step_index, metadata=metadata)

    def step(self, inputs: InferenceInput) -> StepResult:
        """Generate through the shared inference-session API."""
        result = self.generate(inputs)
        if not isinstance(result, StepResult):
            raise TypeError(
                "Session.generate() must return StepResult, got "
                f"{type(result).__name__}."
            )
        return result

    def reset(self, inputs: InferenceInput | None = None) -> None:
        """Reject reset when an application requires a fresh session."""
        del inputs
        raise RuntimeError("Create a new application session instead of resetting.")

    def close(self) -> None:
        """Destroy the session through the shared inference-session API."""
        self.destroy()

    def session_info(self) -> "SessionInfo":
        """Return output information to shared FlashDreams drivers."""
        from flashdreams.runtime.demo import SessionInfo

        return SessionInfo(
            steady_output_frame_count=self.steady_output_frame_count,
        )


DriveSession = Callable[
    [Runtime, InputHandler, OutputHandler], tuple[OutputArtifact, ...]
]
"""Runner-owned function that drives one application session."""


@runtime_checkable
class IOHandler(Protocol):
    """Runner mode that owns input acquisition and output presentation."""

    @property
    def name(self) -> str:
        """Return the stable command-line mode name."""
        ...

    def run(
        self,
        runtime: Runtime,
        drive_session: DriveSession,
    ) -> tuple[OutputArtifact, ...]:
        """Run the mode with an initialized application runtime."""
        ...


@runtime_checkable
class Application(Protocol):
    """ABI exposed by an installed FlashDreams application module."""

    def create_runtime(self, arguments: ApplicationArguments) -> Runtime:
        """Parse application arguments and return an uninitialized runtime."""
        ...


__all__ = [
    "AppConfig",
    "Application",
    "ApplicationArguments",
    "DriveSession",
    "IOHandler",
    "InputHandler",
    "OutputHandler",
    "Runtime",
    "Session",
]
