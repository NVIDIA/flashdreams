# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batch input and output handlers for runner modes."""

from __future__ import annotations

from pathlib import Path

from flashdreams.runtime import (
    InferenceInput,
    NullOutputTarget,
    OutputArtifact,
    StepResult,
)
from flashdreams.runtime.video_output import Mp4VideoOutputTarget

from .contracts import AppConfig


class FiniteInput:
    """Supply empty per-step inputs for a fixed number of iterations."""

    def __init__(self, *, total_steps: int) -> None:
        if total_steps <= 0:
            raise ValueError("FiniteInput.total_steps must be > 0.")
        self._total_steps = total_steps
        self._step_index = 0
        self._opened = False

    def open(self) -> None:
        """Reset the finite input sequence."""
        self._step_index = 0
        self._opened = True

    def initial_input(self) -> InferenceInput:
        """Return an empty initial input for application-owned defaults."""
        if not self._opened:
            raise RuntimeError("Cannot read from a closed input handler.")
        return InferenceInput()

    def read(self) -> InferenceInput | None:
        """Return one empty iteration input until the configured limit."""
        if not self._opened:
            raise RuntimeError("Cannot read from a closed input handler.")
        if self._step_index >= self._total_steps:
            return None
        self._step_index += 1
        return InferenceInput()

    def close(self) -> None:
        """Close the finite input sequence."""
        self._opened = False


class FileOutput:
    """Collect generated chunks and write one MP4 file."""

    def __init__(self, *, path: Path, enabled: bool = True) -> None:
        self._path = path
        self._enabled = enabled
        self._target: Mp4VideoOutputTarget | None = None

    def open(self, config: AppConfig) -> None:
        """Open a video target using application output information."""
        if self._target is not None:
            raise RuntimeError("FileOutput is already open.")
        self._target = Mp4VideoOutputTarget(
            output_path=self._path,
            fps=config.fps,
            output_layout=config.output_layout,
            enabled=self._enabled,
        )
        self._target.open()

    def write(self, result: StepResult) -> None:
        """Append one generated output chunk."""
        if self._target is None:
            raise RuntimeError("Cannot write to a closed FileOutput.")
        self._target.write(result)

    def close(self) -> tuple[OutputArtifact, ...]:
        """Finalize the MP4 and return its artifact metadata."""
        if self._target is None:
            return ()
        target = self._target
        self._target = None
        return tuple(target.close())


class NullOutput:
    """Discard generated outputs for the ``none`` mode."""

    def __init__(self) -> None:
        self._target = NullOutputTarget()

    def open(self, config: AppConfig) -> None:
        """Open the headless output target."""
        del config
        self._target.open()

    def write(self, result: StepResult) -> None:
        """Discard one generated output while recording its count."""
        self._target.write(result)

    def close(self) -> tuple[OutputArtifact, ...]:
        """Close the headless output target without creating artifacts."""
        return tuple(self._target.close())


__all__ = ["FileOutput", "FiniteInput", "NullOutput"]
