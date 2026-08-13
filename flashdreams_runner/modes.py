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

"""Runner-owned I/O mode selection and batch mode implementations."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from flashdreams.runtime import OutputArtifact

from .contracts import DriveSession, IOHandler, Runtime
from .outputs import FileOutput, FiniteInput, NullOutput
from .webrtc import WebRTCMode

MODE_MP4 = "mp4"
"""Compatibility name for a replay written to an MP4 file."""

MODE_REPLAY = "replay"
"""Finite replay mode that writes an MP4 file."""

MODE_WEBRTC = "webrtc"
"""Live WebRTC serving mode."""

MODE_NONE = "none"
"""Finite headless mode that discards generated output."""

MODE_NAMES = (MODE_MP4, MODE_REPLAY, MODE_WEBRTC, MODE_NONE)
"""I/O modes currently implemented by the application runner."""


@dataclass(frozen=True, slots=True)
class ReplayMode:
    """Drive a finite session and persist its output as MP4."""

    output: Path
    """Destination MP4 path."""

    steps: int | None
    """Iteration override; ``None`` uses the application's default."""

    enabled: bool = True
    """Whether this process writes the output artifact."""

    name: str = MODE_REPLAY
    """Stable mode name."""

    def run(
        self,
        runtime: Runtime,
        drive_session: DriveSession,
    ) -> tuple[OutputArtifact, ...]:
        """Run one finite application session through MP4 handlers."""
        total_steps = _resolve_steps(self.steps, runtime)
        return drive_session(
            runtime,
            FiniteInput(total_steps=total_steps),
            FileOutput(path=self.output, enabled=self.enabled),
        )


@dataclass(frozen=True, slots=True)
class NoneMode:
    """Drive a finite session while discarding all generated output."""

    steps: int | None
    """Iteration override; ``None`` uses the application's default."""

    name: str = MODE_NONE
    """Stable mode name."""

    def run(
        self,
        runtime: Runtime,
        drive_session: DriveSession,
    ) -> tuple[OutputArtifact, ...]:
        """Run one finite application session through headless handlers."""
        total_steps = _resolve_steps(self.steps, runtime)
        return drive_session(
            runtime,
            FiniteInput(total_steps=total_steps),
            NullOutput(),
        )


def add_mode_arguments(parser: argparse.ArgumentParser, mode: str) -> None:
    """Add runner-owned arguments for the selected I/O mode."""
    parser.add_argument("--device", default="cuda", help="Runtime device")
    if mode in (MODE_MP4, MODE_REPLAY):
        parser.add_argument("--output", type=Path, required=True, help="MP4 path")
        parser.add_argument(
            "--steps",
            type=int,
            help="Generation iterations (defaults to the application preset)",
        )
        return
    if mode == MODE_NONE:
        parser.add_argument(
            "--steps",
            type=int,
            help="Generation iterations (defaults to the application preset)",
        )
        return
    if mode == MODE_WEBRTC:
        parser.add_argument("--host", default="0.0.0.0", help="WebRTC bind address")
        parser.add_argument("--port", type=int, default=8080, help="WebRTC bind port")
        return
    raise ValueError(f"Unsupported application mode: {mode!r}.")


def create_io_handler(
    mode: str,
    options: argparse.Namespace,
    *,
    device: str,
    world_rank: int,
) -> IOHandler:
    """Create the selected runner-owned I/O handler."""
    if mode in (MODE_MP4, MODE_REPLAY):
        return ReplayMode(
            output=options.output,
            steps=options.steps,
            enabled=world_rank == 0,
            name=mode,
        )
    if mode == MODE_NONE:
        return NoneMode(steps=options.steps)
    if mode == MODE_WEBRTC:
        return WebRTCMode(
            host=options.host,
            port=options.port,
            device=device,
            world_rank=world_rank,
        )
    raise ValueError(f"Unsupported application mode: {mode!r}.")


def _resolve_steps(value: int | None, runtime: Runtime) -> int:
    total_steps = runtime.config.default_steps if value is None else value
    if total_steps is None:
        raise ValueError(
            "Finite modes require --steps or an application default step count."
        )
    if total_steps <= 0:
        raise ValueError("steps must be > 0.")
    return total_steps


__all__ = [
    "MODE_NAMES",
    "MODE_NONE",
    "MODE_REPLAY",
    "MODE_WEBRTC",
    "NoneMode",
    "ReplayMode",
    "add_mode_arguments",
    "create_io_handler",
]
