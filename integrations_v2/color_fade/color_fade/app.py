# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Application fading a frame from red to green, for end-to-end file output."""

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.loop import IModelLoop
from flashdreams.api_v2.session import ISession
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

_DEFAULT_SECONDS = 10.0
"""How long the fade takes by default."""

_DEFAULT_FRAMES_PER_STEP = 8
"""Frames one step generates by default. More than one, because a model
generates several frames a step rather than a single frame."""

_RED_CHANNEL = 0
"""Channel at full intensity when the fade starts."""

_GREEN_CHANNEL = 1
"""Channel at full intensity when the fade ends."""

_FULL_INTENSITY = 1.0
"""Full intensity for a channel, in the ``[-1, 1]`` range a model emits."""

_NO_INTENSITY = -1.0
"""No intensity for a channel, which is black across all three."""


@dataclass(frozen=True, slots=True)
class ColorFadeConfig:
    """Resolved settings for one colour fade application."""

    seconds: float
    """How long the fade from red to green takes."""

    frames_per_step: int
    """Frames one step generates."""


## Session


@dataclass(slots=True)
class ColorFadeModelState:
    """Mutable fade state owned by the model loop."""

    config: ColorFadeConfig
    session_desc: SessionDesc
    total_steps: int
    steps_generated: int = 0


class ColorFadeModelLoop(IModelLoop[ColorFadeModelState]):
    """Generate color-fade frames through the standard model loop."""

    def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]:
        del events
        self.state.steps_generated += 1
        return [
            StepResult(
                step_index=step_index,
                output=_frames(self.state, step_index),
                frame_count=self.state.config.frames_per_step,
                output_layout=self.state.session_desc.output_layout,
            )
        ]

    def is_finished(self) -> bool:
        return self.state.steps_generated >= self.state.total_steps

    def reset(self) -> None:
        self.state.steps_generated = 0


class ColorFadeSession(ISession):
    """Emit solid frames fading from red to green, then stay green.

    Each frame's colour comes from when it plays rather than from which step
    produced it: a frame's time is its position in the run divided by
    ``frames_per_second_for_step``. The fade therefore takes the same time
    however many frames a step generates.

    The session finishes once it has generated the fade, so a run against it
    ends on its own rather than on a count the caller had to work out. A caller
    that stops it earlier gets part of the fade.

    Pixels are ``[-1, 1]`` floats, which is what FlashDreams models emit and
    what an output sink expects of a floating point result.
    """

    def __init__(self, config: ColorFadeConfig, session_desc: SessionDesc) -> None:
        """
        Args:
            config: Resolved settings shared with the owning application.
            session_desc: Description of the session the runtime asked for.
                Honoured as-is; this application can produce any frame size.

        Raises:
            ValueError: ``session_desc`` requests a layout other than ``bcthw``.
        """
        if session_desc.output_layout is not VideoTensorLayout.bcthw:
            raise ValueError(
                "Colour fade only produces bcthw output, got "
                f"{session_desc.output_layout.value}."
            )
        self._config = config
        self._session_desc = session_desc
        # One step past the fade's last whole frame, so the run ends on a frame
        # that is fully green rather than a shade short of it.
        frames = 1 + math.floor(
            config.seconds * session_desc.frames_per_second_for_step
        )
        self._total_steps = math.ceil(frames / config.frames_per_step)

    def init(self) -> None:
        """Register the color-fade model loop; the default UI blits it."""
        self.register_model_loop(
            ColorFadeModelLoop,
            state=ColorFadeModelState(
                config=self._config,
                session_desc=self._session_desc,
                total_steps=self._total_steps,
            ),
        )

    @property
    def session_desc(self) -> SessionDesc:
        return self._session_desc


def _frames(state: ColorFadeModelState, step_index: int) -> Tensor:
    frames_per_step = state.config.frames_per_step
    seconds_per_frame = 1.0 / state.session_desc.frames_per_second_for_step
    frame_times = (
        torch.arange(frames_per_step, dtype=torch.float32)
        + step_index * frames_per_step
    ) * seconds_per_frame
    progress = (frame_times / state.config.seconds).clamp(max=1.0)

    channels = torch.full((3, frames_per_step), _NO_INTENSITY, dtype=torch.float32)
    span = _FULL_INTENSITY - _NO_INTENSITY
    channels[_RED_CHANNEL] = _FULL_INTENSITY - span * progress
    channels[_GREEN_CHANNEL] = _NO_INTENSITY + span * progress
    # Expand each frame's color over its height and width.
    return (
        channels.view(1, 3, frames_per_step, 1, 1)
        .expand(
            -1,
            -1,
            -1,
            state.session_desc.video_height,
            state.session_desc.video_width,
        )
        .contiguous()
    )


## Application


class ColorFadeApplication(IApplication):
    """Application generating a red-to-green fade, ignoring input."""

    def __init__(self) -> None:
        self._config: ColorFadeConfig | None = None

    def init(self, commandline_args: Sequence[str]) -> None:
        """Parse the fade length and how many frames a step generates.

        Args:
            commandline_args: Application-specific arguments.

        Raises:
            ValueError: An argument is not positive.
        """
        parser = argparse.ArgumentParser(
            prog="color-fade",
            description="Fade a solid frame from red to green.",
        )
        parser.add_argument("--seconds", type=float, default=_DEFAULT_SECONDS)
        parser.add_argument(
            "--frames-per-step", type=int, default=_DEFAULT_FRAMES_PER_STEP
        )
        args = parser.parse_args(list(commandline_args))

        # Not just a sign check: a fade of nan seconds makes every frame nan,
        # which reaches a sink as a picture rather than as an error.
        if not math.isfinite(args.seconds) or args.seconds <= 0:
            raise ValueError(f"--seconds must be finite and > 0, got {args.seconds}.")
        if args.frames_per_step <= 0:
            raise ValueError(
                f"--frames-per-step must be > 0, got {args.frames_per_step}."
            )
        self._config = ColorFadeConfig(
            seconds=args.seconds, frames_per_step=args.frames_per_step
        )

    def create_session(self, session_desc: SessionDesc) -> ISession:
        """Create one uninitialized colour fade session.

        Raises:
            RuntimeError: :meth:`init` has not run yet.
        """
        if self._config is None:
            raise RuntimeError(
                "ColorFadeApplication.init() must run before create_session()."
            )
        return ColorFadeSession(self._config, session_desc)


def create_app() -> IApplication:
    """Return a new colour fade application."""
    return ColorFadeApplication()
