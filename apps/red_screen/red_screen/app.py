# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Key-driven red screen application for end-to-end v2 API testing."""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.loop import IModelLoop
from flashdreams.api_v2.session import ISession
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    KeyboardInputState,
    KeyboardUserInputEventData,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

_DEFAULT_ACTIVATION_KEY = "r"
"""Key that turns the screen red while held."""

_RED_CHANNEL = 0
"""Channel index set to full intensity while the activation key is held."""

_FULL_INTENSITY = 1.0
"""Full intensity for a channel, in the ``[-1, 1]`` range a model emits."""

_NO_INTENSITY = -1.0
"""No intensity for a channel, which is black across all three."""


@dataclass(frozen=True, slots=True)
class RedScreenConfig:
    """Resolved settings for one red screen application."""

    activation_key: str
    """Key whose held state selects red over black."""


## Session


@dataclass(slots=True)
class RedScreenModelState:
    """Mutable key state owned by the model loop."""

    config: RedScreenConfig
    session_desc: SessionDesc
    key_held: bool = False
    color_intensity: float = 0.0


class RedScreenModelLoop(IModelLoop[RedScreenModelState]):
    """Generate key-controlled frames through the standard model loop."""

    def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]:
        _apply_events(self.state, events)
        return [
            StepResult(
                step_index=step_index,
                output=_frame(self.state),
                frame_count=1,
                output_layout=self.state.session_desc.output_layout,
            )
        ]

    def reset(self) -> None:
        self.state.key_held = False
        self.state.color_intensity = 0.0


class RedScreenSession(ISession):
    """Emit red frames controlled by activation and intensity keys."""

    def __init__(self, config: RedScreenConfig, session_desc: SessionDesc) -> None:
        """
        Args:
            config: Resolved settings shared with the owning application.
            session_desc: Session the runtime asked for. Honoured as-is; this
                application can produce any frame size.

        Raises:
            ValueError: ``session_desc`` requests a layout other than ``bcthw``.
        """
        if session_desc.output_layout is not VideoTensorLayout.bcthw:
            raise ValueError(
                "Red screen only produces bcthw output, got "
                f"{session_desc.output_layout.value}."
            )
        self._config = config
        self._session_desc = session_desc

    def init(self) -> None:
        """Reset key state and color intensity to start on a black frame."""
        self.register_model_loop(
            RedScreenModelLoop,
            state=RedScreenModelState(
                config=self._config,
                session_desc=self._session_desc,
            ),
        )

    @property
    def session_desc(self) -> SessionDesc:
        return self._session_desc


def _apply_events(state: RedScreenModelState, events: UserInputEvents) -> None:
    received_events = events.get_events()
    if not received_events:
        return
    data = received_events[-1].get_event_data()
    if not isinstance(data, KeyboardUserInputEventData):
        return
    if data.key == state.config.activation_key:
        state.key_held = data.state is KeyboardInputState.PRESSED
    elif data.state is KeyboardInputState.PRESSED and data.key.lower() == "w":
        state.color_intensity = min(1.0, state.color_intensity + 0.1)
    elif data.state is KeyboardInputState.PRESSED and data.key.lower() == "s":
        state.color_intensity = max(0.0, state.color_intensity - 0.1)


def _frame(state: RedScreenModelState) -> Tensor:
    frame = torch.full(
        (1, 3, 1, state.session_desc.video_height, state.session_desc.video_width),
        _NO_INTENSITY,
        dtype=torch.float32,
    )
    frame[:, _RED_CHANNEL] = (
        _FULL_INTENSITY if state.key_held else 2.0 * state.color_intensity - 1.0
    )
    return frame


## Application


class RedScreenApplication(IApplication):
    """Application producing red frames whose intensity responds to key input."""

    def __init__(self) -> None:
        self._config: RedScreenConfig | None = None

    def init(self, commandline_args: Sequence[str]) -> None:
        """Parse the activation key.

        Neither the frame size nor the length of the run is an application
        argument: the runtime supplies the width and height per session through
        :class:`SessionDesc`, and decides how many steps to run when it drives
        the session.

        Args:
            commandline_args: Application-specific arguments.
        """
        parser = argparse.ArgumentParser(
            prog="red-screen",
            description="Turn the screen red while a key is held.",
        )
        parser.add_argument("--key", default=_DEFAULT_ACTIVATION_KEY)
        args = parser.parse_args(list(commandline_args))

        self._config = RedScreenConfig(activation_key=args.key)

    def create_session(self, session_desc: SessionDesc) -> ISession:
        """Create one uninitialized red screen session.

        Raises:
            RuntimeError: :meth:`init` has not run yet.
        """
        if self._config is None:
            raise RuntimeError(
                "RedScreenApplication.init() must run before create_session()."
            )
        return RedScreenSession(self._config, session_desc)


def create_app() -> IApplication:
    """Return a new red screen application."""
    return RedScreenApplication()
