# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Key-driven red screen application for end-to-end v2 API testing."""

import argparse
from dataclasses import dataclass

import torch
from torch import Tensor

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.session import ISession
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import KeyboardUserInputEventData
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

_DEFAULT_ACTIVATION_KEY = "r"
"""Key that turns the screen red while held."""

_RED_CHANNEL = 0
"""Channel index set to full intensity while the activation key is held."""


@dataclass(frozen=True, slots=True)
class RedScreenConfig:
    """Resolved settings for one red screen application."""

    activation_key: str
    """Key whose held state selects red over black."""


## Session


class RedScreenSession(ISession):
    """Emit a red frame while the activation key is held, black otherwise."""

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
        self._key_held = False

    def init(self) -> None:
        """Release any held key so the session starts on a black frame."""
        self._key_held = False

    @property
    def session_desc(self) -> SessionDesc:
        return self._session_desc

    def step(self, step_index: int, events: UserInputEvents) -> StepResult:
        """Apply the events, then emit one frame for ``step_index``.

        Args:
            step_index: Zero-based index of this step.
            events: Events collected since the previous step.

        Returns:
            Result carrying a single ``[1, 3, 1, H, W]`` frame.
        """
        self._apply_events(events)
        return StepResult(
            step_index=step_index,
            output=self._frame(),
            frame_count=1,
            output_layout=self._session_desc.output_layout,
        )

    def reset(self) -> None:
        """Restart the session so it can produce another generation."""
        self.init()

    def _apply_events(self, events: UserInputEvents) -> None:
        # Events are edges, not levels: a key stays held across steps that carry
        # no events for it, so only the last edge per step changes the state.
        for event in events.get_events():
            data = event.get_event_data()
            if (
                isinstance(data, KeyboardUserInputEventData)
                and data.key == self._config.activation_key
            ):
                self._key_held = data.pressed

    def _frame(self) -> Tensor:
        frame = torch.zeros(
            (1, 3, 1, self._session_desc.video_height, self._session_desc.video_width),
            dtype=torch.float32,
        )
        if self._key_held:
            frame[:, _RED_CHANNEL] = 1.0
        return frame


## Application


class RedScreenApplication(IApplication):
    """Application producing solid red or black frames from key input."""

    def __init__(self) -> None:
        self._config: RedScreenConfig | None = None

    def init(self, commandline_args: list[str]) -> None:
        """Parse the activation key.

        Neither frame geometry nor rollout length is an application argument: the
        runtime supplies geometry per session through :class:`SessionDesc`, and
        decides how long to run when it drives the session.

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
