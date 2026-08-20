# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Key-driven red screen application for end-to-end v2 API testing."""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.session import ISession
from flashdreams.runtime_v2.application_runner import ApplicationRunner
from flashdreams.runtime_v2.client_window_factory import create_client_window
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import KeyboardUserInputEventData
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout
from flashdreams.runtime_v2.webrtc_client_window import WebRTCClientWindow

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
        self._key_held = False
        self._color_intensity = 0.0

    def init(self) -> None:
        """Reset key state and color intensity to start on a black frame."""
        self._key_held = False
        self._color_intensity = 0.0

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
        import time

        # Simulate the real model inference time
        time.sleep(0.1)
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
        received_events = events.get_events()
        if not received_events:
            return
        data = received_events[-1].get_event_data()
        if not isinstance(data, KeyboardUserInputEventData):
            return
        if data.key == self._config.activation_key:
            self._key_held = data.pressed
        elif data.pressed and data.key.lower() == "w":
            self._color_intensity = min(1.0, self._color_intensity + 0.1)
        elif data.pressed and data.key.lower() == "s":
            self._color_intensity = max(0.0, self._color_intensity - 0.1)

    def _frame(self) -> Tensor:
        frame = torch.full(
            (1, 3, 1, self._session_desc.video_height, self._session_desc.video_width),
            _NO_INTENSITY,
            dtype=torch.float32,
        )
        frame[:, _RED_CHANNEL] = (
            _FULL_INTENSITY if self._key_held else 2.0 * self._color_intensity - 1.0
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


def _parse_args(commandline_args: Sequence[str] | None) -> argparse.Namespace:
    """Parse runtime arguments and preserve application arguments."""
    parser = argparse.ArgumentParser(
        prog="red-screen-webrtc",
        description="Serve the red-screen v2 application.",
    )
    parser.add_argument("--mode", choices=("webrtc",), default="webrtc")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "application_args",
        nargs=argparse.REMAINDER,
        help="Arguments after -- are passed to the red-screen application.",
    )
    return parser.parse_args(commandline_args)


def main(commandline_args: Sequence[str] | None = None) -> int:
    """Run red screen until the client disconnects or the process is interrupted."""
    args = _parse_args(commandline_args)
    application_args = list(args.application_args)
    if application_args[:1] == ["--"]:
        application_args = application_args[1:]

    window = create_client_window(args)
    app = create_app()
    if isinstance(window, WebRTCClientWindow):
        print(f"Open {window.server.url} in a browser.", flush=True)
    try:
        # ApplicationRunner is a FlashDreams runtime component that takes an IApplication instance, a IClientWindow instance,
        # and drives the main loop.

        # TODO: in production, commandline argument parsing and IClientWindow creation should be done by flashdreams-run, a CLI tool
        # basically, we need to generailze this main function to be shared by all applications
        ApplicationRunner(app, window).run(
            SessionDesc(
                output_layout=VideoTensorLayout.bcthw,
                frames_per_second_for_ui=args.fps,
                frames_per_second_for_step=args.fps,
                video_width=args.width,
                video_height=args.height,
            ),
            application_args,
        )
    except KeyboardInterrupt:
        return 130
    finally:
        window.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
