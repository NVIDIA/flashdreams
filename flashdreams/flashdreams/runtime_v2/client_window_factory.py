# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create v2 client windows from runtime arguments.

A mode is one way to watch a run: an MP4 file, a browser, whatever comes after
them. Each mode owns the arguments it takes and what to tell the user about
where its output went, so a command line offering the modes never has to know
what any of them are, and adding one is adding it here.
"""

import argparse
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, cast

from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.runtime_v2.mp4_client_window import Mp4ClientWindow

if TYPE_CHECKING:
    from flashdreams.runtime_v2.webrtc_client_window import WebRTCClientWindow


class ClientWindowMode(ABC):
    """One way to present a run."""

    name: str
    """What ``--mode`` calls this."""

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add the arguments this mode takes and no other does."""

    def check_arguments(self, parsed_args: argparse.Namespace) -> None:
        """Report a usage error before anything is built.

        A command calls this while it can still print its usage, rather than
        after loading a model to find out the run had nowhere to go.

        Raises:
            ValueError: The arguments do not describe a window this can create.
        """
        del parsed_args

    @abstractmethod
    def create(self, parsed_args: argparse.Namespace) -> IClientWindow:
        """Create the window.

        Raises:
            ValueError: Whatever :meth:`check_arguments` reports.
        """

    def starting(self, client_window: IClientWindow) -> str | None:
        """Return what to tell the user before the run, such as where to watch."""
        del client_window
        return None

    def finished(self, client_window: IClientWindow) -> str | None:
        """Return what to tell the user after a run that generated everything."""
        del client_window
        return None


class _Mp4Mode(ClientWindowMode):
    """Write the run to a file, with nobody watching it happen."""

    name = "mp4"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--output-path", type=Path, help="MP4 file to write. Required for mp4."
        )
        parser.add_argument(
            "--stats-path",
            type=Path,
            default=None,
            help=(
                "JSON file to record what each step measured in, for a benchmark "
                "to read. Nothing is measured unless this is asked for."
            ),
        )

    def check_arguments(self, parsed_args: argparse.Namespace) -> None:
        if parsed_args.output_path is None:
            raise ValueError("--output-path is required when writing an MP4.")

    def create(self, parsed_args: argparse.Namespace) -> IClientWindow:
        self.check_arguments(parsed_args)
        return Mp4ClientWindow(
            parsed_args.output_path, stats_path=parsed_args.stats_path
        )

    def finished(self, client_window: IClientWindow) -> str | None:
        """Return the file, now that there is something in it to watch."""
        return str(cast(Mp4ClientWindow, client_window).path)


class _WebRTCMode(ClientWindowMode):
    """Stream the run to a browser."""

    name = "webrtc"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--host", default="127.0.0.1", help="Interface to serve on."
        )
        parser.add_argument("--port", type=int, default=0, help="Port to serve on.")

    def create(self, parsed_args: argparse.Namespace) -> IClientWindow:
        # Imported here so a run writing a file needs none of the serving stack.
        from flashdreams.runtime_v2.webrtc_client_window import WebRTCClientWindow

        return WebRTCClientWindow(host=parsed_args.host, port=parsed_args.port)

    def starting(self, client_window: IClientWindow) -> str | None:
        """Return where to connect, which nobody can guess when the port is free."""
        server = cast("WebRTCClientWindow", client_window).server
        return f"Open {server.url} in a browser."


_MODES: tuple[ClientWindowMode, ...] = (_Mp4Mode(), _WebRTCMode())
"""Modes a run can be presented through, the first being the default."""


def add_client_window_arguments(parser: argparse.ArgumentParser) -> None:
    """Add ``--mode`` and the arguments each mode takes.

    Every mode's arguments are added, since which mode was asked for is not
    known until they are parsed. A mode reads its own and no others.
    """
    parser.add_argument(
        "--mode",
        choices=tuple(mode.name for mode in _MODES),
        default=_MODES[0].name,
        help="Where the run goes. Default: %(default)s.",
    )
    for mode in _MODES:
        mode.add_arguments(parser)


def client_window_mode(name: str) -> ClientWindowMode:
    """Return the mode of that name.

    Raises:
        ValueError: Nothing here presents a run that way.
    """
    for mode in _MODES:
        if mode.name == name:
            return mode
    raise ValueError(f"Unsupported client-window mode: {name!r}.")


def create_client_window(parsed_args: argparse.Namespace) -> IClientWindow:
    """Create the client window selected by the presentation mode.

    Args:
        parsed_args: Runtime arguments. Mode-specific fields are read only by
            the selected mode.

    Returns:
        Client window for the selected mode.

    Raises:
        ValueError: ``mode`` is unsupported, or its arguments are incomplete.
    """
    return client_window_mode(parsed_args.mode).create(parsed_args)
