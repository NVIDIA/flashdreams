# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command line running one v2 application.

``flashdreams-run-v2`` finds an application by slug, gives it the arguments
after ``--``, and hands it to :class:`ApplicationRunner` along with the window
``--mode`` asked for: an MP4 file, or a client over WebRTC.

The session it asks for is the one the application says it would generate, with
whatever the frame arguments here override. An application that would generate
nothing in particular is described by those arguments alone.

A run can also record what each step measured, for a benchmark to compare runs
by speed as well as by how they look.
"""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.runtime_v2.application_runner import ApplicationRunner
from flashdreams.runtime_v2.applications import (
    create_application,
    registered_application_slugs,
)
from flashdreams.runtime_v2.client_window_factory import create_client_window
from flashdreams.runtime_v2.mp4_client_window import Mp4ClientWindow
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout
from flashdreams.runtime_v2.webrtc_client_window import WebRTCClientWindow

_ARGUMENT_SEPARATOR = "--"
"""What separates this command's arguments from the application's.

An application declares whatever arguments it likes, including ones this command
also has, so the split is stated rather than guessed.
"""

_MP4_MODE = "mp4"
"""Mode writing a file, and the only one with nobody watching."""

_WEBRTC_MODE = "webrtc"
"""Mode streaming to a browser."""


def entrypoint(argv: Sequence[str] | None = None) -> None:
    """Run the command, reporting where to watch what it generates."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    own_args, application_args = split_arguments(arguments)
    parser = _parser()
    parsed = parser.parse_args(own_args)
    if parsed.mode == _MP4_MODE and parsed.output_path is None:
        parser.error("--output-path is required when writing an MP4.")

    # Before the window, so a slug this cannot run costs nothing to find out.
    application = create_application(parsed.slug)
    session_desc = _session_desc(application, parsed)
    window = _client_window(parsed)
    if isinstance(window, WebRTCClientWindow):
        print(f"Open {window.server.url} in a browser.", flush=True)
    # Nothing here says how long the run is. A session generates what it was
    # configured to and reports itself finished, and a window with a client on
    # the other end ends the run when that client goes away.
    ApplicationRunner(application, window).run(session_desc, application_args)
    if parsed.mode == _MP4_MODE:
        print(parsed.output_path)


def split_arguments(arguments: Sequence[str]) -> tuple[list[str], list[str]]:
    """Split this command's arguments from the application's at ``--``.

    Args:
        arguments: Everything after the command name.

    Returns:
        This command's arguments, then the application's. Everything belongs to
        this command when there is no separator, so an application taking no
        arguments needs no ``--``.
    """
    arguments = list(arguments)
    if _ARGUMENT_SEPARATOR not in arguments:
        return arguments, []
    index = arguments.index(_ARGUMENT_SEPARATOR)
    return arguments[:index], arguments[index + 1 :]


def _parser() -> argparse.ArgumentParser:
    """Return the parser for this command's own arguments."""
    installed = ", ".join(registered_application_slugs()) or "(none)"
    parser = argparse.ArgumentParser(
        prog="flashdreams-run-v2",
        description="Run a FlashDreams application, to a file or to a browser.",
        epilog=(
            f"Installed applications: {installed}. Arguments after -- go to the "
            "application, so `flashdreams-run-v2 SLUG -- --help` describes it."
        ),
    )
    parser.add_argument("slug", help="Application to run.")
    parser.add_argument(
        "--mode",
        choices=(_MP4_MODE, _WEBRTC_MODE),
        default=_MP4_MODE,
        help="Where the run goes. Default: %(default)s.",
    )
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
    parser.add_argument("--host", default="127.0.0.1", help="Interface to serve on.")
    parser.add_argument("--port", type=int, default=0, help="Port to serve on.")
    _add_session_arguments(parser)
    return parser


def _add_session_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the arguments describing the session to ask the application for.

    Every one of them defaults to asking for nothing, so a run that says none of
    them gets what the application generates, which for a model is the clip it
    was trained for.
    """
    parser.add_argument(
        "--pixel-width", type=int, default=None, help="Frame width to generate."
    )
    parser.add_argument(
        "--pixel-height", type=int, default=None, help="Frame height to generate."
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Rate the generated frames are meant to play at.",
    )
    parser.add_argument(
        "--layout",
        type=VideoTensorLayout,
        choices=tuple(VideoTensorLayout),
        default=None,
        metavar="{" + ",".join(layout.value for layout in VideoTensorLayout) + "}",
        help="Tensor layout to generate results in.",
    )


def _session_desc(
    application: IApplication, parsed_args: argparse.Namespace
) -> SessionDesc:
    """Return the session to ask for: the application's, with the arguments on top.

    An application that describes no session of its own is described by the
    arguments alone, and by :class:`SessionDesc`'s own defaults for the rest.
    """
    asked_for: dict[str, Any] = {
        field: value
        for field, value in (
            ("output_layout", parsed_args.layout),
            ("frames_per_second_for_step", parsed_args.fps),
            ("video_width", parsed_args.pixel_width),
            ("video_height", parsed_args.pixel_height),
        )
        if value is not None
    }
    described = application.session_desc()
    if described is None:
        return SessionDesc(**asked_for)
    return replace(described, **asked_for)


def _client_window(parsed_args: argparse.Namespace) -> IClientWindow:
    """Create the window the arguments ask for."""
    if parsed_args.mode == _MP4_MODE:
        return Mp4ClientWindow(
            parsed_args.output_path, stats_path=parsed_args.stats_path
        )
    return create_client_window(parsed_args)
