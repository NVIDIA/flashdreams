# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command line running one text-to-video application.

``flashdreams-run-v2`` loads an application by slug, asks it for the session it
would generate, and drives that session against the window its arguments ask
for: an MP4 file, or a client over WebRTC.

This lives beside the text-to-video application rather than in the runtime,
because asking an application what session it would generate is something only
:class:`T2VApplication` offers. A command line for any v2 application needs that
on ``IApplication`` first, and then this becomes an argument parser over it.

A run can also record what each step measured, for a benchmark to compare runs
by speed as well as by how they look.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.runtime_v2.applications import (
    create_application,
    registered_application_slugs,
)
from flashdreams.runtime_v2.client_window_factory import create_client_window
from flashdreams.runtime_v2.mp4_client_window import Mp4ClientWindow
from flashdreams.runtime_v2.session_runner import run_session
from flashdreams.runtime_v2.webrtc_client_window import WebRTCClientWindow
from flashdreams.t2v_v2.application import T2VApplication

_ARGUMENT_SEPARATOR = "--"
"""What separates this command's arguments from the application's.

An application declares whatever arguments it likes, including ones this command
also has, so the split is stated rather than guessed.
"""

_MP4_MODE = "mp4"
"""Mode writing a file, and the only one with nobody watching."""

_WEBRTC_MODE = "webrtc"
"""Mode streaming to a browser."""


def run_application(
    application: T2VApplication,
    client_window: IClientWindow,
    *,
    application_args: Sequence[str] = (),
) -> None:
    """Initialize an application and run one session of it against a window.

    Nothing here says how long the run is. A text-to-video session generates the
    rollout it was configured for and reports itself finished, and a window with
    a client on the other end ends the run when that client goes away.

    The application is closed before this returns, whether or not the run
    finished, since this owns the one it was given for the length of the call.

    Args:
        application: Uninitialized application to run.
        client_window: Window presenting or writing what the run generates.
        application_args: Arguments for the application, such as a prompt.
    """
    try:
        # Inside, as the loop closes a session that failed to init: an
        # application that got half way through holds whatever it acquired.
        application.init(list(application_args))
        # Not through ApplicationRunner, which initializes the application
        # itself: the description has to come from one already initialized.
        run_session(
            application.create_session(application.session_desc()), client_window
        )
    finally:
        application.close()


def entrypoint(argv: Sequence[str] | None = None) -> None:
    """Run the command, reporting where to watch what it generates."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    own_args, application_args = split_arguments(arguments)
    parser = _parser()
    parsed = parser.parse_args(own_args)
    if parsed.mode == _MP4_MODE and parsed.output_path is None:
        parser.error("--output-path is required when writing an MP4.")

    # Before the window, so a slug this cannot run costs nothing to find out.
    application = _t2v_application(parsed.slug)
    window = _client_window(parsed)
    if isinstance(window, WebRTCClientWindow):
        print(f"Open {window.server.url} in a browser.", flush=True)
    run_application(application, window, application_args=application_args)
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
        description="Generate video from text, to a file or to a browser.",
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
    return parser


def _client_window(parsed_args: argparse.Namespace) -> IClientWindow:
    """Create the window the arguments ask for."""
    if parsed_args.mode == _MP4_MODE:
        return Mp4ClientWindow(
            parsed_args.output_path, stats_path=parsed_args.stats_path
        )
    return create_client_window(parsed_args)


def _t2v_application(slug: str) -> T2VApplication:
    """Load the application ``slug`` names, as a text-to-video one.

    Raises:
        TypeError: It is not one. This command asks an application what session
            it would generate, which is a text-to-video question until the
            protocol itself carries it.
    """
    application = create_application(slug)
    if not isinstance(application, T2VApplication):
        raise TypeError(
            f"{slug} is a {type(application).__name__}, and this command only "
            "runs text-to-video applications."
        )
    return application
