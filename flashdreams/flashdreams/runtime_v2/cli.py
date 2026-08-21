# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command line running one v2 application.

``flashdreams-run-v2`` finds an application by slug, gives it the arguments
after ``--``, and hands it to :class:`ApplicationRunner` along with the window
``--mode`` asked for. The session it asks for is the one the application says it
would generate, with whatever the frame arguments here override.

What the modes are, and what each one takes, belongs to
:mod:`flashdreams.runtime_v2.client_window_factory`. Nothing here reads an
argument that only one of them uses.
"""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from flashdreams.api_v2.application import IApplication
from flashdreams.runtime_v2.application_registry import (
    create_application,
    registered_application_slugs,
)
from flashdreams.runtime_v2.application_runner import ApplicationRunner
from flashdreams.runtime_v2.client_window_factory import (
    add_client_window_arguments,
    client_window_mode,
)
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

_ARGUMENT_SEPARATOR = "--"
"""What separates this command's arguments from the application's.

An application declares whatever arguments it likes, including ones this
command also has, so the split is stated rather than guessed.
"""


def entrypoint(argv: Sequence[str] | None = None) -> None:
    """Run the command, reporting where to watch what it generates."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    own_args, application_args = split_arguments(arguments)
    parser = _parser()
    parsed = parser.parse_args(own_args)

    mode = client_window_mode(parsed.mode)
    try:
        mode.check_arguments(parsed)
    except ValueError as error:
        parser.error(str(error))

    # Before the window, so a slug this cannot run costs nothing to find out.
    application = create_application(parsed.slug)
    session_desc = _session_desc(application, parsed)
    window = mode.create(parsed)
    _report(mode.starting(window))
    # Nothing here says how long the run is: a session reports itself finished,
    # and a window ends the run when its client goes away.
    ApplicationRunner(application, window).run(session_desc, application_args)
    _report(mode.finished(window))


def split_arguments(arguments: Sequence[str]) -> tuple[list[str], list[str]]:
    """Split this command's arguments from the application's at ``--``.

    Args:
        arguments: Everything after the command name.

    Returns:
        This command's arguments, then the application's. Everything belongs to
        this command when there is no separator.
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
    add_client_window_arguments(parser)
    _add_session_arguments(parser)
    return parser


def _report(message: str | None) -> None:
    """Print what the mode has to say about the run, if it has anything."""
    if message is not None:
        print(message, flush=True)


def _add_session_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the arguments describing the session to ask the application for.

    Each defaults to asking for nothing, so a run that names none of them gets
    what the application generates.
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

    An application describing no session of its own gets the arguments alone,
    over :class:`SessionDesc`'s own defaults.
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
