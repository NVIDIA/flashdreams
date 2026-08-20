# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command line running one application and writing what it generates to a file.

``flashdreams-run-v2`` is the v2 command line, and writes MP4. The interactive
path has no window to offer yet: ``IClientWindow`` is a protocol that nothing
outside the tests implements, so there is no ``--output local-window`` here yet.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from flashdreams.api_v2.application import IApplication
from flashdreams.runtime_v2.applications import (
    create_application,
    registered_application_slugs,
)
from flashdreams.runtime_v2.batch_runner import run_batch
from flashdreams.runtime_v2.mp4_output_sink import Mp4OutputSink
from flashdreams.t2v_v2.application import T2VApplication

_ARGUMENT_SEPARATOR = "--"
"""What separates this command's arguments from the application's.

An application declares whatever arguments it likes, including ones this command
also has, so the split is stated rather than guessed.
"""


def run_application(
    application: IApplication,
    *,
    output_path: str | Path,
    steps: int | None = None,
    application_args: Sequence[str] = (),
) -> Path:
    """Initialize an application, generate from it, and write an MP4.

    The application is closed before this returns, whether or not the run
    finished, since this owns the one it was given for the length of the call.

    Args:
        application: Uninitialized application to run.
        output_path: File to write.
        steps: Steps to generate, or ``None`` to generate the rollout the
            application says it normally would.
        application_args: Arguments for the application, such as a prompt.

    Returns:
        The file that was written.

    Raises:
        TypeError: The application cannot describe the session it would
            generate, so there is nothing to ask it for.
        ValueError: ``steps`` is not positive.
    """
    if steps is not None and steps <= 0:
        raise ValueError(f"--steps must be > 0, got {steps}.")

    try:
        # Inside, as the batch loop closes a session that failed to init: an
        # application that got half way through holds whatever it acquired.
        application.init(list(application_args))
        describes_itself = _as_t2v(application)
        run_batch(
            application.create_session(describes_itself.session_desc()),
            Mp4OutputSink(output_path),
            steps=steps if steps is not None else describes_itself.total_blocks,
        )
    finally:
        application.close()
    return Path(output_path)


def run(
    slug: str,
    *,
    output_path: str | Path,
    steps: int | None = None,
    application_args: Sequence[str] = (),
) -> Path:
    """Find the application ``slug`` names and run it. See :func:`run_application`."""
    return run_application(
        create_application(slug),
        output_path=output_path,
        steps=steps,
        application_args=application_args,
    )


def entrypoint(argv: Sequence[str] | None = None) -> None:
    """Run the command, printing the file it wrote."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    own_args, application_args = split_arguments(arguments)
    parsed = _parser().parse_args(own_args)
    written = run(
        parsed.slug,
        output_path=parsed.output_path,
        steps=parsed.steps,
        application_args=application_args,
    )
    print(written)


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
        description="Generate from an application and write the result to an MP4.",
        epilog=(
            f"Installed applications: {installed}. Arguments after -- go to the "
            "application, so `flashdreams-run-v2 SLUG -- --help` describes it."
        ),
    )
    parser.add_argument("slug", help="Application to run.")
    parser.add_argument(
        "--output-path", required=True, type=Path, help="MP4 file to write."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Steps to generate. Defaults to the application's rollout length.",
    )
    return parser


def _as_t2v(application: IApplication) -> T2VApplication:
    """Return the application as one that can describe its own session.

    Raises:
        TypeError: It cannot. Only text-to-video says what session it would
            generate, through :class:`T2VApplication`; the protocol itself does
            not carry it, so anything else has nothing to be asked.
    """
    if not isinstance(application, T2VApplication):
        raise TypeError(
            f"{type(application).__name__} cannot describe the session it would "
            "generate, so it cannot be run from the command line yet. Only "
            "text-to-video applications can, through T2VApplication."
        )
    return application
