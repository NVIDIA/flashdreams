# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command line running one v2 application.

``flashdreams-run-v2`` finds an application by slug, gives it the arguments
after ``--``, and hands it to :class:`ApplicationRunner` along with the window
``--mode`` asked for. The session it asks for is the one the application says it
would generate, with whatever the frame arguments here override.

What the modes are, and what each one takes, belongs to
:mod:`flashdreams.runtime_v2.client_window_factory`. Nothing here reads an
argument that only one of them uses. ``--timeout`` and ``--total-model-steps``
belong to this command; ``--mode mp4`` requires that at least one of them is
present. A named steps count is remaining for the whole application run.
"""

import argparse
import math
import os
import sys
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from flashdreams.api_v2.application import IApplication
from flashdreams.core.distributed import (
    get_global_rank_for_logging,
)
from flashdreams.core.distributed import (
    shutdown as shutdown_distributed,
)
from flashdreams.runtime_v2.application_registry import (
    create_application,
    registered_application_slugs,
)
from flashdreams.runtime_v2.application_runner import (
    ApplicationFlags,
    ApplicationRunner,
    Unbound,
)
from flashdreams.runtime_v2.client_window_factory import (
    add_client_window_arguments,
    client_window_mode,
)
from flashdreams.runtime_v2.metrics_output_sink import MetricsOutputSink
from flashdreams.runtime_v2.session_desc import (
    BackpressureMode,
    PresentationMode,
    SessionDesc,
)
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

_ARGUMENT_SEPARATOR = "--"
"""What separates this command's arguments from the application's.

An application declares whatever arguments it likes, including ones this
command also has, so the split is stated rather than guessed.
"""

_HELP_FLAGS = frozenset({"-h", "--help"})
"""What an application's own arguments use to ask for its help."""


def entrypoint(argv: Sequence[str] | None = None) -> None:
    """Run the command, reporting where to watch what it generates."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    own_args, application_args = split_arguments(arguments)
    parser = _parser()
    parsed = parser.parse_args(own_args)
    if parsed.stats_path is not None:
        os.environ["FLASHDREAMS_SYNC_AND_PROFILE"] = "1"
    if parsed.preload_application and any(
        argument == "--mode" or argument.startswith("--mode=") for argument in own_args
    ):
        parser.error("--mode cannot be used with --preload-application.")
    if parsed.skip_preload_validation and not parsed.preload_application:
        parser.error("--skip-preload-validation requires --preload-application.")
    if parsed.preload_application:
        parsed.mode = "null"

    # Asking an application what it takes is answered by the application alone,
    # so a run that only wants its help neither checks the arguments for a
    # window nor opens one.
    wants_application_help = bool(_HELP_FLAGS.intersection(application_args))
    # Before the window, so a slug this cannot run costs nothing to find out.
    application = create_application(parsed.slug)
    if wants_application_help:
        # Parsing is init's first job, so this prints the help and exits.
        application.init(application_args)
        return
    if parsed.mode is None:
        preferred_mode = application.default_client_window_mode()
        parsed.mode = "mp4" if preferred_mode is None else preferred_mode
    try:
        mode = client_window_mode(parsed.mode, modes=parsed._client_window_modes)
        mode.check_arguments(parsed)
    except ValueError as error:
        parser.error(str(error))
    if parsed.mode == "mp4" and parsed.presentation_mode is None:
        parsed.presentation_mode = PresentationMode.ON_DEMAND
    if (
        parsed.mode == "mp4"
        and parsed.timeout is None
        and parsed.total_model_steps is None
    ):
        parser.error("--mode mp4 requires --timeout and/or --total-model-steps.")
    session_desc = _session_desc(application, parsed)
    worker = get_global_rank_for_logging() != 0
    window = None if worker else mode.create(parsed)
    if window is not None:
        _report(mode.starting(window))
    # The session's UI and client input decide when the run ends.
    metrics_output_sink = (
        None
        if worker or parsed.stats_path is None
        else MetricsOutputSink(parsed.stats_path)
    )
    completed = False
    try:
        ApplicationRunner(
            application,
            window,
            metrics_output_sink=metrics_output_sink,
            application_flags=ApplicationFlags(
                preload=parsed.preload_application,
                skip_preload_validation=parsed.skip_preload_validation,
            ),
        ).run(
            session_desc,
            application_args,
            timeout_seconds=(
                Unbound.unbound if parsed.timeout is None else parsed.timeout
            ),
            steps=(
                Unbound.unbound
                if parsed.total_model_steps is None
                else parsed.total_model_steps
            ),
        )
        if window is not None:
            _report(mode.finished(window))
        completed = True
    finally:
        shutdown_distributed(
            synchronize=completed,
            terminate_process=completed,
        )


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
        description=(
            "Run a FlashDreams application to a file, browser, or native window."
        ),
        epilog=(
            f"Installed applications: {installed}. Arguments after -- go to the "
            "application, so `flashdreams-run-v2 SLUG -- --help` describes it."
        ),
    )
    parser.add_argument("slug", help="Application to run.")
    parser.add_argument(
        "--preload-application",
        action="store_true",
        help="Initialize the application and validate one runtime step.",
    )
    parser.add_argument(
        "--skip-preload-validation",
        action="store_true",
        help="Initialize a preloaded application without validating a runtime step.",
    )
    parser.add_argument(
        "--timeout",
        type=_parse_timeout_seconds,
        default=None,
        metavar=f"{{SECONDS,{Unbound.unbound.value}}}",
        help=(
            "Stop the application after this many seconds. "
            f"{Unbound.unbound.value} means no time-limit. Remaining time is "
            "what a replacement session receives."
        ),
    )
    parser.add_argument(
        "--total-model-steps",
        type=_parse_total_model_steps,
        default=None,
        metavar=f"{{N,{Unbound.unbound.value}}}",
        help=(
            "Maximum model steps for the application run. "
            f"{Unbound.unbound.value} means no steps limit. Remaining "
            "steps are what a replacement session receives."
        ),
    )
    add_client_window_arguments(parser)
    _add_session_arguments(parser)
    return parser


def _report(message: str | None) -> None:
    """Print what the mode has to say about the run, if it has anything."""
    if message is not None:
        print(message, flush=True)


def _parse_timeout_seconds(value: str) -> float | Unbound:
    """Parse ``--timeout`` as seconds, or ``unbound`` as no time-limit."""
    if value == Unbound.unbound.value:
        return Unbound.unbound
    try:
        seconds = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"--timeout must be a finite number greater than zero, or "
            f"{Unbound.unbound.value}."
        ) from error
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError(
            f"--timeout must be a finite number greater than zero, or "
            f"{Unbound.unbound.value}."
        )
    return seconds


def _parse_total_model_steps(value: str) -> int | Unbound:
    """Parse ``--total-model-steps`` as a count, or ``unbound`` as no steps limit."""
    if value == Unbound.unbound.value:
        return Unbound.unbound
    try:
        steps = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"--total-model-steps must be a non-negative integer, or "
            f"{Unbound.unbound.value}."
        ) from error
    if steps < 0:
        raise argparse.ArgumentTypeError(
            f"--total-model-steps must be a non-negative integer, or "
            f"{Unbound.unbound.value}."
        )
    return steps


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
    parser.add_argument(
        "--backpressure-mode",
        type=BackpressureMode,
        choices=tuple(BackpressureMode),
        default=None,
        metavar="{" + ",".join(mode.value for mode in BackpressureMode) + "}",
        help="How the model thread handles a full presentation queue.",
    )
    parser.add_argument(
        "--presentation-mode",
        type=PresentationMode,
        choices=tuple(PresentationMode),
        default=None,
        metavar="{" + ",".join(mode.value for mode in PresentationMode) + "}",
        help=(
            "Whether the UI runs continuously or only for newly selected model frames."
        ),
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
            ("backpressure_mode", parsed_args.backpressure_mode),
            ("presentation_mode", parsed_args.presentation_mode),
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
