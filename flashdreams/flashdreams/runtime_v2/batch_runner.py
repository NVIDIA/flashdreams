# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batch loop generating a fixed number of steps and writing each one out."""

import logging
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.user_input_events import UserInputEvents

_LOGGER = logging.getLogger(__name__)
"""Logger for a close that failed after the run had already failed."""


def run_batch(
    application: IApplication,
    window: IClientWindow,
    session_desc: SessionDesc,
    *,
    steps: int,
    commandline_args: Sequence[str] = (),
) -> None:
    """Generate ``steps`` results from an application and write each one out.

    The loop for output that is a file: generate a step, write it, repeat. One
    thread, no input, and no pacing. A file has no client to take input from or
    to keep up with, so frames are generated as fast as they can be, and the
    file plays back at the rate its session declared.

    Each kind of output has its own loop. An interactive window brings the loop
    its platform expects, a WebRTC event loop or an OS message pump, and what
    those have in common with this one is the session they step rather than the
    way they step it.
    :func:`flashdreams.runtime_v2.session_runner.run_session` is the polling
    loop the interactive path uses today.

    The application, its session and the window are all closed before this
    returns, whether the run finished or failed part way through. One call is
    therefore one load: a caller wanting several files out of one loaded
    application needs a loop that keeps it, which nothing needs yet.

    A run that fails raises what failed it, rather than anything that then went
    wrong closing up, since the first failure is the one that explains the
    rest. A close that fails on its own is raised: for a file window that is
    the encode failing to finish, which leaves the file unusable.

    Args:
        application: Application to run. Initialized here, not by the caller.
        window: Window to write results to. Its input is never read, since a
            batch run has nobody to take input from.
        session_desc: Session to ask the application for.
        steps: Number of steps to generate. A batch run is bounded up front,
            since a window that reports no input can never report a close.
        commandline_args: Application-specific arguments.

    Raises:
        ValueError: ``steps`` is negative, or the application cannot honour
            ``session_desc``.
    """
    if steps < 0:
        raise ValueError(f"steps must be >= 0, got {steps}.")

    # No input source, so every step is handed the same empty batch.
    no_events = UserInputEvents([])
    # Nested so each one is closed before whatever created it, and so an outer
    # one is closed even when what it holds never started.
    with _closing(application.close, "application"):
        application.init(commandline_args)
        session = application.create_session(session_desc)
        with _closing(session.close, "session"):
            session.init()
            with _closing(window.close, "window"):
                window.open(session.session_desc)
                for step_index in range(steps):
                    window.write(session.step(step_index, no_events))


@contextmanager
def _closing(close: Callable[[], None], what: str) -> Iterator[None]:
    """Run a block, then close what it was using.

    Closing is where a file window finishes the encode, so a block that raised
    still leaves what it managed to generate, and one that did not is told when
    finishing the file failed.

    A close that fails after the block has already failed is logged rather than
    raised: the caller wants to see what ended the run, not what went wrong
    tidying up after it.

    Args:
        close: Called on the way out, whichever way that is.
        what: Name for the thing being closed, for the log line.
    """
    try:
        yield
    except BaseException:
        try:
            close()
        except Exception:
            _LOGGER.exception("Closing the %s failed after a failed run.", what)
        raise
    close()
