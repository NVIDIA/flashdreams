# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batch loop generating a fixed number of steps and writing each one out."""

from collections.abc import Sequence

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.user_input_events import UserInputEvents


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
    try:
        application.init(commandline_args)
        session = application.create_session(session_desc)
        try:
            session.init()
            try:
                window.open(session.session_desc)
                for step_index in range(steps):
                    window.write(session.step(step_index, no_events))
            finally:
                # Closing is where a file window finishes the encode, so a run
                # that raised still leaves what it managed to generate.
                window.close()
        finally:
            session.close()
    finally:
        # An application that failed part way through init still holds whatever
        # it managed to load.
        application.close()
