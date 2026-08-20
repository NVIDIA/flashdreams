# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batch loop generating a fixed number of steps and writing each to a sink."""

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from flashdreams.api_v2.output_sink import OutputSink
from flashdreams.api_v2.session import ISession
from flashdreams.runtime_v2.user_input_events import UserInputEvents

_LOGGER = logging.getLogger(__name__)
"""Logger for a close that failed after the run had already failed."""


def run_batch(session: ISession, output: OutputSink, *, steps: int) -> None:
    """Generate ``steps`` results from a session and write each one out.

    Generate a step, write it, repeat, on the model generation thread. Nothing
    paces the loop, so frames are generated as fast as they can be and the file
    plays back at the rate its session declared.

    Running a session as a batch rather than interactively is a way of driving
    it, not a kind of session: the same session produces the same results either
    way. Every step here is handed an empty batch of events, since nothing is
    reading input yet.
    :func:`flashdreams.runtime_v2.session_runner.run_session` is the interactive
    counterpart.

    The session and the sink are both closed before this returns, whether the
    run finished or failed part way through. The application the session came
    from belongs to the caller, so one loaded application can write several
    files.

    A run that fails raises what failed it, rather than anything that then went
    wrong closing up, since the first failure is the one that explains the
    rest. A close that fails on its own is raised: for a file that is the
    encode failing to finish, which leaves the file unusable.

    Args:
        session: Uninitialized session to drive.
        output: Sink to write results to.
        steps: Number of steps to generate. A batch run is bounded up front,
            since a sink has no way to ask for the run to end.

    Raises:
        ValueError: ``steps`` is negative.
    """
    if steps < 0:
        raise ValueError(f"steps must be >= 0, got {steps}.")

    # Nothing reads input here, so every step is handed the same empty batch.
    no_events = UserInputEvents([])
    # Nested so the sink is closed before the session it was opened for, and so
    # a session that failed to start is still closed.
    with _closing(session.close, "session"):
        session.init()
        with _closing(output.close, "output"):
            output.open(session.session_desc)
            for step_index in range(steps):
                output.write(session.step(step_index, no_events))


@contextmanager
def _closing(close: Callable[[], None], what: str) -> Iterator[None]:
    """Run a block, then close what it was using.

    Closing is where a file sink finishes the encode, so a block that raised
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
