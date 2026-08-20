# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Application lifecycle runner for the v2 runtime."""

import logging
import sys
from collections.abc import Sequence

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.session_runner import run_session

_LOGGER = logging.getLogger(__name__)
"""Logger for an application that could not be closed."""


class ApplicationRunner:
    """Create and run one application session against one client window."""

    def __init__(self, application: IApplication, client_window: IClientWindow) -> None:
        """
        Args:
            application: Long-lived application that creates the session.
            client_window: Window that supplies input and presents generated output.
        """
        self._application = application
        self._client_window = client_window

    def run(
        self,
        session_desc: SessionDesc,
        commandline_args: Sequence[str] = (),
        *,
        steps: int | None = None,
    ) -> None:
        """Initialize the application, create one session, and run it.

        The application is closed before this method returns or raises.

        Args:
            session_desc: Output shape and timing requested for the session.
            commandline_args: Arguments owned and parsed by the application.
            steps: Steps to generate, or ``None`` to run until the window
                reports a close. A window with a client on the other end reports
                one when that client goes away; a window writing a file never
                does, so a run like that needs a count to end it.
        """
        try:
            self._application.init(commandline_args)
            session = self._application.create_session(session_desc)
            run_session(session, self._client_window, steps=steps)
        finally:
            _close_application(
                self._application, run_failed=sys.exc_info()[0] is not None
            )


def _close_application(application: IApplication, *, run_failed: bool) -> None:
    """Close an application, keeping its close from hiding an earlier failure.

    This is ``session_runner._close_session`` for the application: whatever
    failed first is what a run reports, and a failure while cleaning up after it
    is logged.

    Args:
        application: Application to close.
        run_failed: Whether something has already failed the run. A close that
            fails is logged when it has, and raised when it has not, since then
            it is the only thing that went wrong.

    Raises:
        Whatever the application raises, when nothing has failed yet.
    """
    try:
        application.close()
    except Exception:
        if not run_failed:
            raise
        _LOGGER.exception(
            "The application failed to close after the run had already failed."
        )
