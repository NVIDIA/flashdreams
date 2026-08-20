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
"""Logger for an application or window that could not be closed."""


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
        self, session_desc: SessionDesc, commandline_args: Sequence[str] = ()
    ) -> None:
        """Initialize the application, create one session, and run it.

        The run ends when the window reports a close or the session reports that
        it has finished.

        The application is closed before this method returns or raises.

        The window is closed too when the run never starts, since ``run_session``
        is what otherwise owns it, and a window may already be serving a client
        before the application has loaded anything.

        Args:
            session_desc: Output shape and timing requested for the session.
            commandline_args: Arguments owned and parsed by the application.
        """
        run_started = False
        try:
            self._application.init(commandline_args)
            session = self._application.create_session(session_desc)
            run_started = True
            run_session(session, self._client_window)
        finally:
            if not run_started:
                _close_client_window(self._client_window)
            _close_application(
                self._application, run_failed=sys.exc_info()[0] is not None
            )


def _close_client_window(client_window: IClientWindow) -> None:
    """Close a window the run never reached, so what it was serving goes with it.

    The run has already failed by the time this is called, so a failure here is
    logged rather than raised over the top of it.
    """
    try:
        client_window.close()
    except Exception:
        _LOGGER.exception(
            "The client window failed to close after a run that never started."
        )


def _close_application(application: IApplication, *, run_failed: bool) -> None:
    """Close an application, keeping its close from hiding an earlier failure.

    This is ``session_runner._close_session`` for the application: whatever
    failed first is what a run reports, and a failure while cleaning up after it
    is logged.

    Args:
        application: Application to close.
        run_failed: Whether something has already failed the run. When it has,
            a failing close is logged rather than raised over the top of it.

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
