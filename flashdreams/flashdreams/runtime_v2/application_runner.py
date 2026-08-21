# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Application lifecycle runner for the v2 runtime."""

import logging
import sys
from collections.abc import Sequence

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.runtime_v2.session_desc import SessionDesc, SessionDescRequest
from flashdreams.runtime_v2.session_runner import run_session, wait_for_new_session

_LOGGER = logging.getLogger(__name__)
"""Logger for an application or window that could not be closed."""


class ApplicationRunner:
    """Hold one initialized application and run its sessions."""

    def __init__(self, application: IApplication) -> None:
        """
        Args:
            application: Long-lived application that creates sessions.
        """
        self._application = application
        self._initialized = False
        self._initialization_attempted = False
        self._closed = False

    def init(self, commandline_args: Sequence[str] = ()) -> None:
        """Initialize the application once.

        Args:
            commandline_args: Arguments owned and parsed by the application.

        Raises:
            RuntimeError: The application has already been initialized or closed.
        """
        if self._closed:
            raise RuntimeError("ApplicationRunner is closed.")
        if self._initialization_attempted:
            raise RuntimeError("ApplicationRunner is already initialized.")
        self._initialization_attempted = True
        self._application.init(commandline_args)
        self._initialized = True

    def run(
        self,
        session_desc_request: SessionDescRequest,
        client_window: IClientWindow,
    ) -> None:
        """Create sessions against ``client_window`` until the run ends.

        The window decides whether to start immediately with the resolved
        description or stay open and wait for a client request. A replacement
        description returned by the session loop starts another session after
        the current one has closed. A persistent window returns to waiting when
        a session finishes or its client disconnects.

        The session and window are closed before this method returns or raises.
        The application remains initialized, so callers can run another session
        without reloading its shared state. Call :meth:`close` when no further
        sessions are needed.

        Args:
            session_desc_request: Explicit overrides to apply to the
                application's initialized default description.
            client_window: Window that supplies input and presents generated output.
        """
        try:
            persistent_window = client_window.keeps_open_between_sessions
            current_session_desc = self._resolve_session_desc(session_desc_request)
            if persistent_window:
                client_window.open(current_session_desc)
                next_session_desc = wait_for_new_session(
                    client_window, current_session_desc
                )
            else:
                next_session_desc = current_session_desc

            while next_session_desc is not None:
                session = self._application.create_session(next_session_desc)
                current_session_desc = session.session_desc
                next_session_desc = run_session(
                    session,
                    client_window,
                    keep_window_open=persistent_window,
                )
                if persistent_window and next_session_desc is None:
                    next_session_desc = wait_for_new_session(
                        client_window, current_session_desc
                    )
        finally:
            _close_client_window(client_window)

    def _resolve_session_desc(
        self, session_desc_request: SessionDescRequest
    ) -> SessionDesc:
        """Resolve one request against the initialized application's default."""
        if self._closed:
            raise RuntimeError("ApplicationRunner is closed.")
        if not self._initialized:
            raise RuntimeError("ApplicationRunner.init() must run first.")
        default = self._application.default_session_desc() or SessionDesc()
        return session_desc_request.resolve(default)

    def close(self) -> None:
        """Release the application and the state it shares across sessions."""
        if self._closed:
            return
        self._closed = True
        _close_application(self._application, run_failed=sys.exc_info()[0] is not None)


def _close_client_window(client_window: IClientWindow) -> None:
    """Ensure a window closes without hiding the run's meaningful result.

    ``IClientWindow.close`` is idempotent. The session loop normally performs
    the first close on its I/O thread so file-finalization errors can fail the
    run. This is the application runner's fallback for setup failures,
    persistent windows, and interrupted ownership handoffs.
    """
    try:
        client_window.close()
    except Exception:
        _LOGGER.exception(
            "The client window failed to close while the runner was stopping."
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
