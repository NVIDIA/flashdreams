# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Application lifecycle runner for the v2 runtime."""

import logging
import sys
from collections.abc import Sequence

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.api_v2.output_sink import OutputSink
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.session_runner import run_session

_LOGGER = logging.getLogger(__name__)
"""Logger for an application or window that could not be closed."""


class ApplicationRunner:
    """Initialize one application and run its requested sessions."""

    def __init__(
        self,
        application: IApplication,
        client_window: IClientWindow,
        *,
        metrics_output_sink: OutputSink | None = None,
    ) -> None:
        """
        Args:
            application: Long-lived application that creates the session.
            client_window: Window that supplies input and presents generated output.
            metrics_output_sink: Optional sink for model-step metrics. It is
                opened and closed once for each session.
        """
        self._application = application
        self._client_window = client_window
        self._metrics_output_sink = metrics_output_sink

    def run(
        self, session_desc: SessionDesc, commandline_args: Sequence[str] = ()
    ) -> None:
        """Initialize the application and run sessions until the window exits.

        A new-session request closes the current session and creates its
        replacement from the description returned by ``run_session``. A close
        request or a completed session returns no replacement and ends the run.

        The application is closed before this method returns or raises.

        The runner closes the window when the first session never starts or a
        replacement cannot be created. ``run_session`` otherwise owns it, and
        may leave it open only for a successful replacement handoff.

        Args:
            session_desc: Output shape and timing requested for the session.
            commandline_args: Arguments owned and parsed by the application.
        """
        session_run_started = False
        window_needs_close = True
        try:
            self._application.init(commandline_args)
            next_session_desc: SessionDesc | None = session_desc
            while next_session_desc is not None:
                session = self._application.create_session(next_session_desc)
                session_run_started = True
                try:
                    next_session_desc = run_session(
                        session,
                        self._client_window,
                        metrics_output_sink=self._metrics_output_sink,
                    )
                except BaseException:
                    # ``run_session`` closes the window on every failure.
                    window_needs_close = False
                    raise
                window_needs_close = next_session_desc is not None
        finally:
            if window_needs_close:
                _close_client_window(self._client_window)
            if not session_run_started and self._metrics_output_sink is not None:
                _close_output_sink(self._metrics_output_sink)
            _close_application(
                self._application, run_failed=sys.exc_info()[0] is not None
            )


def _close_client_window(client_window: IClientWindow) -> None:
    """Close a window still owned by the application runner.

    This covers initial setup and the gap between sessions. The run has already
    failed by the time this is called, so a failure here is logged rather than
    raised over the top of it.
    """
    try:
        client_window.close()
    except Exception:
        _LOGGER.exception("The client window failed to close while stopping.")


def _close_output_sink(output_sink: OutputSink) -> None:
    """Close a metrics sink after a run that never reached it."""
    try:
        output_sink.close()
    except Exception:
        _LOGGER.exception(
            "The metrics output sink failed to close after a run that never started."
        )


def _close_application(application: IApplication, *, run_failed: bool) -> None:
    """Close an application, keeping its close from hiding an earlier failure.

    The same rule ``run_session`` follows for the session and its sinks:
    whatever failed first is what a run reports, and a failure while cleaning up
    after it is logged.

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
