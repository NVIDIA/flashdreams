# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the v2 application runner."""

import logging
from collections.abc import Sequence

import pytest
import torch
from numpy import uint64

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.api_v2.session import ISession
from flashdreams.runtime_v2.application_runner import ApplicationRunner
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    CloseUserInputEventData,
    UserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu

_RUNNER_LOGGER = "flashdreams.runtime_v2.application_runner"


class _Session(ISession):
    def __init__(
        self, session_desc: SessionDesc, calls: list[str], *, length: int | None = None
    ) -> None:
        """
        Args:
            session_desc: Description this session reports as resolved.
            calls: Shared log every fake records into.
            length: Steps to generate before reporting that it has finished, or
                ``None`` for a session that runs until its window ends it.
        """
        self._session_desc = session_desc
        self._calls = calls
        self._length = length
        self._generated = 0

    def init(self) -> None:
        self._calls.append("session.init")

    @property
    def session_desc(self) -> SessionDesc:
        return self._session_desc

    def is_finished(self) -> bool:
        return self._length is not None and self._generated >= self._length

    def step(self, step_index: int, events: UserInputEvents) -> StepResult:
        del events
        self._calls.append(f"session.step({step_index})")
        self._generated += 1
        return StepResult(
            step_index=step_index,
            output=torch.zeros((1, 3, 1, 2, 2)),
            frame_count=1,
            output_layout=VideoTensorLayout.bcthw,
        )

    def close(self) -> None:
        self._calls.append("session.close")


class _Application(IApplication):
    def __init__(
        self,
        calls: list[str],
        *,
        fail_to_init: bool = False,
        fail_to_close: bool = False,
        session_length: int | None = None,
    ) -> None:
        self._calls = calls
        self._fail_to_init = fail_to_init
        self._fail_to_close = fail_to_close
        self._session_length = session_length

    def init(self, commandline_args: Sequence[str]) -> None:
        self._calls.append(f"application.init({list(commandline_args)!r})")
        if self._fail_to_init:
            raise RuntimeError("application init failed")

    def create_session(self, session_desc: SessionDesc) -> ISession:
        self._calls.append("application.create_session")
        return _Session(session_desc, self._calls, length=self._session_length)

    def close(self) -> None:
        self._calls.append("application.close")
        if self._fail_to_close:
            raise RuntimeError("application close failed")


class _Window(IClientWindow):
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls
        self.results: list[StepResult] = []
        self._reported_close = False

    def get_user_input_events(self) -> UserInputEvents:
        if not self._reported_close:
            self._reported_close = True
            return UserInputEvents(
                [
                    UserInputEvent(
                        timestamp=uint64(0),
                        event_data=CloseUserInputEventData(),
                    )
                ]
            )
        return UserInputEvents([])

    def open(self, session_desc: SessionDesc) -> None:
        del session_desc
        self._calls.append("window.open")

    def write(self, result: StepResult) -> None:
        self.results.append(result)
        self._calls.append(f"window.write({result.step_index})")

    def close(self) -> None:
        self._calls.append("window.close")


class _SilentWindow(_Window):
    """Report nothing, as a window writing a file does."""

    def get_user_input_events(self) -> UserInputEvents:
        return UserInputEvents([])


def _session_desc() -> SessionDesc:
    return SessionDesc(
        output_layout=VideoTensorLayout.bcthw,
        frames_per_second_for_ui=100,
        frames_per_second_for_step=30,
        video_width=2,
        video_height=2,
    )


def test_application_runner_drives_complete_lifecycle() -> None:
    calls: list[str] = []
    application = _Application(calls)
    window = _Window(calls)

    ApplicationRunner(application, window).run(_session_desc(), ["--model-option"])

    assert window.results == []
    assert calls[0:3] == [
        "application.init(['--model-option'])",
        "application.create_session",
        "session.init",
    ]
    assert calls[-3:] == ["window.close", "session.close", "application.close"]


def test_application_runner_closes_both_when_the_run_never_starts() -> None:
    """The window is closed by the loop, which a failure here never reaches, and
    a window may already be serving a client by then."""
    calls: list[str] = []
    application = _Application(calls, fail_to_init=True)

    with pytest.raises(RuntimeError, match="application init failed"):
        ApplicationRunner(application, _Window(calls)).run(_session_desc())

    assert calls == ["application.init([])", "window.close", "application.close"]


def test_application_runner_ends_a_run_a_window_cannot_end() -> None:
    """A window with no client never reports a close, so the session ends it."""
    calls: list[str] = []
    window = _SilentWindow(calls)

    ApplicationRunner(_Application(calls, session_length=3), window).run(
        _session_desc()
    )

    assert [result.step_index for result in window.results] == [0, 1, 2]
    assert calls[-3:] == ["window.close", "session.close", "application.close"]


def test_application_runner_reports_the_run_rather_than_the_close(
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls: list[str] = []
    application = _Application(calls, fail_to_init=True, fail_to_close=True)

    with caplog.at_level(logging.ERROR, logger=_RUNNER_LOGGER):
        with pytest.raises(RuntimeError, match="application init failed"):
            ApplicationRunner(application, _Window(calls)).run(_session_desc())

    assert "application close failed" in caplog.text
