# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the v2 application runner."""

import logging
import threading
from collections.abc import Sequence

import pytest
import torch
from numpy import uint64

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.api_v2.session import ISession
from flashdreams.runtime_v2.application_runner import ApplicationRunner
from flashdreams.runtime_v2.session_desc import SessionDesc, SessionDescRequest
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    CloseUserInputEventData,
    NewSessionUserInputEventData,
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
        self.created_session_descs: list[SessionDesc] = []

    def init(self, commandline_args: Sequence[str]) -> None:
        self._calls.append(f"application.init({list(commandline_args)!r})")
        if self._fail_to_init:
            raise RuntimeError("application init failed")

    def create_session(self, session_desc: SessionDesc) -> ISession:
        self._calls.append("application.create_session")
        self.created_session_descs.append(session_desc)
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


class _ScriptedWindow(_Window):
    """Report one scripted event batch each time the runner polls."""

    def __init__(self, calls: list[str], events: list[UserInputEvents]) -> None:
        super().__init__(calls)
        self._events = list(events)

    def get_user_input_events(self) -> UserInputEvents:
        if self._events:
            return self._events.pop(0)
        return UserInputEvents([])


class _ServingWindow(_Window):
    """Request two sessions, then interrupt the persistent runner."""

    def __init__(self, calls: list[str]) -> None:
        super().__init__(calls)
        self._prompts = ["A cat surfing", "A dog snowboarding"]

    def get_user_input_events(self) -> UserInputEvents:
        if threading.current_thread() is not threading.main_thread():
            return UserInputEvents([])
        completed_sessions = self._calls.count("session.close")
        if completed_sessions == len(self._prompts):
            raise KeyboardInterrupt
        return UserInputEvents(
            [
                UserInputEvent(
                    timestamp=uint64(completed_sessions),
                    event_data=NewSessionUserInputEventData(
                        metadata={"prompt": self._prompts[completed_sessions]}
                    ),
                )
            ]
        )


def _session_desc_request() -> SessionDescRequest:
    return SessionDescRequest(
        output_layout=VideoTensorLayout.bcthw,
        frames_per_second_for_ui=100,
        frames_per_second_for_step=30,
        video_width=2,
        video_height=2,
    )


def test_application_runner_keeps_the_application_open_for_another_session() -> None:
    calls: list[str] = []
    application = _Application(calls, session_length=1)
    runner = ApplicationRunner(application)
    first_window = _SilentWindow(calls)
    second_window = _SilentWindow(calls)

    runner.init(["--model-option"])
    runner.run(_session_desc_request(), first_window)
    runner.run(_session_desc_request(), second_window)

    assert [result.step_index for result in first_window.results] == [0]
    assert [result.step_index for result in second_window.results] == [0]
    assert calls.count("application.create_session") == 2
    assert calls.count("application.close") == 0
    assert calls[0:3] == [
        "application.init(['--model-option'])",
        "application.create_session",
        "session.init",
    ]
    runner.close()
    assert calls[-1] == "application.close"


def test_application_runner_replaces_a_session_from_window_metadata() -> None:
    calls: list[str] = []
    application = _Application(calls)
    runner = ApplicationRunner(application)
    new_session = UserInputEvents(
        [
            UserInputEvent(
                timestamp=uint64(0),
                event_data=NewSessionUserInputEventData(
                    metadata={"prompt": "A dog snowboarding"}
                ),
            )
        ]
    )
    close = UserInputEvents(
        [
            UserInputEvent(
                timestamp=uint64(1),
                event_data=CloseUserInputEventData(),
            )
        ]
    )
    window = _ScriptedWindow(calls, [new_session, close])

    runner.init()
    runner.run(_session_desc_request(), window)
    runner.close()

    assert len(application.created_session_descs) == 2
    assert application.created_session_descs[0].metadata == {}
    assert application.created_session_descs[1].metadata == {
        "prompt": "A dog snowboarding"
    }
    assert calls.count("session.close") == 2
    assert calls.count("window.open") == 2
    assert calls.count("window.close") == 1
    first_creation = calls.index("application.create_session")
    second_creation = calls.index("application.create_session", first_creation + 1)
    assert calls.index("session.close") < second_creation


def test_application_runner_serves_sessions_until_it_is_interrupted() -> None:
    calls: list[str] = []
    application = _Application(calls, session_length=1)
    runner = ApplicationRunner(application)
    window = _ServingWindow(calls)

    runner.init()
    with pytest.raises(KeyboardInterrupt):
        runner.run(_session_desc_request(), window, serve_sessions=True)

    assert [desc.metadata for desc in application.created_session_descs] == [
        {"prompt": "A cat surfing"},
        {"prompt": "A dog snowboarding"},
    ]
    assert calls.count("session.close") == 2
    assert calls.count("window.open") == 3
    assert calls.count("window.close") == 1
    assert calls.count("application.close") == 0
    runner.close()
    assert calls[-1] == "application.close"


def test_application_runner_closes_the_window_when_a_session_cannot_start() -> None:
    calls: list[str] = []
    runner = ApplicationRunner(_Application(calls))
    window = _Window(calls)

    with pytest.raises(RuntimeError, match=r"init\(\) must run first"):
        runner.run(_session_desc_request(), window)

    assert calls == ["window.close"]


def test_serving_closes_the_window_when_the_session_request_is_invalid() -> None:
    calls: list[str] = []
    runner = ApplicationRunner(_Application(calls))
    window = _Window(calls)
    runner.init()

    with pytest.raises(ValueError, match="frames_per_second_for_ui"):
        runner.run(
            SessionDescRequest(frames_per_second_for_ui=0),
            window,
            serve_sessions=True,
        )

    assert calls == ["application.init([])", "window.close"]
    runner.close()


def test_application_runner_rejects_a_second_initialization() -> None:
    calls: list[str] = []
    runner = ApplicationRunner(_Application(calls))

    runner.init()
    with pytest.raises(RuntimeError, match="already initialized"):
        runner.init()
    runner.close()

    assert calls == ["application.init([])", "application.close"]


def test_application_runner_reports_the_run_rather_than_the_close(
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls: list[str] = []
    application = _Application(calls, fail_to_init=True, fail_to_close=True)
    runner = ApplicationRunner(application)

    with caplog.at_level(logging.ERROR, logger=_RUNNER_LOGGER):
        with pytest.raises(RuntimeError, match="application init failed"):
            try:
                runner.init()
            finally:
                runner.close()

    assert "application close failed" in caplog.text
