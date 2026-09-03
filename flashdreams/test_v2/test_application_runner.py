# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the v2 application runner."""

import logging
from collections.abc import Sequence
from dataclasses import replace

import pytest
import torch
from numpy import uint64

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.api_v2.loop import IModelLoop, IUILoop
from flashdreams.api_v2.session import ISession
from flashdreams.runtime_v2.application_runner import ApplicationRunner
from flashdreams.runtime_v2.session_desc import PresentationMode, SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import CloseUserInputEvent
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu

_RUNNER_LOGGER = "flashdreams.runtime_v2.application_runner"


class _ModelLoop(IModelLoop["_Session"]):
    def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]:
        return [self.state.step(step_index, events)]

    def is_finished(self) -> bool:
        return self.state.is_finished()


class _ReplacementUILoop(IUILoop["_Session"]):
    def step(self, step_index: int, events: UserInputEvents) -> None:
        del step_index, events
        self.request_new_session(self.session_desc)

    def reset(self) -> None:
        return


class _Session(ISession):
    def __init__(
        self,
        session_desc: SessionDesc,
        calls: list[str],
        *,
        length: int | None = None,
        request_replacement: bool = False,
    ) -> None:
        """
        Args:
            session_desc: Description this session reports as resolved.
            calls: Shared log every fake records into.
            length: Steps to generate before reporting that it has finished, or
                ``None`` for a session that runs until its window ends it.
            request_replacement: Whether the UI requests one replacement session.
        """
        self._session_desc = session_desc
        self._calls = calls
        self._length = length
        self._request_replacement = request_replacement
        self._generated = 0

    def init(self) -> None:
        self._calls.append("session.init")
        if self._request_replacement:
            self.register_ui_loop(_ReplacementUILoop, state=self)
        self.register_model_loop(_ModelLoop, state=self)

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
        replace_first_session: bool = False,
    ) -> None:
        self._calls = calls
        self._fail_to_init = fail_to_init
        self._fail_to_close = fail_to_close
        self._session_length = session_length
        self._replace_first_session = replace_first_session
        self.requested_session_descs: list[SessionDesc] = []
        self.sessions: list[_Session] = []

    def init(self, commandline_args: Sequence[str]) -> None:
        self._calls.append(f"application.init({list(commandline_args)!r})")
        if self._fail_to_init:
            raise RuntimeError("application init failed")

    def create_session(self, session_desc: SessionDesc) -> ISession:
        self._calls.append("application.create_session")
        self.requested_session_descs.append(session_desc)
        session = _Session(
            session_desc,
            self._calls,
            length=self._session_length,
            request_replacement=self._replace_first_session and not self.sessions,
        )
        self.sessions.append(session)
        return session

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
                    CloseUserInputEvent(
                        timestamp=uint64(0),
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


class _ClosingAfterWritesWindow(_SilentWindow):
    """Close after receiving the expected model output."""

    def __init__(self, calls: list[str], expected_writes: int) -> None:
        super().__init__(calls)
        self._expected_writes = expected_writes

    def get_user_input_events(self) -> UserInputEvents:
        if len(self.results) == self._expected_writes:
            return _Window.get_user_input_events(self)
        return UserInputEvents([])


class _SecondSessionClosingWindow(_Window):
    """Stay open for one replacement, then close its session."""

    def __init__(self, calls: list[str]) -> None:
        super().__init__(calls)
        self._sessions_opened = 0

    def get_user_input_events(self) -> UserInputEvents:
        if self._sessions_opened < 2:
            return UserInputEvents([])
        return super().get_user_input_events()

    def open(self, session_desc: SessionDesc) -> None:
        super().open(session_desc)
        self._sessions_opened += 1


class _MetricsSink:
    """Record model results delivered independently of the client window."""

    def __init__(self, calls: list[str]) -> None:
        self._calls = calls
        self.results: list[StepResult] = []

    def open(self, session_desc: SessionDesc) -> None:
        del session_desc
        self._calls.append("metrics.open")

    def write(self, result: StepResult) -> None:
        self.results.append(result)
        self._calls.append(f"metrics.write({result.step_index})")

    def close(self) -> None:
        self._calls.append("metrics.close")


def _session_desc() -> SessionDesc:
    return SessionDesc(
        output_layout=VideoTensorLayout.bcthw,
        presentation_mode=PresentationMode.ON_DEMAND,
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


def test_application_runner_keeps_running_until_the_window_closes() -> None:
    """Completing model inference does not bypass the client lifecycle."""
    calls: list[str] = []
    window = _ClosingAfterWritesWindow(calls, 3)

    ApplicationRunner(_Application(calls, session_length=3), window).run(
        _session_desc()
    )

    # The UI ticks once while inference is NOT_STARTED, then presents the three
    # model frames on its following ticks.
    assert [result.step_index for result in window.results] == [1, 2, 3]
    assert calls[-3:] == ["window.close", "session.close", "application.close"]


def test_application_timeout_signals_the_session_and_closes_everything() -> None:
    calls: list[str] = []
    application = _Application(calls)

    ApplicationRunner(application, _SilentWindow(calls)).run(
        _session_desc(),
        timeout_seconds=0.02,
    )

    assert len(application.sessions) == 1
    assert application.sessions[0]._shutdown_event.is_set()
    assert calls[-3:] == ["window.close", "session.close", "application.close"]


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("nan"), float("inf")])
def test_application_timeout_must_be_positive_and_finite(timeout: float) -> None:
    calls: list[str] = []

    with pytest.raises(ValueError, match="finite and greater than zero"):
        ApplicationRunner(_Application(calls), _SilentWindow(calls)).run(
            _session_desc(),
            timeout_seconds=timeout,
        )

    assert calls == []


def test_application_runner_keeps_metrics_output_separate_from_the_window() -> None:
    calls: list[str] = []
    window = _ClosingAfterWritesWindow(calls, 2)
    metrics = _MetricsSink(calls)

    ApplicationRunner(
        _Application(calls, session_length=2),
        window,
        metrics_output_sink=metrics,
    ).run(_session_desc())

    assert [result.step_index for result in window.results] == [1, 2]
    assert [result.step_index for result in metrics.results] == [0, 1]
    assert calls.index("metrics.open") < calls.index("metrics.write(0)")
    assert calls.index("metrics.write(1)") < calls.index("metrics.close")


def test_application_runner_replaces_a_session_before_closing_the_window() -> None:
    calls: list[str] = []
    application = _Application(calls, replace_first_session=True)
    window = _SecondSessionClosingWindow(calls)
    metrics = _MetricsSink(calls)
    session_desc = _session_desc()

    ApplicationRunner(
        application,
        window,
        metrics_output_sink=metrics,
    ).run(session_desc)

    create_indexes = [
        index
        for index, call in enumerate(calls)
        if call == "application.create_session"
    ]
    close_indexes = [
        index for index, call in enumerate(calls) if call == "session.close"
    ]
    assert len(create_indexes) == 2
    assert len(close_indexes) == 2
    assert close_indexes[0] < create_indexes[1]
    assert calls.count("window.open") == 2
    assert calls.count("window.close") == 1
    assert calls.count("metrics.open") == 2
    assert calls.count("metrics.close") == 2
    assert application.requested_session_descs == [session_desc, session_desc]
    assert application.requested_session_descs[1] is session_desc
    assert calls.count("application.init([])") == 1
    assert calls.count("application.close") == 1


def test_application_runner_closes_a_preserved_window_if_replacement_fails() -> None:
    calls: list[str] = []

    class FailingReplacementApplication(_Application):
        def create_session(self, session_desc: SessionDesc) -> ISession:
            if calls.count("application.create_session") == 1:
                calls.append("application.create_session")
                raise RuntimeError("replacement failed")
            return super().create_session(session_desc)

    with pytest.raises(RuntimeError, match="replacement failed"):
        ApplicationRunner(
            FailingReplacementApplication(calls, replace_first_session=True),
            _SecondSessionClosingWindow(calls),
        ).run(_session_desc())

    assert calls.count("application.create_session") == 2
    assert calls.count("session.close") == 1
    assert calls.count("window.close") == 1
    assert calls[-1] == "application.close"


def test_replacement_stops_if_per_session_metrics_fail_to_close() -> None:
    calls: list[str] = []

    class FailingMetricsSink(_MetricsSink):
        def close(self) -> None:
            super().close()
            raise RuntimeError("metrics close failed")

    with pytest.raises(RuntimeError, match="metrics close failed"):
        ApplicationRunner(
            _Application(calls, replace_first_session=True),
            _SecondSessionClosingWindow(calls),
            metrics_output_sink=FailingMetricsSink(calls),
        ).run(_session_desc())

    assert calls.count("application.create_session") == 1
    assert calls.count("session.close") == 1
    assert calls.count("window.close") == 1
    assert calls[-1] == "application.close"


def test_setup_failure_closes_every_runner_owned_resource() -> None:
    calls: list[str] = []
    bad_trace_desc = replace(
        _session_desc(),
        metadata={"trace_chunk_lifecycle": True},
    )

    with pytest.raises(TypeError, match="trace_chunk_lifecycle_path"):
        ApplicationRunner(
            _Application(calls),
            _Window(calls),
            metrics_output_sink=_MetricsSink(calls),
        ).run(bad_trace_desc)

    assert calls == [
        "application.init([])",
        "application.create_session",
        "metrics.close",
        "window.close",
        "session.close",
        "application.close",
    ]


def test_application_runner_reports_the_run_rather_than_the_close(
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls: list[str] = []
    application = _Application(calls, fail_to_init=True, fail_to_close=True)

    with caplog.at_level(logging.ERROR, logger=_RUNNER_LOGGER):
        with pytest.raises(RuntimeError, match="application init failed"):
            ApplicationRunner(application, _Window(calls)).run(_session_desc())

    assert "application close failed" in caplog.text
