# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU test for the batch loop that runs an application into a file window."""

from collections.abc import Sequence

import pytest
import torch

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.api_v2.session import ISession
from flashdreams.runtime_v2.batch_runner import run_batch
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu


class FakeSession(ISession):
    """Emit one blank frame per step, recording the calls the loop makes."""

    def __init__(
        self,
        session_desc: SessionDesc,
        calls: list[str],
        *,
        fail_to_init: bool = False,
        fail_at: int | None = None,
    ) -> None:
        """
        Args:
            session_desc: Description this session reports as resolved.
            calls: Shared log, also written by the application and the window.
            fail_to_init: Whether :meth:`init` raises.
            fail_at: Step index to raise at, for exercising cleanup on failure.
        """
        self._session_desc = session_desc
        self._calls = calls
        self._fail_to_init = fail_to_init
        self._fail_at = fail_at
        self.observed_events: list[UserInputEvents] = []

    def init(self) -> None:
        self._calls.append("session.init")
        if self._fail_to_init:
            raise RuntimeError("session init failed")

    @property
    def session_desc(self) -> SessionDesc:
        return self._session_desc

    def step(self, step_index: int, events: UserInputEvents) -> StepResult:
        self._calls.append(f"session.step({step_index})")
        self.observed_events.append(events)
        if step_index == self._fail_at:
            raise RuntimeError("step failed")
        return StepResult(
            step_index=step_index,
            output=torch.zeros((1, 3, 1, 1, 1), dtype=torch.float32),
            frame_count=1,
            output_layout=self._session_desc.output_layout,
        )

    def close(self) -> None:
        self._calls.append("session.close")


class FakeApplication(IApplication):
    """Create fake sessions, recording its own lifetime calls."""

    def __init__(
        self,
        *,
        fail_to_init: bool = False,
        reject: bool = False,
        session_fails_to_init: bool = False,
        session_fails_at: int | None = None,
    ) -> None:
        """
        Args:
            fail_to_init: Whether :meth:`init` raises, as an application that
                cannot load what its sessions share does.
            reject: Whether :meth:`create_session` rejects the description.
            session_fails_to_init: Whether the sessions it creates fail to init.
            session_fails_at: Step index the sessions it creates raise at.
        """
        self._fail_to_init = fail_to_init
        self._reject = reject
        self._session_fails_to_init = session_fails_to_init
        self._session_fails_at = session_fails_at
        self.calls: list[str] = []
        self.commandline_args: Sequence[str] | None = None
        self.sessions: list[FakeSession] = []

    def init(self, commandline_args: Sequence[str]) -> None:
        self.calls.append("app.init")
        self.commandline_args = commandline_args
        if self._fail_to_init:
            raise RuntimeError("app init failed")

    def create_session(self, session_desc: SessionDesc) -> ISession:
        self.calls.append("app.create_session")
        if self._reject:
            raise ValueError("cannot honour that description")
        session = FakeSession(
            session_desc,
            self.calls,
            fail_to_init=self._session_fails_to_init,
            fail_at=self._session_fails_at,
        )
        self.sessions.append(session)
        return session

    def close(self) -> None:
        self.calls.append("app.close")


class RecordingClientWindow(IClientWindow):
    """Record what a batch run writes, as a file window would encode it."""

    def __init__(
        self,
        calls: list[str],
        *,
        fail_to_open: bool = False,
        fail_to_close: bool = False,
    ) -> None:
        """
        Args:
            calls: Shared log, so window calls can be ordered against the rest.
            fail_to_open: Whether :meth:`open` raises.
            fail_to_close: Whether :meth:`close` raises, as a file window whose
                encoder could not finish the file does.
        """
        self._calls = calls
        self._fail_to_open = fail_to_open
        self._fail_to_close = fail_to_close
        self.session_descs: list[SessionDesc] = []
        self.results: list[StepResult] = []

    def get_user_input_events(self) -> UserInputEvents:
        self._calls.append("window.get_user_input_events")
        return UserInputEvents([])

    def open(self, session_desc: SessionDesc) -> None:
        self._calls.append("window.open")
        if self._fail_to_open:
            raise RuntimeError("open failed")
        self.session_descs.append(session_desc)

    def write(self, result: StepResult) -> None:
        self._calls.append("window.write")
        self.results.append(result)

    def close(self) -> None:
        self._calls.append("window.close")
        if self._fail_to_close:
            raise RuntimeError("close failed")


## Helpers


def _session_desc() -> SessionDesc:
    return SessionDesc(
        output_layout=VideoTensorLayout.bcthw,
        frames_per_second_for_ui=60,
        frames_per_second_for_step=30,
        video_width=1,
        video_height=1,
    )


## Tests


def test_run_generates_and_writes_each_step_in_turn() -> None:
    app = FakeApplication()
    window = RecordingClientWindow(app.calls)

    run_batch(app, window, _session_desc(), steps=2)

    assert app.calls == [
        "app.init",
        "app.create_session",
        "session.init",
        "window.open",
        "session.step(0)",
        "window.write",
        "session.step(1)",
        "window.write",
        "window.close",
        "session.close",
        "app.close",
    ]
    assert window.session_descs == [_session_desc()]
    assert [result.step_index for result in window.results] == [0, 1]


def test_run_never_reads_the_windows_input() -> None:
    # A batch run has nobody to take input from, so every step is handed the
    # same empty batch and the window's input side is left alone.
    app = FakeApplication()
    window = RecordingClientWindow(app.calls)

    run_batch(app, window, _session_desc(), steps=2)

    assert "window.get_user_input_events" not in app.calls
    assert [events.get_events() for events in app.sessions[0].observed_events] == [
        [],
        [],
    ]


def test_run_passes_the_commandline_arguments_to_the_application() -> None:
    app = FakeApplication()

    run_batch(
        app,
        RecordingClientWindow(app.calls),
        _session_desc(),
        steps=1,
        commandline_args=["--seconds", "2"],
    )

    assert list(app.commandline_args or []) == ["--seconds", "2"]


def test_a_run_of_no_steps_still_opens_and_closes() -> None:
    app = FakeApplication()
    window = RecordingClientWindow(app.calls)

    run_batch(app, window, _session_desc(), steps=0)

    assert app.calls == [
        "app.init",
        "app.create_session",
        "session.init",
        "window.open",
        "window.close",
        "session.close",
        "app.close",
    ]
    assert window.results == []


def test_run_rejects_a_negative_step_count() -> None:
    app = FakeApplication()

    with pytest.raises(ValueError, match="steps"):
        run_batch(app, RecordingClientWindow(app.calls), _session_desc(), steps=-1)

    assert app.calls == []


def test_a_failed_step_still_closes_the_window_and_the_session() -> None:
    # Closing the window is what finishes the file, so a run that failed part
    # way through still leaves what it managed to generate.
    app = FakeApplication(session_fails_at=1)
    window = RecordingClientWindow(app.calls)

    with pytest.raises(RuntimeError, match="step failed"):
        run_batch(app, window, _session_desc(), steps=3)

    assert len(window.results) == 1
    assert app.calls[-3:] == ["window.close", "session.close", "app.close"]


def test_a_window_that_fails_to_close_reports_it() -> None:
    # For a file window this is the encode failing to finish, so the run cannot
    # be called a success: the file it was writing is unusable.
    app = FakeApplication()
    window = RecordingClientWindow(app.calls, fail_to_close=True)

    with pytest.raises(RuntimeError, match="close failed"):
        run_batch(app, window, _session_desc(), steps=1)

    assert app.calls[-2:] == ["session.close", "app.close"]


def test_a_failed_run_reports_what_failed_it_rather_than_the_close(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Both the step and the close fail. The step is the one that explains the
    # run, so it is raised and the close is only logged.
    app = FakeApplication(session_fails_at=0)
    window = RecordingClientWindow(app.calls, fail_to_close=True)

    with pytest.raises(RuntimeError, match="step failed"):
        run_batch(app, window, _session_desc(), steps=1)

    assert "close failed" in caplog.text
    assert app.calls[-3:] == ["window.close", "session.close", "app.close"]


def test_a_window_that_fails_to_open_is_still_closed() -> None:
    app = FakeApplication()
    window = RecordingClientWindow(app.calls, fail_to_open=True)

    with pytest.raises(RuntimeError, match="open failed"):
        run_batch(app, window, _session_desc(), steps=1)

    # A partly opened window still holds whatever it acquired.
    assert app.calls == [
        "app.init",
        "app.create_session",
        "session.init",
        "window.open",
        "window.close",
        "session.close",
        "app.close",
    ]


def test_a_session_that_fails_to_init_is_still_closed() -> None:
    app = FakeApplication(session_fails_to_init=True)
    window = RecordingClientWindow(app.calls)

    with pytest.raises(RuntimeError, match="session init failed"):
        run_batch(app, window, _session_desc(), steps=1)

    # Nothing was opened, because there was no run to open it for.
    assert app.calls == [
        "app.init",
        "app.create_session",
        "session.init",
        "session.close",
        "app.close",
    ]


def test_an_application_that_fails_to_init_is_still_closed() -> None:
    app = FakeApplication(fail_to_init=True)

    with pytest.raises(RuntimeError, match="app init failed"):
        run_batch(app, RecordingClientWindow(app.calls), _session_desc(), steps=1)

    assert app.calls == ["app.init", "app.close"]


def test_run_reports_a_description_the_application_rejects() -> None:
    app = FakeApplication(reject=True)
    window = RecordingClientWindow(app.calls)

    with pytest.raises(ValueError, match="cannot honour"):
        run_batch(app, window, _session_desc(), steps=1)

    assert window.session_descs == []
    assert app.calls == ["app.init", "app.create_session", "app.close"]
