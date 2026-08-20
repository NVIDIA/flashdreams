# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU test for the batch loop that writes a session's results to a sink."""

import pytest
import torch

from flashdreams.api_v2.output_sink import OutputSink
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
            calls: Shared log, also written by the sink.
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


class RecordingOutputSink(OutputSink):
    """Record what a batch run writes, as a file sink would encode it."""

    def __init__(
        self,
        calls: list[str],
        *,
        fail_to_open: bool = False,
        fail_to_close: bool = False,
    ) -> None:
        """
        Args:
            calls: Shared log, so sink calls can be ordered against the rest.
            fail_to_open: Whether :meth:`open` raises.
            fail_to_close: Whether :meth:`close` raises, as a file sink whose
                encoder could not finish the file does.
        """
        self._calls = calls
        self._fail_to_open = fail_to_open
        self._fail_to_close = fail_to_close
        self.session_descs: list[SessionDesc] = []
        self.results: list[StepResult] = []

    def open(self, session_desc: SessionDesc) -> None:
        self._calls.append("output.open")
        if self._fail_to_open:
            raise RuntimeError("open failed")
        self.session_descs.append(session_desc)

    def write(self, result: StepResult) -> None:
        self._calls.append("output.write")
        self.results.append(result)

    def close(self) -> None:
        self._calls.append("output.close")
        if self._fail_to_close:
            raise RuntimeError("close failed")


## Helpers


def _session_desc() -> SessionDesc:
    return SessionDesc(
        output_layout=VideoTensorLayout.bcthw,
        frames_per_second_for_ui=60,
        frames_per_second_for_step=30,
        video_width=2,
        video_height=2,
    )


def _session(
    calls: list[str], *, fail_to_init: bool = False, fail_at: int | None = None
) -> FakeSession:
    return FakeSession(
        _session_desc(), calls, fail_to_init=fail_to_init, fail_at=fail_at
    )


## Tests


def test_run_generates_and_writes_each_step_in_turn() -> None:
    calls: list[str] = []
    session = _session(calls)
    output = RecordingOutputSink(calls)

    run_batch(session, output, steps=2)

    assert calls == [
        "session.init",
        "output.open",
        "session.step(0)",
        "output.write",
        "session.step(1)",
        "output.write",
        "output.close",
        "session.close",
    ]
    assert output.session_descs == [_session_desc()]
    assert [result.step_index for result in output.results] == [0, 1]


def test_run_hands_every_step_an_empty_batch() -> None:
    # A batch run has nobody to take input from, so a session sees no events at
    # all rather than seeing whatever the previous run left behind.
    calls: list[str] = []
    session = _session(calls)

    run_batch(session, RecordingOutputSink(calls), steps=2)

    assert [events.get_events() for events in session.observed_events] == [[], []]


def test_a_run_of_no_steps_still_opens_and_closes() -> None:
    calls: list[str] = []
    output = RecordingOutputSink(calls)

    run_batch(_session(calls), output, steps=0)

    assert calls == ["session.init", "output.open", "output.close", "session.close"]
    assert output.results == []


def test_run_rejects_a_negative_step_count() -> None:
    calls: list[str] = []

    with pytest.raises(ValueError, match="steps"):
        run_batch(_session(calls), RecordingOutputSink(calls), steps=-1)

    assert calls == []


def test_a_failed_step_still_closes_the_sink_and_the_session() -> None:
    # Closing the sink is what finishes the file, so a run that failed part way
    # through still leaves what it managed to generate.
    calls: list[str] = []
    output = RecordingOutputSink(calls)

    with pytest.raises(RuntimeError, match="step failed"):
        run_batch(_session(calls, fail_at=1), output, steps=3)

    assert len(output.results) == 1
    assert calls[-2:] == ["output.close", "session.close"]


def test_a_sink_that_fails_to_close_reports_it() -> None:
    # For a file sink this is the encode failing to finish, so the run cannot be
    # called a success: the file it was writing is unusable.
    calls: list[str] = []

    with pytest.raises(RuntimeError, match="close failed"):
        run_batch(
            _session(calls), RecordingOutputSink(calls, fail_to_close=True), steps=1
        )

    assert calls[-1] == "session.close"


def test_a_failed_run_reports_what_failed_it_rather_than_the_close(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Both the step and the close fail. The step is the one that explains the
    # run, so it is raised and the close is only logged.
    calls: list[str] = []

    with pytest.raises(RuntimeError, match="step failed"):
        run_batch(
            _session(calls, fail_at=0),
            RecordingOutputSink(calls, fail_to_close=True),
            steps=1,
        )

    assert "close failed" in caplog.text
    assert calls[-2:] == ["output.close", "session.close"]


def test_a_sink_that_fails_to_open_is_still_closed() -> None:
    # A partly opened sink still holds whatever it acquired.
    calls: list[str] = []

    with pytest.raises(RuntimeError, match="open failed"):
        run_batch(
            _session(calls), RecordingOutputSink(calls, fail_to_open=True), steps=1
        )

    assert calls == ["session.init", "output.open", "output.close", "session.close"]


def test_a_session_that_fails_to_init_is_still_closed() -> None:
    # Nothing was opened, because there was no run to open it for.
    calls: list[str] = []

    with pytest.raises(RuntimeError, match="session init failed"):
        run_batch(
            _session(calls, fail_to_init=True), RecordingOutputSink(calls), steps=1
        )

    assert calls == ["session.init", "session.close"]
