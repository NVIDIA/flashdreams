# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for writing one run's results to several sinks."""

import pytest
import torch

from flashdreams.runtime_v2.composite_output_sink import CompositeOutputSink
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu


class RecordingSink:
    """A sink that remembers what it was asked to do, and can refuse to close."""

    def __init__(self, *, close_error: Exception | None = None) -> None:
        """
        Args:
            close_error: Raised on close, for a sink that fails to finish.
        """
        self.opened: list[SessionDesc] = []
        self.written: list[StepResult] = []
        self.closed = 0
        self._close_error = close_error

    def open(self, session_desc: SessionDesc) -> None:
        self.opened.append(session_desc)

    def write(self, result: StepResult) -> None:
        self.written.append(result)

    def close(self) -> None:
        self.closed += 1
        if self._close_error is not None:
            raise self._close_error


def _session_desc() -> SessionDesc:
    return SessionDesc(
        output_layout=VideoTensorLayout.tchw,
        frames_per_second_for_ui=60,
        frames_per_second_for_step=16,
        video_width=128,
        video_height=64,
    )


def _result() -> StepResult:
    return StepResult(
        step_index=0,
        output=torch.zeros((4, 3, 64, 128)),
        frame_count=4,
        output_layout=VideoTensorLayout.tchw,
    )


def test_every_sink_is_given_the_same_session_and_the_same_results() -> None:
    first, second = RecordingSink(), RecordingSink()
    session_desc = _session_desc()
    result = _result()
    composite = CompositeOutputSink(first, second)

    composite.open(session_desc)
    composite.write(result)
    composite.close()

    for sink in (first, second):
        assert sink.opened == [session_desc]
        # The same object, so a sink keeping what it was given keeps what the
        # others were given.
        assert sink.written[0] is result
        assert sink.closed == 1


def test_a_sink_that_fails_to_close_does_not_leave_the_others_open() -> None:
    """For a file that is the difference between one unusable output and two."""
    failing = RecordingSink(close_error=RuntimeError("encode failed"))
    other = RecordingSink()
    composite = CompositeOutputSink(failing, other)
    composite.open(_session_desc())

    with pytest.raises(RuntimeError, match="encode failed"):
        composite.close()

    assert other.closed == 1


def test_the_first_failure_to_close_is_the_one_reported() -> None:
    composite = CompositeOutputSink(
        RecordingSink(close_error=RuntimeError("first")),
        RecordingSink(close_error=RuntimeError("second")),
    )

    with pytest.raises(RuntimeError, match="first"):
        composite.close()


def test_a_sink_that_cannot_take_a_result_ends_the_run() -> None:
    # Half of what a run was asked to write is not a run that succeeded.
    class Refusing(RecordingSink):
        def write(self, result: StepResult) -> None:
            raise RuntimeError("cannot write that")

    composite = CompositeOutputSink(RecordingSink(), Refusing())
    composite.open(_session_desc())

    with pytest.raises(RuntimeError, match="cannot write that"):
        composite.write(_result())
