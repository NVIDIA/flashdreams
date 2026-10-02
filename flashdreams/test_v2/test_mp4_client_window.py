# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the client window that writes an MP4.

The window accepts requested resets and writes every UI result to the video file.
Encoding is covered in ``test_mp4_output_sink.py``; metrics output remains a
separate sink and is covered in ``test_metrics_output_sink.py``.
"""

import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from flashdreams.api_v2.loop import IModelLoop
from flashdreams.api_v2.session import ISession
from flashdreams.runtime_v2 import null_client_window
from flashdreams.runtime_v2.metrics_output_sink import MetricsOutputSink
from flashdreams.runtime_v2.mp4_client_window import Mp4ClientWindow
from flashdreams.runtime_v2.null_client_window import NullClientWindow
from flashdreams.runtime_v2.session_desc import (
    BackpressureMode,
    PresentationMode,
    SessionDesc,
)
from flashdreams.runtime_v2.session_runner import run_session
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import ResetUserInputEvent
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="encoding and reading an MP4 back needs ffmpeg on PATH",
)

_WIDTH = 16
"""Frame width. Not square, so a transposed frame cannot pass unnoticed."""

_HEIGHT = 8
"""Frame height."""

_FRAMES_PER_STEP = 2
"""Frames each step generates, above one so a frame count is not a step count."""


class FakeModelLoop(IModelLoop["FakeSession"]):
    def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]:
        return [self.state.step(step_index, events)]

    def reset(self) -> None:
        return


class FakeSession(ISession):
    """Emit blank frames, recording the input each step was given."""

    def __init__(self, session_desc: SessionDesc) -> None:
        self._session_desc = session_desc
        self.observed_events: list[UserInputEvents] = []

    def init(self) -> None:
        self.register_model_loop(FakeModelLoop, state=self)

    @property
    def session_desc(self) -> SessionDesc:
        return self._session_desc

    def step(self, step_index: int, events: UserInputEvents) -> StepResult:
        self.observed_events.append(events)
        return StepResult(
            step_index=step_index,
            output=torch.zeros(
                (1, 3, _FRAMES_PER_STEP, _HEIGHT, _WIDTH), dtype=torch.float32
            ),
            frame_count=_FRAMES_PER_STEP,
            output_layout=self._session_desc.output_layout,
        )

    def reset(self) -> None:
        return

    def close(self) -> None:
        return


def _session_desc() -> SessionDesc:
    return SessionDesc(
        output_layout=VideoTensorLayout.bcthw,
        backpressure_mode=BackpressureMode.BLOCK,
        presentation_mode=PresentationMode.ON_DEMAND,
        frames_per_second_for_ui=100,
        frames_per_second_for_step=30,
        video_width=_WIDTH,
        video_height=_HEIGHT,
    )


def _frame_count(path: Path) -> int:
    """Return how many frames an MP4 holds, by reading it back."""
    raw = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        check=True,
        capture_output=True,
    ).stdout
    return len(raw) // (_WIDTH * _HEIGHT * 3)


def test_there_is_no_unsolicited_input_to_report(tmp_path: Path) -> None:
    """A run polls on every tick, and a file has no client to answer."""
    window = Mp4ClientWindow(tmp_path / "clip.mp4")

    assert window.get_user_input_events().get_events() == []
    assert window.get_user_input_events().get_events() == []


@needs_ffmpeg
def test_a_run_writing_a_file_encodes_a_step_at_a_time(tmp_path: Path) -> None:
    """Every step reaches the file, and no step is given any input."""
    session = FakeSession(_session_desc())
    path = tmp_path / "clip.mp4"

    run_session(session, Mp4ClientWindow(path), steps=3)

    assert _frame_count(path) == 3 * _FRAMES_PER_STEP
    assert [events.get_events() for events in session.observed_events] == [[], [], []]
    assert not list(tmp_path.glob("*.json"))


@needs_ffmpeg
def test_a_run_can_write_video_and_metrics_separately(
    tmp_path: Path,
) -> None:
    path = tmp_path / "clip.mp4"
    stats_path = tmp_path / "stats.json"

    run_session(
        FakeSession(_session_desc()),
        Mp4ClientWindow(path),
        metrics_output_sink=MetricsOutputSink(stats_path),
        steps=3,
    )

    assert _frame_count(path) == 3 * _FRAMES_PER_STEP
    assert stats_path.is_file()


@pytest.mark.parametrize("window_type", [NullClientWindow, Mp4ClientWindow])
def test_headless_reset_requests_are_drained_once_per_session(
    window_type: type[NullClientWindow],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now_ns = 1_000_000
    monkeypatch.setattr(null_client_window, "monotonic_ns", lambda: now_ns)
    window = (
        Mp4ClientWindow(tmp_path / "clip.mp4")
        if window_type is Mp4ClientWindow
        else NullClientWindow()
    )
    with pytest.raises(RuntimeError, match="Open the client window"):
        window.request_reset()
    window.open(_session_desc())
    try:
        now_ns = 1_001_000
        window.request_reset()
        now_ns = 1_003_000
        window.request_reset()
        events = window.get_user_input_events().get_events()
        assert len(events) == 2
        assert all(isinstance(event, ResetUserInputEvent) for event in events)
        assert [event.get_timestamp() for event in events] == [1, 3]
        assert window.get_user_input_events().get_events() == []

        window.request_reset()
        now_ns = 2_000_000
        window.open(_session_desc())
        now_ns = 2_002_000
        window.request_reset()
        (event,) = window.get_user_input_events().get_events()
        assert isinstance(event, ResetUserInputEvent)
        assert event.get_timestamp() == 2
        window.request_reset()
    finally:
        window.close()
    assert window.get_user_input_events().get_events() == []
    with pytest.raises(RuntimeError, match="Open the client window"):
        window.request_reset()
