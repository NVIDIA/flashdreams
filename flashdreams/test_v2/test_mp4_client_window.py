# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the client window that writes an MP4.

The window reports no input and writes every UI result to the video file.
Encoding is covered in ``test_mp4_output_sink.py``; metrics output remains a
separate sink and is covered in ``test_metrics_output_sink.py``.
"""

import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from flashdreams.api_v2.loop import IModelLoop
from flashdreams.api_v2.output_sink import AbortableOutputSink
from flashdreams.api_v2.session import ISession
from flashdreams.runtime_v2.metrics_output_sink import MetricsOutputSink
from flashdreams.runtime_v2.mp4_client_window import Mp4ClientWindow
from flashdreams.runtime_v2.session_desc import (
    BackpressureMode,
    PresentationMode,
    SessionDesc,
)
from flashdreams.runtime_v2.session_runner import run_session
from flashdreams.runtime_v2.step_result import StepResult
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
            metrics={"total_ms": 1.5},
        )

    def reset(self) -> None:
        return

    def close(self) -> None:
        return


class FailingSession(FakeSession):
    """Fail generation after the transactional window has opened."""

    def step(self, step_index: int, events: UserInputEvents) -> StepResult:
        del step_index, events
        raise RuntimeError("model failed")


def _session_desc() -> SessionDesc:
    return SessionDesc(
        output_layout=VideoTensorLayout.bcthw,
        backpressure_mode=BackpressureMode.BLOCK,
        presentation_mode=PresentationMode.ONLY_PRESENT_NEW,
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


def test_there_is_never_any_input_to_report(tmp_path: Path) -> None:
    """A run polls on every tick, and a file has no client to answer."""
    window = Mp4ClientWindow(tmp_path / "clip.mp4")

    assert isinstance(window, AbortableOutputSink)
    assert window.get_user_input_events().get_events() == []
    assert window.get_user_input_events().get_events() == []


def test_failed_generation_preserves_target_and_removes_staging(
    tmp_path: Path,
) -> None:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"existing target")

    with pytest.raises(RuntimeError, match="model failed"):
        run_session(FailingSession(_session_desc()), Mp4ClientWindow(path), steps=1)

    assert path.read_bytes() == b"existing target"
    assert list(tmp_path.glob(f".{path.name}.*")) == []


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
