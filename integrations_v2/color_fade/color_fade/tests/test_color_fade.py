# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU test for the colour fade application and the MP4 file it writes."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from color_fade import create_app

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.session import ISession
from flashdreams.runtime_v2.application_runner import ApplicationRunner
from flashdreams.runtime_v2.mp4_client_window import Mp4ClientWindow
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="encoding and reading an MP4 back needs ffmpeg on PATH",
)

_WIDTH = 16
"""Frame width for the tests that only read tensors. Not square, so a transposed
frame cannot pass unnoticed."""

_HEIGHT = 8
"""Frame height for those tests."""

_PLAYABLE_WIDTH = 854
"""Frame width for the test that writes a file, so the file is watchable."""

_PLAYABLE_HEIGHT = 480
"""Frame height for it. A player will not open a video of a few pixels, and a
run nobody can watch is a poor end-to-end test."""

_FRAMES_PER_SECOND = 10
"""Generation rate the tests describe, in frames per second."""

_SECONDS = 1.0
"""Fade length the tests ask for. Short, so a whole fade is a few steps."""

_STEPS_FOR_THE_FADE = 3
"""Steps a whole fade takes at five frames a step. One second at ten frames a
second needs eleven frames, so the last step runs a little past the fade."""

_RED = (1.0, -1.0, -1.0)
"""Colour the fade starts at, in the ``[-1, 1]`` range the frames carry."""

_GREEN = (-1.0, 1.0, -1.0)
"""Colour the fade ends at."""


## Helpers


def _session_desc(
    layout: VideoTensorLayout = VideoTensorLayout.bcthw,
    *,
    width: int = _WIDTH,
    height: int = _HEIGHT,
) -> SessionDesc:
    return SessionDesc(
        output_layout=layout,
        frames_per_second_for_ui=60,
        frames_per_second_for_step=_FRAMES_PER_SECOND,
        video_width=width,
        video_height=height,
    )


def _session(*, frames_per_step: int = 5, seconds: float = _SECONDS) -> ISession:
    app = create_app()
    app.init(["--seconds", str(seconds), "--frames-per-step", str(frames_per_step)])
    session = app.create_session(_session_desc())
    session.init()
    return session


def _colours(result: StepResult) -> list[tuple[float, float, float]]:
    """Return each frame's colour, checking every pixel in it agrees."""
    frames = result.output[0]
    colours = []
    for index in range(result.frame_count):
        frame = frames[:, index]
        red, green, blue = (float(frame[channel, 0, 0]) for channel in range(3))
        assert torch.allclose(frame, torch.tensor([red, green, blue]).view(3, 1, 1)), (
            "every pixel of a frame carries the same colour"
        )
        colours.append((red, green, blue))
    return colours


def _step_colours(session: ISession, steps: int) -> list[tuple[float, float, float]]:
    """Run ``steps`` steps and return every frame's colour, oldest first."""
    colours: list[tuple[float, float, float]] = []
    for step_index in range(steps):
        colours.extend(_colours(session.step(step_index, UserInputEvents([]))))
    return colours


def _decode(path: Path, *, width: int, height: int) -> np.ndarray:
    """Read an MP4 of ``width`` by ``height`` back as ``[T, H, W, C]`` uint8 frames."""
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
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width, 3)


def _mean_colour(frame: np.ndarray) -> tuple[float, float, float]:
    """Return one decoded frame's mean red, green and blue."""
    red, green, blue = (float(frame[:, :, channel].mean()) for channel in range(3))
    return red, green, blue


## Tests of the fade itself


def test_fade_starts_red_and_ends_green() -> None:
    # One second of fade at ten frames a second, so the eleventh frame is the
    # first one the fade has finished by.
    colours = _step_colours(_session(frames_per_step=11), steps=1)

    assert colours[0] == pytest.approx(_RED)
    assert colours[-1] == pytest.approx(_GREEN)


def test_fade_passes_through_the_midpoint() -> None:
    colours = _step_colours(_session(frames_per_step=11), steps=1)

    assert colours[5] == pytest.approx((0.0, 0.0, -1.0))


def test_fade_moves_from_red_to_green_without_going_back() -> None:
    colours = _step_colours(_session(frames_per_step=6), steps=2)

    reds = [red for red, _, _ in colours]
    greens = [green for _, green, _ in colours]
    assert reds == sorted(reds, reverse=True)
    assert greens == sorted(greens)


def test_a_run_longer_than_the_fade_stays_green() -> None:
    colours = _step_colours(_session(frames_per_step=10, seconds=0.5), steps=2)

    assert all(colour == pytest.approx(_GREEN) for colour in colours[6:])


def test_a_frames_colour_depends_on_when_it_plays_not_on_the_chunk_size() -> None:
    # Same instant, different chunking: the fade is timed in seconds, so it
    # takes the same time whatever a step generates.
    in_chunks_of_two = _step_colours(_session(frames_per_step=2), steps=3)
    in_chunks_of_three = _step_colours(_session(frames_per_step=3), steps=2)

    assert in_chunks_of_two == pytest.approx(in_chunks_of_three)


def test_frames_match_the_session_desc() -> None:
    session = _session(frames_per_step=4)

    result = session.step(0, UserInputEvents([]))

    assert result.output.shape == (1, 3, 4, _HEIGHT, _WIDTH)
    assert result.output.dtype is torch.float32
    assert result.frame_count == 4
    assert result.output_layout is VideoTensorLayout.bcthw
    assert result.step_index == 0


def test_the_fade_ignores_input_and_repeats_after_a_reset() -> None:
    session = _session(frames_per_step=3)
    first = _colours(session.step(0, UserInputEvents([])))
    session.step(1, UserInputEvents([]))

    session.reset()

    assert _colours(session.step(0, UserInputEvents([]))) == pytest.approx(first)


def test_the_session_finishes_once_it_has_generated_the_fade() -> None:
    session = _session(frames_per_step=5)

    assert not session.is_finished()
    for step_index in range(_STEPS_FOR_THE_FADE):
        session.step(step_index, UserInputEvents([]))

    assert session.is_finished()
    # A client asking to start over gets the fade again, not a finished session.
    session.reset()
    assert not session.is_finished()


def test_session_desc_available_before_any_output_is_opened() -> None:
    assert _session().session_desc == _session_desc()


def test_create_session_rejects_unsupported_layout() -> None:
    app = create_app()
    app.init([])

    with pytest.raises(ValueError, match="bcthw"):
        app.create_session(_session_desc(layout=VideoTensorLayout.tchw))


def test_create_session_before_init_raises() -> None:
    with pytest.raises(RuntimeError, match="init"):
        create_app().create_session(_session_desc())


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--seconds", "0"], "--seconds"),
        (["--seconds", "-1"], "--seconds"),
        # A fade of nan seconds is nan the whole way through, and nan frames
        # reach a sink as a picture rather than as an error.
        (["--seconds", "nan"], "--seconds"),
        (["--seconds", "inf"], "--seconds"),
        (["--frames-per-step", "0"], "--frames-per-step"),
    ],
)
def test_init_rejects_settings_that_generate_nothing_watchable(
    args: list[str], message: str
) -> None:
    app: IApplication = create_app()

    with pytest.raises(ValueError, match=message):
        app.init(args)


## Test of the file a run produces


@needs_ffmpeg
def test_a_run_writes_the_whole_fade_to_an_mp4(tmp_path: Path) -> None:
    # Written at a size a player will open, so the file this leaves behind is
    # one a person can watch as well as one this test can read back.
    path = tmp_path / "fade.mp4"
    frames_per_step = 5

    # No step count: the session knows how long its fade is and ends the run.
    ApplicationRunner(create_app(), Mp4ClientWindow(path)).run(
        _session_desc(width=_PLAYABLE_WIDTH, height=_PLAYABLE_HEIGHT),
        ["--seconds", str(_SECONDS), "--frames-per-step", str(frames_per_step)],
    )

    frames = _decode(path, width=_PLAYABLE_WIDTH, height=_PLAYABLE_HEIGHT)
    assert len(frames) == _STEPS_FOR_THE_FADE * frames_per_step
    red, green, blue = _mean_colour(frames[0])
    assert red > 180
    assert green < 80
    assert blue < 80
    red, green, blue = _mean_colour(frames[-1])
    assert green > 180
    assert red < 80
    assert blue < 80
