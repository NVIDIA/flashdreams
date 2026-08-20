# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU test for the client window that writes an MP4 file."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import Tensor

from flashdreams.runtime_v2.mp4_client_window import Mp4ClientWindow
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
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

_RED = (1.0, -1.0, -1.0)
"""Full red, in the ``[-1, 1]`` range a floating point result carries."""

_BLACK = (-1.0, -1.0, -1.0)
"""Black in the same range."""


## Helpers


def _session_desc(
    layout: VideoTensorLayout = VideoTensorLayout.bcthw,
    *,
    width: int = _WIDTH,
    height: int = _HEIGHT,
) -> SessionDesc:
    return SessionDesc(
        output_layout=layout,
        frames_per_second_for_ui=30,
        frames_per_second_for_step=30,
        video_width=width,
        video_height=height,
    )


def _in_layout(frames: Tensor, layout: VideoTensorLayout) -> Tensor:
    """Lay a ``[T, C, H, W]`` tensor out as ``layout`` says."""
    if layout is VideoTensorLayout.tchw:
        return frames
    if layout is VideoTensorLayout.btchw:
        return frames.unsqueeze(0)
    if layout is VideoTensorLayout.bcthw:
        return frames.permute(1, 0, 2, 3).unsqueeze(0)
    if layout is VideoTensorLayout.bvtchw:
        return frames.unsqueeze(0).unsqueeze(0)
    raise AssertionError(f"no test layout for {layout.value}")


def _result(
    colours: list[tuple[float, float, float]],
    *,
    step_index: int = 0,
    layout: VideoTensorLayout = VideoTensorLayout.bcthw,
    dtype: torch.dtype = torch.float32,
) -> StepResult:
    """Return a result of solid frames, one per colour."""
    frames = torch.zeros((len(colours), 3, _HEIGHT, _WIDTH), dtype=dtype)
    for index, colour in enumerate(colours):
        for channel, value in enumerate(colour):
            frames[index, channel] = value
    return StepResult(
        step_index=step_index,
        output=_in_layout(frames, layout),
        frame_count=len(colours),
        output_layout=layout,
    )


def _decode(path: Path) -> np.ndarray:
    """Read an MP4 back as ``[T, H, W, C]`` uint8 frames."""
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
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, _HEIGHT, _WIDTH, 3)


def _mean_colour(frame: np.ndarray) -> tuple[float, float, float]:
    """Return one frame's mean red, green and blue.

    The frames written are solid and encoding is lossy, so a mean is what a
    colour is recognisable by rather than an exact value.
    """
    red, green, blue = (float(frame[:, :, channel].mean()) for channel in range(3))
    return red, green, blue


## Tests that do not encode


def test_window_reports_no_input() -> None:
    window = Mp4ClientWindow("unused.mp4")

    assert window.get_user_input_events().get_events() == []


def test_write_before_open_raises(tmp_path: Path) -> None:
    window = Mp4ClientWindow(tmp_path / "out.mp4")

    with pytest.raises(RuntimeError, match="open"):
        window.write(_result([_RED]))


def test_write_rejects_a_layout_the_window_was_not_opened_for(tmp_path: Path) -> None:
    window = Mp4ClientWindow(tmp_path / "out.mp4")
    window.open(_session_desc(VideoTensorLayout.bcthw))

    with pytest.raises(ValueError, match="tchw"):
        window.write(_result([_RED], layout=VideoTensorLayout.tchw))


def test_write_rejects_frames_of_another_size(tmp_path: Path) -> None:
    window = Mp4ClientWindow(tmp_path / "out.mp4")
    window.open(_session_desc(width=_WIDTH * 2))

    with pytest.raises(ValueError, match=f"{_WIDTH}x{_HEIGHT}"):
        window.write(_result([_RED]))


def test_write_rejects_a_frame_count_the_tensor_does_not_carry(tmp_path: Path) -> None:
    window = Mp4ClientWindow(tmp_path / "out.mp4")
    window.open(_session_desc())
    carrying_two = _result([_RED, _BLACK])
    claiming_five = StepResult(
        step_index=0,
        output=carrying_two.output,
        frame_count=5,
        output_layout=carrying_two.output_layout,
    )

    with pytest.raises(ValueError, match="5 frames"):
        window.write(claiming_five)


def test_write_rejects_output_with_more_than_one_batch(tmp_path: Path) -> None:
    window = Mp4ClientWindow(tmp_path / "out.mp4")
    window.open(_session_desc())
    one = _result([_RED])
    two_batches = StepResult(
        step_index=0,
        output=torch.cat([one.output, one.output]),
        frame_count=1,
        output_layout=one.output_layout,
    )

    with pytest.raises(ValueError, match="batch"):
        window.write(two_batches)


def test_a_run_that_generated_nothing_writes_no_file(tmp_path: Path) -> None:
    path = tmp_path / "out.mp4"
    window = Mp4ClientWindow(path)

    window.open(_session_desc())
    window.close()

    assert not path.exists()


def test_close_tolerates_a_window_that_was_never_opened(tmp_path: Path) -> None:
    Mp4ClientWindow(tmp_path / "out.mp4").close()


## Tests that encode


@needs_ffmpeg
@pytest.mark.parametrize(
    "layout",
    [
        VideoTensorLayout.tchw,
        VideoTensorLayout.btchw,
        VideoTensorLayout.bcthw,
        VideoTensorLayout.bvtchw,
    ],
)
def test_window_writes_one_frame_per_result_in_order(
    tmp_path: Path, layout: VideoTensorLayout
) -> None:
    path = tmp_path / "out.mp4"
    window = Mp4ClientWindow(path)

    window.open(_session_desc(layout))
    window.write(_result([_RED], step_index=0, layout=layout))
    window.write(_result([_BLACK], step_index=1, layout=layout))
    window.close()

    frames = _decode(path)
    assert len(frames) == 2
    red, green, blue = _mean_colour(frames[0])
    assert red > 180
    assert green < 80
    assert blue < 80
    assert max(_mean_colour(frames[1])) < 40


@needs_ffmpeg
def test_window_writes_every_frame_a_result_carries(tmp_path: Path) -> None:
    path = tmp_path / "out.mp4"
    window = Mp4ClientWindow(path)

    window.open(_session_desc())
    window.write(_result([_RED, _BLACK, _RED]))
    window.close()

    assert len(_decode(path)) == 3


@needs_ffmpeg
def test_a_floating_point_result_is_read_as_minus_one_to_one(tmp_path: Path) -> None:
    # Zero is the middle of that range, so a frame of zeros is mid grey rather
    # than black. This is the same rule the WebRTC window applies.
    path = tmp_path / "out.mp4"
    window = Mp4ClientWindow(path)

    window.open(_session_desc())
    window.write(_result([(0.0, 0.0, 0.0)]))
    window.close()

    assert min(_mean_colour(_decode(path)[0])) > 100
    assert max(_mean_colour(_decode(path)[0])) < 155


@needs_ffmpeg
def test_an_integer_result_is_read_as_raw_bytes(tmp_path: Path) -> None:
    path = tmp_path / "out.mp4"
    window = Mp4ClientWindow(path)

    window.open(_session_desc())
    window.write(_result([(17.0, 17.0, 17.0)], dtype=torch.uint8))
    window.close()

    assert max(_mean_colour(_decode(path)[0])) < 25


@needs_ffmpeg
def test_a_single_channel_result_is_written_as_grey(tmp_path: Path) -> None:
    path = tmp_path / "out.mp4"
    window = Mp4ClientWindow(path)
    grey = torch.zeros((1, 1, _HEIGHT, _WIDTH), dtype=torch.float32)

    window.open(_session_desc(VideoTensorLayout.tchw))
    window.write(
        StepResult(
            step_index=0,
            output=grey,
            frame_count=1,
            output_layout=VideoTensorLayout.tchw,
        )
    )
    window.close()

    red, green, blue = _mean_colour(_decode(path)[0])
    assert abs(red - green) < 10
    assert abs(green - blue) < 10


@needs_ffmpeg
def test_close_can_run_twice(tmp_path: Path) -> None:
    path = tmp_path / "out.mp4"
    window = Mp4ClientWindow(path)
    window.open(_session_desc())
    window.write(_result([_RED]))

    window.close()
    window.close()

    assert len(_decode(path)) == 1
