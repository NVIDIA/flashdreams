# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MP4 encoding through the ffmpeg executable on ``PATH``."""

import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

_RGB_CHANNELS = 3
"""Colour channels an encoded frame carries."""

_ERROR_CHUNK_BYTES = 8192
"""How much of ffmpeg's diagnostic output to read at a time."""


class Mp4Encoder:
    """Encode RGB frames into one MP4 file, feeding ffmpeg as they arrive.

    Nothing is buffered here, so a long run costs no more memory than the frames
    of one write. ffmpeg starts on the first write, so an encoder that was never
    written to leaves no file behind.

    Call this from one thread at a time: it holds a pipe and a subprocess, and
    does no locking of its own.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        width: int,
        height: int,
        frames_per_second: int,
    ) -> None:
        """
        Args:
            path: File to write. Parent directories are created.
            width: Frame width in pixels, which every write must match.
            height: Frame height in pixels, which every write must match.
            frames_per_second: Rate the file plays back at.

        Raises:
            ValueError: A dimension is odd. Rounding one up would write a file
                of a size nobody asked for, so it is refused instead.
        """
        # yuv420p stores one chroma sample per two pixels in each direction.
        if width % 2 or height % 2:
            raise ValueError(
                f"An MP4 needs even frame dimensions, got {width}x{height}."
            )
        self._path = Path(path)
        self._width = width
        self._height = height
        self._frames_per_second = frames_per_second
        self._process: subprocess.Popen[bytes] | None = None
        self._errors: list[bytes] = []
        self._error_reader: threading.Thread | None = None

    def write(self, frames: npt.NDArray[np.uint8]) -> None:
        """Encode ``[T, H, W, C]`` uint8 frames.

        Args:
            frames: Frames to encode, matching the width and height this encoder
                was created for.

        Raises:
            RuntimeError: ffmpeg is not installed, or it stopped early.
            ValueError: The frames are not the shape this encoder was told to
                expect.
        """
        _, height, width, channels = frames.shape
        if (width, height, channels) != (self._width, self._height, _RGB_CHANNELS):
            raise ValueError(
                f"Expected {self._width}x{self._height} frames with "
                f"{_RGB_CHANNELS} channels, got {width}x{height} with {channels}."
            )
        process = self._process or self._start()
        assert process.stdin is not None
        try:
            process.stdin.write(frames.tobytes())
        except BrokenPipeError as error:
            raise RuntimeError(self._failure()) from error

    def close(self) -> None:
        """Finish the file.

        Does nothing when nothing was ever written, and can be called twice.

        Raises:
            RuntimeError: ffmpeg failed, so the file is unusable.
        """
        process = self._process
        if process is None:
            return
        self._process = None
        assert process.stdin is not None
        try:
            process.stdin.close()
        except BrokenPipeError:
            # ffmpeg gave up first; its exit code and diagnostics say why.
            pass
        exit_code = process.wait()
        if self._error_reader is not None:
            self._error_reader.join()
            self._error_reader = None
        if exit_code != 0:
            raise RuntimeError(self._failure())

    def _start(self) -> subprocess.Popen[bytes]:
        """Start ffmpeg, reading raw frames from its standard input."""
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("Writing an MP4 needs an ffmpeg executable on PATH.")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        process = subprocess.Popen(
            self._command(ffmpeg), stdin=subprocess.PIPE, stderr=subprocess.PIPE
        )
        # Drain the diagnostics on a thread of their own: ffmpeg blocks once
        # that pipe fills, and only a failure reads them, at the end of the run.
        self._errors = []
        self._error_reader = threading.Thread(
            target=_read_errors,
            args=(process.stderr, self._errors),
            name="flashdreams-mp4-errors",
            daemon=True,
        )
        self._error_reader.start()
        self._process = process
        return process

    def _command(self, ffmpeg: str) -> list[str]:
        """Return the ffmpeg invocation for raw frames in, H.264 out."""
        return [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self._width}x{self._height}",
            "-r",
            str(self._frames_per_second),
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-preset",
            "medium",
            "-movflags",
            "+faststart",
            str(self._path),
        ]

    def _failure(self) -> str:
        """Describe what ffmpeg reported before it stopped."""
        reported = b"".join(self._errors).decode("utf-8", errors="replace").strip()
        return f"ffmpeg failed while writing {self._path}: {reported or 'no output'}"


def result_to_rgb24_frames(
    result: StepResult, session_desc: SessionDesc
) -> npt.NDArray[np.uint8]:
    """Convert one result to the ``[T, H, W, C]`` uint8 frames an encoder reads.

    A pixel's value is read by dtype: a floating point tensor holds ``[-1, 1]``,
    which is what FlashDreams models emit, and an integer tensor holds raw
    ``0``-``255`` values. A result carrying one colour channel has it repeated
    across all three.

    Args:
        result: Generated output for one step.
        session_desc: Description the output is expected to match.

    Returns:
        Frames as uint8 RGB, oldest first.

    Raises:
        ValueError: ``result`` does not match ``session_desc``, carries more than
            one sequence of frames, or disagrees with itself over how many frames
            it carries.
    """
    if result.output_layout is not session_desc.output_layout:
        raise ValueError(
            f"Output was described as {session_desc.output_layout.value} but "
            f"arrived as {result.output_layout.value}."
        )
    frames = _to_tchw(result.output.detach(), result.output_layout)
    if frames.shape[0] != result.frame_count:
        raise ValueError(
            f"Result claims {result.frame_count} frames but carries {frames.shape[0]}."
        )
    if frames.shape[1] not in (1, _RGB_CHANNELS):
        raise ValueError(
            f"Expected one or {_RGB_CHANNELS} colour channels, got {frames.shape[1]}."
        )
    if frames.shape[2:] != (session_desc.video_height, session_desc.video_width):
        height, width = frames.shape[2:]
        described = f"{session_desc.video_width}x{session_desc.video_height}"
        raise ValueError(
            f"Output was described as {described} but arrived as {width}x{height}."
        )

    if frames.shape[1] == 1:
        frames = frames.repeat(1, _RGB_CHANNELS, 1, 1)
    if frames.is_floating_point():
        frames = ((frames.to(torch.float32).clamp(-1.0, 1.0) + 1.0) * 127.5).round()
    frames = frames.clamp(0, 255).to(torch.uint8)
    return frames.permute(0, 2, 3, 1).contiguous().cpu().numpy()


def _to_tchw(output: Tensor, layout: VideoTensorLayout) -> Tensor:
    """Return ``output`` as ``[T, C, H, W]``, whatever ``layout`` it arrived in.

    Raises:
        ValueError: The tensor carries more than one sequence of frames, or does
            not have the shape its layout claims.
    """
    if layout is VideoTensorLayout.tchw:
        _require_dimensions(output, layout, 4)
        return output
    if layout is VideoTensorLayout.btchw:
        _require_dimensions(output, layout, 5)
        _require_one(output.shape[0], "batch", layout)
        return output[0]
    if layout is VideoTensorLayout.bcthw:
        _require_dimensions(output, layout, 5)
        _require_one(output.shape[0], "batch", layout)
        return output[0].permute(1, 0, 2, 3)
    if layout is VideoTensorLayout.bvtchw:
        _require_dimensions(output, layout, 6)
        _require_one(output.shape[0], "batch", layout)
        _require_one(output.shape[1], "view", layout)
        return output[0, 0]
    raise ValueError(f"Unsupported output layout: {layout.value}.")


def _require_dimensions(
    output: Tensor, layout: VideoTensorLayout, expected: int
) -> None:
    """Check that a tensor has as many dimensions as its layout names."""
    if output.ndim != expected:
        raise ValueError(
            f"Layout {layout.value} expects {expected} dimensions, got "
            f"{tuple(output.shape)}."
        )


def _require_one(size: int, name: str, layout: VideoTensorLayout) -> None:
    """If a file can only hold one dimension, confirm it is exactly one."""
    if size != 1:
        raise ValueError(
            f"An MP4 holds one sequence of frames, so {layout.value} output "
            f"must have a {name} of 1, got {size}."
        )


def _read_errors(stream: Any, chunks: list[bytes]) -> None:
    """Read ffmpeg's diagnostics until it closes, collecting them into ``chunks``."""
    while True:
        chunk = stream.read(_ERROR_CHUNK_BYTES)
        if not chunk:
            return
        chunks.append(chunk)
