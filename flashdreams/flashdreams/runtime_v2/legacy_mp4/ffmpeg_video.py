# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""External FFmpeg video encoding for legacy MP4 output."""

import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from flashdreams.core.exceptions import add_exception_note

_RGB_CHANNELS = 3
"""Colour channels an encoded frame carries."""

_ERROR_CHUNK_BYTES = 8192
"""Maximum bytes read from FFmpeg diagnostics at once."""


class Mp4Encoder:
    """Encode RGB frames into one legacy MP4 file through host FFmpeg.

    Nothing is buffered here, so a long run costs no more memory than the frames
    of one write. FFmpeg starts on the first write, so an encoder that was never
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
            RuntimeError: FFmpeg is not installed, or it stopped early.
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
            RuntimeError: FFmpeg failed, so the file is unusable.
        """
        process = self._process
        if process is None:
            return
        assert process.stdin is not None
        failure: BaseException | None = None
        try:
            process.stdin.close()
        except BrokenPipeError:
            # FFmpeg gave up first; its exit code and diagnostics say why.
            pass
        except BaseException as error:
            failure = error
        exit_code: int | None = None
        waited = False
        try:
            exit_code = process.wait()
            waited = True
        except BaseException as error:
            if failure is None:
                failure = error
            else:
                add_exception_note(
                    failure, f"Waiting for ffmpeg also failed: {error!r}"
                )
        if waited:
            error_reader = self._error_reader
            if error_reader is not None:
                try:
                    error_reader.join()
                    self._error_reader = None
                except BaseException as error:
                    if failure is None:
                        failure = error
                    else:
                        add_exception_note(
                            failure,
                            f"Joining the ffmpeg error reader also failed: {error!r}",
                        )
            self._process = None
        if failure is not None:
            raise failure
        if exit_code != 0:
            raise RuntimeError(self._failure())

    def abort(self) -> None:
        """Terminate an active encoder without treating its file as complete.

        Does nothing before the process starts or after it has stopped, and can
        be called more than once.
        """
        process = self._process
        if process is None:
            return
        failure: BaseException | None = None
        waited = False
        try:
            if process.poll() is None:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except (BrokenPipeError, OSError, ValueError):
                    pass
            try:
                process.wait()
                waited = True
            except BaseException as error:
                failure = error
        finally:
            if self._error_reader is not None:
                if waited:
                    try:
                        self._error_reader.join()
                        self._error_reader = None
                    except BaseException as error:
                        if failure is None:
                            failure = error
                        else:
                            add_exception_note(
                                failure,
                                "Joining the ffmpeg error reader also failed: "
                                f"{error!r}",
                            )
            if waited:
                self._process = None
        if failure is not None:
            raise failure

    def _start(self) -> subprocess.Popen[bytes]:
        """Start FFmpeg, reading raw frames from its standard input."""
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("Writing an MP4 needs an ffmpeg executable on PATH.")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        process = subprocess.Popen(
            self._command(ffmpeg), stdin=subprocess.PIPE, stderr=subprocess.PIPE
        )
        self._process = process
        # Drain diagnostics concurrently because FFmpeg blocks once its pipe fills.
        self._errors = []
        self._error_reader = threading.Thread(
            target=_read_errors,
            args=(process.stderr, self._errors),
            name="flashdreams-mp4-errors",
            daemon=True,
        )
        try:
            self._error_reader.start()
        except BaseException as error:
            self._error_reader = None
            try:
                if process.poll() is None:
                    process.terminate()
                if process.stdin is not None:
                    process.stdin.close()
                process.wait()
                self._process = None
            except BaseException as cleanup_error:
                add_exception_note(
                    error, f"FFmpeg startup cleanup also failed: {cleanup_error!r}"
                )
            raise
        return process

    def _command(self, ffmpeg: str) -> list[str]:
        """Return the FFmpeg invocation for raw frames in, H.264 out."""
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
        """Describe what FFmpeg reported before it stopped."""
        reported = b"".join(self._errors).decode("utf-8", errors="replace").strip()
        return f"ffmpeg failed while writing {self._path}: {reported or 'no output'}"


def _read_errors(stream: Any, chunks: list[bytes]) -> None:
    """Read FFmpeg diagnostics until the stream closes."""
    while True:
        chunk = stream.read(_ERROR_CHUNK_BYTES)
        if not chunk:
            return
        chunks.append(chunk)


__all__ = ["Mp4Encoder"]
