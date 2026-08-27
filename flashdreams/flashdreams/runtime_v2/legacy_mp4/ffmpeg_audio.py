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

"""External FFmpeg audio encoding and muxing for legacy MP4 output."""

import math
import shutil
import subprocess
from pathlib import Path

from flashdreams.core.exceptions import add_exception_note

_FLOAT32_BYTES = 4
"""Bytes in one staged ``f32le`` channel sample."""

_AAC_FRAME_SAMPLES = 1024
"""Samples in one AAC-LC frame used for the preflight encode."""

_AUDIO_CODEC_ARGUMENTS = (
    "-c:a",
    "aac",
    "-profile:a",
    "aac_low",
    "-b:a",
    "192k",
)
"""Fixed public MP4 audio encoding: AAC-LC at 192 kbit/s."""

_PREFLIGHT_TIMEOUT_SECONDS = 10
"""Maximum time the tiny external AAC capability check may take."""

_MUX_MINIMUM_TIMEOUT_SECONDS = 30.0
"""Minimum wall time allowed for a real audio mux."""

_MUX_TIMEOUT_PER_MEDIA_SECOND = 2.0
"""Additional mux allowance per second of staged media."""

_PROCESS_STOP_TIMEOUT_SECONDS = 5.0
"""Bound for each terminate and kill wait during cleanup."""


def preflight_audio_codec(*, sample_rate: int, channels: int) -> str:
    """Resolve host FFmpeg and prove the legacy audio encoding works.

    This check runs before model generation so a missing product dependency
    does not waste a rollout only to fail during the final mux.

    Args:
        sample_rate: PCM sample rate the session will publish.
        channels: PCM channel count the session will publish.

    Returns:
        Resolved path to the host ``ffmpeg`` executable.

    Raises:
        RuntimeError: FFmpeg is unavailable or cannot encode the session format
            as AAC-LC at the approved bitrate within the timeout.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("Writing MP4 audio needs an ffmpeg executable on PATH.")
    ffmpeg_path = str(Path(ffmpeg).resolve())
    try:
        completed = subprocess.run(
            [
                ffmpeg_path,
                "-hide_banner",
                "-nostdin",
                "-loglevel",
                "error",
                "-f",
                "f32le",
                "-ar",
                str(sample_rate),
                "-ac",
                str(channels),
                "-i",
                "pipe:0",
                "-frames:a",
                "1",
                *_AUDIO_CODEC_ARGUMENTS,
                "-f",
                "null",
                "-",
            ],
            input=b"\0" * (_AAC_FRAME_SAMPLES * channels * _FLOAT32_BYTES),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=_PREFLIGHT_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            "Timed out while checking the host ffmpeg AAC-LC encoder."
        ) from error
    except OSError as error:
        raise RuntimeError(
            f"Could not check the host ffmpeg AAC-LC encoder: {error}"
        ) from error
    if completed.returncode != 0:
        reported = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            "Host ffmpeg cannot encode this session as AAC-LC at 192 kbit/s: "
            f"{reported or 'no output'}"
        )
    return ffmpeg_path


class Mp4AudioMuxer:
    """Mux staged video and raw audio through the host FFmpeg executable."""

    def __init__(
        self,
        *,
        video_path: str | Path,
        audio_path: str | Path,
        output_path: str | Path,
        sample_rate: int,
        channels: int,
        ffmpeg_path: str,
        duration_seconds: float,
    ) -> None:
        """
        Args:
            video_path: Complete staged video-only MP4.
            audio_path: Complete private interleaved ``f32le`` PCM.
            output_path: Staged synchronized MP4 to create.
            sample_rate: PCM sample rate.
            channels: PCM channel count.
            ffmpeg_path: Host executable resolved by the early exact preflight.
            duration_seconds: Staged media duration used to scale the mux
                deadline.

        Raises:
            ValueError: ``duration_seconds`` is negative or non-finite.
        """
        if not math.isfinite(duration_seconds) or duration_seconds < 0:
            raise ValueError("Audio mux duration must be finite and non-negative.")
        self._video_path = Path(video_path)
        self._audio_path = Path(audio_path)
        self._output_path = Path(output_path)
        self._sample_rate = sample_rate
        self._channels = channels
        self._ffmpeg_path = ffmpeg_path
        self._timeout_seconds = max(
            _MUX_MINIMUM_TIMEOUT_SECONDS,
            duration_seconds * _MUX_TIMEOUT_PER_MEDIA_SECOND,
        )
        self._process: subprocess.Popen[bytes] | None = None
        self._finished = False

    def close(self) -> None:
        """Run the mux and require the external process to succeed.

        Raises:
            RuntimeError: FFmpeg is unavailable or the mux fails.
        """
        if self._finished:
            return
        try:
            process = subprocess.Popen(
                self._command(self._ffmpeg_path),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except OSError as error:
            raise RuntimeError(f"Could not start ffmpeg audio mux: {error}") from error
        self._process = process
        try:
            _, errors = process.communicate(timeout=self._timeout_seconds)
        except subprocess.TimeoutExpired as error:
            failure = RuntimeError(
                "Timed out after "
                f"{self._timeout_seconds:g} seconds while muxing "
                f"{self._output_path}."
            )
            try:
                self._terminate_and_reap(process)
            except BaseException as cleanup_error:
                add_exception_note(
                    failure,
                    f"FFmpeg mux timeout cleanup also failed: {cleanup_error!r}",
                )
            raise failure from error
        self._process = None
        if process.returncode != 0:
            reported = errors.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"ffmpeg failed while muxing {self._output_path}: "
                f"{reported or 'no output'}"
            )
        self._finished = True

    def abort(self) -> None:
        """Terminate and wait for an active mux process, if any.

        Each wait is bounded. Ownership is retained if the process cannot be
        reaped, so a later abort can retry without deleting files it may use.
        """
        process = self._process
        if process is None:
            return
        self._terminate_and_reap(process)

    def _terminate_and_reap(self, process: subprocess.Popen[bytes]) -> None:
        """Stop ``process`` with bounded terminate/kill escalation."""
        failure: BaseException | None = None
        reaped = False
        if process.poll() is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            except BaseException as error:
                failure = error
        try:
            process.communicate(timeout=_PROCESS_STOP_TIMEOUT_SECONDS)
            reaped = True
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            except BaseException as error:
                failure = _append_secondary_failure(
                    failure, error, "Killing the ffmpeg audio mux also failed"
                )
            try:
                process.communicate(timeout=_PROCESS_STOP_TIMEOUT_SECONDS)
                reaped = True
            except BaseException as error:
                failure = _append_secondary_failure(
                    failure, error, "Reaping the killed ffmpeg audio mux also failed"
                )
        except BaseException as error:
            failure = _append_secondary_failure(
                failure, error, "Reaping the ffmpeg audio mux also failed"
            )
        if reaped:
            self._process = None
        if failure is not None:
            raise failure

    def _command(self, ffmpeg: str) -> list[str]:
        """Return the external invocation that copies video and encodes audio."""
        return [
            ffmpeg,
            "-y",
            "-nostdin",
            "-loglevel",
            "error",
            "-i",
            str(self._video_path),
            "-f",
            "f32le",
            "-ar",
            str(self._sample_rate),
            "-ac",
            str(self._channels),
            "-i",
            str(self._audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            *_AUDIO_CODEC_ARGUMENTS,
            "-movflags",
            "+faststart",
            str(self._output_path),
        ]


def _append_secondary_failure(
    failure: BaseException | None,
    error: BaseException,
    message: str,
) -> BaseException:
    """Keep the first cleanup failure and annotate later ones."""
    if failure is None:
        return error
    add_exception_note(failure, f"{message}: {error!r}")
    return failure


__all__ = ["Mp4AudioMuxer", "preflight_audio_codec"]
