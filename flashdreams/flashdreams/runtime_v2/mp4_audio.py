# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private PCM staging and external-FFmpeg audio muxing for MP4 output."""

import shutil
import subprocess
from pathlib import Path
from typing import BinaryIO

from flashdreams.runtime_v2.audio_output import AudioOutput, normalized_pcm

_FLOAT32_BYTES = 4
"""Bytes in one staged ``f32le`` channel sample."""


class F32leAudioStager:
    """Write normalized audio to a private interleaved ``f32le`` file."""

    def __init__(self, path: str | Path, *, sample_rate: int, channels: int) -> None:
        """
        Args:
            path: Private staging file to create.
            sample_rate: Sample rate every audio payload must carry.
            channels: Channel count every audio payload must carry.
        """
        self._path = Path(path)
        self._sample_rate = sample_rate
        self._channels = channels
        self._stream: BinaryIO | None = self._path.open("wb")
        self._samples_written = 0

    @property
    def path(self) -> Path:
        """Return the private staging path."""
        return self._path

    @property
    def samples_written(self) -> int:
        """Return the number of samples written per channel."""
        return self._samples_written

    def write(self, audio: AudioOutput) -> None:
        """Write one channel-major PCM payload at its absolute offset.

        Forward gaps become silence. Overlapping or backward payloads are
        rejected instead of silently changing samples already staged.

        Args:
            audio: Normalized PCM and its position on the session timeline.

        Raises:
            RuntimeError: The staging file has already been finished or aborted.
            ValueError: The payload disagrees with the declared audio format or
                overlaps samples already written.
        """
        stream = self._require_stream()
        if audio.sample_rate != self._sample_rate:
            raise ValueError(
                f"Expected audio at {self._sample_rate} Hz, got {audio.sample_rate} Hz."
            )
        pcm = normalized_pcm(audio)
        if pcm.shape[0] != self._channels:
            raise ValueError(
                f"Expected {self._channels} audio channels, got {pcm.shape[0]}."
            )
        if audio.sample_offset < self._samples_written:
            raise ValueError(
                f"Audio at offset {audio.sample_offset} overlaps the "
                f"{self._samples_written} samples already written."
            )
        self._write_silence(audio.sample_offset - self._samples_written, stream)
        interleaved = pcm.transpose(0, 1).contiguous().numpy()
        stream.write(interleaved.tobytes())
        self._samples_written += pcm.shape[1]

    def finish(self, expected_samples: int) -> None:
        """Pad or truncate to exactly ``expected_samples`` and close the file.

        Args:
            expected_samples: Samples per channel required by the video timeline.

        Raises:
            RuntimeError: The staging file has already been finished or aborted.
            ValueError: ``expected_samples`` is negative.
        """
        if expected_samples < 0:
            raise ValueError("Expected audio sample count must be non-negative.")
        stream = self._require_stream()
        if self._samples_written < expected_samples:
            self._write_silence(expected_samples - self._samples_written, stream)
        elif self._samples_written > expected_samples:
            stream.truncate(expected_samples * self._channels * _FLOAT32_BYTES)
            self._samples_written = expected_samples
        stream.close()
        self._stream = None

    def abort(self) -> None:
        """Close the private staging file without treating it as complete."""
        stream = self._stream
        if stream is not None:
            stream.close()
            self._stream = None

    def _write_silence(self, samples: int, stream: BinaryIO) -> None:
        """Append ``samples`` of zero-valued interleaved PCM."""
        if samples:
            stream.write(b"\0" * (samples * self._channels * _FLOAT32_BYTES))
            self._samples_written += samples

    def _require_stream(self) -> BinaryIO:
        """Return the open staging file or report that the transaction ended."""
        if self._stream is None:
            raise RuntimeError("Audio staging has already finished or aborted.")
        return self._stream


class Mp4AudioMuxer:
    """Mux staged video and raw audio through the host ``ffmpeg`` executable."""

    def __init__(
        self,
        *,
        video_path: str | Path,
        audio_path: str | Path,
        output_path: str | Path,
        sample_rate: int,
        channels: int,
        audio_codec: str,
    ) -> None:
        """
        Args:
            video_path: Complete staged video-only MP4.
            audio_path: Complete private interleaved ``f32le`` PCM.
            output_path: Staged synchronized MP4 to create.
            sample_rate: PCM sample rate.
            channels: PCM channel count.
            audio_codec: Host FFmpeg encoder selected for the public MP4.

        Raises:
            ValueError: ``audio_codec`` is empty.
        """
        if not audio_codec:
            raise ValueError("An audio codec must be selected explicitly.")
        self._video_path = Path(video_path)
        self._audio_path = Path(audio_path)
        self._output_path = Path(output_path)
        self._sample_rate = sample_rate
        self._channels = channels
        self._audio_codec = audio_codec
        self._process: subprocess.Popen[bytes] | None = None
        self._finished = False

    def close(self) -> None:
        """Run the mux and require the external process to succeed.

        Raises:
            RuntimeError: ``ffmpeg`` is unavailable or the mux fails.
        """
        if self._finished:
            return
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("Writing MP4 audio needs an ffmpeg executable on PATH.")
        try:
            process = subprocess.Popen(
                self._command(ffmpeg),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except OSError as error:
            raise RuntimeError(f"Could not start ffmpeg audio mux: {error}") from error
        self._process = process
        _, errors = process.communicate()
        self._process = None
        if process.returncode != 0:
            reported = errors.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"ffmpeg failed while muxing {self._output_path}: "
                f"{reported or 'no output'}"
            )
        self._finished = True

    def abort(self) -> None:
        """Terminate and wait for an active mux process, if any."""
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
        process.communicate()
        self._process = None

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
            "-c:a",
            self._audio_codec,
            "-movflags",
            "+faststart",
            str(self._output_path),
        ]


__all__ = ["F32leAudioStager", "Mp4AudioMuxer"]
