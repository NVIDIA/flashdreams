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

"""PCM staging shared by file output backends."""

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


__all__ = ["F32leAudioStager"]
