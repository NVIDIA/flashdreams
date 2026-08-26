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

"""Model-neutral host media decoding helpers."""

from __future__ import annotations

import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np


def read_video_fps(path: str | Path) -> float:
    """Read the first video stream's frame rate through host FFprobe."""
    _, _, fps = _probe_video_stream(path)
    return fps


def read_video_rgb_with_fps(path: str | Path) -> tuple[np.ndarray, float]:
    """Decode RGB frames and their rate through host FFmpeg and FFprobe.

    Args:
        path: Input video file.

    Returns:
        Contiguous uint8 ``[time, height, width, 3]`` frames and a positive
        frames-per-second value.

    Raises:
        ValueError: Stream metadata or decoded bytes are malformed or empty.
        RuntimeError: Host tools are absent, probing fails, or decoding fails.
    """
    width, height, fps = _probe_video_stream(path)
    command = [
        _find_ffmpeg_binary(),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    process = subprocess.run(command, capture_output=True, check=False)
    if process.returncode != 0:
        diagnostic = process.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"ffmpeg could not decode a video stream from {path}: "
            f"{diagnostic or 'no diagnostic output'}"
        )
    frame_bytes = width * height * 3
    if not process.stdout:
        raise ValueError(f"decoded video stream is empty: {path}")
    if len(process.stdout) % frame_bytes:
        raise ValueError(f"decoded video bytes contain a truncated frame: {path}")
    frames = np.frombuffer(process.stdout, dtype=np.uint8).reshape(-1, height, width, 3)
    return frames.copy(), fps


def read_audio_f32(
    path: str | Path,
    *,
    sample_rate: int,
    channels: int = 2,
) -> np.ndarray:
    """Decode one audio stream through host FFmpeg.

    Args:
        path: Input audio file or audio-bearing video file.
        sample_rate: Positive output sampling rate.
        channels: Output channel count, currently mono or stereo.

    Returns:
        Contiguous finite float32 samples shaped ``[channels, samples]``.

    Raises:
        ValueError: The output format or decoded samples are invalid.
        RuntimeError: Host FFmpeg is absent or cannot decode an audio stream.
    """
    _validate_audio_output_format(sample_rate=sample_rate, channels=channels)
    return _decode_audio_f32(path, sample_rate=sample_rate, channels=channels)


def read_optional_audio_f32(
    path: str | Path,
    *,
    sample_rate: int,
    channels: int = 2,
) -> np.ndarray | None:
    """Decode a first audio stream when an input contains one.

    FFprobe distinguishes a genuinely absent stream from a malformed or
    unreadable input. FFmpeg then performs the same finite float32 decode as
    :func:`read_audio_f32`.

    Args:
        path: Input audio file or optionally audio-bearing video file.
        sample_rate: Positive output sampling rate.
        channels: Output channel count, currently mono or stereo.

    Returns:
        Contiguous ``[channels, samples]`` float32 audio, or ``None`` when no
        audio stream is present.

    Raises:
        ValueError: The output format or decoded samples are invalid.
        RuntimeError: Host tools are absent, probing fails, or decoding fails.
    """
    _validate_audio_output_format(sample_rate=sample_rate, channels=channels)
    if not _has_audio_stream(path):
        return None
    return _decode_audio_f32(path, sample_rate=sample_rate, channels=channels)


def _probe_video_stream(path: str | Path) -> tuple[int, int, float]:
    """Return first-video-stream width, height, and rate through host FFprobe."""
    command = [
        _find_ffprobe_binary(),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    process = subprocess.run(command, capture_output=True, check=False)
    if process.returncode != 0:
        diagnostic = process.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"ffprobe could not inspect a video stream in {path}: "
            f"{diagnostic or 'no diagnostic output'}"
        )
    try:
        payload = json.loads(process.stdout)
        stream = payload["streams"][0]
        width = int(stream["width"])
        height = int(stream["height"])
        rate = stream.get("avg_frame_rate") or stream["r_frame_rate"]
        fps = float(Fraction(rate))
    except (IndexError, KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError(
            f"ffprobe returned invalid video metadata for {path}"
        ) from error
    if width <= 0 or height <= 0 or not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"ffprobe returned invalid video metadata for {path}")
    return width, height, fps


def _validate_audio_output_format(*, sample_rate: int, channels: int) -> None:
    """Validate shared host-decoder output format options."""
    if type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    if type(channels) is not int or channels not in (1, 2):
        raise ValueError("channels must be 1 or 2")


def _decode_audio_f32(
    path: str | Path,
    *,
    sample_rate: int,
    channels: int,
) -> np.ndarray:
    """Decode a known audio stream through host FFmpeg."""
    command = [
        _find_ffmpeg_binary(),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "-c:a",
        "pcm_f32le",
        "pipe:1",
    ]
    process = subprocess.run(command, capture_output=True, check=False)
    if process.returncode != 0:
        diagnostic = process.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"ffmpeg could not decode an audio stream from {path}: "
            f"{diagnostic or 'no diagnostic output'}"
        )
    if len(process.stdout) % np.dtype("<f4").itemsize:
        raise ValueError(f"decoded audio bytes contain a truncated sample: {path}")
    samples = np.frombuffer(process.stdout, dtype="<f4")
    if samples.size == 0:
        raise ValueError(f"decoded audio stream is empty: {path}")
    if samples.size % channels:
        raise ValueError(
            f"decoded audio sample count {samples.size} is not divisible by "
            f"{channels} channels"
        )
    samples = samples.reshape(-1, channels).T.copy()
    if not np.isfinite(samples).all():
        raise ValueError(f"decoded audio contains non-finite samples: {path}")
    return samples


def _has_audio_stream(path: str | Path) -> bool:
    """Return whether host FFprobe reports a first audio stream."""
    command = [
        _find_ffprobe_binary(),
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        str(path),
    ]
    process = subprocess.run(command, capture_output=True, check=False)
    if process.returncode != 0:
        diagnostic = process.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"ffprobe could not inspect audio streams in {path}: "
            f"{diagnostic or 'no diagnostic output'}"
        )
    return bool(process.stdout.strip())


def _find_ffmpeg_binary() -> str:
    """Find host FFmpeg without discovering a bundled replacement."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise RuntimeError(
            "Decoding input media requires an ffmpeg executable installed on the "
            "host and available on PATH."
        )
    return ffmpeg_bin


def _find_ffprobe_binary() -> str:
    """Find host FFprobe without discovering a bundled replacement."""
    ffprobe_bin = shutil.which("ffprobe")
    if ffprobe_bin is None:
        raise RuntimeError(
            "Inspecting input media requires an ffprobe executable installed on the "
            "host and available on PATH."
        )
    return ffprobe_bin


__all__ = [
    "read_audio_f32",
    "read_optional_audio_f32",
    "read_video_fps",
    "read_video_rgb_with_fps",
]
