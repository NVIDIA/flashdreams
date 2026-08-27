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

"""Tests for the isolated legacy FFmpeg MP4 backend."""

import subprocess
import sys
from pathlib import Path

import pytest

from flashdreams.runtime_v2.audio_stager import (
    F32leAudioStager as SharedF32leAudioStager,
)
from flashdreams.runtime_v2.legacy_mp4.ffmpeg_audio import (
    Mp4AudioMuxer as LegacyMp4AudioMuxer,
)
from flashdreams.runtime_v2.legacy_mp4.ffmpeg_audio import (
    preflight_audio_codec as legacy_preflight_audio_codec,
)
from flashdreams.runtime_v2.legacy_mp4.ffmpeg_video import (
    Mp4Encoder as LegacyMp4Encoder,
)
from flashdreams.runtime_v2.mp4_audio import (
    F32leAudioStager as CompatibleF32leAudioStager,
)
from flashdreams.runtime_v2.mp4_audio import (
    Mp4AudioMuxer as CompatibleMp4AudioMuxer,
)
from flashdreams.runtime_v2.mp4_audio import (
    preflight_audio_codec as compatible_preflight_audio_codec,
)
from flashdreams.runtime_v2.video_encoder import (
    Mp4Encoder as CompatibleMp4Encoder,
)
from flashdreams.runtime_v2.video_encoder import (
    result_to_rgb24_frames as compatible_result_to_rgb24_frames,
)
from flashdreams.runtime_v2.video_frames import (
    result_to_rgb24_frames as shared_result_to_rgb24_frames,
)

pytestmark = pytest.mark.ci_cpu


def test_importing_webm_does_not_load_legacy_ffmpeg() -> None:
    """Native WebM remains independent from the legacy FFmpeg backend."""
    code = """
import sys

import flashdreams.runtime_v2.webm_output_sink

loaded = sorted(
    name
    for name in sys.modules
    if name.startswith("flashdreams.runtime_v2.legacy_mp4")
)
if loaded:
    raise SystemExit(f"WebM loaded legacy modules: {loaded}")
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_legacy_mp4_compatibility_exports_are_preserved() -> None:
    """The former public import paths continue to expose the moved classes."""
    assert CompatibleMp4Encoder is LegacyMp4Encoder
    assert CompatibleMp4AudioMuxer is LegacyMp4AudioMuxer
    assert compatible_preflight_audio_codec is legacy_preflight_audio_codec
    assert CompatibleF32leAudioStager is SharedF32leAudioStager
    assert compatible_result_to_rgb24_frames is shared_result_to_rgb24_frames


def test_legacy_mp4_video_command_is_preserved(tmp_path: Path) -> None:
    """The isolated backend retains the established H.264 command settings."""
    output_path = tmp_path / "video.mp4"
    encoder = LegacyMp4Encoder(
        output_path,
        width=16,
        height=8,
        frames_per_second=24,
    )

    assert encoder._command("/usr/bin/ffmpeg") == [
        "/usr/bin/ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        "16x8",
        "-r",
        "24",
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
        str(output_path),
    ]
