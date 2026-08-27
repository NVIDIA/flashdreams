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

"""CPU smoke tests for native VPx, Opus, and WebM output."""

import json
import shutil
import struct
import subprocess
from pathlib import Path

import pytest
from flashdreams_webm import WebmWriter, versions

pytestmark = pytest.mark.ci_cpu

_WIDTH = 16
_HEIGHT = 8


def _frame(seed: int) -> bytes:
    """Return one deterministic RGB24 frame."""
    return bytes(
        (x * 7 + y * 11 + channel * 53 + seed * 31) % 256
        for y in range(_HEIGHT)
        for x in range(_WIDTH)
        for channel in range(3)
    )


@pytest.mark.parametrize("codec", ["vp9", "vp8"])
def test_native_writer_finalizes_each_vpx_codec(tmp_path: Path, codec: str) -> None:
    path = tmp_path / f"{codec}.webm"
    writer = WebmWriter(path, _WIDTH, _HEIGHT, 24, codec)

    writer.write_video(_frame(0))
    writer.write_video(_frame(1))
    writer.close()
    writer.close()
    writer.abort()

    assert path.read_bytes().startswith(b"\x1aE\xdf\xa3")
    assert writer.codec == codec
    assert writer.closed is True
    assert list(tmp_path.iterdir()) == [path]


def test_native_versions_identify_every_wrapped_library() -> None:
    assert set(versions()) == {"libvpx", "libopus", "libwebm"}


@pytest.mark.skipif(
    shutil.which("ffprobe") is None,
    reason="validating native WebM streams needs ffprobe on PATH",
)
def test_native_writer_muxes_vp9_and_opus(tmp_path: Path) -> None:
    path = tmp_path / "audio.webm"
    audio_path = tmp_path / "audio.f32le"
    sample_rate = 48_000
    samples = 4_000
    audio_path.write_bytes(
        b"".join(
            struct.pack("<f", 0.1 if sample % 2 == 0 else -0.1)
            for sample in range(samples)
        )
    )
    writer = WebmWriter(path, _WIDTH, _HEIGHT, 24, "vp9", sample_rate, 1)
    writer.write_video(_frame(0))
    writer.write_video(_frame(1))
    writer.close(audio_path)

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_name,codec_type,sample_rate,channels,width,height,r_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(result.stdout)["streams"]

    assert streams == [
        {
            "codec_name": "vp9",
            "codec_type": "video",
            "width": _WIDTH,
            "height": _HEIGHT,
            "r_frame_rate": "24/1",
        },
        {
            "codec_name": "opus",
            "codec_type": "audio",
            "sample_rate": str(sample_rate),
            "channels": 1,
            "r_frame_rate": "0/0",
        },
    ]
