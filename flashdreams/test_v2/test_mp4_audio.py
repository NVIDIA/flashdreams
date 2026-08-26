# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for private PCM staging and external MP4 audio muxing."""

import struct
from pathlib import Path

import pytest
import torch

import flashdreams.runtime_v2.mp4_audio as mp4_audio_module
from flashdreams.runtime_v2.audio_output import AudioOutput
from flashdreams.runtime_v2.mp4_audio import F32leAudioStager, Mp4AudioMuxer

pytestmark = pytest.mark.ci_cpu


def _audio(
    channels: list[list[float]], *, sample_rate: int = 8_000, offset: int = 0
) -> AudioOutput:
    """Return one normalized audio payload."""
    return AudioOutput(
        samples=torch.tensor(channels, dtype=torch.float32),
        sample_rate=sample_rate,
        sample_offset=offset,
    )


def _floats(path: Path) -> tuple[float, ...]:
    """Read a little-endian float32 staging file."""
    data = path.read_bytes()
    return struct.unpack(f"<{len(data) // 4}f", data)


def test_audio_stager_interleaves_channels_and_exact_offsets(tmp_path: Path) -> None:
    path = tmp_path / "audio.f32le"
    stager = F32leAudioStager(path, sample_rate=8_000, channels=2)

    stager.write(_audio([[0.1, 0.2], [-0.1, -0.2]]))
    stager.write(_audio([[0.3], [-0.3]], offset=2))
    stager.finish(3)

    assert _floats(path) == pytest.approx((0.1, -0.1, 0.2, -0.2, 0.3, -0.3))
    assert stager.samples_written == 3


def test_audio_stager_zero_fills_forward_gaps_and_padding(tmp_path: Path) -> None:
    path = tmp_path / "audio.f32le"
    stager = F32leAudioStager(path, sample_rate=8_000, channels=1)

    stager.write(_audio([[0.5]], offset=2))
    stager.finish(5)

    assert _floats(path) == pytest.approx((0.0, 0.0, 0.5, 0.0, 0.0))
    assert stager.samples_written == 5


def test_audio_stager_truncates_to_the_video_timeline(tmp_path: Path) -> None:
    path = tmp_path / "audio.f32le"
    stager = F32leAudioStager(path, sample_rate=8_000, channels=2)
    stager.write(_audio([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]]))

    stager.finish(2)

    assert _floats(path) == pytest.approx((0.1, -0.1, 0.2, -0.2))
    assert stager.samples_written == 2


def test_audio_stager_rejects_overlap_without_changing_staged_audio(
    tmp_path: Path,
) -> None:
    path = tmp_path / "audio.f32le"
    stager = F32leAudioStager(path, sample_rate=8_000, channels=1)
    stager.write(_audio([[0.1, 0.2]]))

    with pytest.raises(ValueError, match="overlaps"):
        stager.write(_audio([[0.3]], offset=1))
    stager.finish(2)

    assert _floats(path) == pytest.approx((0.1, 0.2))


@pytest.mark.parametrize(
    ("audio", "message"),
    [
        (_audio([[0.1]], sample_rate=16_000), "8000 Hz"),
        (_audio([[0.1], [0.2]]), "1 audio channels"),
    ],
)
def test_audio_stager_rejects_a_payload_of_another_format(
    tmp_path: Path, audio: AudioOutput, message: str
) -> None:
    stager = F32leAudioStager(tmp_path / "audio.f32le", sample_rate=8_000, channels=1)

    with pytest.raises(ValueError, match=message):
        stager.write(audio)
    stager.abort()


def test_audio_muxer_builds_external_stream_copy_command(tmp_path: Path) -> None:
    muxer = Mp4AudioMuxer(
        video_path=tmp_path / "video.mp4",
        audio_path=tmp_path / "audio.f32le",
        output_path=tmp_path / "output.mp4",
        sample_rate=32_000,
        channels=2,
        audio_codec="reviewed-codec",
    )

    assert muxer._command("/usr/bin/ffmpeg") == [
        "/usr/bin/ffmpeg",
        "-y",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(tmp_path / "video.mp4"),
        "-f",
        "f32le",
        "-ar",
        "32000",
        "-ac",
        "2",
        "-i",
        str(tmp_path / "audio.f32le"),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "reviewed-codec",
        "-movflags",
        "+faststart",
        str(tmp_path / "output.mp4"),
    ]


def test_audio_muxer_surfaces_external_process_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Process:
        returncode = 7

        def communicate(self) -> tuple[None, bytes]:
            return None, b"codec failed"

    monkeypatch.setattr(mp4_audio_module.shutil, "which", lambda name: f"/{name}")
    monkeypatch.setattr(
        mp4_audio_module.subprocess, "Popen", lambda *args, **kw: Process()
    )
    muxer = Mp4AudioMuxer(
        video_path=tmp_path / "video.mp4",
        audio_path=tmp_path / "audio.f32le",
        output_path=tmp_path / "output.mp4",
        sample_rate=8_000,
        channels=1,
        audio_codec="reviewed-codec",
    )

    with pytest.raises(RuntimeError, match="codec failed"):
        muxer.close()


def test_audio_muxer_abort_terminates_and_waits_for_ffmpeg(tmp_path: Path) -> None:
    calls: list[str] = []

    class Process:
        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            calls.append("terminate")

        def communicate(self) -> tuple[None, bytes]:
            calls.append("communicate")
            return None, b""

    muxer = Mp4AudioMuxer(
        video_path=tmp_path / "video.mp4",
        audio_path=tmp_path / "audio.f32le",
        output_path=tmp_path / "output.mp4",
        sample_rate=8_000,
        channels=1,
        audio_codec="reviewed-codec",
    )
    muxer._process = Process()  # type: ignore[assignment]

    muxer.abort()
    muxer.abort()

    assert calls == ["terminate", "communicate"]


def test_audio_muxer_retains_process_when_abort_wait_fails(tmp_path: Path) -> None:
    """A second abort can retry a child whose first wait was interrupted."""
    calls: list[str] = []

    class Process:
        communicate_calls = 0

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            calls.append("terminate")

        def communicate(self) -> tuple[None, bytes]:
            calls.append("communicate")
            self.communicate_calls += 1
            if self.communicate_calls == 1:
                raise RuntimeError("mux wait interrupted")
            return None, b""

    process = Process()
    muxer = Mp4AudioMuxer(
        video_path=tmp_path / "video.mp4",
        audio_path=tmp_path / "audio.f32le",
        output_path=tmp_path / "output.mp4",
        sample_rate=8_000,
        channels=1,
        audio_codec="reviewed-codec",
    )
    muxer._process = process  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="mux wait interrupted"):
        muxer.abort()
    assert muxer._process is process

    muxer.abort()

    assert muxer._process is None
    assert calls == ["terminate", "communicate", "terminate", "communicate"]
