# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for private PCM staging and external MP4 audio muxing."""

import struct
from pathlib import Path

import pytest
import torch

import flashdreams.runtime_v2.legacy_mp4.ffmpeg_audio as ffmpeg_audio_module
from flashdreams.runtime_v2.audio_output import AudioOutput
from flashdreams.runtime_v2.audio_stager import F32leAudioStager
from flashdreams.runtime_v2.legacy_mp4.ffmpeg_audio import (
    Mp4AudioMuxer,
    preflight_audio_codec,
)

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


def test_audio_stager_retains_stream_when_abort_close_fails(tmp_path: Path) -> None:
    """A later abort can retry a staging handle whose close was interrupted."""
    calls: list[str] = []

    class Stream:
        def close(self) -> None:
            calls.append("close")
            if len(calls) == 1:
                raise RuntimeError("audio close interrupted")

    stager = F32leAudioStager(tmp_path / "audio.f32le", sample_rate=8_000, channels=1)
    assert stager._stream is not None
    stager._stream.close()
    stream = Stream()
    stager._stream = stream  # ty: ignore[invalid-assignment]

    with pytest.raises(RuntimeError, match="audio close interrupted"):
        stager.abort()
    assert stager._stream is stream

    stager.abort()

    assert stager._stream is None
    assert calls == ["close", "close"]


def test_audio_codec_preflight_encodes_the_exact_session_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    class Completed:
        returncode = 0
        stderr = b""

    def run(command: list[str], **kwargs: object) -> Completed:
        calls.append((command, kwargs))
        return Completed()

    monkeypatch.setattr(
        ffmpeg_audio_module.shutil, "which", lambda name: "/host/ffmpeg"
    )
    monkeypatch.setattr(ffmpeg_audio_module.subprocess, "run", run)

    assert preflight_audio_codec(sample_rate=32_000, channels=2) == "/host/ffmpeg"
    assert calls[0][0] == [
        "/host/ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-f",
        "f32le",
        "-ar",
        "32000",
        "-ac",
        "2",
        "-i",
        "pipe:0",
        "-frames:a",
        "1",
        "-c:a",
        "aac",
        "-profile:a",
        "aac_low",
        "-b:a",
        "192k",
        "-f",
        "null",
        "-",
    ]
    options = calls[0][1]
    assert options["input"] == b"\0" * (1024 * 2 * 4)
    assert options["stdout"] == ffmpeg_audio_module.subprocess.DEVNULL
    assert options["stderr"] == ffmpeg_audio_module.subprocess.PIPE
    assert options["timeout"] == 10
    assert options["check"] is False


def test_audio_codec_preflight_canonicalizes_a_relative_executable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commands: list[list[str]] = []

    class Completed:
        returncode = 0
        stderr = b""

    def run(command: list[str], **kwargs: object) -> Completed:
        del kwargs
        commands.append(command)
        return Completed()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ffmpeg_audio_module.shutil, "which", lambda name: "bin/ffmpeg")
    monkeypatch.setattr(ffmpeg_audio_module.subprocess, "run", run)
    expected = str((tmp_path / "bin/ffmpeg").resolve())

    assert preflight_audio_codec(sample_rate=32_000, channels=2) == expected
    assert commands[0][0] == expected


def test_audio_codec_preflight_rejects_failed_aac_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Completed:
        returncode = 1
        stderr = b"Unknown encoder 'aac'"

    monkeypatch.setattr(
        ffmpeg_audio_module.shutil, "which", lambda name: "/host/ffmpeg"
    )
    monkeypatch.setattr(
        ffmpeg_audio_module.subprocess, "run", lambda *args, **kwargs: Completed()
    )

    with pytest.raises(RuntimeError, match="Unknown encoder 'aac'"):
        preflight_audio_codec(sample_rate=32_000, channels=2)


def test_audio_codec_preflight_rejects_a_missing_executable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ffmpeg_audio_module.shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match="ffmpeg executable on PATH"):
        preflight_audio_codec(sample_rate=32_000, channels=2)


def test_audio_codec_preflight_bounds_a_hung_external_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def hang(*args: object, **kwargs: object) -> None:
        raise ffmpeg_audio_module.subprocess.TimeoutExpired("ffmpeg", 10)

    monkeypatch.setattr(
        ffmpeg_audio_module.shutil, "which", lambda name: "/host/ffmpeg"
    )
    monkeypatch.setattr(ffmpeg_audio_module.subprocess, "run", hang)

    with pytest.raises(RuntimeError, match="Timed out"):
        preflight_audio_codec(sample_rate=32_000, channels=2)


def test_audio_codec_preflight_surfaces_an_external_spawn_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_spawn(*args: object, **kwargs: object) -> None:
        raise OSError("process table full")

    monkeypatch.setattr(
        ffmpeg_audio_module.shutil, "which", lambda name: "/host/ffmpeg"
    )
    monkeypatch.setattr(ffmpeg_audio_module.subprocess, "run", fail_spawn)

    with pytest.raises(RuntimeError, match="process table full"):
        preflight_audio_codec(sample_rate=32_000, channels=2)


def test_audio_muxer_builds_external_stream_copy_command(tmp_path: Path) -> None:
    muxer = Mp4AudioMuxer(
        video_path=tmp_path / "video.mp4",
        audio_path=tmp_path / "audio.f32le",
        output_path=tmp_path / "output.mp4",
        sample_rate=32_000,
        channels=2,
        ffmpeg_path="/usr/bin/ffmpeg",
        duration_seconds=5.0,
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
        "aac",
        "-profile:a",
        "aac_low",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(tmp_path / "output.mp4"),
    ]


def test_audio_muxer_surfaces_external_process_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Process:
        returncode = 7

        def communicate(self, timeout: float) -> tuple[None, bytes]:
            assert timeout == 30.0
            return None, b"codec failed"

    monkeypatch.setattr(ffmpeg_audio_module.shutil, "which", lambda name: f"/{name}")
    monkeypatch.setattr(
        ffmpeg_audio_module.subprocess, "Popen", lambda *args, **kw: Process()
    )
    muxer = Mp4AudioMuxer(
        video_path=tmp_path / "video.mp4",
        audio_path=tmp_path / "audio.f32le",
        output_path=tmp_path / "output.mp4",
        sample_rate=8_000,
        channels=1,
        ffmpeg_path="/ffmpeg",
        duration_seconds=5.0,
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

        def communicate(self, timeout: float) -> tuple[None, bytes]:
            calls.append(f"communicate({timeout:g})")
            return None, b""

    muxer = Mp4AudioMuxer(
        video_path=tmp_path / "video.mp4",
        audio_path=tmp_path / "audio.f32le",
        output_path=tmp_path / "output.mp4",
        sample_rate=8_000,
        channels=1,
        ffmpeg_path="/usr/bin/ffmpeg",
        duration_seconds=5.0,
    )
    muxer._process = Process()  # ty: ignore[invalid-assignment]

    muxer.abort()
    muxer.abort()

    assert calls == ["terminate", "communicate(5)"]


def test_audio_muxer_retains_process_when_abort_wait_fails(tmp_path: Path) -> None:
    """A second abort can retry a child whose first wait was interrupted."""
    calls: list[str] = []

    class Process:
        communicate_calls = 0

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            calls.append("terminate")

        def communicate(self, timeout: float) -> tuple[None, bytes]:
            calls.append(f"communicate({timeout:g})")
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
        ffmpeg_path="/usr/bin/ffmpeg",
        duration_seconds=5.0,
    )
    muxer._process = process  # ty: ignore[invalid-assignment]

    with pytest.raises(RuntimeError, match="mux wait interrupted"):
        muxer.abort()
    assert muxer._process is process

    muxer.abort()

    assert muxer._process is None
    assert calls == [
        "terminate",
        "communicate(5)",
        "terminate",
        "communicate(5)",
    ]


def test_audio_muxer_timeout_escalates_to_kill_and_reaps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    class Process:
        returncode = None
        communicate_calls = 0

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            calls.append("terminate")

        def kill(self) -> None:
            calls.append("kill")

        def communicate(self, timeout: float) -> tuple[None, bytes]:
            calls.append(f"communicate({timeout:g})")
            self.communicate_calls += 1
            if self.communicate_calls < 3:
                raise ffmpeg_audio_module.subprocess.TimeoutExpired("ffmpeg", timeout)
            return None, b""

    process = Process()
    monkeypatch.setattr(
        ffmpeg_audio_module.subprocess, "Popen", lambda *args, **kwargs: process
    )
    muxer = Mp4AudioMuxer(
        video_path=tmp_path / "video.mp4",
        audio_path=tmp_path / "audio.f32le",
        output_path=tmp_path / "output.mp4",
        sample_rate=8_000,
        channels=1,
        ffmpeg_path="/usr/bin/ffmpeg",
        duration_seconds=20.0,
    )

    with pytest.raises(RuntimeError, match="Timed out after 40 seconds"):
        muxer.close()

    assert muxer._process is None
    assert calls == [
        "communicate(40)",
        "terminate",
        "communicate(5)",
        "kill",
        "communicate(5)",
    ]


def test_audio_muxer_timeout_retains_unreaped_child_for_abort_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    class Process:
        returncode = None
        allow_reap = False

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            calls.append("terminate")

        def kill(self) -> None:
            calls.append("kill")

        def communicate(self, timeout: float) -> tuple[None, bytes]:
            calls.append(f"communicate({timeout:g})")
            if not self.allow_reap:
                raise ffmpeg_audio_module.subprocess.TimeoutExpired("ffmpeg", timeout)
            return None, b""

    process = Process()
    monkeypatch.setattr(
        ffmpeg_audio_module.subprocess, "Popen", lambda *args, **kwargs: process
    )
    muxer = Mp4AudioMuxer(
        video_path=tmp_path / "video.mp4",
        audio_path=tmp_path / "audio.f32le",
        output_path=tmp_path / "output.mp4",
        sample_rate=8_000,
        channels=1,
        ffmpeg_path="/usr/bin/ffmpeg",
        duration_seconds=5.0,
    )

    with pytest.raises(RuntimeError, match="Timed out after 30 seconds"):
        muxer.close()

    assert muxer._process is process
    process.allow_reap = True
    muxer.abort()
    assert muxer._process is None
    assert calls == [
        "communicate(30)",
        "terminate",
        "communicate(5)",
        "kill",
        "communicate(5)",
        "terminate",
        "communicate(5)",
    ]


@pytest.mark.parametrize("duration", [-1.0, float("nan"), float("inf")])
def test_audio_muxer_rejects_invalid_duration(tmp_path: Path, duration: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        Mp4AudioMuxer(
            video_path=tmp_path / "video.mp4",
            audio_path=tmp_path / "audio.f32le",
            output_path=tmp_path / "output.mp4",
            sample_rate=8_000,
            channels=1,
            ffmpeg_path="/usr/bin/ffmpeg",
            duration_seconds=duration,
        )
