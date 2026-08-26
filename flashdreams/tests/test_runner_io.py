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

"""CPU tests for shared runner I/O helpers."""

from __future__ import annotations

import io
import shutil
import sys
import threading
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import flashdreams.core.io.media_input as media_input
import flashdreams.infra.runner_io as runner_io
from flashdreams.core.io.media_input import (
    read_audio_f32,
    read_optional_audio_f32,
    read_video_fps,
    read_video_rgb_with_fps,
)
from flashdreams.infra.runner_io import (
    ensure_output_dir,
    load_first_frame_tensor,
    read_first_frame_rgb,
    resolve_input_path,
    resolve_prompt_value,
    runner_artifact_path,
    runner_stats_path,
    video_tensor_to_uint8,
    write_runner_stats,
    write_video_tensor,
)

pytestmark = pytest.mark.ci_cpu


def test_runner_io_keeps_host_media_reader_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep V1 runner imports while V2 applications use the core I/O boundary."""
    assert runner_io.read_audio_f32 is read_audio_f32
    assert runner_io.read_optional_audio_f32 is read_optional_audio_f32
    assert runner_io.read_video_rgb_with_fps is read_video_rgb_with_fps
    monkeypatch.setattr(runner_io, "_read_host_video_fps", lambda _path: 24.0)
    assert runner_io.read_video_fps("clip.mp4", install_hint="unused") == 24.0


def test_resolve_prompt_value_reads_first_non_empty_line(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("\n  first prompt  \nsecond prompt\n")

    assert resolve_prompt_value("inline") == "inline"
    assert resolve_prompt_value(prompt_path) == "first prompt"


def test_resolve_prompt_value_rejects_empty_values(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("\n  \n")

    with pytest.raises(ValueError, match="has no non-empty lines"):
        resolve_prompt_value(prompt_path)
    with pytest.raises(ValueError, match="must be a non-empty string"):
        resolve_prompt_value("")


def test_resolve_input_path_passes_local_paths_through(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"

    assert resolve_input_path(tmp_path / "frame.png", cache_dir=cache_dir) == (
        tmp_path / "frame.png"
    )
    assert resolve_input_path("relative.mp4", cache_dir=cache_dir) == Path(
        "relative.mp4"
    )


def test_resolve_input_path_downloads_urls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, Path, str | None, runner_io.InputAssetValidator | None]] = []

    def validator(path: Path) -> object:
        return path

    def fake_download_to_cache(
        url: str,
        *,
        cache_dir: Path,
        filename: str | None = None,
        validator: runner_io.InputAssetValidator | None = None,
    ) -> Path:
        calls.append((url, cache_dir, filename, validator))
        return cache_dir / (filename or "asset.bin")

    monkeypatch.setattr(runner_io, "_download_to_cache", fake_download_to_cache)

    resolved = resolve_input_path(
        "https://example.test/asset.png",
        cache_dir=tmp_path / "cache",
        filename="input.png",
        validator=validator,
    )

    assert resolved == tmp_path / "cache" / "input.png"
    assert calls == [
        ("https://example.test/asset.png", tmp_path / "cache", "input.png", validator)
    ]


def test_runner_artifact_and_stats_paths(tmp_path: Path) -> None:
    output_dir = ensure_output_dir(tmp_path / "nested")

    assert output_dir.is_dir()
    assert runner_artifact_path(output_dir, "demo-runner", "mp4") == (
        output_dir / "demo-runner.mp4"
    )
    assert runner_artifact_path(output_dir, "demo-runner", ".mp4") == (
        output_dir / "demo-runner.mp4"
    )
    assert runner_stats_path(output_dir, "demo-runner") == (
        output_dir / "stats_demo-runner.json"
    )


def test_write_runner_stats_matches_existing_json_format(tmp_path: Path) -> None:
    stats_path = write_runner_stats(
        tmp_path, "demo", [{"autoregressive_index": 0, "total_ms": 12.5}]
    )

    assert stats_path == tmp_path / "stats_demo.json"
    assert stats_path.read_text() == (
        '[\n  {\n    "autoregressive_index": 0,\n    "total_ms": 12.5\n  }\n]'
    )


def test_video_tensor_to_uint8_converts_tchw_layout() -> None:
    video = torch.tensor(
        [
            [
                [[-1.0, 0.0], [1.0, 2.0]],
                [[-2.0, 0.5], [0.0, 1.0]],
                [[1.0, -1.0], [0.0, 0.0]],
            ],
        ],
        dtype=torch.float32,
    )

    frames = video_tensor_to_uint8(video, layout="tchw")

    assert frames.dtype == np.uint8
    assert frames.shape == (1, 2, 2, 3)
    np.testing.assert_array_equal(
        frames,
        np.array(
            [[[[0, 0, 255], [127, 191, 0]], [[255, 127, 127], [255, 255, 127]]]],
            dtype=np.uint8,
        ),
    )


def test_video_tensor_to_uint8_converts_thwc_layout() -> None:
    video = torch.tensor(
        [
            [
                [[-1.0, 0.0, 1.0], [0.5, -0.5, 0.0]],
                [[1.0, 1.0, -1.0], [2.0, -2.0, 0.0]],
            ],
        ],
        dtype=torch.float32,
    )

    frames = video_tensor_to_uint8(video, layout="thwc")

    assert frames.dtype == np.uint8
    assert frames.shape == (1, 2, 2, 3)
    np.testing.assert_array_equal(
        frames,
        np.array(
            [[[[0, 127, 255], [191, 63, 127]], [[255, 255, 0], [255, 0, 127]]]],
            dtype=np.uint8,
        ),
    )


def test_video_tensor_to_uint8_converts_bcthw_layout() -> None:
    video = torch.full((1, 3, 2, 4, 5), -1.0, dtype=torch.float32)

    frames = video_tensor_to_uint8(video, layout="bcthw")

    assert frames.shape == (2, 4, 5, 3)
    assert frames.dtype == np.uint8
    assert frames.max() == 0


def test_video_tensor_to_uint8_converts_btchw_layout() -> None:
    video = torch.full((1, 2, 3, 4, 5), 1.0, dtype=torch.float32)

    frames = video_tensor_to_uint8(video, layout="btchw")

    assert frames.shape == (2, 4, 5, 3)
    assert frames.dtype == np.uint8
    assert frames.min() == 255


def test_video_tensor_to_uint8_rejects_multi_batch_bcthw() -> None:
    video = torch.zeros((2, 3, 1, 4, 5), dtype=torch.float32)

    with pytest.raises(ValueError, match="expects a single batch element"):
        video_tensor_to_uint8(video, layout="bcthw")


def test_video_tensor_to_uint8_rejects_multi_batch_btchw() -> None:
    video = torch.zeros((2, 1, 3, 4, 5), dtype=torch.float32)

    with pytest.raises(ValueError, match="expects a single batch element"):
        video_tensor_to_uint8(video, layout="btchw")


def test_read_first_frame_rgb_rejects_empty_video(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_media = types.ModuleType("mediapy")

    def read_video(path: str) -> np.ndarray:
        return np.empty((0, 2, 2, 3), dtype=np.uint8)

    setattr(fake_media, "read_video", read_video)
    monkeypatch.setitem(sys.modules, "mediapy", fake_media)

    with pytest.raises(ValueError, match="video has no frames"):
        read_first_frame_rgb(Path("empty.mp4"))


def test_read_video_fps_uses_host_ffprobe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def run(command: list[str], **kwargs: Any) -> Any:
        commands.append(command)
        assert kwargs == {"capture_output": True, "check": False}
        return types.SimpleNamespace(
            returncode=0,
            stdout=b'{"streams":[{"width":16,"height":8,"avg_frame_rate":"24000/1001"}]}',
            stderr=b"",
        )

    monkeypatch.setattr(media_input, "_find_ffprobe_binary", lambda: "/host/ffprobe")
    monkeypatch.setattr(media_input.subprocess, "run", run)

    assert read_video_fps(Path("clip.mp4")) == pytest.approx(24_000 / 1_001)
    assert commands[0][0] == "/host/ffprobe"
    assert commands[0][-1] == "clip.mp4"


def test_read_video_with_fps_uses_host_executables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []
    frames = np.arange(2 * 8 * 16 * 3, dtype=np.uint8).reshape(2, 8, 16, 3)

    def run(command: list[str], **kwargs: Any) -> Any:
        commands.append(command)
        assert kwargs == {"capture_output": True, "check": False}
        if command[0] == "/host/ffprobe":
            stdout = b'{"streams":[{"width":16,"height":8,"avg_frame_rate":"24/1"}]}'
        else:
            stdout = frames.tobytes()
        return types.SimpleNamespace(returncode=0, stdout=stdout, stderr=b"")

    monkeypatch.setattr(media_input, "_find_ffprobe_binary", lambda: "/host/ffprobe")
    monkeypatch.setattr(media_input, "_find_ffmpeg_binary", lambda: "/host/ffmpeg")
    monkeypatch.setattr(media_input.subprocess, "run", run)

    decoded, fps = read_video_rgb_with_fps(Path("clip.mp4"))

    assert fps == 24.0
    assert np.array_equal(decoded, frames)
    assert [command[0] for command in commands] == ["/host/ffprobe", "/host/ffmpeg"]


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="real video decode needs host ffmpeg and ffprobe on PATH",
)
def test_video_round_trip_uses_real_host_executables(tmp_path: Path) -> None:
    path = tmp_path / "clip.mp4"
    video = torch.stack(
        (
            torch.full((3, 8, 16), -1.0),
            torch.full((3, 8, 16), 1.0),
        )
    )

    write_video_tensor(video, path, fps=24, layout="tchw")
    decoded, fps = read_video_rgb_with_fps(path)

    assert decoded.shape == (2, 8, 16, 3)
    assert decoded.dtype == np.uint8
    assert fps == 24.0
    assert decoded[0].mean() < 10
    assert decoded[1].mean() > 245


def test_read_audio_f32_uses_host_ffmpeg_and_deinterleaves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []
    expected = np.array([[0.25, 0.5], [-0.25, -0.5]], dtype=np.float32)
    stdout = expected.T.astype("<f4").tobytes()

    def run(command: list[str], **kwargs: Any) -> Any:
        commands.append(command)
        assert kwargs == {"capture_output": True, "check": False}
        return types.SimpleNamespace(returncode=0, stdout=stdout, stderr=b"")

    monkeypatch.setattr(media_input, "_find_ffmpeg_binary", lambda: "/host/ffmpeg")
    monkeypatch.setattr(media_input.subprocess, "run", run)

    actual = read_audio_f32("reference.mp4", sample_rate=32000, channels=2)

    assert actual.flags.c_contiguous
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)
    assert commands == [
        [
            "/host/ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "reference.mp4",
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "2",
            "-ar",
            "32000",
            "-f",
            "f32le",
            "-c:a",
            "pcm_f32le",
            "pipe:1",
        ]
    ]


def test_read_audio_f32_surfaces_host_decode_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(media_input, "_find_ffmpeg_binary", lambda: "ffmpeg")
    monkeypatch.setattr(
        media_input.subprocess,
        "run",
        lambda *_args, **_kwargs: types.SimpleNamespace(
            returncode=1,
            stdout=b"",
            stderr=b"Stream map '0:a:0' matches no streams",
        ),
    )
    with pytest.raises(RuntimeError, match="matches no streams"):
        read_audio_f32("silent.mp4", sample_rate=32000)


@pytest.mark.parametrize(
    ("probe_stdout", "expected"),
    [(b"", None), (b"1\n", np.array([[0.25], [-0.25]], dtype=np.float32))],
)
def test_read_optional_audio_f32_distinguishes_absent_stream(
    monkeypatch: pytest.MonkeyPatch,
    probe_stdout: bytes,
    expected: np.ndarray | None,
) -> None:
    """Probe absence explicitly and decode only a present audio stream."""
    commands: list[list[str]] = []

    def run(command: list[str], **kwargs: Any) -> Any:
        commands.append(command)
        assert kwargs == {"capture_output": True, "check": False}
        if command[0] == "/host/ffprobe":
            return types.SimpleNamespace(returncode=0, stdout=probe_stdout, stderr=b"")
        stdout = np.array([0.25, -0.25], dtype="<f4").tobytes()
        return types.SimpleNamespace(returncode=0, stdout=stdout, stderr=b"")

    monkeypatch.setattr(media_input, "_find_ffprobe_binary", lambda: "/host/ffprobe")
    monkeypatch.setattr(media_input, "_find_ffmpeg_binary", lambda: "/host/ffmpeg")
    monkeypatch.setattr(media_input.subprocess, "run", run)

    actual = read_optional_audio_f32("reference.mp4", sample_rate=32000, channels=2)

    if expected is None:
        assert actual is None
        assert len(commands) == 1
    else:
        assert actual is not None
        np.testing.assert_array_equal(actual, expected)
        assert len(commands) == 2
    assert commands[0] == [
        "/host/ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        "reference.mp4",
    ]


def test_read_optional_audio_f32_surfaces_probe_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not misreport unreadable input as a video without audio."""
    monkeypatch.setattr(media_input, "_find_ffprobe_binary", lambda: "ffprobe")
    monkeypatch.setattr(
        media_input.subprocess,
        "run",
        lambda *_args, **_kwargs: types.SimpleNamespace(
            returncode=1,
            stdout=b"",
            stderr=b"invalid data",
        ),
    )

    with pytest.raises(RuntimeError, match="invalid data"):
        read_optional_audio_f32("broken.mp4", sample_rate=32000)


@pytest.mark.parametrize(
    ("sample_rate", "channels", "message"),
    [
        (True, 2, "sample_rate"),
        (0, 2, "sample_rate"),
        (32000, True, "channels"),
        (32000, 3, "channels"),
    ],
)
def test_read_audio_f32_rejects_invalid_format(
    sample_rate: int,
    channels: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        read_audio_f32("reference.wav", sample_rate=sample_rate, channels=channels)


@pytest.mark.parametrize(
    ("stdout", "message"),
    [
        (b"", "empty"),
        (b"\x00", "truncated"),
        (np.array([0.0], dtype="<f4").tobytes(), "not divisible"),
        (
            np.array([0.0, float("nan")], dtype="<f4").tobytes(),
            "non-finite",
        ),
    ],
)
def test_read_audio_f32_rejects_invalid_decoded_samples(
    monkeypatch: pytest.MonkeyPatch,
    stdout: bytes,
    message: str,
) -> None:
    monkeypatch.setattr(media_input, "_find_ffmpeg_binary", lambda: "ffmpeg")
    monkeypatch.setattr(
        media_input.subprocess,
        "run",
        lambda *_args, **_kwargs: types.SimpleNamespace(
            returncode=0, stdout=stdout, stderr=b""
        ),
    )
    with pytest.raises(ValueError, match=message):
        read_audio_f32("broken.wav", sample_rate=32000, channels=2)


def test_write_video_tensor_streams_to_host_ffmpeg(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    commands: list[list[str]] = []
    process = types.SimpleNamespace(
        stdin=io.BytesIO(),
        stderr=io.BytesIO(),
        wait=lambda: 0,
    )

    def popen(cmd: list[str], **_kwargs: Any) -> Any:
        commands.append(cmd)
        return process

    monkeypatch.setattr(runner_io, "_find_ffmpeg_binary", lambda: "ffmpeg")
    monkeypatch.setattr(runner_io.subprocess, "Popen", popen)

    out_path = tmp_path / "out.mp4"
    returned = write_video_tensor(
        torch.zeros((1, 3, 2, 2), dtype=torch.float32),
        out_path,
        fps=16,
        layout="tchw",
    )

    assert returned == out_path
    assert commands[0][commands[0].index("-s") + 1] == "2x2"
    assert commands[0][commands[0].index("-r") + 1] == "16"
    assert commands[0][-1] == str(out_path)


def test_write_video_tensor_drains_ffmpeg_stderr_while_streaming(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    drain_started = threading.Event()

    class RecordingStderr:
        def __init__(self) -> None:
            self._chunks = [b"ffmpeg diagnostic", b""]

        def read(self, _size: int = -1) -> bytes:
            drain_started.set()
            return self._chunks.pop(0)

    class GuardedStdin(io.BytesIO):
        def write(self, data: bytes) -> int:
            assert drain_started.wait(timeout=1.0)
            return super().write(data)

    process = types.SimpleNamespace(
        stdin=GuardedStdin(),
        stderr=RecordingStderr(),
        wait=lambda: 0,
    )

    monkeypatch.setattr(runner_io, "_find_ffmpeg_binary", lambda: "ffmpeg")
    monkeypatch.setattr(
        runner_io.subprocess, "Popen", lambda *_args, **_kwargs: process
    )

    write_video_tensor(
        torch.zeros((2, 3, 2, 2), dtype=torch.float32),
        tmp_path / "out.mp4",
        fps=16,
        layout="tchw",
    )


def test_load_first_frame_tensor_uses_requested_resize_interpolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_media = types.ModuleType("mediapy")
    fake_cv2 = types.ModuleType("cv2")
    calls: dict[str, Any] = {}

    setattr(fake_cv2, "INTER_CUBIC", 2)
    setattr(fake_cv2, "INTER_NEAREST", 0)
    setattr(fake_cv2, "INTER_LINEAR", 1)
    setattr(fake_cv2, "INTER_AREA", 3)
    setattr(fake_cv2, "INTER_LANCZOS4", 4)

    def read_image(path: str) -> np.ndarray:
        calls["path"] = path
        return np.full((2, 3, 4), 127, dtype=np.uint8)

    def resize(image: np.ndarray, dsize: tuple[int, int], **kwargs: int) -> np.ndarray:
        calls["dsize"] = dsize
        calls["kwargs"] = kwargs
        width, height = dsize
        return np.full((height, width, 3), image[0, 0, 0], dtype=image.dtype)

    setattr(fake_media, "read_image", read_image)
    setattr(fake_cv2, "resize", resize)
    monkeypatch.setitem(sys.modules, "mediapy", fake_media)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    tensor = load_first_frame_tensor(
        Path("frame.png"),
        pixel_height=4,
        pixel_width=5,
        device=torch.device("cpu"),
        dtype=torch.float32,
        interpolation="cubic",
    )

    assert calls == {
        "path": "frame.png",
        "dsize": (5, 4),
        "kwargs": {"interpolation": 2},
    }
    assert tensor.shape == (1, 3, 4, 5)
    assert tensor.dtype == torch.float32
    assert torch.allclose(tensor, torch.full_like(tensor, 127.0 / 127.5 - 1.0))
