# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WebM duration and resize regression check without FFmpeg."""

import subprocess
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout
from flashdreams.runtime_v2.webm_client_window import WebmClientWindow

pytestmark = pytest.mark.ci_cpu


def _result(value: int = 120, count: int = 1) -> StepResult:
    return StepResult(
        step_index=0,
        output=torch.full((count, 3, 72, 80), value, dtype=torch.uint8),
        frame_count=count,
        output_layout=VideoTensorLayout.tchw,
    )


def test_webm_encoder_is_silent(
    tmp_path: Path,
    capfd: pytest.CaptureFixture[str],
) -> None:
    from flashdreams_webm import Writer

    writer = Writer(str(tmp_path / "silent.webm"), 64, 64, 24, 1)
    writer.write(b"\x10" * (64 * 64) + b"\x80" * (64 * 64 // 2))
    writer.close()

    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == ""


@pytest.mark.parametrize("native_rgb", [False, True])
def test_webm_recording(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native_rgb: bool
) -> None:
    """Include the final frame duration and preserve frames through canvas growth."""
    from flashdreams_webm import Reader, Writer

    def no_process(*args, **kwargs):
        raise AssertionError("WebM must not start an executable")

    monkeypatch.setattr(subprocess, "Popen", no_process)
    monkeypatch.setenv("PATH", "")
    for count, fps in [(1, 24), (96, 30000 / 1001)]:
        path = tmp_path / f"recording-{count}.webm"
        path.write_bytes(b"original")
        window = WebmClientWindow(path, native_rgb=native_rgb)
        window.open(
            SessionDesc(
                video_width=80,
                video_height=72,
                frames_per_second_for_step=cast(int, fps),
            )
        )
        for i in range(count):
            window.write(_result(value=40 + i))
        assert path.read_bytes() == b"original"
        window.close()
        reader = Reader(str(path))
        try:
            assert reader.duration_ns / 1e9 == pytest.approx(count / fps, abs=1e-6)
            decoded = 0
            while (planes := reader.read()) is not None:
                assert (
                    np.abs(
                        np.frombuffer(planes[0], dtype=np.uint8).astype(float)
                        - (16 + (40 + decoded) * 219 / 255)
                    ).mean()
                    < 3
                )
                decoded += 1
            assert decoded == count
        finally:
            reader.close()

    path = tmp_path / "resized.webm"
    window = WebmClientWindow(path, native_rgb=native_rgb)
    window.open(
        SessionDesc(video_width=80, video_height=72, frames_per_second_for_step=24)
    )
    sizes = [(19, 9), (95, 77), (21, 81)]
    for size in sizes:
        window.request_new_window_size(size)
        window.write(_result(220))
    window.close()
    reader = Reader(str(path))
    try:
        assert (reader.width, reader.height) == (95, 81)
        assert reader.duration_ns / 1e9 == pytest.approx(len(sizes) / 24, abs=1e-6)
        for width, height in sizes:
            planes = reader.read()
            assert planes is not None
            y = np.frombuffer(planes[0], dtype=np.uint8).reshape(81, 95)
            assert np.abs(y[:height, :width].astype(float) - 205).mean() < 5
            if width < 95:
                assert np.abs(y[:, width:].astype(float) - 16).mean() < 5
        assert reader.read() is None
    finally:
        reader.close()

    if native_rgb:
        path = tmp_path / "buffer.webm"
        writer = Writer(str(path), 80, 72, 24, 1)
        rgb = np.full((72, 80, 3), 120, dtype=np.uint8)
        with pytest.raises(ValueError, match="contiguous"):
            writer.write_rgb(rgb[:, ::2])
        with pytest.raises(ValueError, match="contiguous"):
            writer.write_rgb(rgb.astype(np.float32))
        writer.write_rgb(rgb)
        rgb.fill(0)
        writer.close()
        reader = Reader(str(path))
        try:
            planes = reader.read()
            assert planes is not None
            assert (
                np.abs(
                    np.frombuffer(planes[0], dtype=np.uint8).astype(float) - 119
                ).mean()
                < 3
            )
            assert reader.read() is None
        finally:
            reader.close()
        with pytest.raises(RuntimeError, match="closed"):
            writer.write_rgb(rgb)
