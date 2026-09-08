# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the shared streaming video writer used by OmniDreams."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest
import torch

import flashdreams.infra.runner_io as runner_module

pytestmark = pytest.mark.ci_cpu


class _RecordingStdin:
    def __init__(self, *, fail_write: bool = False, fail_close: bool = False) -> None:
        self.fail_write = fail_write
        self.fail_close = fail_close
        self.writes: list[bytes] = []
        self.closed = False

    def write(self, data: bytes) -> None:
        if self.fail_write:
            raise BrokenPipeError("write failed")
        self.writes.append(data)

    def close(self) -> None:
        self.closed = True
        if self.fail_close:
            raise BrokenPipeError("close failed")


class _FakeProcess:
    def __init__(
        self,
        *,
        returncode: int = 0,
        stderr: bytes = b"",
        fail_write: bool = False,
        fail_close: bool = False,
    ) -> None:
        self.stdin = _RecordingStdin(
            fail_write=fail_write,
            fail_close=fail_close,
        )
        self.stderr = io.BytesIO(stderr)
        self.returncode = returncode
        self.wait_calls = 0

    def wait(self) -> int:
        self.wait_calls += 1
        return self.returncode


def _canvas(*, height: int = 3, width: int = 5) -> torch.Tensor:
    return torch.zeros(1, height, width, 3)


def test_find_ffmpeg_binary_fails_loudly_when_host_binary_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner_module.shutil, "which", lambda _name: None)

    with pytest.raises(RuntimeError, match="installed on the host.*PATH"):
        runner_module._find_ffmpeg_binary()


def test_write_video_pads_odd_dimensions_for_yuv420p(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    process = _FakeProcess()
    commands: list[list[str]] = []

    def _popen(cmd: list[str], **_kwargs: Any) -> _FakeProcess:
        commands.append(cmd)
        return process

    monkeypatch.setattr(runner_module, "_find_ffmpeg_binary", lambda: "ffmpeg")
    monkeypatch.setattr(runner_module.subprocess, "Popen", _popen)

    runner_module.write_video_tensor(
        _canvas(), tmp_path / "odd.mp4", fps=24, layout="thwc"
    )

    assert commands[0][commands[0].index("-vf") + 1] == (
        "pad=ceil(iw/2)*2:ceil(ih/2)*2"
    )
    assert commands[0][commands[0].index("-s") + 1] == "5x3"
    assert process.wait_calls == 1


def test_write_video_reaps_ffmpeg_and_preserves_broken_pipe_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    process = _FakeProcess(
        returncode=7,
        stderr=b"encoder rejected frame",
        fail_write=True,
        fail_close=True,
    )
    monkeypatch.setattr(runner_module, "_find_ffmpeg_binary", lambda: "ffmpeg")
    monkeypatch.setattr(
        runner_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )

    with pytest.raises(RuntimeError, match="exit code 7.*encoder rejected frame"):
        runner_module.write_video_tensor(
            _canvas(), tmp_path / "broken.mp4", fps=24, layout="thwc"
        )

    assert process.stdin.closed
    assert process.wait_calls == 1
