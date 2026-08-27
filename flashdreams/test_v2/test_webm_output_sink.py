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

"""CPU tests for transactional native WebM output."""

from pathlib import Path
from types import SimpleNamespace

import flashdreams.runtime_v2.webm_output_sink as webm_sink_module
import numpy as np
import pytest
import torch
from flashdreams.runtime_v2.audio_output import AudioOutput
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout
from flashdreams.runtime_v2.webm_output_sink import WebmOutputSink

pytestmark = pytest.mark.ci_cpu

_WIDTH = 16
_HEIGHT = 8


class _FakeWriter:
    """Capture native arguments and create a recognizable staged WebM."""

    def __init__(
        self,
        path: str | Path,
        width: int,
        height: int,
        frames_per_second: int,
        codec: str,
        audio_sample_rate: int,
        audio_channels: int,
        *,
        fail_close: bool = False,
        fail_abort_once: bool = False,
    ) -> None:
        self.path = Path(path)
        self.width = width
        self.height = height
        self.frames_per_second = frames_per_second
        self.codec = codec
        self.audio_sample_rate = audio_sample_rate
        self.audio_channels = audio_channels
        self.fail_close = fail_close
        self.fail_abort_once = fail_abort_once
        self.frames_written = 0
        self.abort_calls = 0
        self.audio_bytes: bytes | None = None

    def write_video(self, frames: np.ndarray) -> None:
        """Record the submitted frames and leave an incomplete staged file."""
        self.frames_written += len(frames)
        self.path.write_bytes(b"partial WebM")

    def close(self, audio_path: str | Path | None = None) -> None:
        """Capture normalized PCM and emulate native container finalization."""
        if audio_path is not None:
            self.audio_bytes = Path(audio_path).read_bytes()
        self.path.write_bytes(b"partial finalized WebM")
        if self.fail_close:
            raise RuntimeError("native close failed")
        self.path.write_bytes(b"complete WebM")

    def abort(self) -> None:
        """Emulate a native cleanup that can retain ownership once."""
        self.abort_calls += 1
        if self.fail_abort_once and self.abort_calls == 1:
            raise RuntimeError("native abort interrupted")


def _install_backend(
    monkeypatch: pytest.MonkeyPatch,
    *,
    codec: str = "vp9",
    fail_close: bool = False,
    fail_abort_once: bool = False,
) -> list[_FakeWriter]:
    """Install a fake optional companion and return its created writers."""
    writers: list[_FakeWriter] = []

    def create_writer(
        path: str | Path,
        width: int,
        height: int,
        frames_per_second: int,
        selected_codec: str,
        audio_sample_rate: int,
        audio_channels: int,
    ) -> _FakeWriter:
        writer = _FakeWriter(
            path,
            width,
            height,
            frames_per_second,
            selected_codec,
            audio_sample_rate,
            audio_channels,
            fail_close=fail_close,
            fail_abort_once=fail_abort_once,
        )
        writers.append(writer)
        return writer

    backend = SimpleNamespace(
        WebmWriter=create_writer,
        select_video_codec=lambda: codec,
    )
    monkeypatch.setattr(webm_sink_module, "_load_webm_backend", lambda: backend)
    return writers


def _session_desc(
    *,
    audio: bool = False,
    frames_per_second: int = 24,
    audio_sample_rate: int = 48_000,
) -> SessionDesc:
    """Return the small deterministic session used by sink tests."""
    return SessionDesc(
        output_layout=VideoTensorLayout.tchw,
        frames_per_second_for_ui=frames_per_second,
        frames_per_second_for_step=frames_per_second,
        video_width=_WIDTH,
        video_height=_HEIGHT,
        audio_sample_rate=audio_sample_rate if audio else None,
        audio_channels=1 if audio else None,
    )


def _result(
    frame_count: int,
    *,
    audio: AudioOutput | None = None,
) -> StepResult:
    """Return black floating-point frames with optional normalized PCM."""
    return StepResult(
        step_index=0,
        output=torch.zeros((frame_count, 3, _HEIGHT, _WIDTH)),
        frame_count=frame_count,
        output_layout=VideoTensorLayout.tchw,
        audio=audio,
    )


def _staging_paths(path: Path) -> list[Path]:
    """Return private sibling transactions belonging to one target."""
    return list(path.parent.glob(f".{path.name}.*"))


def test_missing_companion_has_an_installation_focused_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing_companion(name: str) -> None:
        assert name == "flashdreams_webm"
        raise ImportError("not installed")

    monkeypatch.setattr(webm_sink_module.importlib, "import_module", missing_companion)

    with pytest.raises(RuntimeError, match=r"pip install 'flashdreams\[webm\]'"):
        WebmOutputSink(tmp_path / "clip.webm")


def test_selected_codec_and_declared_format_reach_the_native_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writers = _install_backend(monkeypatch, codec="vp8")
    sink = WebmOutputSink(tmp_path / "clip.webm")

    sink.open(_session_desc(audio=True, audio_sample_rate=24_000))

    assert sink.codec == "vp8"
    assert (
        writers[0].width,
        writers[0].height,
        writers[0].frames_per_second,
        writers[0].codec,
        writers[0].audio_sample_rate,
        writers[0].audio_channels,
    ) == (_WIDTH, _HEIGHT, 24, "vp8", 24_000, 1)
    sink.abort()


def test_target_is_replaced_only_after_successful_native_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writers = _install_backend(monkeypatch)
    path = tmp_path / "clip.webm"
    path.write_bytes(b"existing target")
    sink = WebmOutputSink(path)
    sink.open(_session_desc())
    sink.write(_result(2))

    assert path.read_bytes() == b"existing target"
    assert len(_staging_paths(path)) == 1

    sink.close()
    sink.close()
    sink.abort()

    assert path.read_bytes() == b"complete WebM"
    assert _staging_paths(path) == []
    assert writers[0].abort_calls == 0


def test_native_close_failure_preserves_target_and_can_be_aborted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writers = _install_backend(monkeypatch, fail_close=True)
    path = tmp_path / "clip.webm"
    path.write_bytes(b"existing target")
    sink = WebmOutputSink(path)
    sink.open(_session_desc())
    sink.write(_result(1))

    with pytest.raises(RuntimeError, match="native close failed"):
        sink.close()
    sink.abort()

    assert path.read_bytes() == b"existing target"
    assert _staging_paths(path) == []
    assert writers[0].abort_calls == 1


def test_failed_native_abort_retains_staging_for_an_explicit_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writers = _install_backend(monkeypatch, fail_abort_once=True)
    path = tmp_path / "clip.webm"
    path.write_bytes(b"existing target")
    sink = WebmOutputSink(path)
    sink.open(_session_desc())
    sink.write(_result(1))

    with pytest.raises(RuntimeError, match="native abort interrupted"):
        sink.abort()

    assert sink._writer is writers[0]
    assert path.read_bytes() == b"existing target"
    assert len(_staging_paths(path)) == 1

    sink.abort()

    assert sink._writer is None
    assert path.read_bytes() == b"existing target"
    assert _staging_paths(path) == []
    assert writers[0].abort_calls == 2


def test_audio_is_padded_to_the_written_video_timeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writers = _install_backend(monkeypatch)
    path = tmp_path / "clip.webm"
    sink = WebmOutputSink(path)
    audio = AudioOutput(
        samples=torch.full((1, 100), 0.25),
        sample_rate=24_000,
    )

    sink.open(
        _session_desc(
            audio=True,
            frames_per_second=24,
            audio_sample_rate=24_000,
        )
    )
    sink.write(_result(2, audio=audio))
    sink.close()

    expected_samples = 2_000
    audio_bytes = writers[0].audio_bytes
    assert audio_bytes is not None
    assert len(audio_bytes) == expected_samples * 4
    assert audio_bytes[-(expected_samples - 100) * 4 :] == (
        b"\0" * ((expected_samples - 100) * 4)
    )
    assert path.read_bytes() == b"complete WebM"


def test_empty_session_preserves_target_and_discards_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writers = _install_backend(monkeypatch)
    path = tmp_path / "clip.webm"
    path.write_bytes(b"existing target")
    sink = WebmOutputSink(path)
    sink.open(_session_desc())

    sink.close()

    assert path.read_bytes() == b"existing target"
    assert _staging_paths(path) == []
    assert writers[0].abort_calls == 1


def test_atomic_replace_failure_preserves_target_and_can_be_aborted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_backend(monkeypatch)
    path = tmp_path / "clip.webm"
    path.write_bytes(b"existing target")
    sink = WebmOutputSink(path)
    sink.open(_session_desc())
    sink.write(_result(1))

    def fail_replace(source: Path, target: Path) -> None:
        del source, target
        raise OSError("replace failed")

    monkeypatch.setattr(webm_sink_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        sink.close()
    sink.abort()

    assert path.read_bytes() == b"existing target"
    assert _staging_paths(path) == []
