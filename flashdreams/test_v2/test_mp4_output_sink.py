# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU test for the output sink that writes an MP4 file."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import Tensor

import flashdreams.runtime_v2.mp4_output_sink as mp4_sink_module
import flashdreams.runtime_v2.video_encoder as video_encoder_module
from flashdreams.runtime_v2.audio_output import AudioOutput
from flashdreams.runtime_v2.mp4_output_sink import Mp4OutputSink
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.video_encoder import Mp4Encoder
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="encoding and reading an MP4 back needs ffmpeg on PATH",
)

_WIDTH = 16
"""Frame width. Not square, so a transposed frame cannot pass unnoticed."""

_HEIGHT = 8
"""Frame height."""

_RED = (1.0, -1.0, -1.0)
"""Full red, in the ``[-1, 1]`` range a floating point result carries."""

_BLACK = (-1.0, -1.0, -1.0)
"""Black in the same range."""


## Helpers


class _FakeEncoder:
    """Write a recognizable staged file without starting ffmpeg."""

    def __init__(
        self,
        path: str | Path,
        *,
        fail_write: bool = False,
        fail_close: bool = False,
    ) -> None:
        self.path = Path(path)
        self.fail_write = fail_write
        self.fail_close = fail_close
        self.abort_calls = 0
        self.frames_written = 0

    def write(self, frames: np.ndarray) -> None:
        self.frames_written += len(frames)
        self.path.write_bytes(b"partial video")
        if self.fail_write:
            raise RuntimeError("encode write failed")

    def close(self) -> None:
        if self.frames_written:
            self.path.write_bytes(b"complete video")
        if self.fail_close:
            raise RuntimeError("encode close failed")

    def abort(self) -> None:
        self.abort_calls += 1


class _FakeMuxer:
    """Capture staged PCM and create a recognizable synchronized file."""

    def __init__(
        self,
        *,
        video_path: str | Path,
        audio_path: str | Path,
        output_path: str | Path,
        fail_close: bool = False,
        **kwargs: object,
    ) -> None:
        del kwargs
        self.video_path = Path(video_path)
        self.audio_path = Path(audio_path)
        self.output_path = Path(output_path)
        self.fail_close = fail_close
        self.audio_bytes = b""
        self.abort_calls = 0

    def close(self) -> None:
        self.audio_bytes = self.audio_path.read_bytes()
        self.output_path.write_bytes(b"partial synchronized output")
        if self.fail_close:
            raise RuntimeError("mux close failed")
        self.output_path.write_bytes(b"complete synchronized output")

    def abort(self) -> None:
        self.abort_calls += 1


def _install_fake_encoder(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_write: bool = False,
    fail_close: bool = False,
) -> list[_FakeEncoder]:
    encoders: list[_FakeEncoder] = []

    def create_encoder(path: str | Path, **kwargs: object) -> _FakeEncoder:
        del kwargs
        encoder = _FakeEncoder(
            path,
            fail_write=fail_write,
            fail_close=fail_close,
        )
        encoders.append(encoder)
        return encoder

    monkeypatch.setattr(mp4_sink_module, "Mp4Encoder", create_encoder)
    return encoders


def _install_fake_muxer(
    monkeypatch: pytest.MonkeyPatch, *, fail_close: bool = False
) -> list[_FakeMuxer]:
    muxers: list[_FakeMuxer] = []

    def create_muxer(**kwargs: object) -> _FakeMuxer:
        muxer = _FakeMuxer(**kwargs, fail_close=fail_close)
        muxers.append(muxer)
        return muxer

    monkeypatch.setattr(mp4_sink_module, "Mp4AudioMuxer", create_muxer)
    return muxers


def _staging_paths(path: Path) -> list[Path]:
    return list(path.parent.glob(f".{path.name}.*"))


def _session_desc(
    layout: VideoTensorLayout = VideoTensorLayout.bcthw,
    *,
    width: int = _WIDTH,
    height: int = _HEIGHT,
    audio: bool = False,
    fps: int = 30,
    audio_sample_rate: int = 8_000,
) -> SessionDesc:
    return SessionDesc(
        output_layout=layout,
        frames_per_second_for_ui=fps,
        frames_per_second_for_step=fps,
        video_width=width,
        video_height=height,
        audio_sample_rate=audio_sample_rate if audio else None,
        audio_channels=2 if audio else None,
    )


def _in_layout(frames: Tensor, layout: VideoTensorLayout) -> Tensor:
    """Lay a ``[T, C, H, W]`` tensor out as ``layout`` says."""
    if layout is VideoTensorLayout.tchw:
        return frames
    if layout is VideoTensorLayout.btchw:
        return frames.unsqueeze(0)
    if layout is VideoTensorLayout.bcthw:
        return frames.permute(1, 0, 2, 3).unsqueeze(0)
    if layout is VideoTensorLayout.bvtchw:
        return frames.unsqueeze(0).unsqueeze(0)
    raise AssertionError(f"no test layout for {layout.value}")


def _result(
    colours: list[tuple[float, float, float]],
    *,
    step_index: int = 0,
    layout: VideoTensorLayout = VideoTensorLayout.bcthw,
    dtype: torch.dtype = torch.float32,
    audio: AudioOutput | None = None,
) -> StepResult:
    """Return a result of solid frames, one per colour."""
    frames = torch.zeros((len(colours), 3, _HEIGHT, _WIDTH), dtype=dtype)
    for index, colour in enumerate(colours):
        for channel, value in enumerate(colour):
            frames[index, channel] = value
    return StepResult(
        step_index=step_index,
        output=_in_layout(frames, layout),
        frame_count=len(colours),
        output_layout=layout,
        audio=audio,
    )


def _decode(path: Path) -> np.ndarray:
    """Read an MP4 back as ``[T, H, W, C]`` uint8 frames."""
    raw = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        check=True,
        capture_output=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, _HEIGHT, _WIDTH, 3)


def _mean_colour(frame: np.ndarray) -> tuple[float, float, float]:
    """Return one frame's mean red, green and blue.

    The frames written are solid and encoding is lossy, so a mean is what a
    colour is recognisable by rather than an exact value.
    """
    red, green, blue = (float(frame[:, :, channel].mean()) for channel in range(3))
    return red, green, blue


## Tests that do not encode


def test_target_is_replaced_only_after_successful_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    encoders = _install_fake_encoder(monkeypatch)
    path = tmp_path / "out.mp4"
    path.write_bytes(b"existing target")
    sink = Mp4OutputSink(path)

    sink.open(_session_desc())
    sink.write(_result([_RED]))

    assert path.read_bytes() == b"existing target"
    assert len(_staging_paths(path)) == 1

    sink.close()
    sink.close()
    sink.abort()

    assert path.read_bytes() == b"complete video"
    assert _staging_paths(path) == []
    assert encoders[0].abort_calls == 0


def test_abort_is_idempotent_and_preserves_existing_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    encoders = _install_fake_encoder(monkeypatch)
    path = tmp_path / "out.mp4"
    path.write_bytes(b"existing target")
    sink = Mp4OutputSink(path)
    sink.open(_session_desc())
    sink.write(_result([_RED]))

    sink.abort()
    sink.abort()

    assert path.read_bytes() == b"existing target"
    assert _staging_paths(path) == []
    assert encoders[0].abort_calls == 1


def test_write_failure_can_be_aborted_without_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_encoder(monkeypatch, fail_write=True)
    path = tmp_path / "out.mp4"
    path.write_bytes(b"existing target")
    sink = Mp4OutputSink(path)
    sink.open(_session_desc())

    with pytest.raises(RuntimeError, match="encode write failed"):
        sink.write(_result([_RED]))
    sink.abort()

    assert path.read_bytes() == b"existing target"
    assert _staging_paths(path) == []


def test_close_failure_can_be_aborted_without_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    encoders = _install_fake_encoder(monkeypatch, fail_close=True)
    path = tmp_path / "out.mp4"
    path.write_bytes(b"existing target")
    sink = Mp4OutputSink(path)
    sink.open(_session_desc())
    sink.write(_result([_RED]))

    with pytest.raises(RuntimeError, match="encode close failed"):
        sink.close()
    sink.abort()

    assert path.read_bytes() == b"existing target"
    assert _staging_paths(path) == []
    assert encoders[0].abort_calls == 1


def test_atomic_replace_failure_preserves_target_and_can_be_aborted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_encoder(monkeypatch)
    path = tmp_path / "out.mp4"
    path.write_bytes(b"existing target")
    sink = Mp4OutputSink(path)
    sink.open(_session_desc())
    sink.write(_result([_RED]))

    def fail_replace(source: Path, target: Path) -> None:
        del source, target
        raise OSError("replace failed")

    monkeypatch.setattr(mp4_sink_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        sink.close()
    sink.abort()

    assert path.read_bytes() == b"existing target"
    assert _staging_paths(path) == []


def test_encoder_abort_terminates_and_waits_for_ffmpeg(tmp_path: Path) -> None:
    calls: list[str] = []

    class Stream:
        def close(self) -> None:
            calls.append("stdin.close")

    class Process:
        stdin = Stream()

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            calls.append("process.terminate")

        def wait(self) -> int:
            calls.append("process.wait")
            return -15

    class Reader:
        def join(self) -> None:
            calls.append("reader.join")

    encoder = Mp4Encoder(
        tmp_path / "staged.mp4",
        width=_WIDTH,
        height=_HEIGHT,
        frames_per_second=30,
    )
    encoder._process = Process()  # type: ignore[assignment]
    encoder._error_reader = Reader()  # type: ignore[assignment]

    encoder.abort()
    encoder.abort()

    assert calls == [
        "process.terminate",
        "stdin.close",
        "process.wait",
        "reader.join",
    ]


def test_encoder_wait_failure_keeps_child_owned_until_abort(tmp_path: Path) -> None:
    calls: list[str] = []

    class Stream:
        def close(self) -> None:
            calls.append("stdin.close")

    class Process:
        stdin = Stream()
        waits = 0

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            calls.append("process.terminate")

        def wait(self) -> int:
            self.waits += 1
            calls.append(f"process.wait({self.waits})")
            if self.waits == 1:
                raise InterruptedError("wait interrupted")
            return -15

    class Reader:
        def join(self) -> None:
            calls.append("reader.join")

    encoder = Mp4Encoder(
        tmp_path / "staged.mp4",
        width=_WIDTH,
        height=_HEIGHT,
        frames_per_second=30,
    )
    encoder._process = Process()  # type: ignore[assignment]
    encoder._error_reader = Reader()  # type: ignore[assignment]

    with pytest.raises(InterruptedError, match="wait interrupted"):
        encoder.close()
    encoder.abort()

    assert calls == [
        "stdin.close",
        "process.wait(1)",
        "process.terminate",
        "stdin.close",
        "process.wait(2)",
        "reader.join",
    ]


def test_encoder_reader_start_failure_terminates_spawned_ffmpeg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    class Stream:
        def close(self) -> None:
            calls.append("stdin.close")

        def write(self, data: bytes) -> None:
            del data

    class Process:
        stdin = Stream()
        stderr = object()

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            calls.append("process.terminate")

        def wait(self) -> int:
            calls.append("process.wait")
            return -15

    class Reader:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def start(self) -> None:
            raise RuntimeError("reader start failed")

    monkeypatch.setattr(
        video_encoder_module.subprocess, "Popen", lambda *args, **kwargs: Process()
    )
    monkeypatch.setattr(video_encoder_module.threading, "Thread", Reader)
    encoder = Mp4Encoder(
        tmp_path / "staged.mp4",
        width=_WIDTH,
        height=_HEIGHT,
        frames_per_second=30,
    )

    with pytest.raises(RuntimeError, match="reader start failed"):
        encoder.write(np.zeros((1, _HEIGHT, _WIDTH, 3), dtype=np.uint8))

    assert calls == ["process.terminate", "stdin.close", "process.wait"]
    assert encoder._process is None


def test_write_before_open_raises(tmp_path: Path) -> None:
    sink = Mp4OutputSink(tmp_path / "out.mp4")

    with pytest.raises(RuntimeError, match="open"):
        sink.write(_result([_RED]))


@pytest.mark.parametrize(("width", "height"), [(15, 8), (16, 7), (15, 7)])
def test_open_rejects_odd_frame_dimensions(
    tmp_path: Path, width: int, height: int
) -> None:
    # Rounding up to the even size the encoding needs would write a file of a
    # size the session never declared.
    sink = Mp4OutputSink(tmp_path / "out.mp4")

    with pytest.raises(ValueError, match=f"{width}x{height}"):
        sink.open(_session_desc(width=width, height=height))


def test_open_rejects_audio_without_an_explicit_codec(tmp_path: Path) -> None:
    path = tmp_path / "out.mp4"
    sink = Mp4OutputSink(path)

    with pytest.raises(ValueError, match="explicitly selected audio codec"):
        sink.open(_session_desc(audio=True))

    assert not path.exists()


def test_audio_result_is_rejected_when_the_session_declared_none(
    tmp_path: Path,
) -> None:
    sink = Mp4OutputSink(tmp_path / "out.mp4")
    sink.open(_session_desc())
    audio = AudioOutput(samples=torch.zeros((2, 1)), sample_rate=8_000)

    with pytest.raises(ValueError, match="declared none"):
        sink.write(_result([_RED], audio=audio))
    sink.abort()


@pytest.mark.parametrize(
    ("audio", "message"),
    [
        (AudioOutput(samples=torch.zeros((2, 1)), sample_rate=16_000), "8000 Hz"),
        (
            AudioOutput(samples=torch.zeros((1, 1)), sample_rate=8_000),
            "2 audio channels",
        ),
    ],
)
def test_audio_result_must_match_the_declared_format(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    audio: AudioOutput,
    message: str,
) -> None:
    _install_fake_encoder(monkeypatch)
    sink = Mp4OutputSink(tmp_path / "out.mp4", audio_codec="reviewed-codec")
    sink.open(_session_desc(audio=True))

    with pytest.raises(ValueError, match=message):
        sink.write(_result([_RED], audio=audio))
    sink.abort()


@pytest.mark.parametrize(
    ("frame_count", "native_samples", "expected_samples"),
    [
        (124, 165_600, 165_333),
        (362, 482_400, 482_667),
    ],
)
def test_audio_is_aligned_to_the_written_video_timeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frame_count: int,
    native_samples: int,
    expected_samples: int,
) -> None:
    _install_fake_encoder(monkeypatch)
    muxers = _install_fake_muxer(monkeypatch)
    path = tmp_path / "out.mp4"
    sink = Mp4OutputSink(path, audio_codec="reviewed-codec")
    audio = AudioOutput(
        samples=torch.full((2, native_samples), 0.25),
        sample_rate=32_000,
    )

    sink.open(_session_desc(audio=True, fps=24, audio_sample_rate=32_000))
    sink.write(_result([_RED] * frame_count, audio=audio))
    sink.close()

    assert path.read_bytes() == b"complete synchronized output"
    assert len(muxers[0].audio_bytes) == expected_samples * 2 * 4
    if expected_samples > native_samples:
        assert muxers[0].audio_bytes[-(expected_samples - native_samples) * 8 :] == (
            b"\0" * ((expected_samples - native_samples) * 8)
        )
    assert _staging_paths(path) == []


def test_missing_audio_is_padded_with_timeline_silence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_encoder(monkeypatch)
    muxers = _install_fake_muxer(monkeypatch)
    path = tmp_path / "out.mp4"
    sink = Mp4OutputSink(path, audio_codec="reviewed-codec")

    sink.open(_session_desc(audio=True))
    sink.write(_result([_RED, _BLACK, _RED]))
    sink.close()

    expected_samples = round(3 * 8_000 / 30)
    assert muxers[0].audio_bytes == b"\0" * (expected_samples * 2 * 4)


def test_mux_failure_preserves_target_and_can_be_aborted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_encoder(monkeypatch)
    muxers = _install_fake_muxer(monkeypatch, fail_close=True)
    path = tmp_path / "out.mp4"
    path.write_bytes(b"existing target")
    sink = Mp4OutputSink(path, audio_codec="reviewed-codec")
    audio = AudioOutput(samples=torch.zeros((2, 100)), sample_rate=8_000)
    sink.open(_session_desc(audio=True))
    sink.write(_result([_RED], audio=audio))

    with pytest.raises(RuntimeError, match="mux close failed"):
        sink.close()
    sink.abort()

    assert path.read_bytes() == b"existing target"
    assert _staging_paths(path) == []
    assert muxers[0].abort_calls == 1


def test_write_rejects_a_layout_the_sink_was_not_opened_for(tmp_path: Path) -> None:
    sink = Mp4OutputSink(tmp_path / "out.mp4")
    sink.open(_session_desc(VideoTensorLayout.bcthw))

    with pytest.raises(ValueError, match="tchw"):
        sink.write(_result([_RED], layout=VideoTensorLayout.tchw))


def test_write_rejects_frames_of_another_size(tmp_path: Path) -> None:
    sink = Mp4OutputSink(tmp_path / "out.mp4")
    sink.open(_session_desc(width=_WIDTH * 2))

    with pytest.raises(ValueError, match=f"{_WIDTH}x{_HEIGHT}"):
        sink.write(_result([_RED]))


def test_write_rejects_a_frame_count_the_tensor_does_not_carry(tmp_path: Path) -> None:
    sink = Mp4OutputSink(tmp_path / "out.mp4")
    sink.open(_session_desc())
    carrying_two = _result([_RED, _BLACK])
    claiming_five = StepResult(
        step_index=0,
        output=carrying_two.output,
        frame_count=5,
        output_layout=carrying_two.output_layout,
    )

    with pytest.raises(ValueError, match="5 frames"):
        sink.write(claiming_five)


def test_write_rejects_output_with_more_than_one_batch(tmp_path: Path) -> None:
    sink = Mp4OutputSink(tmp_path / "out.mp4")
    sink.open(_session_desc())
    one = _result([_RED])
    two_batches = StepResult(
        step_index=0,
        output=torch.cat([one.output, one.output]),
        frame_count=1,
        output_layout=one.output_layout,
    )

    with pytest.raises(ValueError, match="batch"):
        sink.write(two_batches)


def test_a_run_that_generated_nothing_writes_no_file(tmp_path: Path) -> None:
    path = tmp_path / "out.mp4"
    sink = Mp4OutputSink(path)

    sink.open(_session_desc())
    sink.close()

    assert not path.exists()
    assert _staging_paths(path) == []


def test_empty_run_preserves_an_existing_target(tmp_path: Path) -> None:
    path = tmp_path / "out.mp4"
    path.write_bytes(b"existing target")
    sink = Mp4OutputSink(path)

    sink.open(_session_desc())
    sink.close()

    assert path.read_bytes() == b"existing target"
    assert _staging_paths(path) == []


def test_close_tolerates_a_sink_that_was_never_opened(tmp_path: Path) -> None:
    Mp4OutputSink(tmp_path / "out.mp4").close()


## Tests that encode


@needs_ffmpeg
@pytest.mark.parametrize(
    "layout",
    [
        VideoTensorLayout.tchw,
        VideoTensorLayout.btchw,
        VideoTensorLayout.bcthw,
        VideoTensorLayout.bvtchw,
    ],
)
def test_sink_writes_one_frame_per_result_in_order(
    tmp_path: Path, layout: VideoTensorLayout
) -> None:
    path = tmp_path / "out.mp4"
    sink = Mp4OutputSink(path)

    sink.open(_session_desc(layout))
    sink.write(_result([_RED], step_index=0, layout=layout))
    sink.write(_result([_BLACK], step_index=1, layout=layout))
    sink.close()

    frames = _decode(path)
    assert len(frames) == 2
    red, green, blue = _mean_colour(frames[0])
    assert red > 180
    assert green < 80
    assert blue < 80
    assert max(_mean_colour(frames[1])) < 40


@needs_ffmpeg
def test_sink_writes_every_frame_a_result_carries(tmp_path: Path) -> None:
    path = tmp_path / "out.mp4"
    sink = Mp4OutputSink(path)

    sink.open(_session_desc())
    sink.write(_result([_RED, _BLACK, _RED]))
    sink.close()

    assert len(_decode(path)) == 3


@needs_ffmpeg
def test_a_floating_point_result_is_read_as_minus_one_to_one(tmp_path: Path) -> None:
    # Zero is the middle of that range, so a frame of zeros is mid grey rather
    # than black, and an application emitting [0, 1] would come out washed out.
    path = tmp_path / "out.mp4"
    sink = Mp4OutputSink(path)

    sink.open(_session_desc())
    sink.write(_result([(0.0, 0.0, 0.0)]))
    sink.close()

    assert min(_mean_colour(_decode(path)[0])) > 100
    assert max(_mean_colour(_decode(path)[0])) < 155


@needs_ffmpeg
def test_an_integer_result_is_read_as_raw_bytes(tmp_path: Path) -> None:
    path = tmp_path / "out.mp4"
    sink = Mp4OutputSink(path)

    sink.open(_session_desc())
    sink.write(_result([(17.0, 17.0, 17.0)], dtype=torch.uint8))
    sink.close()

    assert max(_mean_colour(_decode(path)[0])) < 25


@needs_ffmpeg
def test_a_single_channel_result_is_written_as_grey(tmp_path: Path) -> None:
    path = tmp_path / "out.mp4"
    sink = Mp4OutputSink(path)
    grey = torch.zeros((1, 1, _HEIGHT, _WIDTH), dtype=torch.float32)

    sink.open(_session_desc(VideoTensorLayout.tchw))
    sink.write(
        StepResult(
            step_index=0,
            output=grey,
            frame_count=1,
            output_layout=VideoTensorLayout.tchw,
        )
    )
    sink.close()

    red, green, blue = _mean_colour(_decode(path)[0])
    assert abs(red - green) < 10
    assert abs(green - blue) < 10


@needs_ffmpeg
def test_close_can_run_twice(tmp_path: Path) -> None:
    path = tmp_path / "out.mp4"
    sink = Mp4OutputSink(path)
    sink.open(_session_desc())
    sink.write(_result([_RED]))

    sink.close()
    sink.close()

    assert len(_decode(path)) == 1
