# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU decoding checks for V2V input streams and timing."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import av
import imageio_ffmpeg
import numpy as np
import pytest
import torch
from v2v.v2v import _load_input_video

pytestmark = pytest.mark.ci_cpu


@pytest.mark.parametrize(
    "suffix, frame_count", [(".AVIF", 5), (".avif", 1), (".mp4", 5)]
)
@pytest.mark.parametrize("max_frames", [None, 2])
@pytest.mark.parametrize("extensionless", [False, True])
def test_load_input_video(
    tmp_path: Path,
    suffix: str,
    frame_count: int,
    max_frames: int | None,
    extensionless: bool,
) -> None:
    path = tmp_path / f"input{suffix}"
    writer = imageio_ffmpeg.write_frames(
        str(path),
        size=(16, 16),
        fps=12,
        codec="libaom-av1" if suffix.lower() == ".avif" else "libx264",
        output_params=["-cpu-used", "8"] if suffix.lower() == ".avif" else [],
    )
    try:
        next(writer)
        for index in range(frame_count):
            writer.send(np.full((16, 16, 3), index * 40, dtype=np.uint8))
    finally:
        writer.close()

    if suffix == ".AVIF":
        with av.open(str(path)) as container:
            assert [stream.frames for stream in container.streams.video] == [1, 5]
            assert [stream.average_rate for stream in container.streams.video] == [
                1,
                12,
            ]

    if extensionless:
        path = path.rename(tmp_path / "download")

    av_open = av.open
    container_spy = MagicMock()
    reader_spy = MagicMock()
    reader_spy.__iter__.side_effect = lambda: reader_spy

    def open_reader(path: str) -> MagicMock:
        container = av_open(path)
        container_spy.streams = container.streams
        container_spy.metadata = container.metadata
        container_spy.__enter__.return_value = container_spy
        container_spy.close.side_effect = container.close
        container_spy.__exit__.side_effect = lambda *_: container_spy.close()

        def decode(stream: av.VideoStream) -> MagicMock:
            reader = container.decode(stream)
            reader_spy.__next__.side_effect = reader.__next__
            return reader_spy

        container_spy.decode.side_effect = decode
        return container_spy

    with (
        patch("v2v.v2v._INPUT_CACHE_DIR", tmp_path),
        patch.object(av, "open", side_effect=open_reader),
    ):
        video = _load_input_video(
            "https://example.test/download" if extensionless else path, max_frames
        )

    expected_count = frame_count if max_frames is None else min(frame_count, max_frames)
    assert video.frames.shape == (expected_count, 3, 16, 16)
    assert (video.spec.width, video.spec.height) == (16, 16)
    assert video.spec.fps == (1.0 if frame_count == 1 else 12.0)
    assert video.frames.dtype == torch.float32
    assert video.frames.is_contiguous()
    assert -1.0 <= video.frames.min() <= video.frames.max() <= 1.0
    container_spy.close.assert_called_once_with()
    if max_frames is not None and max_frames < frame_count:
        assert reader_spy.__next__.call_count == max_frames
    if expected_count > 1:
        assert video.frames[0].mean() < video.frames[-1].mean()


@pytest.mark.parametrize("rotation", [90, -90, 180, 45])
def test_load_input_video_preserves_display_rotation(
    tmp_path: Path, rotation: int
) -> None:
    rotated = tmp_path / "rotated.mp4"
    image = np.zeros((32, 16, 3), dtype=np.uint8)
    image[:16, :8] = (255, 0, 0)
    image[16:, 8:] = (0, 255, 0)
    writer = imageio_ffmpeg.write_frames(
        str(rotated),
        size=(16, 32),
        fps=12,
        input_params=["-display_rotation", str(rotation), "-noautorotate"],
    )
    try:
        next(writer)
        writer.send(image)
    finally:
        writer.close()
    with av.open(str(rotated)) as container:
        frame = next(container.decode(video=0))
        assert frame.rotation % 360 == rotation % 360
    # Use an explicit odd-angle filter to avoid imageio's warning parser bug.
    reader = imageio_ffmpeg.read_frames(
        str(rotated),
        pix_fmt="rgb24",
        input_params=["-noautorotate"] if rotation % 90 else [],
        output_params=["-vf", f"rotate={-rotation}*PI/180"] if rotation % 90 else [],
    )
    try:
        metadata = next(reader)
        width, height = metadata["size"]
        expected = np.frombuffer(next(reader), dtype=np.uint8).reshape(height, width, 3)
    finally:
        reader.close()

    video = _load_input_video(rotated, 1)

    assert (video.spec.width, video.spec.height) == (width, height)
    assert video.frames.shape == (1, 3, height, width)
    actual = video.frames[0].permute(1, 2, 0).numpy() * 127.5 + 127.5
    np.testing.assert_allclose(actual, expected, atol=2)


def test_load_input_video_prefers_highest_resolution_track(tmp_path: Path) -> None:
    path = tmp_path / "multiple-tracks.mp4"
    with av.open(str(path), "w") as container:
        streams = [container.add_stream("libx264", rate=12) for _ in range(2)]
        for stream, width in zip(streams, (16, 32), strict=True):
            stream.width, stream.height, stream.pix_fmt = width, 16, "yuv420p"
        for stream in streams:
            image = np.zeros((stream.height, stream.width, 3), dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)

    video = _load_input_video(path, None)

    assert (video.spec.width, video.spec.height) == (32, 16)
    assert video.frames.shape == (1, 3, 16, 32)


def test_load_input_video_falls_back_to_guessed_frame_rate(tmp_path: Path) -> None:
    path = tmp_path / "input"
    path.touch()
    stream = SimpleNamespace(
        width=16, height=16, index=0, average_rate=None, guessed_rate=12
    )
    frame = av.VideoFrame.from_ndarray(np.zeros((16, 16, 3), dtype=np.uint8))
    container = MagicMock()
    container.__enter__.return_value = container
    container.streams.video = [stream]
    container.metadata = {}
    container.decode.return_value = iter([frame])
    with patch.object(av, "open", return_value=container):
        video = _load_input_video(path, None)

    assert video.spec.fps == 12.0
