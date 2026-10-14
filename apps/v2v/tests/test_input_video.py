# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU decoding checks for V2V input streams and timing."""

from pathlib import Path
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

    read_frames = imageio_ffmpeg.read_frames
    reader_spy = MagicMock()
    reader_spy.__iter__.side_effect = lambda: reader_spy

    def open_reader(path: str, *, pix_fmt: str, output_params: list[str]) -> MagicMock:
        reader = read_frames(path, pix_fmt=pix_fmt, output_params=output_params)
        reader_spy.__next__.side_effect = reader.__next__
        reader_spy.close.side_effect = reader.close
        return reader_spy

    with (
        patch("v2v.v2v._INPUT_CACHE_DIR", tmp_path),
        patch.object(imageio_ffmpeg, "read_frames", side_effect=open_reader),
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
    reader_spy.close.assert_called_once_with()
    if max_frames is not None and max_frames < frame_count:
        assert reader_spy.__next__.call_count == 1 + max_frames
    if expected_count > 1:
        assert video.frames[0].mean() < video.frames[-1].mean()
