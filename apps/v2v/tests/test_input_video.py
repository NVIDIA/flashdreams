# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU decoding checks for V2V input streams and timing."""

from pathlib import Path

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
def test_load_input_video(
    tmp_path: Path, suffix: str, frame_count: int, max_frames: int | None
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

    video = _load_input_video(path, max_frames)

    expected_count = frame_count if max_frames is None else min(frame_count, max_frames)
    assert video.frames.shape == (expected_count, 3, 16, 16)
    assert (video.spec.width, video.spec.height) == (16, 16)
    assert video.spec.fps == (1.0 if frame_count == 1 else 12.0)
    assert video.frames.dtype == torch.float32
    assert video.frames.is_contiguous()
    assert -1.0 <= video.frames.min() <= video.frames.max() <= 1.0
    if expected_count > 1:
        assert video.frames[0].mean() < video.frames[-1].mean()
