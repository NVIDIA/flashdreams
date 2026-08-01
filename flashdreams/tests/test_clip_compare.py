# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from flashdreams.quality.clip_compare import (
    ClipComparisonThresholds,
    assert_clip_within_thresholds,
    bottom_half,
    parse_frame_indices,
    read_video_rgb,
    select_frame_indices,
)

pytestmark = pytest.mark.ci_cpu


def test_clip_comparison_accepts_small_sampled_drift() -> None:
    reference = np.zeros((5, 4, 4, 3), dtype=np.uint8)
    candidate = reference.copy()
    candidate[..., 0] = 2

    result = assert_clip_within_thresholds(
        reference,
        candidate,
        frame_indices=(0, 2, 4),
        thresholds=ClipComparisonThresholds(
            max_mean_abs=1.0,
            max_rmse=2.0,
            min_psnr_db=40.0,
            max_frame_mean_abs=1.0,
            max_frame_rmse=2.0,
            max_mean_flip=None,
            max_frame_flip=None,
        ),
    )

    assert result.frame_indices == (0, 2, 4)
    assert result.reference_frame_count == 5
    assert result.candidate_frame_count == 5


def test_clip_comparison_reports_large_drift() -> None:
    reference = np.zeros((3, 4, 4, 3), dtype=np.uint8)
    candidate = np.full_like(reference, 50)

    with pytest.raises(AssertionError, match="mean_abs"):
        assert_clip_within_thresholds(
            reference,
            candidate,
            thresholds=ClipComparisonThresholds(
                max_mean_abs=1.0,
                max_rmse=2.0,
                min_psnr_db=40.0,
                max_frame_mean_abs=1.0,
                max_frame_rmse=2.0,
                max_mean_flip=None,
                max_frame_flip=None,
            ),
        )


def test_clip_comparison_reports_frame_count_regression() -> None:
    reference = np.zeros((4, 4, 4, 3), dtype=np.uint8)
    candidate = np.zeros((3, 4, 4, 3), dtype=np.uint8)

    with pytest.raises(AssertionError, match="frame count changed"):
        assert_clip_within_thresholds(
            reference,
            candidate,
            thresholds=ClipComparisonThresholds(
                max_mean_abs=1.0,
                max_rmse=2.0,
                min_psnr_db=40.0,
                max_frame_mean_abs=1.0,
                max_frame_rmse=2.0,
                max_mean_flip=None,
                max_frame_flip=None,
            ),
        )


def test_frame_index_helpers() -> None:
    assert parse_frame_indices("0, 3, 5") == (0, 3, 5)
    assert parse_frame_indices("") is None
    assert select_frame_indices(10, 10, sample_count=4) == (0, 3, 6, 9)
    assert select_frame_indices(10, 7, frame_indices=(1, 1, 6)) == (1, 6)


def test_bottom_half_extracts_generated_region() -> None:
    video = np.zeros((2, 6, 4, 3), dtype=np.uint8)
    video[:, 3:] = 255

    generated = bottom_half(video)

    assert generated.shape == (2, 3, 4, 3)
    assert np.all(generated == 255)


def test_read_video_rgb_falls_back_to_imageio_ffmpeg_without_host_ffmpeg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_media = types.ModuleType("mediapy")

    def read_video(_path: str) -> np.ndarray:
        raise RuntimeError(
            "Program 'ffmpeg' is not found; perhaps install ffmpeg using "
            "'apt install ffmpeg'."
        )

    def read_frames(path: str, *, pix_fmt: str) -> object:
        assert path == "clip.mp4"
        assert pix_fmt == "rgb24"
        yield {"size": (2, 1)}
        yield bytes([1, 2, 3, 4, 5, 6])

    fake_imageio_ffmpeg = types.ModuleType("imageio_ffmpeg")
    setattr(fake_media, "read_video", read_video)
    setattr(fake_imageio_ffmpeg, "read_frames", read_frames)
    monkeypatch.setitem(sys.modules, "mediapy", fake_media)
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", fake_imageio_ffmpeg)

    video = read_video_rgb(Path("clip.mp4"))

    assert video.shape == (1, 1, 2, 3)
    assert video.dtype == np.uint8
    assert video.tolist() == [[[[1, 2, 3], [4, 5, 6]]]]
