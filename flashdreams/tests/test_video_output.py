# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for shared video output contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from flashdreams.infra.video_output import (
    LazyRGBFrame,
    VideoOutputStream,
    VideoStepResult,
    infer_video_num_frames,
    lazy_rgb_frames_from_video_tensor,
    video_tensor_to_hwc_uint8,
)

pytestmark = pytest.mark.ci_cpu


def test_video_step_result_infers_num_frames_from_layout() -> None:
    video = torch.zeros((1, 2, 4, 3, 5, 6), dtype=torch.float32)

    result = VideoStepResult.from_video_chunk(
        chunk_index=7,
        video_chunk=video,
        layout="bvtchw",
        stats={"total_ms": 12.5},
        metadata={"stream": "rgb"},
    )

    assert result.chunk_index == 7
    assert result.num_frames == 4
    assert result.video_chunk is video
    assert result.stats == {"total_ms": 12.5}
    assert result.layout == "bvtchw"
    assert result.metadata == {"stream": "rgb"}
    assert infer_video_num_frames(video, layout="bvtchw") == 4


def test_video_output_stream_makes_step_result_without_host_copy() -> None:
    video = torch.zeros((3, 3, 4, 5), dtype=torch.float32, requires_grad=True)
    output_stream = VideoOutputStream(
        postprocess_stream=None,
        output_layout="tchw",
        collect_output=False,
        move_to_cpu=False,
    )

    result = output_stream.make_step_result(
        video,
        autoregressive_index=4,
        stats={"decode_ms": 1.5},
    )

    assert isinstance(result, VideoStepResult)
    assert result.chunk_index == 4
    assert result.num_frames == 3
    assert result.video_chunk.device == video.device
    assert result.video_chunk.data_ptr() == video.data_ptr()
    assert result.video_chunk.requires_grad is False
    assert result.layout == "tchw"
    assert result.stats == {"decode_ms": 1.5}


def test_video_tensor_to_hwc_uint8_preserves_device_layout_conversion() -> None:
    video = torch.empty((1, 3, 1, 2, 2), dtype=torch.float32)
    video[:, 0] = -1.0
    video[:, 1] = 0.5
    video[:, 2] = 1.0

    frames = video_tensor_to_hwc_uint8(video, layout="bcthw")

    assert frames.device == video.device
    assert frames.dtype == torch.uint8
    assert frames.shape == (1, 2, 2, 3)
    assert frames[0, 0, 0].tolist() == [0, 191, 255]


def test_lazy_rgb_frames_from_video_tensor_materializes_on_demand() -> None:
    video = torch.zeros((2, 3, 4, 5), dtype=torch.float32)
    video[1, :, 2, 3] = 1.0

    frames = lazy_rgb_frames_from_video_tensor(video, layout="tchw")

    assert len(frames) == 2
    assert isinstance(frames[0], LazyRGBFrame)
    assert frames[1].to_numpy()[2, 3].tolist() == [255, 255, 255]


def test_video_step_result_exposes_lazy_rgb_frames() -> None:
    result = VideoStepResult.from_video_chunk(
        chunk_index=0,
        video_chunk=torch.zeros((1, 2, 1, 3, 4, 5), dtype=torch.float32),
        layout="bvtchw",
    )

    frames = result.lazy_rgb_frames()

    assert len(frames) == 1
    assert frames[0].to_numpy().shape == (4, 5, 3)


def test_video_output_stream_collects_chunks_and_stats() -> None:
    output_stream = VideoOutputStream(
        postprocess_stream=None,
        output_layout="tchw",
        move_to_cpu=False,
    )
    chunk = torch.zeros((2, 3, 4, 5), dtype=torch.float32)

    processed = output_stream.process(
        chunk,
        autoregressive_index=3,
        stats={"total_ms": 8.0},
        stats_extra={"frames": 2, "fps": 250.0},
    )
    collected = output_stream.finish()

    assert collected is not None
    assert collected.shape == chunk.shape
    assert collected.data_ptr() == chunk.data_ptr()
    assert processed is chunk
    assert output_stream.stats_history == [
        {
            "autoregressive_index": 3,
            "total_ms": 8.0,
            "frames": 2,
            "fps": 250.0,
        }
    ]


def test_video_output_stream_collects_noop_chunks_without_postprocess() -> None:
    output_stream = VideoOutputStream(
        postprocess_stream=None,
        output_layout="bcthw",
        move_to_cpu=False,
    )
    first = torch.ones((1, 3, 2, 4, 5))
    empty = torch.empty((1, 3, 0, 4, 5))
    second = torch.full((1, 3, 1, 4, 5), 2.0)

    output_stream.process(first, autoregressive_index=0)
    output_stream.process(empty, autoregressive_index=1)
    output_stream.process(second, autoregressive_index=2)
    output = output_stream.finish()

    assert output is not None
    assert output.shape == (1, 3, 3, 4, 5)
    assert torch.equal(output[:, :, :2], first)
    assert torch.equal(output[:, :, 2:], second)


def test_video_output_stream_finishes_to_mp4_with_multiview_tiling() -> None:
    calls: list[dict[str, Any]] = []

    def fake_writer(
        video: torch.Tensor,
        path: Path,
        *,
        fps: int | float,
        layout: str,
        install_hint: str,
    ) -> Path:
        calls.append(
            {
                "shape": tuple(video.shape),
                "path": path,
                "fps": fps,
                "layout": layout,
                "install_hint": install_hint,
            }
        )
        return path

    output_stream = VideoOutputStream(
        postprocess_stream=None,
        output_layout="bvtchw",
        move_to_cpu=False,
    )
    output_stream.process(
        torch.zeros((1, 2, 3, 3, 4, 5)), autoregressive_index=0
    )

    written = output_stream.finish_to_mp4(
        Path("output.mp4"), fps=24, writer=fake_writer
    )

    assert written is not None
    assert written == Path("output.mp4")
    assert calls[0]["shape"] == (3, 4, 10, 3)
