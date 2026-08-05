# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for shared video output contracts."""

from __future__ import annotations

import pytest
import torch

from flashdreams.infra.video_output import (
    RunnerVideoOutputStream,
    VideoStepResult,
    infer_video_num_frames,
)
from flashdreams.serving.webrtc.manager import WebRTCStepResult, make_webrtc_step_result

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


def test_webrtc_step_result_uses_shared_video_result_contract() -> None:
    result = WebRTCStepResult(
        chunk_index=1,
        num_frames=2,
        video_chunk=torch.zeros((2, 3, 4, 5)),
        stats=None,
    )

    assert isinstance(result, VideoStepResult)
    assert result.num_frames == 2


def test_make_webrtc_step_result_preserves_tensor_and_infers_frames() -> None:
    video = torch.zeros((3, 3, 4, 5), dtype=torch.float32, requires_grad=True)

    result = make_webrtc_step_result(
        chunk_index=4,
        video_chunk=video,
        layout="tchw",
        stats={"decode_ms": 1.5},
    )

    assert isinstance(result, WebRTCStepResult)
    assert result.chunk_index == 4
    assert result.num_frames == 3
    assert result.video_chunk.device == video.device
    assert result.video_chunk.data_ptr() == video.data_ptr()
    assert result.video_chunk.requires_grad is False
    assert result.layout == "tchw"
    assert result.stats == {"decode_ms": 1.5}


def test_runner_video_output_stream_collects_chunks_and_stats() -> None:
    output_stream = RunnerVideoOutputStream(
        postprocess_stream=None,
        output_layout="tchw",
        move_to_cpu=False,
    )
    chunk = torch.zeros((2, 3, 4, 5), dtype=torch.float32)

    output_stream.process(
        chunk,
        autoregressive_index=3,
        stats={"total_ms": 8.0},
        stats_extra={"frames": 2, "fps": 250.0},
    )
    collected = output_stream.finish()

    assert collected is not None
    assert collected.shape == chunk.shape
    assert collected.data_ptr() == chunk.data_ptr()
    assert output_stream.stats_history == [
        {
            "autoregressive_index": 3,
            "total_ms": 8.0,
            "frames": 2,
            "fps": 250.0,
        }
    ]


def test_runner_video_output_stream_collects_noop_chunks_without_postprocess() -> None:
    output_stream = RunnerVideoOutputStream(
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
