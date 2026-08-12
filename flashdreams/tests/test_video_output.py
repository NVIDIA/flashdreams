# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for shared video output contracts."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from flashdreams.infra.video_output import (
    LazyRGBFrame,
    VideoOutputStream,
    VideoResultCollector,
    infer_video_num_frames,
    lazy_rgb_frames_from_video_tensor,
    prepare_video_for_mp4,
    video_tensor_to_hwc_uint8,
)
from flashdreams.runtime import StepResult

pytestmark = pytest.mark.ci_cpu


def test_step_result_infers_video_frame_count_from_layout() -> None:
    video = torch.zeros((1, 2, 4, 3, 5, 6), dtype=torch.float32)

    result = StepResult.from_video_chunk(
        step_index=7,
        video_chunk=video,
        layout="bvtchw",
        metrics={"total_ms": 12.5},
        metadata={"stream": "rgb"},
    )

    assert result.step_index == 7
    assert result.frame_count == 4
    assert result.video_chunk is video
    assert result.metrics == {"total_ms": 12.5}
    assert result.layout == "bvtchw"
    assert result.metadata == {"stream": "rgb"}
    assert infer_video_num_frames(video, layout="bvtchw") == 4


def test_step_result_validates_video_step_and_layout_shape() -> None:
    with pytest.raises(ValueError, match="step_index"):
        StepResult.from_video_chunk(
            step_index=-1,
            video_chunk=torch.zeros((1, 3, 2, 4, 5)),
            layout="bcthw",
        )

    with pytest.raises(ValueError, match="expects a 5D tensor"):
        StepResult.from_video_chunk(
            step_index=0,
            video_chunk=torch.zeros((2, 3, 4, 5)),
            layout="bcthw",
        )


def test_step_result_freezes_video_metadata_and_metrics() -> None:
    metadata = {"stream": "rgb"}
    metrics = {"model_step_s": 0.5}
    result = StepResult.from_video_chunk(
        step_index=0,
        video_chunk=torch.zeros((2, 3, 4, 5)),
        layout="tchw",
        metadata=metadata,
        metrics=metrics,
    )

    metadata["stream"] = "debug"
    metrics["model_step_s"] = 1.0

    assert result.metadata == {"stream": "rgb"}
    assert result.metrics == {"model_step_s": 0.5}
    with pytest.raises(TypeError):
        cast(Any, result.metadata)["stream"] = "debug"


def test_video_output_stream_returns_step_result_without_host_copy() -> None:
    video = torch.zeros((3, 3, 4, 5), dtype=torch.float32, requires_grad=True)
    output_stream = VideoOutputStream(
        postprocess_stream=None,
        output_layout="tchw",
    )

    result = output_stream.process(
        video,
        autoregressive_index=4,
        metrics={"decode_ms": 1.5},
    )

    assert isinstance(result, StepResult)
    assert result.step_index == 4
    assert result.frame_count == 3
    assert result.video_chunk.device == video.device
    assert result.video_chunk.data_ptr() == video.data_ptr()
    assert result.video_chunk.requires_grad is False
    assert result.layout == "tchw"
    assert result.metrics == {"decode_ms": 1.5}


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


def test_step_result_exposes_lazy_rgb_frames() -> None:
    result = StepResult.from_video_chunk(
        step_index=0,
        video_chunk=torch.zeros((1, 2, 1, 3, 4, 5), dtype=torch.float32),
        layout="bvtchw",
    )

    frames = result.lazy_rgb_frames()

    assert len(frames) == 1
    assert frames[0].to_numpy().shape == (4, 5, 3)


def test_video_result_collector_collects_chunks_and_stats() -> None:
    output_stream = VideoOutputStream(
        postprocess_stream=None,
        output_layout="tchw",
    )
    collector = VideoResultCollector(output_layout="tchw", move_to_cpu=False)
    chunk = torch.zeros((2, 3, 4, 5), dtype=torch.float32)

    result = output_stream.process(
        chunk,
        autoregressive_index=3,
        metrics={"total_ms": 8.0, "pipeline_fps": 250.0},
    )
    collector.add(result)
    assert output_stream.finish() is None
    collected = collector.finish()

    assert collected is not None
    assert collected.shape == chunk.shape
    assert collected.data_ptr() == chunk.data_ptr()
    assert result.video_chunk.data_ptr() == chunk.data_ptr()
    assert collector.stats_history == [
        {
            "step_index": 3,
            "frames": 2,
            "total_ms": 8.0,
            "pipeline_fps": 250.0,
        }
    ]


def test_video_result_collector_skips_empty_chunks() -> None:
    output_stream = VideoOutputStream(
        postprocess_stream=None,
        output_layout="bcthw",
    )
    collector = VideoResultCollector(output_layout="bcthw", move_to_cpu=False)
    first = torch.ones((1, 3, 2, 4, 5))
    empty = torch.empty((1, 3, 0, 4, 5))
    second = torch.full((1, 3, 1, 4, 5), 2.0)

    collector.add(output_stream.process(first, autoregressive_index=0))
    collector.add(output_stream.process(empty, autoregressive_index=1))
    collector.add(output_stream.process(second, autoregressive_index=2))
    assert output_stream.finish() is None
    output = collector.finish()

    assert output is not None
    assert output.shape == (1, 3, 3, 4, 5)
    assert torch.equal(output[:, :, :2], first)
    assert torch.equal(output[:, :, 2:], second)


def test_video_output_stream_returns_postprocess_tail_as_step_result() -> None:
    class _TailPostprocess:
        last_process_stats = None

        def process(
            self,
            output: torch.Tensor,
            *,
            autoregressive_index: int,
        ) -> torch.Tensor:
            del autoregressive_index
            return output[:, :, :0]

        def finish(self) -> torch.Tensor:
            return torch.ones((1, 3, 2, 4, 5))

    output_stream = VideoOutputStream(
        postprocess_stream=cast(Any, _TailPostprocess()),
        output_layout="bcthw",
    )
    result = output_stream.process(
        torch.zeros((1, 3, 2, 4, 5)),
        autoregressive_index=6,
    )

    tail = output_stream.finish()

    assert result.frame_count == 0
    assert tail is not None
    assert tail.step_index == 6
    assert tail.frame_count == 2
    assert tail.metadata == {"postprocess_tail": True}


def test_video_output_stream_state_is_isolated_per_session() -> None:
    class _StatefulPostprocess:
        last_process_stats = None

        def __init__(self) -> None:
            self.calls = 0

        def process(
            self,
            output: torch.Tensor,
            *,
            autoregressive_index: int,
        ) -> torch.Tensor:
            del autoregressive_index
            self.calls += 1
            return output + self.calls

        def finish(self) -> None:
            return None

    first = VideoOutputStream(
        postprocess_stream=cast(Any, _StatefulPostprocess()),
        output_layout="tchw",
    )
    second = VideoOutputStream(
        postprocess_stream=cast(Any, _StatefulPostprocess()),
        output_layout="tchw",
    )
    video = torch.zeros((1, 3, 2, 2))

    first_result = first.process(video, autoregressive_index=0)
    second_result = second.process(video, autoregressive_index=0)

    assert torch.equal(first_result.video_chunk, second_result.video_chunk)
    assert first.postprocess_stream is not second.postprocess_stream


def test_prepare_video_for_mp4_tiles_multiview_video() -> None:
    video = torch.zeros((1, 2, 3, 3, 4, 5))

    writable, layout = prepare_video_for_mp4(video, layout="bvtchw")

    assert writable.shape == (3, 4, 10, 3)
    assert layout == "thwc"
