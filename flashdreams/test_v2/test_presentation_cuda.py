# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA stream-ordering tests for cross-thread presentation."""

import pytest
import torch

from flashdreams.runtime_v2.presentation_manager import PresentationManager
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_step_result_rejects_an_unrecorded_output_event() -> None:
    device = torch.device("cuda", torch.cuda.current_device())

    with pytest.raises(ValueError, match="must already be recorded"):
        StepResult(
            step_index=0,
            output=torch.zeros((1, 3, 8, 8), device=device),
            frame_count=1,
            output_layout=VideoTensorLayout.tchw,
            output_ready_event=torch.cuda.Event(),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_step_result_preserves_a_custom_output_event() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    producer = torch.cuda.Stream(device=device)
    with torch.cuda.stream(producer):
        output = torch.zeros((1, 3, 8, 8), device=device)
        ready = torch.cuda.Event()
        ready.record(producer)

    result = StepResult(
        step_index=0,
        output=output,
        frame_count=1,
        output_layout=VideoTensorLayout.tchw,
        output_ready_event=ready,
    )

    assert result._output_ready_event is ready
    assert result.replace(step_index=1)._output_ready_event is ready
    producer.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_step_result_treats_none_as_automatic_output_readiness() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    producer = torch.cuda.Stream(device=device)
    with torch.cuda.stream(producer):
        result = StepResult(
            step_index=0,
            output=torch.zeros((1, 3, 8, 8), device=device),
            frame_count=1,
            output_layout=VideoTensorLayout.tchw,
            output_ready_event=None,
        )

    assert result._output_ready_event is not None
    producer.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_presentation_manager_joins_default_producer_to_consumer_stream() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    producer = torch.cuda.Stream(device=device)
    consumer = torch.cuda.Stream(device=device, priority=-1)
    manager = PresentationManager()

    try:
        with torch.cuda.stream(producer):
            output = torch.empty((1, 3, 8, 8), device=device)
            torch.cuda._sleep(2_000_000)
            output.fill_(0.25)
            result = StepResult(
                step_index=0,
                output=output,
                frame_count=1,
                output_layout=VideoTensorLayout.tchw,
            )
            manager.publish(0, [result])

        assert result._output_ready_event is not None

        assert manager.advance(0)[0]
        with torch.cuda.stream(consumer):
            frame = manager.presented_frame(0, stream=consumer)
            assert frame is not None
            observed = frame.clone()
        consumer.synchronize()

        torch.testing.assert_close(
            observed.cpu(),
            torch.full((3, 8, 8), 0.25),
        )
    finally:
        manager.clear()
        producer.synchronize()
        consumer.synchronize()
