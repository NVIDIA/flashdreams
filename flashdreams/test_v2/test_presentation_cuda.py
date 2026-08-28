# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA stream-ordering tests for cross-thread presentation."""

import pytest
import torch

from flashdreams.runtime_v2.presentation_manager import PresentationManager
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.ui_compositing import prepare_ui_back_buffer
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_default_manager_uses_high_priority_stream_and_joins_producer() -> None:
    device = torch.device(
        "cuda", (torch.cuda.current_device() + 1) % torch.cuda.device_count()
    )
    producer = torch.cuda.Stream(device=device)
    manager = PresentationManager()

    try:
        assert manager._presentation_stream is None
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

        stream = manager._presentation_stream
        assert stream is not None
        assert stream.device == device
        assert stream.priority < torch.cuda.default_stream(device).priority
        assert result._output_ready_event is not None

        assert manager.advance(0)[0]
        with manager.presentation_context():
            assert torch.cuda.current_stream(device) == stream
            frame = manager.presented_frame(0)
            assert frame is not None
            observed = frame.clone()
        manager.close()

        torch.testing.assert_close(
            observed.cpu(),
            torch.full((3, 8, 8), 0.25),
        )
    finally:
        manager.close()
        producer.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_ui_prepares_cpu_integer_frame_on_default_presentation_stream() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    manager = PresentationManager()

    try:
        overlay = torch.zeros((4, 2, 3), device=device)
        manager.composite(None, overlay)
        stream = manager._presentation_stream
        assert stream is not None
        assert stream.device == device
        with manager.presentation_context():
            back_buffer = torch.tensor(
                [0, 255, 128],
                dtype=torch.uint8,
            ).reshape(3, 1, 1)
            prepared = prepare_ui_back_buffer(back_buffer, overlay)
            observed = manager.composite(prepared, overlay).clone()
        manager.close()

        torch.testing.assert_close(
            observed[:, 0, 0].cpu(),
            torch.tensor([-1.0, 1.0, 1.0 / 255.0]),
        )
    finally:
        manager.close()
