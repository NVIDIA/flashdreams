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
def test_presentation_manager_joins_producer_to_its_presentation_stream() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    producer = torch.cuda.Stream(device=device)
    manager = PresentationManager(device=device)

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
        with manager.presentation_context():
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
