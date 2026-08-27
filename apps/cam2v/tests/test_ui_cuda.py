# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA coverage for prioritized Cam2V SlangPy presentation."""

import queue
import threading
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from cam2v import Cam2VSlangPyUILoop, Cam2VUIState

from flashdreams.runtime_v2.presentation_manager import PresentationManager
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_gpu


class _CudaOverlayRenderer:
    """Minimal injected SlangPy renderer for stream-ordering coverage."""

    def __init__(self, *, device: torch.device, width: int, height: int) -> None:
        self.device = device
        self.width = width
        self.height = height
        self.closed = False
        self.render_stream: torch.cuda.Stream | None = None
        self.ui = SimpleNamespace(
            screen=object(),
            Window=lambda *args, **kwargs: object(),
            Text=lambda parent, text: SimpleNamespace(text=text),
        )

    def render(
        self,
        step_index: int,
        events: UserInputEvents,
        step_ui: Any,
    ) -> torch.Tensor:
        """Run the Cam2V widget callback and return a transparent overlay."""
        self.render_stream = torch.cuda.current_stream(self.device)
        step_ui(self.ui, step_index, events)
        return torch.zeros(
            (4, self.height, self.width),
            device=self.device,
            dtype=torch.float32,
        )

    def reset(self) -> None:
        """Match the injected renderer protocol."""

    def close(self) -> None:
        """Record renderer shutdown."""
        self.closed = True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_slangpy_composite_records_high_priority_output_readiness() -> None:
    """Join delayed model output to prioritized UI composition exactly."""
    device = torch.device("cuda", torch.cuda.current_device())
    producer_stream = torch.cuda.Stream(device=device)
    source = torch.empty(
        (1, 3, 64, 96),
        device=device,
        dtype=torch.bfloat16,
    )
    with torch.cuda.stream(producer_stream):
        torch.cuda._sleep(2_000_000)
        source.fill_(0.25)
        source_ready = torch.cuda.Event()
        source_ready.record(producer_stream)
    manager = PresentationManager()
    manager.publish(
        0,
        [
            StepResult(
                step_index=0,
                output=source,
                frame_count=1,
                output_layout=VideoTensorLayout.tchw,
                output_ready_event=source_ready,
            )
        ],
    )
    assert manager.advance(0)[0]
    renderer = _CudaOverlayRenderer(device=device, width=96, height=64)
    loop = Cam2VSlangPyUILoop(
        renderer=renderer,
        device=device,
    )
    loop.register_session_loop_objects(
        state=Cam2VUIState(total_blocks=2, target_fps=16, warmup_blocks=0),
        frequency=60,
        shutdown_event=threading.Event(),
        failure_queue=queue.Queue(),
    )
    loop.register_session_ui_loop_objects(
        output_layout=VideoTensorLayout.tchw,
        presentation_manager=manager,
    )

    closed = False
    try:
        result = loop.step(0, UserInputEvents([]))

        assert result.output_ready_event is not None
        assert renderer.render_stream is not None
        assert (
            renderer.render_stream.priority < torch.cuda.default_stream(device).priority
        )
        loop.close()
        closed = True
        assert result.output_ready_event.query()
        assert result.output.shape == (1, 3, 64, 96)
        assert result.output.dtype is torch.bfloat16
        torch.testing.assert_close(
            result.output[0].cpu(),
            torch.full((3, 64, 96), 0.25, dtype=torch.bfloat16),
        )
    finally:
        if not closed:
            loop.close()
        manager.clear()
        producer_stream.synchronize()

    assert renderer.closed
