# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA lifecycle coverage for the asynchronous Cam2V HUD upload path."""

import queue
import threading
import time

import cam2v.ui as cam2v_ui
import pytest
import torch
from cam2v import (
    Cam2VHUDLoop,
    Cam2VHUDRenderer,
    Cam2VModelStepTiming,
    Cam2VUIState,
    Cam2VUIStatus,
)

from flashdreams.runtime_v2.presentation_manager import PresentationManager
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_step_result_rejects_an_unrecorded_output_event() -> None:
    """Make asynchronous producers publish an explicit recorded dependency."""
    device = torch.device("cuda", torch.cuda.current_device())
    with pytest.raises(ValueError, match="must already be recorded"):
        StepResult(
            step_index=0,
            output=torch.zeros((1, 3, 16, 16), device=device),
            frame_count=1,
            output_layout=VideoTensorLayout.tchw,
            output_ready_event=torch.cuda.Event(),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_hud_uploads_on_presentation_stream_and_preserves_bfloat16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compare async device overlays with CPU and release in-flight storage."""
    device = torch.device("cuda", torch.cuda.current_device())
    state = Cam2VUIState(total_blocks=4, target_fps=16, warmup_blocks=1)
    state.held_keys.update(("w", "e"))
    completed_at = time.perf_counter()
    monkeypatch.setattr(cam2v_ui.time, "perf_counter", lambda: completed_at)
    state.update_status(
        Cam2VUIStatus(
            completed_blocks=2,
            frames_generated=24,
            chunk_fps=13.5,
            recent_model_steps=(
                Cam2VModelStepTiming(
                    completed_at=completed_at,
                    frame_count=13,
                    wall_s=1.0,
                ),
            ),
            model_step_wall_s=0.89,
        )
    )
    cpu_renderer = Cam2VHUDRenderer(width=640, height=360)
    expected = cpu_renderer.render(
        state,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    renderer = Cam2VHUDRenderer(width=640, height=360, device=device)
    consumer = torch.cuda.Stream(device=device)

    with torch.cuda.stream(consumer):
        actual = renderer.render(state, device=device, dtype=torch.bfloat16)
        state.held_keys.remove("e")
        renderer.render(state, device=device, dtype=torch.bfloat16)
        state.held_keys.add("e")
        actual = renderer.render(state, device=device, dtype=torch.bfloat16)
    consumer.synchronize()

    assert actual.device == device
    assert actual.dtype is torch.bfloat16
    torch.testing.assert_close(
        actual.float().cpu(),
        expected,
        atol=0.004,
        rtol=0.004,
    )

    renderer.reset()
    renderer.render(state, device=device, dtype=torch.float32)
    renderer.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_hud_composite_records_high_priority_output_readiness() -> None:
    """Join delayed model output to prioritized HUD composition exactly."""
    device = torch.device("cuda", torch.cuda.current_device())
    producer_stream = torch.cuda.Stream(device=device)
    presentation_stream = torch.cuda.Stream(device=device, priority=-1)
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
    loop = Cam2VHUDLoop(
        width=96,
        height=64,
        device=device,
        presentation_stream=presentation_stream,
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

    try:
        result = loop.step(0, UserInputEvents([]))

        assert result.output_ready_event is not None
        assert presentation_stream.priority < torch.cuda.default_stream(device).priority
        result.output_ready_event.synchronize()
        assert result.output.shape == (1, 3, 64, 96)
        assert result.output.dtype is torch.bfloat16
        expected_renderer = Cam2VHUDRenderer(width=96, height=64)
        expected_overlay = expected_renderer.render(
            loop.state,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )
        expected = manager.composite(
            torch.full((3, 64, 96), 0.25, dtype=torch.bfloat16),
            expected_overlay,
        )
        torch.testing.assert_close(
            result.output[0].cpu(),
            expected,
            atol=0.01,
            rtol=0.01,
        )
        expected_renderer.close()
    finally:
        loop.close()
        manager.clear()
        producer_stream.synchronize()
