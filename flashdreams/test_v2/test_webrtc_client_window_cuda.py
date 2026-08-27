# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA lifecycle tests for write-owned WebRTC frame materialization."""

from typing import Any, cast

import pytest
import torch

pytestmark = pytest.mark.ci_gpu

pytest.importorskip("aiohttp")
pytest.importorskip("aiortc")

from av import VideoFrame

from flashdreams.runtime_v2.serving.webrtc_server import WebRTCServer
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout


class _CapturingTrack:
    """Capture frames after ``WebRTCServer.write`` has prepared them."""

    def __init__(self) -> None:
        self.frames: list[VideoFrame] = []

    def enqueue(self, frame: VideoFrame) -> bool:
        assert isinstance(frame, VideoFrame)
        self.frames.append(frame)
        return True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_write_materializes_owned_cuda_frames_after_the_producer_event() -> None:
    """A write owns stable pixels before its reusable source can be mutated."""
    device = torch.device("cuda", torch.cuda.current_device())
    producer = torch.cuda.Stream(device=device)
    server = WebRTCServer()
    track = _CapturingTrack()
    server.register_input_callback(lambda _event: None)
    server.open(
        SessionDesc(
            output_layout=VideoTensorLayout.tchw,
            frames_per_second_for_ui=60,
            frames_per_second_for_step=16,
            video_width=48,
            video_height=32,
        )
    )
    server._video_track = cast(Any, track)
    server._media_connected.set()

    expected_values = (23, 47, 89)
    try:
        for step_index, value in enumerate(expected_values):
            with torch.cuda.stream(producer):
                source = torch.empty(
                    (1, 3, 32, 48),
                    dtype=torch.uint8,
                    device=device,
                )
                torch.cuda._sleep(2_000_000)
                source.fill_(value)
                source_ready = torch.cuda.Event()
                source_ready.record(producer)

            result = StepResult(
                step_index=step_index,
                output=source,
                frame_count=1,
                output_layout=VideoTensorLayout.tchw,
                output_ready_event=source_ready,
            )
            server.write(result)

            assert len(track.frames) == step_index + 1
            assert source_ready.query()
            source.fill_(255 - value)

        for frame, value in zip(track.frames, expected_values, strict=True):
            pixels = frame.to_ndarray(format="rgb24")
            assert int(pixels.min()) == value
            assert int(pixels.max()) == value
    finally:
        server._video_track = None
        server._media_connected.clear()
        server.close()
        producer.synchronize()
