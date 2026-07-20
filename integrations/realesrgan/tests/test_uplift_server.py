# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Real-ESRGAN uplift gRPC servicer."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import realesrgan.grpc.uplift_server as server_mod
from flashdreams.serving.uplift.protos import uplift_pb2 as pb2

pytestmark = pytest.mark.ci_cpu


class _FakeUpsampler:
    def __init__(
        self,
        *,
        scale: int,
        model_name: str,
        model_path: object,
        tile: int,
        tile_pad: int,
        pre_pad: int,
        half: bool,
        compile_model: bool,
        compile_mode: str,
        device: torch.device,
        load_checkpoint: bool,
    ) -> None:
        self.scale = scale
        self.model_name = model_name

    def upsample_frame_tensor(self, frame: torch.Tensor) -> torch.Tensor:
        return frame.repeat_interleave(self.scale, dim=1).repeat_interleave(
            self.scale,
            dim=2,
        )


@pytest.fixture
def servicer(monkeypatch: pytest.MonkeyPatch) -> server_mod.RealESRGANUplift:
    monkeypatch.setattr(server_mod, "RealESRGANUpsampler", _FakeUpsampler)
    return server_mod.RealESRGANUplift(
        default_scale=2,
        device="cpu",
        warmup=False,
        load_checkpoint=False,
    )


def _raw_request(frames: np.ndarray, *, chunk_index: int = 0) -> pb2.UpscaleChunkRequest:
    total, height, width, _channels = frames.shape
    return pb2.UpscaleChunkRequest(
        num_frames=total,
        height=height,
        width=width,
        chunk_index=chunk_index,
        frame_encoding=pb2.FRAME_ENCODING_RAW_RGB,
        frames_rgb=np.ascontiguousarray(frames).tobytes(),
    )


def test_unary_upscale_chunk_returns_scaled_frames(
    servicer: server_mod.RealESRGANUplift,
) -> None:
    started = servicer.start_session(
        pb2.StartSessionRequest(session_id="s0", scale=2),
        None,
    )
    assert started.success

    frames = np.arange(2 * 2 * 3 * 3, dtype=np.uint8).reshape(2, 2, 3, 3)
    request = _raw_request(frames)
    request.session_id = "s0"
    response = servicer.upscale_chunk(request, None)

    assert not response.error
    assert not response.frames_omitted
    assert response.num_frames == 2
    assert response.height == 4
    assert response.width == 6
    output = np.frombuffer(response.frames_rgb, dtype=np.uint8).reshape(2, 4, 6, 3)
    assert output[0, 0, 0].tolist() == frames[0, 0, 0].tolist()
    assert output[0, 1, 1].tolist() == frames[0, 0, 0].tolist()


def test_display_only_omits_response_frames(
    servicer: server_mod.RealESRGANUplift,
) -> None:
    started = servicer.start_session(
        pb2.StartSessionRequest(session_id="s0", scale=2),
        None,
    )
    assert started.success

    frames = np.zeros((1, 2, 3, 3), dtype=np.uint8)
    request = _raw_request(frames)
    request.session_id = "s0"
    request.display_only = True

    response = servicer.upscale_chunk(request, None)

    assert not response.error
    assert response.frames_omitted
    assert response.frames_rgb == b""
    assert response.height == 4
    assert response.width == 6


def test_streaming_upscale_video_uses_first_request_scale(
    servicer: server_mod.RealESRGANUplift,
) -> None:
    frames = np.zeros((1, 2, 3, 3), dtype=np.uint8)
    request = _raw_request(frames)
    request.session_id = "stream"
    request.scale = 2

    responses = list(servicer.upscale_video(iter([request]), None))

    assert len(responses) == 1
    assert responses[0].session_id == "stream"
    assert responses[0].height == 4
    assert responses[0].width == 6
