# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from flashdreams_app import RuntimeMetadata, webrtc

from flashdreams.runtime import InferenceInput, StepRequest, StepResult

pytestmark = pytest.mark.ci_cpu


class _Session:
    def next_step_request(self) -> StepRequest | None:
        return None

    def step(self, inputs: InferenceInput) -> StepResult:
        raise AssertionError("Server construction must not step a session.")

    def reset(self, inputs: InferenceInput | None = None) -> None:
        del inputs

    def close(self) -> None:
        pass


class _Runtime:
    metadata = RuntimeMetadata(
        model_id="fake-app",
        fps=16,
        output_layout="tchw",
        video_width=96,
        video_height=64,
    )
    initial_input = InferenceInput()

    def prepare_step_input(self, request: object) -> InferenceInput:
        del request
        return InferenceInput()

    def start_session(self, inputs: InferenceInput) -> _Session:
        del inputs
        return _Session()

    def close(self) -> None:
        pass


def test_host_constructs_webrtc_presentation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_serve(**kwargs: object) -> str:
        captured.update(kwargs)
        return "served"

    monkeypatch.setattr(webrtc, "serve_webrtc_demo", fake_serve)
    runtime = _Runtime()
    result = webrtc.serve_webrtc(
        runtime=runtime,
        options=webrtc.WebRTCOptions(
            host="127.0.0.1",
            port=8080,
            warmup_chunks=0,
            warmup_timeout_s=30.0,
            client_liveness_timeout_s=30.0,
            device="cpu",
            encoder_backend="default",
            encoder_bitrate_bps=1_000_000,
            encoder_gop=16,
        ),
        world_rank=0,
    )

    assert result == "served"
    assert captured["model_id"] == "fake-app"
    assert captured["world_rank"] == 0
    session_manager = captured["session_manager"]
    assert isinstance(session_manager, webrtc.BaseWebRTCSessionManager)
    assert session_manager._shared_adapter is None
    assert session_manager._shared_host is not None
    assert session_manager._shared_host.runtime is runtime
    assert session_manager._shared_scenario is not None
    assert session_manager._shared_scenario.initial_inputs is runtime.initial_input
    assert callable(session_manager._shared_model_input_provider_factory)
