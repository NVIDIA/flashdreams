# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the WebRTC application mode."""

from __future__ import annotations

from typing import Any, cast

import pytest

from flashdreams.runtime import InferenceInput, StepResult
from flashdreams.runtime.demo import WebRTCAppResources
from flashdreams_runner import AppConfig, IOHandler, Runtime, Session, webrtc

pytestmark = pytest.mark.ci_cpu


class _Session(Session):
    @property
    def step_index(self) -> int:
        return 0

    def step(self, inputs: InferenceInput) -> StepResult:
        del inputs
        raise AssertionError("Server construction must not run a step.")

    def destroy(self) -> None:
        pass


class _Runtime(Runtime):
    @property
    def config(self) -> AppConfig:
        return AppConfig(
            model_id="fake-app",
            fps=16,
            output_layout="tchw",
            video_width=96,
            video_height=64,
        )

    def initialize(self, *, device: str, io_handler: IOHandler) -> None:
        del device, io_handler

    def create_session(self, initial_input: InferenceInput | None = None) -> Session:
        del initial_input
        return _Session()

    def destroy(self) -> None:
        pass


def test_webrtc_mode_constructs_shared_presentation(
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
        host="127.0.0.1",
        port=8080,
        device="cpu",
        world_rank=0,
    )

    assert result == "served"
    assert captured["model_id"] == "fake-app"
    assert captured["world_rank"] == 0
    output = captured["output"]
    assert isinstance(output, webrtc.WebRTCOutputSpec)
    assert output.video_width == 96
    assert output.warmup_chunks == 0
    session_manager = captured["session_manager"]
    assert isinstance(session_manager, webrtc.BaseWebRTCSessionManager)
    assert session_manager._shared_adapter is None
    assert session_manager._shared_host is not None
    assert session_manager._shared_host.runtime is runtime
    assert session_manager.is_runtime_ready()


def test_webrtc_mode_uses_application_customization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    custom_manager = object()

    class Customization:
        def prepare_initial_input(self) -> InferenceInput:
            return InferenceInput(global_conditioning={"prompt": "custom"})

        def create_session_manager(self, **kwargs: object) -> Any:
            captured["manager_kwargs"] = kwargs
            return custom_manager

        def create_app_resources(self, **kwargs: object) -> WebRTCAppResources:
            captured["resources_kwargs"] = kwargs
            return WebRTCAppResources(preload_name="custom-ui")

    def fake_serve(**kwargs: object) -> str:
        captured.update(kwargs)
        return "served"

    monkeypatch.setattr(webrtc, "serve_webrtc_demo", fake_serve)
    result = webrtc.serve_webrtc(
        runtime=_Runtime(),
        host="127.0.0.1",
        port=8080,
        device="cpu",
        world_rank=0,
        customization=Customization(),
    )

    assert result == "served"
    assert captured["session_manager"] is custom_manager
    resources = captured["app_resources"]
    assert isinstance(resources, WebRTCAppResources)
    assert resources.preload_name == "custom-ui"
    manager_kwargs = cast(dict[str, object], captured["manager_kwargs"])
    scenario = cast(Any, manager_kwargs["scenario"])
    assert scenario.initial_inputs.global_conditioning["prompt"] == "custom"
    assert (
        cast(dict[str, object], captured["resources_kwargs"])["session_manager"]
        is custom_manager
    )
