# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for T2V-specific WebRTC controls and recording."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import torch
from aiohttp import web

from flashdreams.runtime import InferenceInput, OutputArtifact
from flashdreams.runtime.demo import DemoSpec, PreparedScenario, WebRTCOutputSpec
from flashdreams_runner import AppConfig, Runtime
from flashdreams_runner.webrtc import WebRTCMode
from t2v_app import runtime as runtime_module
from t2v_app import session as session_module
from t2v_app.runtime import T2VRuntime
from t2v_app.session import T2VScenario, T2VSession, T2VSessionDefaults
from t2v_app.webrtc import T2VWebRTCCustomization, T2VWebRTCSessionManager

pytestmark = pytest.mark.ci_cpu


class _WebRuntime:
    def __init__(self) -> None:
        self.config = AppConfig(
            model_id="t2v-app",
            fps=12,
            output_layout="tchw",
            video_width=96,
            video_height=64,
            default_steps=2,
        )
        self.latest_artifact = None

    def prepare_session_input(
        self,
        *,
        prompt: str | None = None,
        total_blocks: int | None = None,
    ) -> InferenceInput:
        return InferenceInput(
            global_conditioning={
                "prompt": prompt or "default prompt",
                "total_blocks": total_blocks,
                "pixel_height": 64,
                "pixel_width": 96,
                "fps": 12,
            }
        )

    def blocks_for_duration(self, duration_s: float) -> int:
        return int(duration_s * 2)


def test_t2v_runtime_customizes_runner_webrtc_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Pipeline:
        def to(self, device: object) -> "Pipeline":
            assert device == "cpu"
            return self

        def eval(self) -> "Pipeline":
            return self

    class PipelineConfig:
        def setup(self) -> Pipeline:
            return Pipeline()

    monkeypatch.setattr(runtime_module, "StreamInferencePipeline", Pipeline)
    runtime = T2VRuntime(
        pipeline_config=cast(Any, PipelineConfig()),
        session_defaults=T2VSessionDefaults(
            prompt="default prompt",
            total_blocks=2,
            pixel_height=64,
            pixel_width=96,
            fps=12,
        ),
        config=AppConfig(
            model_id="t2v-app",
            fps=12,
            output_layout="tchw",
            video_width=96,
            video_height=64,
            default_steps=2,
        ),
    )
    mode = WebRTCMode(host="127.0.0.1", port=8080, device="cpu", world_rank=0)

    runtime.initialize(device="cpu", io_handler=mode)

    assert isinstance(mode._customization, T2VWebRTCCustomization)
    runtime.destroy()


def test_t2v_customization_updates_prompt_duration_and_routes() -> None:
    runtime = _WebRuntime()
    customization = T2VWebRTCCustomization(runtime=cast(Any, runtime))
    initial_input = customization.prepare_initial_input()
    assert initial_input.global_conditioning["total_blocks"] == 2

    output = WebRTCOutputSpec(
        host="127.0.0.1",
        port=8080,
        fps=12,
        video_width=96,
        video_height=64,
    )
    spec = DemoSpec(model_id="t2v-app", input_mode="webrtc", output=output)
    scenario = PreparedScenario(initial_inputs=initial_input)

    def provider_factory(spec: DemoSpec, scenario: PreparedScenario) -> Any:
        del spec, scenario
        return object()

    manager = customization.create_session_manager(
        runtime=cast(Runtime, cast(object, runtime)),
        output=output,
        spec=spec,
        scenario=scenario,
        input_provider_factory=provider_factory,
    )

    assert isinstance(manager, T2VWebRTCSessionManager)
    assert manager.is_runtime_ready()
    assert manager._keep_connection_after_completed
    manager.update_generation(prompt="  A waterfall  ", duration_s=3.0)
    prepared = manager._shared_scenario
    assert prepared is not None
    assert prepared.initial_inputs.global_conditioning["prompt"] == "A waterfall"
    assert prepared.initial_inputs.global_conditioning["total_blocks"] == 6

    resources = customization.create_app_resources(session_manager=manager)
    assert resources.model_web_resource is not None
    assert resources.model_web_resource.joinpath("adapter.js").is_file()
    assert resources.configure_app is not None
    app = web.Application()
    resources.configure_app(app)
    routes = {
        resource.canonical
        for route in app.router.routes()
        if (resource := route.resource) is not None
    }
    assert {
        "/api/t2v/config",
        "/api/t2v/prompt",
        "/api/t2v/download",
        "/api/t2v/playback",
    }.issubset(routes)


def test_t2v_session_records_completed_finite_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    class Decoder:
        spatial_compression_ratio = 8

    class Cache:
        def close(self) -> None:
            calls.append("cache.close")

    class Pipeline:
        decoder = Decoder()

        def initialize_cache(self, **kwargs: object) -> Cache:
            del kwargs
            return Cache()

        def get_num_output_frames(self, index: int) -> int:
            del index
            return 3

        def generate(self, *, autoregressive_index: int, cache: object) -> torch.Tensor:
            del autoregressive_index, cache
            return torch.zeros((3, 3, 2, 2))

        def finalize(
            self, *, autoregressive_index: int, cache: object
        ) -> dict[str, float]:
            del autoregressive_index, cache
            return {}

    class OutputTarget:
        def __init__(self, *, output_path: Path, **kwargs: object) -> None:
            del kwargs
            self.output_path = output_path

        def open(self) -> None:
            calls.append("output.open")

        def write(self, result: object) -> None:
            del result
            calls.append("output.write")

        def close(self) -> tuple[OutputArtifact, ...]:
            calls.append("output.close")
            return (OutputArtifact(kind="video/mp4", uri=str(self.output_path)),)

    monkeypatch.setattr(session_module, "StreamingVideoDecoder", Decoder)
    monkeypatch.setattr(session_module, "Mp4VideoOutputTarget", OutputTarget)
    recorded: list[tuple[Path, T2VScenario]] = []
    session = T2VSession(
        pipeline=cast(Any, Pipeline()),
        defaults=T2VSessionDefaults(
            prompt="A waterfall",
            total_blocks=1,
            pixel_height=64,
            pixel_width=96,
            fps=12,
        ),
        initial_input=InferenceInput(),
        output_layout="tchw",
        record_artifact=lambda path, scenario: recorded.append((path, scenario)),
        recording_directory=tmp_path,
    )

    assert session.next_step_request() is not None
    session.generate(InferenceInput())
    assert session.next_step_request() is None
    session.destroy()

    assert calls == ["output.open", "output.write", "output.close", "cache.close"]
    assert len(recorded) == 1
    assert recorded[0][0].parent == tmp_path
    assert recorded[0][1].prompt == "A waterfall"
