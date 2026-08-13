# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest
import torch
from flashdreams_app import (
    PipelineAppRuntime,
    PipelineAppSpec,
    PipelineContract,
    RuntimeMetadata,
    cli,
)

from flashdreams.infra.pipeline import StreamInferencePipelineConfig
from flashdreams.runtime import InferenceInput, StepResult

pytestmark = pytest.mark.ci_cpu


def test_host_drives_runtime_api_and_owns_file_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class Cache:
        def close(self) -> None:
            calls.append("cache.close")

    class Pipeline:
        def __init__(self, config: object) -> None:
            del config
            calls.append("pipeline.init")

        def to(self, device: str) -> "Pipeline":
            assert device == "cpu"
            calls.append("pipeline.to")
            return self

        def eval(self) -> "Pipeline":
            calls.append("pipeline.eval")
            return self

        def generate(self, *, autoregressive_index: int, cache: object) -> torch.Tensor:
            assert autoregressive_index == 0
            assert isinstance(cache, Cache)
            calls.append("pipeline.generate")
            return torch.zeros((1, 3, 2, 2))

        def finalize(
            self, *, autoregressive_index: int, cache: object
        ) -> dict[str, float]:
            assert autoregressive_index == 0
            assert isinstance(cache, Cache)
            calls.append("pipeline.finalize")
            return {"step_ms": 1.0}

        def close(self) -> None:
            calls.append("pipeline.close")

    pipeline_config = StreamInferencePipelineConfig(
        _target=cast(Any, Pipeline),
        name="fake",
        diffusion_model=cast(Any, None),
    )

    def initialize_cache(pipeline: object, inputs: InferenceInput) -> object:
        assert isinstance(pipeline, Pipeline)
        assert inputs.global_conditioning["prompt"] == "test"
        calls.append("contract.initialize_cache")
        return Cache()

    provider = ModuleType("fake_app")
    setattr(
        provider,
        "create_app",
        lambda config: PipelineAppSpec(
            pipeline_config=pipeline_config,
            contract=PipelineContract(initialize_cache=initialize_cache),
            metadata=RuntimeMetadata(
                model_id="fake",
                fps=24,
                output_layout="tchw",
                video_width=64,
                video_height=64,
            ),
            initial_input=InferenceInput(global_conditioning={"prompt": "test"}),
            total_steps=1,
        ),
    )
    monkeypatch.setattr(cli, "load_provider", lambda _: provider)

    class Output:
        def __init__(self, **_: object) -> None:
            calls.append("output.init")

        def open(self) -> None:
            calls.append("output.open")

        def write(self, result: StepResult) -> None:
            calls.append("output.write")

        def close(self) -> tuple[object, ...]:
            calls.append("output.close")
            return ()

    monkeypatch.setattr(cli, "FileOutput", Output)
    cli.run(["fake-app", "mp4", "--device", "cpu", "--output", "result.mp4"])
    assert calls == [
        "pipeline.init",
        "pipeline.to",
        "pipeline.eval",
        "output.init",
        "output.open",
        "contract.initialize_cache",
        "pipeline.generate",
        "pipeline.finalize",
        "output.write",
        "output.close",
        "cache.close",
        "pipeline.close",
    ]


def test_host_exposes_only_supported_output_modes() -> None:
    mode_action = next(
        action for action in cli.build_parser()._actions if action.dest == "mode"
    )
    assert mode_action.choices == ("mp4", "webrtc")


def test_webrtc_path_owns_serving_options_and_runtime_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    captured: dict[str, object] = {}

    class Runtime:
        metadata = RuntimeMetadata(
            model_id="fake",
            fps=24,
            output_layout="tchw",
            video_width=64,
            video_height=64,
        )
        initial_input = InferenceInput()

        def prepare_step_input(self, request: object) -> InferenceInput:
            del request
            return InferenceInput()

        def start_session(self, inputs: InferenceInput) -> Any:
            del inputs
            raise AssertionError("The WebRTC path must not start an MP4 session.")

        def close(self) -> None:
            calls.append("runtime.close")

    def serve(**kwargs: object) -> None:
        calls.append("serve_webrtc")
        captured.update(kwargs)

    monkeypatch.setattr(cli, "serve_webrtc", serve)
    result = cli._run_webrtc(
        runtime=Runtime(),
        args=argparse.Namespace(
            host="127.0.0.1",
            port=9000,
            warmup_chunks=2,
            warmup_timeout_s=30.0,
            client_liveness_timeout_s=10.0,
            encoder_backend="default",
            encoder_bitrate_bps=1_000_000,
            encoder_gop=None,
        ),
        environment=cli._Environment(device="cpu", world_rank=0, world_size=1),
    )

    assert result == ()
    assert calls == ["serve_webrtc", "runtime.close"]
    assert captured["world_rank"] == 0
    options = captured["options"]
    assert isinstance(options, cli.WebRTCOptions)
    assert options.host == "127.0.0.1"
    assert options.port == 9000
    assert options.encoder_gop == 24


def test_host_owns_execution_options() -> None:
    destinations = {action.dest for action in cli.build_parser()._actions}
    assert {"compile", "cuda_graph"} <= destinations


def test_host_applies_execution_options_without_mutating_provider_spec() -> None:
    configured_pipeline: StreamInferencePipelineConfig | None = None

    class Pipeline:
        def __init__(self, config: StreamInferencePipelineConfig) -> None:
            nonlocal configured_pipeline
            configured_pipeline = config

        def to(self, device: str) -> "Pipeline":
            assert device == "cpu"
            return self

        def eval(self) -> "Pipeline":
            return self

    transformer = SimpleNamespace(compile_network=False, use_cuda_graph=False)
    pipeline_config = StreamInferencePipelineConfig(
        _target=cast(Any, Pipeline),
        name="fake",
        diffusion_model=cast(Any, SimpleNamespace(transformer=transformer)),
    )
    spec = PipelineAppSpec(
        pipeline_config=pipeline_config,
        contract=PipelineContract(initialize_cache=lambda pipeline, inputs: object()),
        metadata=RuntimeMetadata(
            model_id="fake",
            fps=24,
            output_layout="tchw",
            video_width=64,
            video_height=64,
        ),
        initial_input=InferenceInput(),
        total_steps=1,
    )

    runtime = PipelineAppRuntime(
        spec=spec,
        device="cpu",
        compile=True,
        cuda_graph=True,
    )
    try:
        assert configured_pipeline is not None
        configured_transformer = configured_pipeline.diffusion_model.transformer
        assert getattr(configured_transformer, "compile_network") is True
        assert getattr(configured_transformer, "use_cuda_graph") is True
        assert transformer.compile_network is False
        assert transformer.use_cuda_graph is False
    finally:
        runtime.close()
