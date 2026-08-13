# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import flashdreams_app
import pytest
import torch
from flashdreams_app import (
    AppConfig,
    AppProvider,
    AppRequest,
    AppSpec,
    Mp4RunSpec,
    PipelineAppSpec,
    WebRTCRunSpec,
    cli,
)

from flashdreams.infra.pipeline import StreamInferencePipelineConfig
from flashdreams.runtime import InferenceInput, StepResult

pytestmark = pytest.mark.ci_cpu


def test_public_package_surface_contains_only_provider_contracts() -> None:
    assert flashdreams_app.__all__ == [
        "AppConfig",
        "AppProvider",
        "AppRequest",
        "AppSpec",
        "Mp4RunSpec",
        "PipelineAppSpec",
        "WebRTCRunSpec",
    ]
    assert "initial_input" not in PipelineAppSpec.__dataclass_fields__
    assert "total_steps" not in PipelineAppSpec.__dataclass_fields__


def test_host_drives_runtime_api_and_owns_file_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class Cache:
        def close(self) -> None:
            calls.append("cache.close")

    class Pipeline:
        def __init__(self, config: object) -> None:
            assert config is pipeline_config
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

    def parse_options(
        parser: argparse.ArgumentParser, argv: Sequence[str]
    ) -> Mapping[str, object]:
        calls.append("provider.parse_options")
        parser.add_argument("--model-option", required=True)
        return vars(parser.parse_args(argv))

    def create_app_spec(request: AppRequest) -> AppSpec:
        assert request.mode == "mp4"
        assert request.options["model_option"] == "enabled"
        calls.append("provider.create_app_spec")
        return AppSpec(
            config=AppConfig(
                model_id="fake",
                fps=24,
                output_layout="tchw",
                video_width=64,
                video_height=64,
            ),
            pipeline=PipelineAppSpec(
                pipeline_config=pipeline_config,
                initialize_cache=initialize_cache,
            ),
            run=Mp4RunSpec(
                initial_input=InferenceInput(global_conditioning={"prompt": "test"}),
                total_steps=1,
            ),
        )

    setattr(provider, "parse_options", parse_options)
    setattr(provider, "create_app_spec", create_app_spec)
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
    cli.run(
        [
            "fake-app",
            "mp4",
            "--device",
            "cpu",
            "--output",
            "result.mp4",
            "--model-option",
            "enabled",
        ]
    )
    assert calls == [
        "provider.parse_options",
        "provider.create_app_spec",
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


def test_run_delegates_options_and_dispatches_webrtc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = ModuleType("fake_app")
    pipeline_config = StreamInferencePipelineConfig(
        _target=cast(Any, object),
        name="fake",
        diffusion_model=cast(Any, None),
    )
    initial_input = InferenceInput(global_conditioning={"prompt": "test"})

    def parse_options(
        parser: argparse.ArgumentParser, argv: Sequence[str]
    ) -> Mapping[str, object]:
        assert tuple(argv) == (
            "--host",
            "127.0.0.1",
            "--port",
            "9000",
            "--prompt",
            "test",
        )
        parser.add_argument("--prompt", required=True)
        return vars(parser.parse_args(argv))

    def create_app_spec(request: AppRequest) -> AppSpec:
        assert request.mode == "webrtc"
        assert request.options["prompt"] == "test"
        return AppSpec(
            config=AppConfig(
                model_id="fake",
                fps=24,
                output_layout="tchw",
                video_width=64,
                video_height=64,
            ),
            pipeline=PipelineAppSpec(
                pipeline_config=pipeline_config,
                initialize_cache=lambda pipeline, inputs: object(),
            ),
            run=WebRTCRunSpec(initial_input=initial_input),
        )

    setattr(provider, "parse_options", parse_options)
    setattr(provider, "create_app_spec", create_app_spec)
    monkeypatch.setattr(cli, "load_provider", lambda _: provider)
    environment = cli._Environment(device="cpu", world_rank=0, world_size=1)
    monkeypatch.setattr(cli, "_initialize_environment", lambda device: environment)
    runtime = object()
    monkeypatch.setattr(cli, "PipelineAppRuntime", lambda **kwargs: runtime)

    captured: dict[str, object] = {}

    def run_webrtc(**kwargs: object) -> tuple[object, ...]:
        captured.update(kwargs)
        return ()

    monkeypatch.setattr(cli, "_run_webrtc", run_webrtc)
    monkeypatch.setattr(
        cli,
        "_run_mp4",
        lambda **kwargs: pytest.fail("WebRTC mode must not launch the MP4 path."),
    )

    assert (
        cli.run(
            [
                "fake-app",
                "webrtc",
                "--host",
                "127.0.0.1",
                "--port",
                "9000",
                "--prompt",
                "test",
            ]
        )
        == ()
    )
    assert captured["runtime"] is runtime
    assert captured["run_spec"] == WebRTCRunSpec(initial_input=initial_input)
    assert captured["environment"] is environment
    args = captured["args"]
    assert isinstance(args, argparse.Namespace)
    assert args.host == "127.0.0.1"
    assert args.port == 9000


def test_app_provider_protocol_requires_both_methods() -> None:
    provider = ModuleType("provider")
    setattr(provider, "parse_options", lambda parser, argv: {})
    setattr(provider, "create_app_spec", lambda config: None)
    assert isinstance(provider, AppProvider)

    delattr(provider, "parse_options")
    assert not isinstance(provider, AppProvider)


def test_load_provider_rejects_module_outside_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("invalid_provider")
    distribution = SimpleNamespace(metadata={"Name": "invalid-app"})
    monkeypatch.setattr(cli.metadata, "distribution", lambda name: distribution)
    monkeypatch.setattr(
        cli.metadata,
        "packages_distributions",
        lambda: {"invalid_provider": ["invalid-app"]},
    )
    monkeypatch.setattr(cli.importlib, "import_module", lambda name: module)

    with pytest.raises(TypeError, match="none satisfy AppProvider"):
        cli.load_provider("invalid-app")


def test_host_exposes_only_supported_output_modes() -> None:
    route = cli._parse_provider_and_mode(["fake-app", "webrtc", "--prompt", "x"])
    assert route.provider == "fake-app"
    assert route.mode == "webrtc"
    assert route.remaining_argv == ("--prompt", "x")

    with pytest.raises(SystemExit):
        cli._parse_provider_and_mode(["fake-app", "unsupported"])


def test_host_does_not_expose_pipeline_execution_options() -> None:
    for mode in ("mp4", "webrtc"):
        destinations = {
            action.dest for action in cli.build_parser("fake-app", mode)._actions
        }
        assert "compile" not in destinations
        assert "cuda_graph" not in destinations


def test_host_exposes_only_minimal_webrtc_options() -> None:
    destinations = {
        action.dest for action in cli.build_parser("fake-app", "webrtc")._actions
    }
    assert {"host", "port"} <= destinations
    assert "output" not in destinations
    assert {
        "warmup_chunks",
        "warmup_timeout_s",
        "client_liveness_timeout_s",
        "encoder_backend",
        "encoder_bitrate_bps",
        "encoder_gop",
    }.isdisjoint(destinations)


def test_mp4_parser_exposes_only_file_presentation_options() -> None:
    destinations = {
        action.dest for action in cli.build_parser("fake-app", "mp4")._actions
    }
    assert {"device", "output"} <= destinations
    assert {"host", "port"}.isdisjoint(destinations)


def test_webrtc_path_owns_serving_options_and_runtime_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    captured: dict[str, object] = {}

    class Runtime:
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
        config=AppConfig(
            model_id="fake",
            fps=24,
            output_layout="tchw",
            video_width=64,
            video_height=64,
        ),
        run_spec=WebRTCRunSpec(initial_input=InferenceInput()),
        args=argparse.Namespace(
            host="127.0.0.1",
            port=9000,
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
    assert captured["device"] == "cpu"
    assert isinstance(captured["initial_input"], InferenceInput)
