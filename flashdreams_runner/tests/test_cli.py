# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the application runner lifecycle and mode boundary."""

from __future__ import annotations

import argparse
from types import ModuleType, SimpleNamespace

import pytest
import torch

import flashdreams_runner
from flashdreams.runtime import InferenceInput, OutputArtifact, StepResult
from flashdreams_runner import (
    AppConfig,
    Application,
    ApplicationArguments,
    DriveSession,
    IOHandler,
    Runtime,
    Session,
    cli,
)

pytestmark = pytest.mark.ci_cpu


def _config() -> AppConfig:
    return AppConfig(
        model_id="fake-app",
        fps=24,
        output_layout="tchw",
        video_width=64,
        video_height=64,
        default_steps=1,
    )


def _result(index: int = 0) -> StepResult:
    return StepResult.from_video_chunk(
        step_index=index,
        video_chunk=torch.zeros((1, 3, 2, 2)),
        layout="tchw",
    )


def test_public_package_surface_is_the_application_abi() -> None:
    assert flashdreams_runner.__all__ == [
        "AppConfig",
        "Application",
        "ApplicationArguments",
        "DriveSession",
        "IOHandler",
        "InputHandler",
        "OutputHandler",
        "Runtime",
        "Session",
    ]


def test_runner_owns_lifecycle_io_and_main_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class FakeSession(Session):
        def __init__(self) -> None:
            self._step_index = 0

        @property
        def step_index(self) -> int:
            return self._step_index

        def generate(self, inputs: InferenceInput) -> StepResult:
            assert not inputs.global_conditioning
            calls.append("session.generate")
            result = _result(self._step_index)
            self._step_index += 1
            return result

        def destroy(self) -> None:
            calls.append("session.destroy")

    class FakeRuntime(Runtime):
        @property
        def config(self) -> AppConfig:
            return _config()

        def initialize(self, *, device: str, io_handler: IOHandler) -> None:
            assert device == "cpu"
            assert io_handler.name == "fake"
            calls.append("runtime.initialize")

        def create_session(
            self, initial_input: InferenceInput | None = None
        ) -> Session:
            assert isinstance(initial_input, InferenceInput)
            calls.append("runtime.create_session")
            return FakeSession()

        def destroy(self) -> None:
            calls.append("runtime.destroy")

    runtime = FakeRuntime()
    application = ModuleType("fake_app")

    def create_runtime(arguments: ApplicationArguments) -> Runtime:
        calls.append("application.create_runtime")
        arguments.parser.add_argument("--model-option", required=True)
        options = arguments.parse_args()
        assert options.model_option == "enabled"
        return runtime

    setattr(application, "create_runtime", create_runtime)
    monkeypatch.setattr(cli, "load_application", lambda _: application)

    class Input:
        def open(self) -> None:
            calls.append("input.open")

        def initial_input(self) -> InferenceInput:
            calls.append("input.initial_input")
            return InferenceInput()

        def read(self) -> InferenceInput | None:
            calls.append("input.read")
            if calls.count("input.read") == 1:
                return InferenceInput()
            return None

        def close(self) -> None:
            calls.append("input.close")

    class Output:
        def open(self, config: AppConfig) -> None:
            assert config.model_id == "fake-app"
            calls.append("output.open")

        def write(self, result: StepResult) -> None:
            assert result.step_index == 0
            calls.append("output.write")

        def close(self) -> tuple[OutputArtifact, ...]:
            calls.append("output.close")
            return ()

    class Mode:
        name = "fake"

        def run(
            self, runtime: Runtime, drive_session: DriveSession
        ) -> tuple[OutputArtifact, ...]:
            calls.append("mode.run")
            return drive_session(runtime, Input(), Output())

    mode = Mode()
    monkeypatch.setattr(cli, "create_io_handler", lambda *args, **kwargs: mode)

    assert (
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
        == ()
    )
    assert calls == [
        "application.create_runtime",
        "runtime.initialize",
        "mode.run",
        "input.open",
        "output.open",
        "input.initial_input",
        "runtime.create_session",
        "input.read",
        "session.generate",
        "output.write",
        "input.read",
        "output.close",
        "session.destroy",
        "input.close",
        "runtime.destroy",
    ]


def test_application_protocol_requires_only_runtime_factory() -> None:
    application = ModuleType("application")
    setattr(application, "create_runtime", lambda arguments: None)
    assert isinstance(application, Application)

    delattr(application, "create_runtime")
    assert not isinstance(application, Application)


def test_load_application_rejects_module_outside_abi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("invalid_application")
    distribution = SimpleNamespace(metadata={"Name": "invalid-app"})
    monkeypatch.setattr(cli.metadata, "distribution", lambda name: distribution)
    monkeypatch.setattr(
        cli.metadata,
        "packages_distributions",
        lambda: {"invalid_application": ["invalid-app"]},
    )
    monkeypatch.setattr(cli.importlib, "import_module", lambda name: module)

    with pytest.raises(TypeError, match="none satisfy Application"):
        cli.load_application("invalid-app")


def test_runner_exposes_mode_names_and_preserves_application_arguments() -> None:
    route = cli._parse_application_and_mode(["fake-app", "webrtc", "--prompt", "x"])
    assert route.application == "fake-app"
    assert route.mode == "webrtc"
    assert route.remaining_argv == ("--prompt", "x")

    for mode in ("mp4", "replay", "webrtc", "none"):
        assert cli._parse_application_and_mode(["fake-app", mode]).mode == mode

    with pytest.raises(SystemExit):
        cli._parse_application_and_mode(["fake-app", "unsupported"])


def test_mode_parsers_keep_transport_options_separate() -> None:
    mp4_destinations = {
        action.dest for action in cli.build_parser("fake-app", "mp4")._actions
    }
    assert {"device", "output", "steps"} <= mp4_destinations
    assert {"host", "port"}.isdisjoint(mp4_destinations)

    webrtc_destinations = {
        action.dest for action in cli.build_parser("fake-app", "webrtc")._actions
    }
    assert {"device", "host", "port"} <= webrtc_destinations
    assert {"output", "steps"}.isdisjoint(webrtc_destinations)


def test_application_arguments_must_be_parsed_by_factory() -> None:
    arguments = ApplicationArguments(
        mode="none",
        parser=argparse.ArgumentParser(),
        argv=(),
    )
    with pytest.raises(RuntimeError, match="must call arguments.parse_args"):
        _ = arguments.options

    assert isinstance(arguments.parse_args(), argparse.Namespace)
    assert arguments.options is arguments.parse_args()


def test_runtime_and_session_bridge_shared_inference_api() -> None:
    calls: list[str] = []

    class FakeSession(Session):
        @property
        def step_index(self) -> int:
            return 3

        def generate(self, inputs: InferenceInput) -> StepResult:
            calls.append("generate")
            return _result(3)

        def destroy(self) -> None:
            calls.append("session.destroy")

    session = FakeSession()

    class FakeRuntime(Runtime):
        @property
        def config(self) -> AppConfig:
            return _config()

        def initialize(self, *, device: str, io_handler: IOHandler) -> None:
            del device, io_handler

        def create_session(
            self, initial_input: InferenceInput | None = None
        ) -> Session:
            calls.append("create_session")
            return session

        def destroy(self) -> None:
            calls.append("runtime.destroy")

    runtime = FakeRuntime()
    assert runtime.start_session(InferenceInput()) is session
    request = session.next_step_request()
    assert request is not None
    assert request.step_index == 3
    assert session.step(InferenceInput()).step_index == 3
    session.close()
    runtime.close()
    assert calls == [
        "create_session",
        "generate",
        "session.destroy",
        "runtime.destroy",
    ]
