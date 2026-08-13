# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from flashdreams.runtime import (
    ApplicationRunner,
    CanonicalInputSchema,
    FlashDreamsApplication,
    IdentityInputMapping,
    InferenceConfig,
    InferenceInputSchema,
    IOHandler,
)
from flashdreams.runtime.demo import NullOutputSpec, OutputSpec
from flashdreams.scripts import cli
from flashdreams.serving import application_launcher
from flashdreams.serving.io_handlers import (
    Mp4IOHandler,
    NullIOHandler,
)

pytestmark = pytest.mark.ci_cpu


class _Application:
    application_name = "example"
    description = "Example application"
    model_id = "example-model"
    inference_input_schema = InferenceInputSchema()
    canonical_input_schema = CanonicalInputSchema()
    config = InferenceConfig(model_id=model_id, device="cpu")
    scenario = None
    fps = 30
    video_width = 16
    video_height = 16
    output_layout = "tchw"
    default_io_handler = "null"
    title = None
    supported_control_keys = frozenset()

    def initialize(self, config: InferenceConfig) -> None:
        del config

    def create_session(self, inputs: Any) -> Any:
        del inputs
        raise AssertionError("session should not be created")

    def close(self) -> None:
        return None

    def default_input_mapping(self) -> IdentityInputMapping:
        return IdentityInputMapping()

    def validate_config(self, config: InferenceConfig) -> None:
        del config

    def create_runtime(self, config: InferenceConfig) -> Any:
        del config
        raise AssertionError("runtime should not be created")

    def prepare_scenario(self, spec: Any) -> Any:
        del spec
        raise AssertionError("scenario should not be prepared")

    def create_model_input_provider(self, spec: Any, scenario: Any) -> Any:
        del spec, scenario
        raise AssertionError("provider should not be created")


@dataclass(frozen=True)
class _EntryPoint:
    name: str
    value: object

    def load(self) -> object:
        return self.value


@dataclass(frozen=True)
class _BrokenEntryPoint:
    name: str

    def load(self) -> object:
        raise RuntimeError("broken application")


class _ReactorIOHandler:
    input_mode = "reactor"
    realtime = True

    @classmethod
    def from_argv(cls, args: Sequence[str]) -> tuple[IOHandler, list[str]]:
        return cls(), list(args)

    def create_output(self, application: FlashDreamsApplication) -> OutputSpec:
        del application
        return NullOutputSpec()

    def run(self, runner: ApplicationRunner) -> object:
        return runner


def _io_handlers() -> dict[str, _EntryPoint]:
    return {
        "mp4": _EntryPoint("mp4", Mp4IOHandler),
        "null": _EntryPoint("null", NullIOHandler),
        "reactor": _EntryPoint("reactor", _ReactorIOHandler),
    }


def _load_io_handler(
    name: str,
    args: Sequence[str],
) -> tuple[IOHandler, list[str]]:
    handler_type = _io_handlers()[name].load()
    assert isinstance(handler_type, type)
    handler, remaining = handler_type.from_argv(args)
    assert isinstance(handler, IOHandler)
    return handler, remaining


def test_application_entry_point_receives_remaining_arguments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    factory_args: list[list[str]] = []
    runners: list[ApplicationRunner] = []

    def factory(args: list[str]) -> _Application:
        factory_args.append(args)
        return _Application()

    monkeypatch.setattr(
        application_launcher,
        "entry_points",
        lambda **_kwargs: [_EntryPoint("example", factory)],
    )
    application_launcher.application_entry_points.cache_clear()
    monkeypatch.setattr(
        application_launcher,
        "io_handler_entry_points",
        _io_handlers,
    )
    monkeypatch.setattr(application_launcher, "load_io_handler", _load_io_handler)
    monkeypatch.setattr(
        ApplicationRunner,
        "run",
        lambda self: runners.append(self),
    )
    output = tmp_path / "result.mp4"

    handled = application_launcher.run_application_from_argv(
        [
            "example",
            "mp4",
            f"--output={output}",
            "--model-flag",
            "value",
        ]
    )

    assert handled
    assert factory_args == [["--model-flag", "value"]]
    assert isinstance(runners[0].io_handler, Mp4IOHandler)
    assert runners[0].io_handler.output_path == output


def test_unknown_application_falls_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(application_launcher, "entry_points", lambda **_kwargs: [])
    application_launcher.application_entry_points.cache_clear()

    assert not application_launcher.run_application_from_argv(["missing"])


def test_application_accepts_discovered_io_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runners: list[ApplicationRunner] = []

    def factory(_args: list[str]) -> _Application:
        return _Application()

    monkeypatch.setattr(
        application_launcher,
        "application_entry_points",
        lambda: {"example": _EntryPoint("example", factory)},
    )
    monkeypatch.setattr(
        application_launcher,
        "io_handler_entry_points",
        _io_handlers,
    )
    monkeypatch.setattr(application_launcher, "load_io_handler", _load_io_handler)
    monkeypatch.setattr(
        ApplicationRunner,
        "run",
        lambda self: runners.append(self),
    )

    assert application_launcher.run_application_from_argv(["example", "reactor"])
    assert isinstance(runners[0].io_handler, _ReactorIOHandler)


def test_application_no_instantiate_skips_execution(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    entry_point = _EntryPoint("example", lambda _args: _Application())
    monkeypatch.setattr(
        application_launcher,
        "application_entry_points",
        lambda: {"example": entry_point},
    )
    monkeypatch.setattr(
        application_launcher,
        "io_handler_entry_points",
        _io_handlers,
    )
    monkeypatch.setattr(application_launcher, "load_io_handler", _load_io_handler)
    monkeypatch.setattr(
        ApplicationRunner,
        "run",
        lambda *_args, **_kwargs: pytest.fail("runner should not run"),
    )

    assert application_launcher.run_application_from_argv(
        ["example", "null", "--no-instantiate"]
    )
    assert "Resolved application" in capsys.readouterr().out


def test_unselected_broken_application_is_not_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        application_launcher,
        "entry_points",
        lambda **_kwargs: [_BrokenEntryPoint("broken")],
    )
    application_launcher.application_entry_points.cache_clear()

    assert not application_launcher.run_application_from_argv(["other"])


def test_application_runner_name_collision_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry_point = _EntryPoint("example", lambda _args: _Application())
    monkeypatch.setattr(
        application_launcher,
        "application_entry_points",
        lambda: {"example": entry_point},
    )
    monkeypatch.setattr(cli, "all_runners", lambda: {"example": object()})

    with pytest.raises(ValueError, match="both an application and a runner"):
        cli.entrypoint(["example"])
