# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
import tomli
from t2v_demo import app
from t2v_demo.runner import RUNNER_T2V, T2VDemoRunnerConfig

from flashdreams.demo import Application, FileOutputSink, ReplayIOHandler
from flashdreams.runtime.demo import RunResult

pytestmark = pytest.mark.ci_cpu


def test_t2v_runner_slug_has_launch_capability() -> None:
    assert RUNNER_T2V.runner_name == "t2v"
    assert RUNNER_T2V.launch_capability == "t2v_demo.launch:LAUNCH_CAPABILITY"


def test_t2v_registers_application_entry_point() -> None:
    pyproject_path = Path(__file__).parents[1] / "pyproject.toml"
    pyproject = tomli.loads(pyproject_path.read_text())

    assert pyproject["project"]["entry-points"]["flashdreams.applications"] == {
        "t2v": "t2v_demo.app:create_app"
    }


def test_t2v_create_app_exposes_public_application() -> None:
    public_app = app.create_app(
        T2VDemoRunnerConfig(
            runner_name="t2v-test",
            description="test",
            backend="self-forcing",
            prompt="A waterfall",
            total_blocks=3,
        )
    )

    assert app.createApp is app.create_app
    assert isinstance(public_app, Application)
    assert isinstance(public_app, app.T2VApplication)
    assert public_app.defaults.backend == "self-forcing"
    assert public_app.defaults.prompt == "A waterfall"
    assert public_app.defaults.total_blocks == 3


def test_runner_mp4_launch_uses_demo_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeRunner:
        def __init__(self, *, io_handler: object, app: object) -> None:
            captured["io_handler"] = io_handler
            captured["app"] = app

        def run(self) -> RunResult:
            return RunResult(status="completed")

    monkeypatch.setattr(app, "Runner", FakeRunner)
    config = T2VDemoRunnerConfig(
        runner_name="t2v",
        description="test",
        backend="self-forcing",
        prompt="A waterfall",
        total_blocks=3,
    )

    app.launch_t2v(
        config=config,
        mode="mp4",
        output_overrides={"path": "outputs/test.mp4", "fps": 24},
    )

    public_app = captured["app"]
    io_handler = captured["io_handler"]
    assert isinstance(public_app, app.T2VApplication)
    assert isinstance(io_handler, ReplayIOHandler)
    assert public_app.defaults.prompt == "A waterfall"
    assert public_app.defaults.total_blocks == 3
    output_sink = io_handler.output_sink
    assert isinstance(output_sink, FileOutputSink)
    assert str(output_sink.output_path) == "outputs/test.mp4"
    assert output_sink.fps == 24
    assert output_sink.output_layout == "tchw"
