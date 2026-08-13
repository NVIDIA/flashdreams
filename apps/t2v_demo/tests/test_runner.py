# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
import tomli
from t2v_demo import app
from t2v_demo.runner import RUNNER_T2V, T2VDemoRunnerConfig

from flashdreams.demo import Application

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


def test_runner_mp4_launch_uses_demo_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = []

    def fake_replay_demo(*, spec: object, adapter: object) -> object:
        captured.append((spec, adapter))
        return type("Result", (), {"status": "completed"})()

    monkeypatch.setattr(app, "run_replay_demo", fake_replay_demo)
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

    spec, _adapter = captured[0]
    assert spec.input_mode == "replay"
    assert spec.scenario["prompt"] == "A waterfall"
    assert spec.scenario["total_blocks"] == 3
    assert str(spec.output.path) == "outputs/test.mp4"
    assert spec.output.fps == 24
