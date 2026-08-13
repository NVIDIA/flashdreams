# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
import t2v.t2v as t2v_shell
import tomli
from t2v_demo import app
from t2v_demo.runner import RUNNER_T2V, T2VDemoRunnerConfig
from t2v_demo.runtime import backend_metadata, model_from_backend

from flashdreams.demo import (
    Application,
    DemoAdapterApplication,
)
from flashdreams.runtime.demo import Mp4OutputSpec, RunResult

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


def test_t2v_demo_no_longer_owns_backend_presets() -> None:
    package_dir = Path(__file__).parents[1]

    assert not (package_dir / "backends.py").exists()
    assert not (package_dir / "presets.py").exists()


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
    assert isinstance(public_app, DemoAdapterApplication)
    spec = public_app.spec
    assert spec.model_id == "self-forcing-t2v"
    assert spec.config is not None
    assert spec.config.runtime_options["backend"] == "self-forcing"
    assert spec.config.runtime_options["application"] == "self-forcing-t2v"
    scenario = spec.scenario
    assert isinstance(scenario, dict)
    assert scenario["prompt"] == "A waterfall"
    assert scenario["total_blocks"] == 3


def test_t2v_backend_bridge_builds_neutral_model_config() -> None:
    model = model_from_backend("self-forcing")

    assert isinstance(model, t2v_shell.T2VModelConfig)
    assert model.model_id == "self-forcing-t2v"
    assert model.runtime_options["backend"] == "self-forcing"


def test_t2v_backend_bridge_supports_integration_owned_preset() -> None:
    model = model_from_backend(
        "self-forcing",
        "self-forcing-wan2.1-t2v-1.3b-sink5-window7-rerope",
    )

    assert model.model_id == "self-forcing-t2v"
    assert model.preset_id == "self-forcing-wan2.1-t2v-1.3b-sink5-window7-rerope"
    assert model.total_blocks == 80


def test_t2v_backend_metadata_is_derived_from_integrations() -> None:
    metadata = {item["key"]: item for item in backend_metadata()}

    assert metadata["self-forcing"]["default_preset"] == (
        "self-forcing-wan2.1-t2v-1.3b"
    )
    assert "self-forcing-wan2.1-t2v-1.3b-taehv" in metadata["self-forcing"]["presets"]
    assert metadata["cosmos-predict2"]["application"] == "cosmos-predict2-t2v"


def test_runner_mp4_launch_uses_demo_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_application_replay(*, app: Application) -> RunResult:
        captured["app"] = app
        return RunResult(status="completed")

    monkeypatch.setattr(app, "run_application_replay", fake_run_application_replay)
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
    assert isinstance(public_app, DemoAdapterApplication)
    scenario = public_app.spec.scenario
    assert isinstance(scenario, dict)
    assert scenario["prompt"] == "A waterfall"
    assert scenario["total_blocks"] == 3
    output = public_app.spec.output
    assert isinstance(output, Mp4OutputSpec)
    assert str(output.path) == "outputs/test.mp4"
    assert output.fps == 24
    assert output.output_layout == "tchw"
