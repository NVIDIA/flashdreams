# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from dataclasses import replace
from types import SimpleNamespace

import pytest
from t2v.t2v import (
    T2VDemoAdapter,
    T2VModelConfig,
    T2VRunDefaults,
    create_t2v_application,
    create_t2v_spec,
    model_config_from_runner,
    t2v_scenario_mapping,
)

from flashdreams.demo import DemoAdapterApplication
from flashdreams.runtime.demo import (
    Mp4OutputSpec,
    NullOutputSpec,
    WebRTCOutputSpec,
)

pytestmark = pytest.mark.ci_cpu


def test_t2v_shell_builds_spec_from_model_defaults() -> None:
    model = _fake_model()

    spec = create_t2v_spec(
        model=model,
        defaults=T2VRunDefaults(device="cuda:0"),
        input_mode="replay",
        output=NullOutputSpec(),
    )

    assert spec.model_id == "fake-t2v"
    assert spec.preset_id == "fake-preset"
    assert spec.config is not None
    assert spec.config.device == "cuda:0"
    assert spec.config.runtime_options["family"] == "fake"
    assert spec.scenario == {
        "prompt": "A test prompt",
        "total_blocks": 2,
        "pixel_height": 32,
        "pixel_width": 64,
        "fps": 8,
    }


def test_t2v_shell_applies_launch_overrides() -> None:
    model = _fake_model()

    scenario = t2v_scenario_mapping(
        model=model,
        defaults=T2VRunDefaults(prompt="Override", total_blocks=4),
    )

    assert scenario["prompt"] == "Override"
    assert scenario["total_blocks"] == 4
    assert scenario["pixel_height"] == 32


def test_t2v_shell_creates_demo_adapter_application() -> None:
    public_app = create_t2v_application(
        model=_fake_model(),
        defaults=T2VRunDefaults(),
        output=Mp4OutputSpec(path="outputs/fake.mp4", fps=8, output_layout="tchw"),
    )

    assert isinstance(public_app, DemoAdapterApplication)
    assert isinstance(public_app.adapter, T2VDemoAdapter)
    assert public_app.spec.output.mode == "mp4"


def test_t2v_shell_builds_model_config_from_runner_config() -> None:
    runner = SimpleNamespace(
        runner_name="fake-runner",
        pipeline=object(),
        prompt="Runner prompt",
        total_blocks=5,
        pixel_height=48,
        pixel_width=96,
        fps=12,
    )

    model = model_config_from_runner(
        model_id="fake-t2v",
        runner=runner,
        runtime_options={"owner": "integration"},
    )

    assert model.model_id == "fake-t2v"
    assert model.preset_id == "fake-runner"
    assert model.pipeline is runner.pipeline
    assert model.prompt == "Runner prompt"
    assert model.total_blocks == 5
    assert model.runtime_options["owner"] == "integration"


def test_t2v_shell_enables_download_artifact_only_for_webrtc() -> None:
    replay_app = create_t2v_application(
        model=_fake_model(),
        output=Mp4OutputSpec(path="outputs/fake.mp4", fps=8, output_layout="tchw"),
    )
    webrtc_app = create_t2v_application(
        model=_fake_model(), input_mode="webrtc", output=WebRTCOutputSpec()
    )

    assert isinstance(replay_app, DemoAdapterApplication)
    assert isinstance(webrtc_app, DemoAdapterApplication)
    replay_adapter = replay_app.adapter
    webrtc_adapter = webrtc_app.adapter
    assert isinstance(replay_adapter, T2VDemoAdapter)
    assert isinstance(webrtc_adapter, T2VDemoAdapter)
    assert replay_adapter.write_download_artifact is False
    assert webrtc_adapter.write_download_artifact is True


def test_t2v_shell_rejects_non_positive_scenario_override() -> None:
    model = _fake_model()
    adapter = T2VDemoAdapter(model=model)
    spec = create_t2v_spec(model=model, input_mode="replay", output=NullOutputSpec())
    spec = replace(spec, scenario={**dict(spec.scenario or {}), "total_blocks": 0})

    with pytest.raises(ValueError, match="total_blocks"):
        adapter.prepare_scenario(spec)


def test_t2v_shell_has_no_legacy_backend_imports() -> None:
    import t2v.t2v as t2v_shell

    source = inspect.getsource(t2v_shell)

    assert "t2v_demo" not in source
    assert "causal_forcing" not in source
    assert "self_forcing" not in source
    assert "cosmos_predict2" not in source


def _fake_model() -> T2VModelConfig:
    return T2VModelConfig(
        model_id="fake-t2v",
        preset_id="fake-preset",
        pipeline=object(),
        prompt="A test prompt",
        total_blocks=2,
        pixel_height=32,
        pixel_width=64,
        fps=8,
        runtime_options={"family": "fake"},
    )
