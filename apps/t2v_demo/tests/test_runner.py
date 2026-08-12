# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from t2v_demo import launch
from t2v_demo.runner import RUNNER_T2V, T2VDemoRunnerConfig
from t2v_demo.runtime import make_adapter

from flashdreams.infra.runner import LaunchOnlyRunner
from flashdreams.runtime import InferenceInput
from flashdreams.runtime.demo import DemoSpec, PreparedScenario
from flashdreams.runtime.demo.spec import WebRTCOutputSpec
from flashdreams.serving.webrtc.runtime import WebRTCSessionConfig

pytestmark = pytest.mark.ci_cpu


def test_t2v_runner_slug_has_launch_capability() -> None:
    assert RUNNER_T2V.runner_name == "t2v"
    assert RUNNER_T2V.launch_capability == "t2v_demo.launch:LAUNCH_CAPABILITY"
    resolved = launch.LAUNCH_CAPABILITY.resolve(
        RUNNER_T2V,
        mode="mp4",
        options=launch.LaunchOptions(),
    )
    assert resolved is not None
    assert resolved.label == "T2V mp4 launch"
    assert isinstance(RUNNER_T2V.setup(), LaunchOnlyRunner)


def test_t2v_runner_inherits_shared_launch_fields() -> None:
    config = T2VDemoRunnerConfig(runner_name="t2v", description="test")
    assert config.prompt is None
    assert config.total_blocks is None
    assert config.pixel_height is None
    assert config.pixel_width is None
    assert config.fps is None
    assert config.compile is None


def test_runner_mp4_launch_uses_demo_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = []

    def fake_replay_demo(*, spec: object, adapter: object) -> object:
        captured.append((spec, adapter))
        return type("Result", (), {"status": "completed"})()

    monkeypatch.setattr(launch, "run_replay_demo", fake_replay_demo)
    config = T2VDemoRunnerConfig(
        runner_name="t2v",
        description="test",
        backend="self-forcing",
        prompt="A waterfall",
        total_blocks=3,
    )

    launch.launch_t2v(
        config,
        "mp4",
        launch.LaunchOptions(output={"path": "outputs/test.mp4", "fps": 24}),
    )

    spec, _adapter = captured[0]
    assert spec.input_mode == "replay"
    assert spec.scenario["prompt"] == "A waterfall"
    assert spec.scenario["total_blocks"] == 3
    assert str(spec.output.path) == "outputs/test.mp4"
    assert spec.output.fps == 24


def test_t2v_uses_shared_demo_and_webrtc_defaults() -> None:
    adapter = make_adapter("self-forcing")
    assert adapter.supported_input_modes() == ("replay", "webrtc")
    assert adapter.supported_output_modes() == ("mp4", "null", "webrtc")
    provider = adapter.create_model_input_provider(
        DemoSpec(
            model_id=adapter.model_id,
            input_mode="replay",
            output=WebRTCOutputSpec(),
        ),
        PreparedScenario(initial_inputs=InferenceInput()),
    )
    assert provider.prepare_initial_input() == InferenceInput()
    assert WebRTCSessionConfig.from_output(WebRTCOutputSpec()) == WebRTCSessionConfig()
