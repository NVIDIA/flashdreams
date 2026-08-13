# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Literal, cast

import pytest
from flashdreams.runtime import (
    InferenceConfig,
    StepRequirements,
    UserInputEvent,
    UserInputs,
)
from flashdreams.runtime.demo import (
    DemoSpec,
    Mp4OutputSpec,
    NativeWindowOutputSpec,
    NullOutputSpec,
    UserInputWindow,
    WebRTCOutputSpec,
)
from flashdreams.serving.launch import LaunchOptions
from flashdreams.serving.native_window import run_native_window_demo
from triangle_app import TriangleScenario, launch
from triangle_app.launch import LAUNCH_CAPABILITY
from triangle_app.runner import (
    RUNNER_TRIANGLE_APP,
    TriangleAppRunnerConfig,
)
from triangle_model import MODEL_ID, TriangleModel

pytestmark = pytest.mark.ci_cpu


def _spec(*, frames: int = 3) -> DemoSpec:
    return DemoSpec(
        model_id=MODEL_ID,
        input_mode="keyboard-driving",
        output=NativeWindowOutputSpec(
            fps=1000,
            video_width=16,
            video_height=16,
            close_timeout_s=1.0,
        ),
        scenario=TriangleScenario(
            width=16,
            height=16,
            fps=1000,
            total_frames=frames,
        ),
        config=InferenceConfig(model_id=MODEL_ID, device="cpu"),
    )


def test_triangle_app_registers_all_output_modes() -> None:
    config = TriangleAppRunnerConfig(
        runner_name="triangle-app",
        description="test",
        device="cpu",
        model="triangle-model",
    )
    assert RUNNER_TRIANGLE_APP.runner_name == "triangle-app"
    assert LAUNCH_CAPABILITY.supported_modes(config, LaunchOptions()) == (
        "mp4",
        "null",
        "webrtc",
        "local-window",
    )


@pytest.mark.parametrize(
    ("mode", "output_type"),
    (("mp4", Mp4OutputSpec), ("null", NullOutputSpec)),
)
def test_triangle_app_uses_shared_replay_outputs(
    mode: Literal["mp4", "null"],
    output_type: type[object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[DemoSpec] = []
    completed = SimpleNamespace(status="completed", reason=None, error=None)
    monkeypatch.setattr(
        launch,
        "run_replay_demo",
        lambda *, spec, adapter: captured.append(spec) or completed,
    )

    result = launch.launch_triangle_app(
        TriangleAppRunnerConfig(
            runner_name="triangle-app",
            description="test",
            device="cpu",
            model="triangle-model",
        ),
        mode=mode,
    )

    assert result is completed
    assert isinstance(captured[0].output, output_type)
    assert captured[0].input_mode == "replay"


def test_triangle_app_builds_webrtc_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager_args: dict[str, Any] = {}

    class _Manager:
        def __init__(self, **kwargs: Any) -> None:
            manager_args.update(kwargs)

    monkeypatch.setattr(launch, "BaseWebRTCSessionManager", _Manager)
    monkeypatch.setattr(
        launch,
        "serve_webrtc_demo",
        lambda **_kwargs: "served",
    )

    result = launch.launch_triangle_app(
        TriangleAppRunnerConfig(
            runner_name="triangle-app",
            description="test",
            device="cpu",
            model="triangle-model",
        ),
        mode="webrtc",
    )

    assert result == "served"
    spec = manager_args["shared_spec"]
    assert isinstance(spec.output, WebRTCOutputSpec)
    assert spec.input_mode == "keyboard-driving"
    manager_args["shared_host"].close()


def test_keyboard_input_changes_model_output() -> None:
    adapter = TriangleModel()
    spec = _spec(frames=1)
    scenario = adapter.prepare_scenario(spec)
    provider = adapter.create_model_input_provider(spec, scenario)
    prepared = provider.prepare_step(
        request=StepRequirements(step_index=0),
        user_window=UserInputWindow(
            start_s=0.0,
            end_s=1.0,
            inputs=UserInputs(
                events=(
                    UserInputEvent(
                        timestamp_s=0.5,
                        event_type="key_down",
                        payload={"key": "g"},
                    ),
                )
            ),
        ),
    )
    assert prepared.inference_input is not None
    assert spec.config is not None
    runtime = adapter.create_runtime(spec.config)
    session = runtime.start_session(scenario.initial_inputs)

    result = session.step(prepared.inference_input)

    pixels = result.video_chunk[0].permute(1, 2, 0)
    colors = pixels[pixels.ne(0).any(dim=-1)].unique(dim=0)
    assert colors.tolist() == [[64, 255, 128]]
    session.close()
    runtime.close()


class _Presenter:
    def __init__(self, **kwargs: object) -> None:
        del kwargs
        self.frames: list[object] = []
        self.closed = False

    @property
    def should_close(self) -> bool:
        return False

    def process_events(self) -> None:
        return None

    def present_frame(self, frame: object) -> None:
        self.frames.append(frame)

    def close(self) -> None:
        self.closed = True


class _KeyboardPresenter(_Presenter):
    def __init__(self, **kwargs: object) -> None:
        self._on_key = cast(Any, kwargs["on_key"])
        self._sent = False
        super().__init__(**kwargs)

    def process_events(self) -> None:
        if not self._sent:
            self._sent = True
            self._on_key("keydown", "g")


def test_triangle_model_composes_application_and_native_backend() -> None:
    presenter = _Presenter()

    result = run_native_window_demo(
        spec=_spec(),
        adapter=TriangleModel(),
        presenter_factory=lambda **_kwargs: presenter,
    )

    assert result.status == "completed"
    assert len(presenter.frames) == 3
    assert presenter.closed


def test_startup_keyboard_input_reaches_model() -> None:
    presenters: list[_KeyboardPresenter] = []

    def presenter_factory(**kwargs: object) -> _KeyboardPresenter:
        presenter = _KeyboardPresenter(**kwargs)
        presenters.append(presenter)
        return presenter

    run_native_window_demo(
        spec=_spec(frames=1),
        adapter=TriangleModel(),
        presenter_factory=presenter_factory,
    )

    frame = cast(Any, presenters[0].frames[0]).to_numpy()
    colors = frame[frame.any(axis=-1)]
    assert colors[0].tolist() == [64, 255, 128]
