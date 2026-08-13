# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, cast

import pytest
from flashdreams.runtime import StepRequirements, UserInputEvent, UserInputs
from flashdreams.runtime.demo import (
    DemoSpec,
    FlashDreamsApplication,
    NativeWindowOutputSpec,
    UserInputWindow,
)
from flashdreams.serving.native_window import run_native_window_demo
from triangle_model import TriangleModel, create_app

pytestmark = pytest.mark.ci_cpu


def _application(*, frames: int = 3) -> TriangleModel:
    return create_app(
        [
            "--width",
            "16",
            "--height",
            "16",
            "--fps",
            "1000",
            "--total-frames",
            str(frames),
        ]
    )


def _spec(application: TriangleModel) -> DemoSpec:
    return DemoSpec(
        model_id=application.model_id,
        input_mode="keyboard-driving",
        output=NativeWindowOutputSpec(
            fps=application.fps,
            video_width=application.video_width,
            video_height=application.video_height,
            close_timeout_s=1.0,
        ),
        scenario=application.scenario,
        config=application.config,
    )


def test_create_app_returns_flashdreams_application() -> None:
    application = _application()

    assert isinstance(application, FlashDreamsApplication)
    assert application.application_name == "triangle-model"
    assert application.video_width == 16
    assert application.default_mode == "local-window"


def test_keyboard_input_changes_model_output() -> None:
    application = _application(frames=1)
    spec = _spec(application)
    scenario = application.prepare_scenario(spec)
    provider = application.create_model_input_provider(spec, scenario)
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
    runtime = application.create_runtime(application.config)
    session = runtime.start_session(scenario.initial_inputs)

    result = session.step(prepared.inference_input)

    pixels = result.video_chunk[0].permute(1, 2, 0)
    colors = pixels[pixels.ne(0).any(dim=-1)].unique(dim=0)
    assert colors.tolist() == [[64, 255, 128]]
    session.close()
    runtime.close()


class _Presenter:
    def __init__(self, **kwargs: object) -> None:
        self._on_key = cast(Any, kwargs["on_key"])
        self._sent = False
        self.frames: list[object] = []

    @property
    def should_close(self) -> bool:
        return False

    def process_events(self) -> None:
        if not self._sent:
            self._sent = True
            self._on_key("keydown", "b")

    def present_frame(self, frame: object) -> None:
        self.frames.append(frame)

    def close(self) -> None:
        return None


def test_application_runs_end_to_end_with_native_input() -> None:
    application = _application(frames=1)
    presenter = _Presenter(on_key=lambda _event, _key: None)

    def presenter_factory(**kwargs: object) -> _Presenter:
        presenter._on_key = cast(Any, kwargs["on_key"])
        return presenter

    result = run_native_window_demo(
        spec=_spec(application),
        adapter=application,
        presenter_factory=presenter_factory,
    )

    assert result.status == "completed"
    frame = cast(Any, presenter.frames[0]).to_numpy()
    colors = frame[frame.any(axis=-1)]
    assert colors[0].tolist() == [64, 128, 255]
