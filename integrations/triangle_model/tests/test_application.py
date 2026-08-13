# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, cast

import pytest
from flashdreams.runtime import (
    ApplicationRunner,
    FlashDreamsApplication,
    UserInputEvent,
    UserInputs,
)
from flashdreams.runtime.demo import (
    DemoSpec,
    NativeWindowOutputSpec,
)
from flashdreams.serving.application_launcher import (
    application_entry_points,
    run_application_from_argv,
)
from flashdreams.serving.io_handlers import NativeWindowIOHandler
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
    assert application.default_io_handler == "local-window"


def test_installed_application_entry_point_runs() -> None:
    application_entry_points.cache_clear()

    assert "triangle-model" in application_entry_points()
    assert run_application_from_argv(
        [
            "triangle-model",
            "null",
            "--width",
            "8",
            "--height",
            "8",
            "--total-frames",
            "1",
        ]
    )


def test_keyboard_input_changes_model_output() -> None:
    application = _application(frames=1)
    application.initialize(application.config)
    session = application.create_session(application.scenario)
    event = session.next_event()
    assert event is not None

    result = session.generate(
        event,
        UserInputs(
            events=(
                UserInputEvent(
                    timestamp_s=0.5,
                    event_type="key_down",
                    payload={"key": "g"},
                ),
            )
        ),
    )

    pixels = result.video_chunk[0].permute(1, 2, 0)
    colors = pixels[pixels.ne(0).any(dim=-1)].unique(dim=0)
    assert colors.tolist() == [[64, 255, 128]]
    session.close()
    application.close()


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

    application.native_presenter_factory = presenter_factory
    result = ApplicationRunner(
        application=application,
        io_handler=NativeWindowIOHandler(),
    ).run()

    assert result.status == "completed"
    frame = cast(Any, presenter.frames[0]).to_numpy()
    colors = frame[frame.any(axis=-1)]
    assert colors[0].tolist() == [64, 128, 255]
