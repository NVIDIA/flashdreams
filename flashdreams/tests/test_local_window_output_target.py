# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest
import torch
from flashdreams.infra.video_output import VideoStepResult
from flashdreams.runtime import RGB_VIDEO, StepResult, TimeWindow
from flashdreams.runtime.demo import DemoSpec, LocalWindowOutputSpec
from flashdreams.runtime.demo.local_window import build_local_window_io
from flashdreams.serving.presentation import DisplayFrame, NullOverlay, WindowConfig
from flashdreams.serving.presentation.output import LocalWindowVideoOutputTarget

pytestmark = pytest.mark.ci_cpu


class _Presenter:
    def __init__(self) -> None:
        self.should_close = False
        self.processed_events = 0
        self.prepared: list[DisplayFrame] = []
        self.presented: list[DisplayFrame] = []
        self.closed = False

    def process_events(self) -> None:
        self.processed_events += 1

    def prepare_frame(self, frame: DisplayFrame) -> None:
        self.prepared.append(frame)

    def present_frame(self, frame: DisplayFrame) -> None:
        self.presented.append(frame)

    def close(self) -> None:
        self.closed = True


def _video_result() -> StepResult:
    return StepResult(
        step_index=0,
        output=VideoStepResult.from_video_chunk(
            chunk_index=0,
            video_chunk=torch.zeros((1, 1, 2, 3, 2, 2)),
            layout="bvtchw",
        ),
        frame_count=2,
        output_window=TimeWindow(start_s=1.0, end_s=2.0),
    )


def _target() -> tuple[LocalWindowVideoOutputTarget, _Presenter]:
    presenter = _Presenter()

    def factory(**kwargs: Any) -> _Presenter:
        assert kwargs["overlay"] is not None
        assert kwargs["config"] == WindowConfig(width=640, height=480)
        return presenter

    target = LocalWindowVideoOutputTarget(
        overlay=NullOverlay(),
        config=WindowConfig(width=640, height=480),
        presenter_factory=factory,
    )
    return target, presenter


def test_target_declares_rgb_video_compatibility() -> None:
    target, _ = _target()

    assert target.output_requirement.modalities == frozenset({RGB_VIDEO})
    assert target.output_requirement.python_type is VideoStepResult


def test_video_chunk_is_presented_as_lazy_display_frames() -> None:
    target, presenter = _target()
    target.open()

    target.write(_video_result())

    assert len(presenter.presented) == 2
    assert presenter.prepared == presenter.presented
    assert [frame.timestamp_us for frame in presenter.presented] == [
        1_000_000,
        1_500_000,
    ]
    assert presenter.processed_events == 2
    target.close()


def test_wrong_step_output_type_is_rejected() -> None:
    target, _ = _target()
    target.open()

    with pytest.raises(TypeError, match="requires StepResult.output"):
        target.write(StepResult(step_index=0, output="not video"))

    target.close()


def test_closed_window_stops_remaining_frames() -> None:
    target, presenter = _target()
    target.open()
    presenter.should_close = True

    target.write(_video_result())

    assert presenter.presented == []
    assert target.should_stop
    target.close()


def test_close_releases_the_presenter() -> None:
    target, presenter = _target()
    target.open()

    assert target.close() == ()

    assert presenter.closed
    with pytest.raises(RuntimeError, match="closed output target"):
        target.poll()


def test_application_can_reuse_one_presenter_across_sessions() -> None:
    presenter = _Presenter()
    factory_calls = 0

    def factory(**_: Any) -> _Presenter:
        nonlocal factory_calls
        factory_calls += 1
        return presenter

    target = LocalWindowVideoOutputTarget(
        overlay=NullOverlay(),
        presenter_factory=factory,
        close_presenter_on_close=False,
    )

    for _ in range(2):
        target.open()
        target.write(_video_result())
        target.close()

    assert len(presenter.presented) == 4
    assert factory_calls == 1
    assert not presenter.closed
    target.shutdown()
    assert presenter.closed


def test_demo_spec_builds_native_io_without_a_model_specific_bridge() -> None:
    presenter = _Presenter()
    captured: dict[str, Any] = {}

    def factory(**kwargs: Any) -> _Presenter:
        captured.update(kwargs)
        return presenter

    io = build_local_window_io(
        spec=DemoSpec(
            model_id="any-video-model",
            input_mode="keyboard",
            output=LocalWindowOutputSpec(
                width=1280,
                height=720,
                title="compatible demo",
            ),
        ),
        overlay=NullOverlay(),
        presenter_factory=factory,
    )
    io.output.open()

    assert captured["config"] == WindowConfig(
        width=1280,
        height=720,
        title="compatible demo",
    )
    assert io.user_inputs.source_schema.supports_event_types({"key_down", "key_up"})
    io.output.close()
