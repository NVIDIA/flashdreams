# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Application-owned controller and inference cleanup after failures."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest
from handdreams.teleop import TeleopControllerPipeline
from handdreams.ui import HandDreamsPhase

from apps.handdreams.tests.helpers import (
    _application as HandDreamsV2Application,
)
from apps.handdreams.tests.helpers import (
    _event,
    _FakeInferencePipeline,
    _FakePipeline,
    _scene,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

pytestmark = pytest.mark.ci_cpu


class _FailingOwner:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        raise self.error


def test_application_close_finishes_authoritative_cleanup_after_failures() -> None:
    pipeline_error = RuntimeError("pipeline failed")
    inference_error = RuntimeError("inference failed")
    pipeline = _FailingOwner(pipeline_error)
    inference_pipeline = _FailingOwner(inference_error)
    app = HandDreamsV2Application()
    app._pipeline = cast(TeleopControllerPipeline, pipeline)
    app._inference_pipeline = inference_pipeline

    with pytest.raises(RuntimeError, match="pipeline failed") as raised:
        app.close()

    assert raised.value is pipeline_error
    assert any(
        "inference failed" in note for note in getattr(raised.value, "__notes__", ())
    )
    assert pipeline.close_calls == inference_pipeline.close_calls == 1
    assert app._pipeline is app._inference_pipeline is None

    app.close()
    assert pipeline.close_calls == inference_pipeline.close_calls == 1


def test_catalog_limit_fails_before_previews_or_services() -> None:
    def unexpected_setup(*args, **kwargs):
        pytest.fail("oversized catalog must fail before preparation")

    app = HandDreamsV2Application(
        scene_catalog=tuple(f"scene-{index}" for index in range(6)),
        pipeline_factory=unexpected_setup,
        inference_pipeline_factory=unexpected_setup,
        scene_factory=unexpected_setup,
    )
    app.defaults = replace(app.defaults, scene_previews=unexpected_setup)
    try:
        with pytest.raises(
            ValueError, match=r"discovered 6 scenes.*limit is 5"
        ) as error:
            app.init([])
        assert "--scene-path" in str(error.value)
        assert "--sample-dir" in str(error.value)
        assert app._prepared_scenes == {}
        assert app._pipeline is app._inference_pipeline is None
    finally:
        app.close()


@pytest.mark.parametrize("scene_count", [4, 5])
def test_catalog_limit_prepares_each_scene_once_and_rejects_reinitialization(
    scene_count: int,
) -> None:
    catalog = tuple(f"scene-{index}" for index in range(scene_count))
    inference = _FakeInferencePipeline()
    app = HandDreamsV2Application(
        scene_catalog=catalog,
        inference_pipeline_factory=lambda _: inference,
    )
    try:
        app.init([])
        assert [scene.name for scene in inference.prepared_scenes] == list(catalog)
        for name in catalog:
            session = app.create_session(
                replace(app.session_desc(), metadata={"sample": name})
            )
            session.init()
            session.model_loop.reset()
            assert session._shutdown_registered_loops() == []
            session.close()
        assert [scene.name for scene in inference.prepared_scenes] == list(catalog)

        app.defaults = replace(
            app.defaults, discover_scene_catalog=lambda _: ("another-scene",)
        )
        with pytest.raises(RuntimeError, match="already initialized"):
            app.init([])
        assert app.scene_catalog == catalog
        assert [scene.name for scene in inference.prepared_scenes] == list(catalog)
    finally:
        app.close()


def test_selected_scene_cannot_add_an_entry_outside_the_catalog() -> None:
    app = HandDreamsV2Application()
    try:
        with pytest.raises(ValueError, match="sample is absent: missing-scene"):
            app.init(["--sample", "missing-scene"])
        assert app._prepared_scenes == {}
        assert app._pipeline is app._inference_pipeline is None
    finally:
        app.close()


@pytest.mark.parametrize("prewarm_all", [False, True])
def test_application_warms_selected_scene_before_calibration_and_reuses_it(
    prewarm_all: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _FakePipeline()
    inference = _FakeInferencePipeline()
    loaded = []
    app = HandDreamsV2Application(
        pipeline_factory=lambda **_: controller,
        inference_pipeline_factory=lambda _: inference,
        scene_factory=lambda scenario, _: (
            loaded.append(_scene(scenario.sample)) or loaded[-1]
        ),
    )
    app.init(
        ["--sample", "scene-b", *(["--prewarm-all-scenes"] if prewarm_all else [])]
    )

    expected_names = ["scene-b", "scene-a"]
    assert [scene.name for scene in loaded] == expected_names
    assert inference.prepared_scenes == loaded
    assert inference.caches == []
    assert controller.calibrations == 0
    assert controller.windows == []

    def unexpected_preparation(*args, **kwargs):
        pytest.fail("scene preparation must not run after application.init")

    app.defaults = replace(app.defaults, scene_factory=unexpected_preparation)
    monkeypatch.setattr(inference, "prepare_scene", unexpected_preparation)

    session = app.create_session(
        replace(app.session_desc(), metadata={"sample": "scene-b"})
    )
    session.init()
    loop = session.model_loop
    assert loop.state.scene is loaded[0]
    assert loop.state.phase is HandDreamsPhase.WAITING
    assert loop.step(0, UserInputEvents([_event(1)])) == []
    first_cache = loop.state.inference_cache
    loop.reset()
    assert loop.step(1, UserInputEvents([_event(2)])) == []

    assert loop.state.inference_cache is not first_cache
    assert inference.prepared_scenes == loaded
    assert len(loaded) == len(expected_names)
    assert controller.calibrations == 0
    assert controller.windows == []
    assert loop.state.rows_consumed == loop.state.steps_generated == 0
    assert len(loop.state.poses) == 0
    assert inference.generate_calls == []
    assert session._shutdown_registered_loops() == []
    session.close()
    replacement = app.create_session(
        replace(app.session_desc(), metadata={"sample": "scene-a"})
    )
    replacement.init()
    assert replacement.model_loop.state.scene is loaded[1]
    assert inference.prepared_scenes == loaded
    assert [scene.name for scene in loaded] == expected_names
    assert replacement._shutdown_registered_loops() == []
    replacement.close()
    app.close()
    assert app._prepared_scenes == {}
    assert controller.closes == inference.close_count == 1


def test_session_rejects_missing_prepared_scene_without_lazy_setup() -> None:
    inference = _FakeInferencePipeline()
    app = HandDreamsV2Application(inference_pipeline_factory=lambda _: inference)
    app.init([])
    app._prepared_scenes.pop("scene-b")
    prepared_count = len(inference.prepared_scenes)
    try:
        with pytest.raises(RuntimeError, match="scene-b.*not prepared during init"):
            app.create_session(
                replace(app.session_desc(), metadata={"sample": "scene-b"})
            )
        assert len(inference.prepared_scenes) == prepared_count
    finally:
        app.close()


@pytest.mark.parametrize("selected", ["scene-a", "scene-b"])
def test_live_startup_requires_scene_selection_before_controller_calibration(
    selected: str,
) -> None:
    controller = _FakePipeline()
    inference = _FakeInferencePipeline()
    app = HandDreamsV2Application(
        pipeline_factory=lambda **_: controller,
        inference_pipeline_factory=lambda _: inference,
    )
    app.init(["--sample", "scene-a"])
    session = app.create_session(app.session_desc())
    session.init()
    loop, ui = session.model_loop, session.ui_loop

    assert loop.state.phase is HandDreamsPhase.SELECTING
    assert ui.state.status.current_scene == ""
    assert [scene.name for scene in inference.prepared_scenes] == ["scene-a", "scene-b"]
    for index in range(3):
        assert loop.step(index, UserInputEvents([_event(index, trigger=1.0)])) == []
    assert inference.caches == []
    assert inference.generate_calls == []
    assert controller.calibrations == 0
    assert controller.windows == []
    assert loop.state.last_input_frame_id == loop.state.last_input_capture_ns == -1
    loop.reset()
    ui.reset()
    assert loop.state.phase is HandDreamsPhase.SELECTING
    assert ui.state.status.phase is HandDreamsPhase.SELECTING
    assert ui.state.status.current_scene == ""

    ui._request_scene_name(selected)
    request = ui.flush_ui_loop_requests()
    assert request is not None and request.new_session is not None
    assert ui.state.status.phase is HandDreamsPhase.LOADING
    assert session._shutdown_registered_loops() == []
    session.close()

    replacement = app.create_session(request.new_session)
    replacement.init()
    loop = replacement.model_loop
    assert loop.state.scene.name == selected
    assert loop.state.phase is HandDreamsPhase.WAITING
    assert replacement.ui_loop.state.status.current_scene == selected
    assert [scene.name for scene in inference.prepared_scenes] == ["scene-a", "scene-b"]
    assert controller.starts == 1
    assert controller.closes == inference.close_count == 0
    # The cache-preparation snapshot is ignored; only fresh input starts setup.
    assert loop.step(0, UserInputEvents([_event(10)])) == []
    assert loop.state.last_input_frame_id == -1
    assert loop.step(1, UserInputEvents([_event(11)])) == []
    assert loop.state.phase is HandDreamsPhase.CALIBRATING
    assert loop.state.last_input_frame_id == 11
    assert inference.generate_calls == []
    loop.reset()
    assert loop.state.phase is HandDreamsPhase.WAITING
    assert replacement._shutdown_registered_loops() == []
    replacement.close()
    app.close()
    assert controller.closes == inference.close_count == 1


def test_auto_start_bypasses_scene_picker() -> None:
    app = HandDreamsV2Application()
    app.init(["--sample", "scene-b", "--auto-start"])
    session = app.create_session(app.session_desc())
    session.init()
    assert session.model_loop.state.phase is HandDreamsPhase.WAITING
    assert session.ui_loop.state.status.current_scene == "scene-b"
    assert session._shutdown_registered_loops() == []
    session.close()
    app.close()


@pytest.mark.parametrize("flags", [[], ["--auto-start"]])
def test_return_to_picker_retains_services_and_reselects_same_scene(flags) -> None:
    controller, inference = _FakePipeline(), _FakeInferencePipeline()
    app = HandDreamsV2Application(
        pipeline_factory=lambda **_: controller,
        inference_pipeline_factory=lambda _: inference,
    )
    app.init(["--sample", "scene-b", *flags])
    session = app.create_session(
        replace(app.session_desc(), metadata={"sample": "scene-b"})
    )
    session.init()
    session.model_loop.step(0, UserInputEvents([]))
    assert len(inference.caches) == 1
    generated = len(inference.generate_calls)
    session.ui_loop._request_scene_name(None)
    request = session.ui_loop.flush_ui_loop_requests()
    assert request is not None and request.new_session is not None
    assert session._shutdown_registered_loops() == []
    session.close()
    assert inference.released_caches == inference.caches

    picker = app.create_session(request.new_session)
    picker.init()
    assert picker.model_loop.state.phase is HandDreamsPhase.SELECTING
    assert picker.ui_loop.state.status.current_scene == ""
    picker.model_loop.reset()
    assert picker.model_loop.state.phase is HandDreamsPhase.SELECTING
    assert picker.model_loop.step(0, UserInputEvents([_event(1)])) == []
    assert len(inference.generate_calls) == generated
    assert len(inference.caches) == 1
    assert controller.calibrations == 0
    assert controller.closes == inference.close_count == 0

    picker.ui_loop._request_scene_name("scene-b")
    selected = picker.ui_loop.flush_ui_loop_requests()
    assert selected is not None and selected.new_session is not None
    assert "select_scene" not in selected.new_session.metadata
    assert picker._shutdown_registered_loops() == []
    picker.close()
    replacement = app.create_session(selected.new_session)
    replacement.init()
    assert replacement.model_loop.state.phase is HandDreamsPhase.WAITING
    assert replacement.ui_loop.state.status.current_scene == "scene-b"
    assert [scene.name for scene in inference.prepared_scenes] == ["scene-b", "scene-a"]
    assert controller.starts == 1
    assert replacement._shutdown_registered_loops() == []
    replacement.close()
    app.close()
    assert inference.close_count == 1


@pytest.mark.parametrize("failing_scene", ["scene-a", "scene-b"])
def test_scene_warmup_failure_releases_application_resources(
    failing_scene: str,
) -> None:
    controller = _FakePipeline()

    class _FailingWarmup(_FakeInferencePipeline):
        def prepare_scene(self, scene) -> None:
            super().prepare_scene(scene)
            if scene.name == failing_scene:
                raise RuntimeError("warmup failed")

    inference = _FailingWarmup()
    app = HandDreamsV2Application(
        pipeline_factory=lambda **_: controller,
        inference_pipeline_factory=lambda _: inference,
    )

    with pytest.raises(RuntimeError, match="warmup failed"):
        app.init(["--sample", "scene-a", "--prewarm-all-scenes"])

    assert controller.starts == 1
    assert [scene.name for scene in inference.prepared_scenes] == (
        ["scene-a"] if failing_scene == "scene-a" else ["scene-a", "scene-b"]
    )
    assert list(app._prepared_scenes) == (
        [] if failing_scene == "scene-a" else ["scene-a"]
    )
    assert controller.calibrations == 0
    assert controller.windows == []
    app.close()
    app.close()
    assert app._prepared_scenes == {}
    assert app._pipeline is app._inference_pipeline is None
    assert controller.closes == inference.close_count == 1
