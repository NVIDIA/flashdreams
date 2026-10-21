# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for live controller transactions and application ownership."""

from __future__ import annotations

import gc
import queue
import threading
import weakref
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from handdreams.config import (
    HandDreamsLaunchConfig,
    HanddreamsScenario,
)
from handdreams.controllers.quest_pose import (
    OPENXR_LOCAL_GRIP_POSE_CONVENTION,
)
from handdreams.input import (
    OverlapWindowBuffer,
    controller_frame,
    controller_status,
)
from handdreams.session import (
    HandDreamsModelLoop,
    HandDreamsModelState,
    HandDreamsSession,
)
from handdreams.ui import HandDreamsPhase

from apps.handdreams.tests.helpers import (
    _application,
    _event,
    _events_from_raw_frame,
    _FakeInferencePipeline,
    _FakePipeline,
    _FakePostprocessStream,
    _FakeRolloutBuilder,
    _pipeline_config,
    _raw_frame,
    _scene,
)
from flashdreams.infra.postprocess import (
    VideoPostprocessChainConfig,
)
from flashdreams.runtime_v2.session_desc import (
    SessionDesc,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu


def test_application_rejects_source_owned_wrist_offset(capsys) -> None:
    app = _application()
    with pytest.raises(SystemExit) as error:
        app.init(["--wrist-offset-m", "0.08"])
    assert error.value.code == 2
    assert "unrecognized arguments: --wrist-offset-m 0.08" in capsys.readouterr().err
    assert app._pipeline is app._inference_pipeline is None


@pytest.mark.parametrize("sample", ["", "   "])
def test_scenario_rejects_empty_sample(sample: str) -> None:
    with pytest.raises(ValueError, match="sample must be non-empty"):
        HanddreamsScenario(sample=sample)


def test_application_selects_first_discovered_scene_without_default_name() -> None:
    app = _application(scene_catalog=("new-scene", "another-scene"))
    try:
        app.init([])
        assert app.launch_config.scenario.sample == "new-scene"
    finally:
        app.close()


def test_application_rejects_empty_catalog_before_starting() -> None:
    app = _application(
        scene_catalog=(),
        inference_pipeline_factory=lambda _: pytest.fail("empty catalog loaded model"),
        pipeline_factory=lambda **_: pytest.fail("empty catalog started Teleop"),
    )
    with pytest.raises(ValueError, match="at least one scene"):
        app.init([])


@pytest.mark.parametrize("field", ["scene_path", "sample_dir"])
def test_scenario_rejects_missing_local_assets(tmp_path: Path, field: str) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        options: dict[str, Any] = {field: tmp_path / "missing"}
        HanddreamsScenario(**options)


def test_scenario_rejects_wrong_local_asset_types(tmp_path: Path) -> None:
    source = tmp_path / "scene.json"
    source.touch()
    with pytest.raises(FileNotFoundError, match="does not exist"):
        HanddreamsScenario(scene_path=source)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        HanddreamsScenario(sample_dir=source)


@pytest.mark.parametrize("missing", ["first_frame.jpg", "scene.json"])
def test_scenario_requires_both_scene_assets(tmp_path: Path, missing: str) -> None:
    for name in {"first_frame.jpg", "scene.json"} - {missing}:
        (tmp_path / name).touch()
    with pytest.raises(FileNotFoundError, match=missing):
        HanddreamsScenario(scene_path=tmp_path)


def test_local_scene_arguments_validate_assets_and_exclude_catalog_root(
    tmp_path: Path,
) -> None:
    for name in ("first_frame.jpg", "scene.json"):
        (tmp_path / name).touch()
    assert HanddreamsScenario(scene_path=tmp_path).scene_path == tmp_path
    assert HanddreamsScenario(sample_dir=tmp_path).sample_dir == tmp_path
    with pytest.raises(ValueError, match="mutually exclusive"):
        HanddreamsScenario(scene_path=tmp_path, sample_dir=tmp_path)
    app = _application()
    with pytest.raises(SystemExit) as error:
        app.init(["--scene-path", str(tmp_path), "--sample-dir", str(tmp_path)])
    assert error.value.code == 2
    assert app._pipeline is app._inference_pipeline is None
    app.init(["--scene-path", str(tmp_path)])
    assert app.launch_config.scenario.scene_path == tmp_path
    assert app._pipeline.starts == 1
    app.close()


def test_scene_preview_hook_receives_the_selected_source(tmp_path: Path) -> None:
    app = _application()
    requests = []
    app.defaults = replace(
        app.defaults,
        scene_previews=lambda scenario, catalog: requests.append((scenario, catalog))
        or (),
    )
    try:
        app.init(["--sample", "scene-b", "--sample-dir", str(tmp_path)])
        assert requests == [(app.launch_config.scenario, app.scene_catalog)]
        assert requests[0][0].sample_dir == tmp_path
    finally:
        app.close()


def test_controller_event_preserves_raw_frame_without_face_button_commands() -> None:
    event = _event(4)
    event = replace(
        event,
        left=replace(event.left, secondary_button=True),
        right=replace(event.right, primary_button=True, secondary_button=True),
    )
    frame = controller_frame(event)
    assert frame["source_frame_id"] == 4
    assert frame["pose_convention"] == OPENXR_LOCAL_GRIP_POSE_CONVENTION
    assert frame["left"]["position"] == [4.0, 0.0, 0.0]
    assert frame["left"]["battery"] == 0.75
    assert "next_scene" not in frame["left"]
    assert "previous_scene" not in frame["left"]
    assert "recenter" not in frame["left"]
    assert "next_scene" not in frame["right"]
    assert "previous_scene" not in frame["right"]
    assert "recenter" not in frame["right"]
    assert "primary_button" not in frame["left"]
    assert "secondary_button" not in frame["right"]
    frame["left"]["position"][0] = 99
    assert event.left.position == (4.0, 0.0, 0.0)
    assert event.left.secondary_button and event.right.primary_button


def test_controller_status_and_overlap_window_contract() -> None:
    frame = _raw_frame(0)
    frame["left"]["trigger"] = 0.0
    frame["right"]["trigger"] = 0.8
    frame["right"]["tracked"] = False
    assert controller_status(frame, threshold=0.1) == (True, False, True, False)
    windows = OverlapWindowBuffer()
    for frame_id in range(3):
        windows.append(_raw_frame(frame_id))
    assert [frame["source_frame_id"] for frame in windows.peek(1)] == [0, 1]
    windows.consume(1)
    assert [frame["source_frame_id"] for frame in windows.peek(1)] == [1, 2]


def _live_loop():
    inference_pipeline = _FakeInferencePipeline()
    controller_pipeline = _FakePipeline()
    state = HandDreamsModelState(
        builder=_FakeRolloutBuilder(
            pipeline=inference_pipeline,
            scene=_scene(),
        ),
        config=HandDreamsLaunchConfig(
            scenario=HanddreamsScenario(sample="scene-a"),
            pipeline_config=_pipeline_config(),
            total_steps=2,
            auto_start=True,
        ),
        session_desc=SessionDesc(output_layout=VideoTensorLayout.tchw),
        pipeline=controller_pipeline,
        scene_catalog=("scene-a",),
    )
    state.builder.prepare_scene()
    loop = HandDreamsModelLoop()
    loop.register_session_loop_objects(
        state=state,
        frequency=36,
        shutdown_event=threading.Event(),
        failure_queue=queue.Queue(),
    )
    assert loop.step(0, UserInputEvents([])) == []
    neutral = _raw_frame(0)
    neutral["right"]["trigger"] = 0.0
    assert loop.step(1, _events_from_raw_frame(neutral, frame_id=0)) == []
    assert state.phase is HandDreamsPhase.LIVE
    return loop, state, inference_pipeline, controller_pipeline


def test_live_loop_uses_canonical_targets_and_retains_endpoint() -> None:
    loop, state, inference_pipeline, controller_pipeline = _live_loop()
    first_id, second_id = 27_777_778, 55_555_556
    first = loop.step(2, _events_from_raw_frame(_raw_frame(1), frame_id=first_id))
    second = loop.step(3, _events_from_raw_frame(_raw_frame(2), frame_id=second_id))
    assert len(first) == len(second) == 1
    assert first[0].read_output().shape == (1, 3, 4, 6)
    assert first[0].metrics is not None
    assert first[0].metrics["fake_s"] == 0.25
    assert first[0].metrics["action_start"] == 0
    assert first[0].metrics["action_rows"] == 1
    assert {
        "input_prepare_s",
        "model_wall_s",
        "decode_s",
        "total_s",
    } <= first[0].metrics.keys()
    assert controller_pipeline.calibrations == 1
    assert controller_pipeline.windows == [[0, first_id], [first_id, second_id]]
    assert controller_pipeline.commits == 2
    assert controller_pipeline.aborts == 0
    assert len(state.poses) == 1
    first_cache = inference_pipeline.caches[0]
    assert first_cache.actions[0].shape == (1, 64)
    assert inference_pipeline.generate_calls == [(0, first_cache), (1, first_cache)]
    assert inference_pipeline.finalize_calls == [(0, first_cache), (1, first_cache)]
    assert state.rows_consumed == 2
    assert loop.is_finished()
    loop.reset()
    assert controller_pipeline.resets == 1
    assert inference_pipeline.released_caches == [first_cache]
    assert state.inference_cache is None
    assert state.phase is HandDreamsPhase.WAITING
    assert len(state.poses) == 0
    assert loop.step(4, UserInputEvents([])) == []
    assert len(inference_pipeline.caches) == 2
    assert inference_pipeline.caches[1] is state.inference_cache
    assert inference_pipeline.prepared_scenes == [state.scene]


def test_model_loop_processes_outputs_and_finishes_final_chunk() -> None:
    loop, state, _inference_pipeline, _controller_pipeline = _live_loop()
    stream = _FakePostprocessStream(
        postprocess=VideoPostprocessChainConfig(), output_layout="tchw"
    )
    state.postprocess_stream = stream
    state.postprocess_enabled = True

    first = loop.step(2, _events_from_raw_frame(_raw_frame(1), frame_id=27_777_778))
    second = loop.step(3, _events_from_raw_frame(_raw_frame(2), frame_id=55_555_556))

    assert first[0].read_output().shape == (1, 3, 8, 12)
    assert second[0].read_output().shape == (1, 3, 8, 12)
    assert stream.processed == [0, 1]
    assert stream.finished == 1
    assert first[0].metrics is not None
    assert "postprocess_submit_s" in first[0].metrics
    loop.reset()
    assert stream.resets == 1


def test_model_loop_toggles_presentation_without_pausing_postprocessing() -> None:
    loop, state, _inference_pipeline, _controller_pipeline = _live_loop()

    class _VisiblePostprocessStream(_FakePostprocessStream):
        def process(self, output: torch.Tensor, *, autoregressive_index: int):
            return (
                super().process(output, autoregressive_index=autoregressive_index) + 1
            )

    stream = _VisiblePostprocessStream(
        postprocess=VideoPostprocessChainConfig(), output_layout="tchw"
    )
    state.postprocess_stream = stream
    state.stage_width = 12
    state.stage_height = 8

    raw = loop.step(2, _events_from_raw_frame(_raw_frame(1), frame_id=27_777_778))[0]
    state.set_postprocess_enabled(True)
    processed = loop.step(
        3, _events_from_raw_frame(_raw_frame(2), frame_id=55_555_556)
    )[0]

    assert stream.processed == [0, 1]
    assert raw.read_output().shape == processed.read_output().shape == (1, 3, 8, 12)
    assert torch.count_nonzero(raw.read_output()) == 0
    assert torch.all(processed.read_output() == 1)
    assert raw.metrics["postprocess_enabled"] == 0
    assert processed.metrics["postprocess_enabled"] == 1


def test_cosmos_rejection_aborts_without_consuming_controller_window() -> None:
    loop, state, inference_pipeline, controller_pipeline = _live_loop()
    inference_pipeline.reject_before_callback = True
    with pytest.raises(RuntimeError, match="Cosmos rejected action"):
        loop.step(2, _events_from_raw_frame(_raw_frame(1), frame_id=27_777_778))
    assert controller_pipeline.aborts == 1
    assert controller_pipeline.commits == 0
    assert state.rows_consumed == 0
    assert len(state.poses) == 2
    assert inference_pipeline.finalize_calls == []


def test_decode_failure_preserves_accepted_controller_transaction() -> None:
    loop, state, inference_pipeline, controller_pipeline = _live_loop()
    inference_pipeline.fail_decode_after_callback = True

    with pytest.raises(RuntimeError, match="decoder failed"):
        loop.step(2, _events_from_raw_frame(_raw_frame(1), frame_id=27_777_778))

    assert controller_pipeline.commits == 1
    assert controller_pipeline.aborts == 0
    assert state.rows_consumed == 1
    assert state.steps_generated == 1
    assert len(state.poses) == 1
    assert inference_pipeline.finalize_calls == []


class _SessionPayload:
    pass


def _closed_session_payload_ref() -> weakref.ReferenceType[_SessionPayload]:
    payload = _SessionPayload()
    scene = _scene()
    scene.retained_payload = payload
    session = HandDreamsSession(
        builder=_FakeRolloutBuilder(
            pipeline=_FakeInferencePipeline(),
            scene=scene,
        ),
        config=HandDreamsLaunchConfig(
            scenario=HanddreamsScenario(sample="scene-a"),
            pipeline_config=_pipeline_config(),
        ),
        session_desc=SessionDesc(
            output_layout=VideoTensorLayout.tchw,
            video_width=640,
            video_height=360,
        ),
        scene_catalog=("scene-a",),
        pipeline=_FakePipeline(),
    )
    session.init()
    session._take_loops()
    session._shutdown_registered_loops()
    session.close()
    return weakref.ref(payload)


def test_session_close_releases_scene_payload_without_cyclic_gc() -> None:
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        assert _closed_session_payload_ref()() is None
    finally:
        if gc_was_enabled:
            gc.enable()
