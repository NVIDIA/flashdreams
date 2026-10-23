# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Automatic live startup through the normal V2 controller pipeline."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from flashdreams_controller_input.events import ControllerFrameEvent
from handdreams import session as session_module
from handdreams.config import (
    HandDreamsLaunchConfig,
    HanddreamsScenario,
)
from handdreams.session import HandDreamsSession
from handdreams.ui import HandDreamsPhase, HandDreamsUILoop

from apps.handdreams.tests.helpers import (
    _application as HandDreamsV2Application,
)
from apps.handdreams.tests.helpers import (
    _event,
    _FakeInferencePipeline,
    _FakePipeline,
    _FakeRolloutBuilder,
    _pipeline_config,
    _scene,
)
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.user_input_events import UserInputEvents

pytestmark = pytest.mark.ci_cpu


def _config():
    return HandDreamsLaunchConfig(
        scenario=HanddreamsScenario(sample="scene-a"),
        pipeline_config=_pipeline_config(),
        auto_start=True,
    )


def _session(*, prime: bool = True):
    builder = _FakeRolloutBuilder(pipeline=_FakeInferencePipeline(), scene=_scene())
    builder.prepare_scene()
    session = HandDreamsSession(
        builder=builder,
        config=_config(),
        session_desc=SessionDesc(video_width=640, video_height=360),
        scene_catalog=("scene-a",),
        pipeline=_FakePipeline(),
    )
    session.init()
    if prime:
        assert session.model_loop.step(0, UserInputEvents([])) == []
    return session


def _events(
    index: int,
    *,
    left_trigger: float = 0.0,
    right_trigger: float = 0.0,
    left_tracked: bool = True,
    right_tracked: bool = True,
    right_secondary: bool = False,
    disconnected: bool = False,
):
    capture_ns = 1 + round(index * 1e9 / 36)
    timestamp = np.uint64(capture_ns // 1000)
    if disconnected:
        return UserInputEvents(
            [ControllerFrameEvent(timestamp=timestamp, action="disconnected")]
        )
    event = _event(index)
    return UserInputEvents(
        [
            replace(
                event,
                timestamp=timestamp,
                client_capture_ns=capture_ns,
                left=replace(
                    event.left,
                    trigger=left_trigger,
                    grip=0.0,
                    tracked=left_tracked,
                    primary_button=False,
                ),
                right=replace(
                    event.right,
                    trigger=right_trigger,
                    grip=0.0,
                    tracked=right_tracked,
                    secondary_button=right_secondary,
                ),
            )
        ]
    )


def test_generation_reset_retains_guided_setup_for_a_new_source() -> None:
    session = _session()
    loop = session.model_loop
    loop.state.config = replace(loop.state.config, auto_start=False)
    try:
        assert loop.step(0, _events(0)) == []
        assert loop.state.phase is HandDreamsPhase.CALIBRATING
        assert loop.state.pipeline.calibrations == 0
        loop.reset()
        # The runtime retains input across reset; warmup drops that snapshot.
        stale = _events(1).get_events()
        boundary = _events(2, disconnected=True).get_events()
        current = _events(3).get_events()
        assert loop.step(0, UserInputEvents(stale + boundary + current)) == []
        assert loop.state.phase is HandDreamsPhase.WAITING
        assert loop.state.controller_calibration is None
        assert loop.state.last_input_frame_id == -1
        assert loop.step(1, _events(4)) == []
        assert loop.state.phase is HandDreamsPhase.CALIBRATING
        assert loop.state.pipeline.calibrations == 0
        assert len(loop.state.poses) == 0
    finally:
        loop.close()


def test_session_automatically_starts_with_normal_v2_ui() -> None:
    automatic = _session()
    state = automatic.model_loop.state
    assert state.phase is HandDreamsPhase.WAITING
    assert isinstance(automatic.ui_loop, HandDreamsUILoop)
    assert automatic.ui_loop.state.stage_width == state.stage_width
    assert automatic.ui_loop.state.stage_height == state.stage_height
    automatic.model_loop.close()


@pytest.mark.parametrize(
    "invalid",
    [
        {"left_tracked": False},
        {"right_tracked": False},
        {"left_trigger": 0.10001},
        {"right_trigger": 0.10001},
    ],
)
def test_auto_live_waits_for_both_tracked_open_claws(invalid) -> None:
    loop = _session().model_loop
    assert loop.step(0, _events(0, **invalid)) == []
    assert loop.state.phase is HandDreamsPhase.WAITING
    assert loop.state.pipeline.calibrations == 0
    assert len(loop.state.poses) == 0

    assert loop.step(1, _events(1, left_trigger=0.1, right_trigger=0.1)) == []
    assert loop.state.phase is HandDreamsPhase.LIVE
    assert loop.state.pipeline.calibrations == 1
    assert len(loop.state.poses) == 1
    loop.close()


def test_auto_live_does_not_calibrate_a_missing_hand() -> None:
    loop = _session().model_loop
    with pytest.raises(ValueError):
        replace(_event(0), right=None)
    assert loop.step(0, UserInputEvents([])) == []
    assert loop.state.phase is HandDreamsPhase.WAITING
    assert loop.state.pipeline.calibrations == 0
    assert loop.step(1, _events(1)) == []
    assert loop.state.pipeline.calibrations == 1
    loop.close()


def test_startup_uses_only_latest_pair_then_waits_for_new_poses() -> None:
    loop = _session().model_loop
    batch = _events(0).get_events() + _events(1, right_trigger=1.0).get_events()
    assert loop.step(0, UserInputEvents(batch)) == []
    assert loop.state.pipeline.calibrations == 0

    batch = _events(2).get_events() + _events(3).get_events()
    assert loop.step(1, UserInputEvents(batch)) == []
    assert loop.state.pipeline.calibrations == 1
    assert len(loop.state.poses) == 1
    assert loop.state.poses.peek(0)[0]["source_frame_id"] == 3
    assert loop.state.builder.pipeline.caches[0].actions == []

    assert len(loop.step(2, _events(4))) == 1
    assert len(loop.step(3, _events(5))) == 1
    assert loop.state.pipeline.windows == [[3, 4], [4, 5]]
    assert loop.state.rows_consumed == 2
    assert loop.state.pipeline.calibrations == 1
    assert len(loop.state.poses) == 1
    assert all(
        action.shape == (1, 64)
        for action in loop.state.builder.pipeline.caches[0].actions
    )
    loop.close()


def test_auto_live_discards_the_input_snapshot_used_during_cache_setup() -> None:
    loop = _session(prime=False).model_loop
    assert loop.step(0, _events(0)) == []
    assert loop.state.phase is HandDreamsPhase.WAITING
    assert loop.state.pipeline.calibrations == 0
    assert len(loop.state.poses) == 0

    batch = _events(1).get_events() + _events(2).get_events()
    assert loop.step(1, UserInputEvents(batch)) == []
    assert loop.state.pipeline.calibrations == 1
    assert loop.state.poses.peek(0)[0]["source_frame_id"] == 2
    assert len(loop.step(2, _events(3))) == 1
    assert loop.state.pipeline.windows == [[2, 3]]
    loop.close()


def test_two_resets_require_new_open_anchor_without_closing_pipeline() -> None:
    loop = _session().model_loop
    for attempt in range(3):
        offset = attempt * 4
        assert loop.step(offset, _events(offset)) == []
        assert loop.state.pipeline.calibrations == attempt + 1
        assert len(loop.step(offset + 1, _events(offset + 1))) == 1
        if attempt < 2:
            loop.reset()
            assert loop.state.phase is HandDreamsPhase.WAITING
            assert len(loop.state.poses) == 0
            assert loop.step(offset + 2, UserInputEvents([])) == []
            assert loop.step(offset + 2, _events(offset + 2, right_trigger=1.0)) == []
            assert loop.state.pipeline.calibrations == attempt + 1
            assert loop.state.pipeline.closes == 0
    loop.close()


@pytest.mark.parametrize("batched", [False, True])
def test_tracking_reconnect_and_controller_b_preserve_calibration(batched) -> None:
    loop = _session().model_loop
    loop.step(0, _events(0))
    calibration = loop.state.controller_calibration
    loop.step(1, _events(1))
    assert loop.step(2, _events(2, disconnected=True)) == []
    assert loop.state.last_wrist_observed_at is None
    assert loop.state.phase is HandDreamsPhase.LIVE
    assert len(loop.state.poses) == 0
    assert loop.state.pipeline.calibrations == 1

    assert loop.step(3, _events(3, right_trigger=1.0)) == []
    assert len(loop.step(4, _events(4, right_trigger=1.0))) == 1
    assert loop.state.pipeline.calibrations == 1
    frames = [
        _events(index, right_secondary=pressed)
        for index, pressed in enumerate((True, True, False, True), start=5)
    ]
    if batched:
        batch = UserInputEvents(
            [event for frame in frames for event in frame.get_events()]
        )
        assert len(loop.step(5, batch)) == 1
        for index in range(6, 9):
            assert len(loop.step(index, UserInputEvents([]))) == 1
    else:
        for index, frame in enumerate(frames, start=5):
            assert len(loop.step(index, frame)) == 1
    assert loop.state.pipeline.windows == [
        [0, 1],
        [3, 4],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 8],
    ]
    assert loop.state.controller_calibration is calibration
    assert loop.state.pipeline.calibrations == 1
    assert len(loop.state.poses) == 1
    loop.close()


def test_source_silence_drops_auto_live_poses_without_recalibrating(
    monkeypatch,
) -> None:
    clock = [0.0]
    monkeypatch.setattr(session_module.time, "perf_counter", lambda: clock[0])
    loop = _session().model_loop
    loop.step(0, _events(0))
    assert loop.state.last_wrist_observed_at == 0.0
    assert len(loop.state.poses) == 1

    clock[0] = 0.6
    assert loop.step(1, UserInputEvents([])) == []
    assert loop.state.last_wrist_observed_at is None
    assert loop.state.phase is HandDreamsPhase.LIVE
    assert len(loop.state.poses) == 0
    assert loop.state.pipeline.calibrations == 1

    clock[0] = 0.7
    assert loop.step(2, _events(25, right_trigger=1.0)) == []
    assert len(loop.step(3, _events(26, right_trigger=1.0))) == 1
    assert loop.state.pipeline.calibrations == 1
    assert loop.state.pipeline.windows == [[25, 26]]
    loop.close()


def test_scene_handoff_waits_again_and_reuses_app_resources(monkeypatch) -> None:
    inference_pipeline = _FakeInferencePipeline()
    pipeline = _FakePipeline()
    inference_pipelines_loaded = []
    app = HandDreamsV2Application(
        inference_pipeline_factory=lambda _: (
            inference_pipelines_loaded.append(inference_pipeline) or inference_pipeline
        ),
        pipeline_factory=lambda **_: pipeline,
        scene_factory=lambda scenario, _: _scene(scenario.sample),
    )
    app.init(["--device", "cpu", "--sample", "scene-a", "--auto-start"])
    assert [scene.name for scene in inference_pipeline.prepared_scenes] == [
        "scene-a",
        "scene-b",
    ]
    assert inference_pipeline.caches == []
    session_caches = []
    for scene_index, scene in enumerate(("scene-a", "scene-b", "scene-a")):
        session = app.create_session(
            SessionDesc(video_width=640, video_height=360, metadata={"sample": scene})
        )
        assert [item.name for item in inference_pipeline.prepared_scenes] == [
            "scene-a",
            "scene-b",
        ]
        assert pipeline.calibrations == scene_index
        session.init()
        loop = session.model_loop
        assert loop.state.phase is HandDreamsPhase.WAITING
        assert loop.state.rows_consumed == loop.state.steps_generated == 0
        assert len(loop.state.poses) == 0
        assert loop.step(0, UserInputEvents([])) == []
        session_caches.append(loop.state.inference_cache)
        assert len({id(cache) for cache in session_caches}) == len(session_caches)
        assert loop.step(0, _events(scene_index * 10, left_trigger=1.0)) == []
        assert pipeline.calibrations == scene_index
        assert loop.step(1, _events(scene_index * 10 + 1)) == []
        assert pipeline.calibrations == scene_index + 1
        loop.close()
        assert pipeline.closes == 0
        assert inference_pipeline.released_caches == session_caches
    assert inference_pipelines_loaded == [inference_pipeline]
    assert pipeline.starts == 1
    assert [scene.name for scene in inference_pipeline.prepared_scenes] == [
        "scene-a",
        "scene-b",
    ]
    app.close()
    assert pipeline.closes == 1
    assert inference_pipeline.close_count == 1
