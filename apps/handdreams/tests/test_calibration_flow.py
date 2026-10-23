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

"""Guided calibration transitions through the V2 model loop without hardware."""

from dataclasses import replace

import numpy as np
import pytest
from flashdreams_controller_input.events import ControllerFrameEvent
from flashdreams_controller_input.pose import WRIST_PROXY_POSE_CONVENTION
from handdreams import session as session_module
from handdreams.config import (
    HandDreamsLaunchConfig,
    HanddreamsScenario,
)
from handdreams.session import HandDreamsSession
from handdreams.ui import HandDreamsPhase

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


@pytest.fixture
def clock(monkeypatch):
    value = [0.0]
    monkeypatch.setattr(session_module.time, "perf_counter", lambda: value[0])
    return value


def _session(*, prime=True, scene=None, **config_options):
    builder = _FakeRolloutBuilder(
        pipeline=_FakeInferencePipeline(), scene=_scene() if scene is None else scene
    )
    builder.prepare_scene()
    session = HandDreamsSession(
        builder=builder,
        config=HandDreamsLaunchConfig(
            scenario=HanddreamsScenario(sample="scene-a"),
            pipeline_config=_pipeline_config(),
            **config_options,
        ),
        session_desc=SessionDesc(video_width=640, video_height=360),
        scene_catalog=("scene-a",),
        pipeline=_FakePipeline(),
    )
    session.init()
    if prime:
        assert session.model_loop.step(0, UserInputEvents([])) == []
    return session


def _frame(index, *, left=None, right=None):
    event = _event(index)
    return replace(
        event,
        timestamp=np.uint64(index),
        client_capture_ns=1 + index * 100_000_000,
        pose_convention=WRIST_PROXY_POSE_CONVENTION,
        left=replace(
            event.left,
            **{
                "position": (-0.2, 0.0, 0.0),
                "grip": 0.0,
                "primary_button": False,
                **(left or {}),
            },
        ),
        right=replace(
            event.right,
            **{
                "position": (0.2, 0.0, 0.0),
                "grip": 0.0,
                **(right or {}),
            },
        ),
    )


def _step(loop, clock, now, *events):
    clock[0] = now
    return loop.step(0, UserInputEvents(list(events)))


def _held(index, **kwargs):
    frame = _frame(index, **kwargs)
    return replace(
        frame,
        left=replace(frame.left, trigger=0.9),
        right=replace(frame.right, trigger=0.9),
    )


def _capture(loop, clock, *, start=0.0, first=0):
    for offset in range(4):
        assert _step(loop, clock, start + offset / 3, _held(first + offset)) == []
    assert loop.state.phase is HandDreamsPhase.ALIGNING
    assert loop.state.calibration_since is None
    assert loop.state.alignment_since is None


def _hold(loop, clock, *, start=1.0, first=4):
    for offset in range(4):
        assert _step(loop, clock, start + offset / 3, _frame(first + offset)) == []


def test_default_requires_dwell_and_seeds_only_final_neutral_pose(clock):
    loop = _session().model_loop
    for index, now in enumerate((0.0, 0.2, 0.4)):
        assert _step(loop, clock, now, _held(index)) == []
        assert loop.state.phase is HandDreamsPhase.CALIBRATING
        assert loop.state.alignment is None
        assert loop.state.pipeline.calibrations == 0
        assert len(loop.state.poses) == 0
        assert loop.state.inference_cache.actions == []
        assert loop.state.calibration_progress == pytest.approx(now / 0.5)
    assert _step(loop, clock, 0.5, _held(3)) == []
    assert loop.state.phase is HandDreamsPhase.ALIGNING
    assert loop.state.calibration_since is None
    assert loop.state.alignment_since is None
    assert loop.state.pipeline.calibrations == 0
    assert loop.state.calibration_progress == 1.0
    _hold(loop, clock)
    assert loop.state.phase is HandDreamsPhase.LIVE
    assert loop.state.pipeline.calibrations == 1
    np.testing.assert_allclose(
        loop.state.pipeline.calibration_bases[0], np.diag([1, -1, -1])
    )
    calibration = loop.state.controller_calibration
    assert calibration is not None
    assert calibration.position_gain == loop.state.config.position_gain
    for side, neutral, target in zip(
        ("left", "right"), calibration.neutral, calibration.targets, strict=True
    ):
        np.testing.assert_allclose(neutral.position, getattr(_frame(7), side).position)
        scene_pose = loop.state.scene.wrist_poses[side]
        np.testing.assert_allclose(target.position, scene_pose[:3, 3])
        np.testing.assert_allclose(
            np.reshape(target.rotation, (3, 3)), scene_pose[:3, :3]
        )
    assert loop.state.poses.peek(0)[0]["source_frame_id"] == 7
    assert loop.state.alignment is None
    assert loop.state.alignment_preview is None
    assert len(_step(loop, clock, 2.1, _frame(8))) == 1
    assert loop.state.pipeline.windows == [[7, 8]]
    loop.close()


def test_tracked_open_controllers_never_capture_a_reference(clock):
    loop = _session().model_loop
    for index in range(8):
        assert _step(loop, clock, index * 0.25, _frame(index)) == []
        assert loop.state.phase is HandDreamsPhase.CALIBRATING
        assert loop.state.calibration_since is None
        assert loop.state.alignment is None
    assert loop.state.pipeline.calibrations == 0
    assert len(loop.state.poses) == 0
    loop.close()


def test_capture_requires_half_a_second_of_closed_claws(clock):
    loop = _session().model_loop
    for index, now in enumerate((0.0, 0.25, 0.499)):
        _step(loop, clock, now, _held(index))
        assert loop.state.alignment is None
        assert loop.state.calibration_progress == pytest.approx(now / 0.5)
    _step(loop, clock, 0.5, _held(3))
    assert loop.state.phase is HandDreamsPhase.ALIGNING
    assert loop.state.calibration_progress == 1.0
    loop.close()


def test_alignment_requires_one_full_second_of_open_pointing_controllers(clock):
    loop = _session().model_loop
    _capture(loop, clock)
    for index, now in enumerate((1.0, 1.5, 1.999), start=4):
        _step(loop, clock, now, _frame(index))
        assert loop.state.phase is HandDreamsPhase.ALIGNING
        assert loop.state.pipeline.calibrations == 0
    _step(loop, clock, 2.0, _frame(7))
    assert loop.state.phase is HandDreamsPhase.LIVE
    assert loop.state.pipeline.calibrations == 1
    loop.close()


@pytest.mark.parametrize(
    "orientation",
    [
        (0.0, 2**-0.5, 0.0, 2**-0.5),
        (0.0, 1.0, 0.0, 0.0),
        (2**-0.5, 0.0, 0.0, 2**-0.5),
    ],
    ids=["sideways", "backwards", "vertical"],
)
def test_wrong_initial_pointing_cannot_become_an_aligned_reference(clock, orientation):
    loop = _session().model_loop
    hands = {side: {"orientation": orientation} for side in ("left", "right")}
    for index in range(4):
        _step(loop, clock, index / 3, _held(index, **hands))
    assert loop.state.phase is HandDreamsPhase.ALIGNING
    assert not loop.state.alignment_preview.aligned
    for index in range(4):
        _step(loop, clock, 1.0 + index / 3, _frame(4 + index, **hands))
        assert loop.state.phase is HandDreamsPhase.ALIGNING
        assert loop.state.alignment_since is None
        assert loop.state.pipeline.calibrations == 0
    _hold(loop, clock, start=2.25, first=8)
    assert loop.state.phase is HandDreamsPhase.LIVE
    assert loop.state.pipeline.calibrations == 1
    loop.close()


def test_live_calibration_retains_held_separation_basis_after_alignment_movement(clock):
    loop = _session().model_loop
    initial_yaw = np.deg2rad(45.0)
    initial = (0.0, np.sin(initial_yaw / 2), 0.0, np.cos(initial_yaw / 2))
    for index in range(4):
        _step(
            loop,
            clock,
            index / 3,
            _held(index, left={"orientation": initial}, right={"orientation": initial}),
        )
    assert not loop.state.alignment_preview.aligned
    held_basis = loop.state.alignment.participant_basis
    np.testing.assert_allclose(held_basis, np.diag([1, -1, -1]))

    final_yaw = np.deg2rad(10.0)
    final = (0.0, np.sin(final_yaw / 2), 0.0, np.cos(final_yaw / 2))
    placement_yaw = np.deg2rad(20.0)
    separation = 0.2 * np.array([np.cos(placement_yaw), 0.0, -np.sin(placement_yaw)])
    for index in range(4):
        _step(
            loop,
            clock,
            1.25 + index / 3,
            _frame(
                4 + index,
                left={"position": tuple(-separation), "orientation": final},
                right={"position": tuple(separation), "orientation": final},
            ),
        )
    assert loop.state.phase is HandDreamsPhase.LIVE
    np.testing.assert_array_equal(loop.state.pipeline.calibration_bases[0], held_basis)
    calibration = loop.state.controller_calibration
    assert calibration is not None
    np.testing.assert_allclose(
        np.reshape(calibration.participant_basis, (3, 3)), held_basis
    )
    np.testing.assert_allclose(calibration.neutral[1].position, separation)
    np.testing.assert_allclose(calibration.neutral[1].orientation, final)
    neutral = loop.state.poses.peek(0)[0]
    np.testing.assert_allclose(neutral["right"]["position"], separation)
    np.testing.assert_allclose(neutral["right"]["orientation"], final)
    loop.close()


def test_capture_progress_does_not_advance_without_fresh_input(clock):
    loop = _session().model_loop
    _step(loop, clock, 0.0, _held(0))
    _step(loop, clock, 0.25, _held(1))
    for events in ((), (_held(1),)):
        _step(loop, clock, 0.5, *events)
        assert loop.state.calibration_progress == 0.5
    _step(loop, clock, 0.6, _frame(2), _held(3))
    assert loop.state.calibration_progress == 0.0
    assert loop.state.calibration_since == 0.6
    _step(loop, clock, 0.9, _held(4))
    assert loop.state.calibration_progress == pytest.approx(0.6)
    _step(loop, clock, 1.5)
    assert loop.state.calibration_progress == 0.0
    assert loop.state.calibration_since is None
    loop.close()


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize(
    "invalid",
    [
        {"left": {"trigger": 0.89999}},
        {"right": {"trigger": 0.89999}},
        {"left": {"tracked": False}},
        {"right": {"tracked": False}},
        {"right": {"position": (-0.2, 0.0, 0.0)}},
    ],
)
def test_interrupted_trigger_hold_requires_a_new_full_dwell(clock, invalid, batched):
    loop = _session().model_loop
    _step(loop, clock, 0.0, _held(0))
    _step(loop, clock, 0.4, _held(1))
    left = {"trigger": 0.9, **invalid.get("left", {})}
    right = {"trigger": 0.9, **invalid.get("right", {})}
    interrupted = _frame(2, left=left, right=right)
    if batched:
        _step(loop, clock, 0.8, interrupted, _held(3))
    else:
        _step(loop, clock, 0.6, interrupted)
        _step(loop, clock, 0.8, _held(3))
    assert loop.state.alignment is None
    restarted = 0.8
    assert loop.state.calibration_since == restarted
    assert loop.state.pipeline.calibrations == 0
    for index, now in enumerate((1.0, 1.299), start=4):
        _step(loop, clock, now, _held(index))
        assert loop.state.alignment is None
    _step(loop, clock, 1.3, _held(6))
    assert loop.state.phase is HandDreamsPhase.ALIGNING
    assert len(loop.state.poses) == 0
    loop.close()


def test_alignment_follows_current_midpoint_without_changing_frozen_basis(clock):
    loop = _session().model_loop
    for index, now in enumerate((0.0, 0.25, 0.499)):
        _step(loop, clock, now, _held(index))
    held = _held(
        3, left={"position": (0.8, 0.0, 0.0)}, right={"position": (1.2, 0.0, 0.0)}
    )
    _step(loop, clock, 0.5, held)
    guide = loop.state.alignment
    assert loop.state.alignment_preview.aligned
    displaced = _held(
        4, left={"position": (0.9, 0.0, 0.0)}, right={"position": (1.3, 0.0, 0.0)}
    )
    _step(loop, clock, 0.75, displaced)
    assert loop.state.alignment is guide
    assert loop.state.alignment_preview.left.position_error_m == pytest.approx(0.04)
    assert loop.state.alignment_preview.aligned
    assert loop.state.alignment_since is None
    assert len(loop.state.poses) == 0
    for index in range(4):
        _step(
            loop,
            clock,
            1.0 + index / 3,
            _frame(
                5 + index,
                left={"position": (0.9, 0.0, 0.0)},
                right={"position": (1.3, 0.0, 0.0)},
            ),
        )
    assert loop.state.phase is HandDreamsPhase.LIVE
    assert loop.state.poses.peek(0)[0]["source_frame_id"] == 8
    np.testing.assert_allclose(
        loop.state.pipeline.calibration_bases[0], np.diag([1, -1, -1])
    )
    assert loop.state.inference_cache.actions == []
    loop.close()


@pytest.mark.parametrize(
    "invalid",
    [
        {"left": {"tracked": False}},
        {"right": {"tracked": False}},
        {"left": {"trigger": 0.10001}},
        {"right": {"trigger": 0.10001}},
        {"right": {"position": (0.8, 0.0, 0.0)}},
    ],
)
def test_invalid_pose_or_closed_claw_restarts_hold(clock, invalid):
    loop = _session().model_loop
    _capture(loop, clock)
    _step(loop, clock, 1.0, _frame(4))
    _step(loop, clock, 1.4, _frame(5))
    _step(loop, clock, 1.6, _frame(6, **invalid))
    assert loop.state.alignment_since is None
    _step(loop, clock, 1.8, _frame(7))
    assert loop.state.phase is not HandDreamsPhase.LIVE
    if loop.state.alignment is None:
        _capture(loop, clock, start=2.0, first=8)
        _hold(loop, clock, start=3.0, first=12)
    else:
        _hold(loop, clock, start=2.0, first=8)
    assert loop.state.phase is HandDreamsPhase.LIVE
    loop.close()


@pytest.mark.parametrize(
    "invalid",
    [
        {"left": {"tracked": False}},
        {"right": {"trigger": 1.0}},
        {"right": {"position": (0.8, 0.0, 0.0)}},
    ],
)
def test_intermediate_invalid_state_cannot_hide_inside_a_batch(clock, invalid):
    loop = _session().model_loop
    _capture(loop, clock)
    _step(loop, clock, 1.0, _frame(4))
    _step(loop, clock, 1.4, _frame(5))
    _step(loop, clock, 1.8, _frame(6, **invalid), _frame(7))
    assert loop.state.phase is not HandDreamsPhase.LIVE
    if invalid.get("left", {}).get("tracked") is False:
        assert loop.state.alignment is None
        assert loop.state.calibration_since is None
        assert loop.state.alignment_since is None
    else:
        assert loop.state.alignment_since == 1.8
    assert loop.state.pipeline.calibrations == 0
    loop.close()


def test_warmup_and_queued_setup_movement_do_not_select_initial_anchor(clock):
    loop = _session(prime=False).model_loop
    _step(loop, clock, 0.0, _held(0, right={"position": (2.0, 0.0, 0.0)}))
    assert loop.state.alignment is None
    assert loop.state.last_input_frame_id == -1
    _step(loop, clock, 1.0, _held(1, right={"position": (2.0, 0.0, 0.0)}), _held(2))
    assert loop.state.alignment is None
    assert loop.state.calibration_since == 1.0
    assert loop.state.last_input_frame_id == 2
    _capture(loop, clock, start=1.0, first=3)
    assert loop.state.alignment_preview.aligned
    assert len(loop.state.poses) == 0
    assert loop.state.pipeline.calibrations == 0
    loop.close()


@pytest.mark.parametrize("stale_ms", [500.0, None])
@pytest.mark.parametrize("empty_poll", [False, True])
@pytest.mark.parametrize("captured", [False, True])
def test_source_silence_requires_a_new_hold_even_without_empty_polls(
    clock,
    stale_ms,
    empty_poll,
    captured,
):
    loop = _session(stale_ms=stale_ms).model_loop
    if captured:
        _capture(loop, clock)
    else:
        _step(loop, clock, 1.0, _held(0))
    if empty_poll:
        _step(loop, clock, 1.6)
        assert loop.state.alignment is None
        assert loop.state.calibration_since is None
    _step(loop, clock, 1.8, _held(4))
    assert loop.state.phase is HandDreamsPhase.CALIBRATING
    assert loop.state.alignment is None
    assert loop.state.calibration_since == 1.8
    assert loop.state.alignment_since is None
    assert loop.state.pipeline.calibrations == 0
    loop.close()


def test_disconnect_discards_prior_states_and_preserves_ordering(clock):
    loop = _session().model_loop
    _capture(loop, clock)
    _step(loop, clock, 1.0, _frame(4))
    _step(loop, clock, 1.4, _frame(5))
    disconnect = ControllerFrameEvent(timestamp=np.uint64(3), action="disconnected")
    _step(loop, clock, 1.8, _frame(6), disconnect, _held(8))
    assert loop.state.last_input_frame_id == 8
    assert loop.state.alignment is None
    assert loop.state.alignment_since is None
    assert loop.state.calibration_since == 1.8
    assert loop.state.pipeline.calibrations == 0
    _step(loop, clock, 2.0, _frame(1))
    assert loop.state.last_input_frame_id == 8
    loop.close()


@pytest.mark.parametrize("captured", [False, True])
def test_duplicate_frames_do_not_advance_hold(clock, captured):
    loop = _session().model_loop
    if captured:
        _capture(loop, clock)
    frame = _frame(4) if captured else _held(4)
    _step(loop, clock, 1.0, frame)
    for now in (1.25, 1.5, 1.75, 2.0):
        _step(loop, clock, now, frame)
    assert loop.state.pipeline.calibrations == 0
    assert loop.state.last_input_frame_id == 4
    assert loop.state.alignment is None
    assert loop.state.calibration_since is None
    _step(loop, clock, 2.1, _held(5))
    assert loop.state.calibration_since == 2.1
    assert loop.state.alignment_since is None
    loop.close()


@pytest.mark.parametrize("captured", [False, True])
@pytest.mark.parametrize("batched", [False, True])
def test_controller_b_does_not_restart_trigger_hold_or_alignment(
    clock, captured, batched
):
    loop = _session().model_loop
    if captured:
        _capture(loop, clock)
    first, start = (4, 1.0) if captured else (0, 0.0)
    frame = _frame if captured else _held
    _step(loop, clock, start, frame(first))
    original = loop.state.alignment
    for offset, pressed in enumerate((True, True, False, True), start=1):
        index = first + offset * 2
        events = [frame(index, right={"secondary_button": pressed})]
        if batched:
            events.insert(0, frame(index - 1, right={"secondary_button": True}))
        _step(loop, clock, start + offset * (0.25 if captured else 0.125), *events)
        assert loop.state.last_input_frame_id == index
        if offset < 4:
            assert loop.state.alignment is original
            assert loop.state.calibration_since == (None if captured else start)
            assert loop.state.alignment_since == (start if captured else None)
            assert loop.state.pipeline.calibrations == 0
    assert loop.state.phase is (
        HandDreamsPhase.LIVE if captured else HandDreamsPhase.ALIGNING
    )
    assert loop.state.pipeline.calibrations == int(captured)
    loop.close()


def test_degenerate_controller_separation_waits_without_crashing(clock):
    loop = _session().model_loop
    _step(loop, clock, 0.0, _held(0, right={"position": (-0.2, 0.0, 0.0)}))
    assert loop.state.phase is HandDreamsPhase.CALIBRATING
    assert loop.state.alignment is None
    assert loop.state.pipeline.calibrations == 0
    assert loop.state.calibration_since is None
    _capture(loop, clock, start=0.25, first=1)
    _hold(loop, clock, start=1.25, first=5)
    assert loop.state.phase is HandDreamsPhase.LIVE
    loop.close()


def test_live_hud_reference_survives_tracking_loss_and_controller_b(clock):
    """Keep the UI's latest input while publishing each acknowledged calibration."""
    session = _session(auto_start=True)
    loop, ui = session.model_loop, session.ui_loop
    _step(loop, clock, 0.0, _frame(0))
    original = loop.state.controller_calibration
    assert original is not None

    latest = _frame(1, left={"trigger": 0.7}, right={"tracked": False})
    ui.state.observe_events(UserInputEvents([latest]))
    ui._run_message_batch()
    assert ui.state.controller_preview.calibration is original
    assert ui.state.controller_frame is latest
    assert ui.state.status.left_trigger == 0.7
    assert not ui.state.status.right_tracked

    disconnected = ControllerFrameEvent(timestamp=np.uint64(2), action="disconnected")
    ui.state.observe_events(UserInputEvents([disconnected]))
    _step(loop, clock, 0.1, disconnected)
    ui._run_message_batch()
    assert loop.state.controller_calibration is original
    assert ui.state.controller_preview.calibration is original
    assert not ui.state.status.left_tracked and not ui.state.status.right_tracked

    resumed = _frame(
        2,
        left={"position": (-0.2, 0.15, 0.0)},
        right={"position": (0.2, 0.15, 0.0), "secondary_button": True},
    )
    _step(loop, clock, 0.2, resumed)
    assert loop.state.controller_calibration is original
    assert loop.state.pipeline.calibrations == 1
    np.testing.assert_allclose(original.neutral[0].position, _frame(0).left.position)
    ui._run_message_batch()
    assert ui.state.controller_preview.calibration is original
    assert ui.state.controller_frame is disconnected

    loop.reset()
    ui._run_message_batch()
    assert loop.state.controller_calibration is None
    assert ui.state.controller_preview is None
    loop.close()


def test_two_resets_discard_guide_and_identity_without_closing_input(clock):
    loop = _session().model_loop
    for attempt in range(3):
        _capture(loop, clock, start=attempt * 2.0)
        _hold(loop, clock, start=attempt * 2.0 + 1.0)
        assert loop.state.phase is HandDreamsPhase.LIVE
        assert loop.state.pipeline.calibrations == attempt + 1
        if attempt < 2:
            loop.reset()
            assert loop.state.phase is HandDreamsPhase.WAITING
            assert loop.state.alignment is None
            assert loop.state.controller_calibration is None
            assert loop.state.calibration_since is None
            assert loop.state.alignment_since is None
            assert loop.state.last_input_frame_id == -1
            assert loop.state.last_input_capture_ns == -1
            assert len(loop.state.poses) == 0
            assert loop.state.pipeline.closes == 0
            _step(loop, clock, attempt * 2.0 + 2.0)
    loop.close()


def test_scene_handoff_repeats_alignment_using_shared_app_resources(clock, monkeypatch):
    inference_pipeline, pipeline = _FakeInferencePipeline(), _FakePipeline()
    loads = []
    app = HandDreamsV2Application(
        inference_pipeline_factory=lambda _: (
            loads.append(inference_pipeline) or inference_pipeline
        ),
        pipeline_factory=lambda **_: pipeline,
        scene_factory=lambda scenario, _: _scene(scenario.sample),
    )
    app.init(["--device", "cpu", "--sample", "scene-a"])
    assert not app.launch_config.auto_start
    for index, name in enumerate(("scene-a", "scene-b")):
        session = app.create_session(
            SessionDesc(
                video_width=640,
                video_height=360,
                metadata={"sample": name},
            )
        )
        session.init()
        loop = session.model_loop
        assert loop.state.alignment is None
        assert loop.state.controller_calibration is None
        _step(loop, clock, index * 2.0)
        _capture(loop, clock, start=index * 2.0)
        _hold(loop, clock, start=index * 2.0 + 1.0)
        assert pipeline.calibrations == index + 1
        loop.close()
        assert loop.state.controller_calibration is None
        assert pipeline.closes == 0
    assert loads == [inference_pipeline]
    assert pipeline.starts == 1
    app.close()
    assert pipeline.closes == 1


def test_explicit_auto_start_preserves_immediate_desktop_flow(clock):
    app = HandDreamsV2Application()
    app.init(["--auto-start"])
    assert app.launch_config.auto_start
    app.close()
    loop = _session(auto_start=True).model_loop
    _step(loop, clock, 0.0, _frame(0))
    assert loop.state.phase is HandDreamsPhase.LIVE
    assert loop.state.pipeline.calibrations == 1
    assert loop.state.alignment is None
    loop.close()


def _virtual_frame(index, **kwargs):
    return replace(_frame(index, **kwargs), is_emulated=True)


@pytest.mark.parametrize("side", ["left", "right"])
def test_virtual_controller_hold_then_release_skips_alignment(clock, side):
    session = _session()
    loop, ui = session.model_loop, session.ui_loop
    backwards = (0.0, 1.0, 0.0, 0.0)
    hands = {
        hand: {"orientation": backwards, "trigger": 0.9 if hand == side else 0.0}
        for hand in ("left", "right")
    }
    for index, now in enumerate((0.0, 0.25, 0.5, 0.75)):
        assert _step(loop, clock, now, _virtual_frame(index, **hands)) == []
        assert loop.state.phase is HandDreamsPhase.CALIBRATING
        assert loop.state.alignment is None
        assert loop.state.pipeline.calibrations == 0
    assert loop.state.calibration_progress == 1.0
    ui._run_message_batch()
    assert ui.state.status.is_emulated
    assert ui.state.status.calibration_progress == 1.0
    released = _virtual_frame(
        4,
        left={"position": (-0.3, 0.1, 0.0), "orientation": backwards},
        right={"position": (0.1, 0.1, 0.0), "orientation": backwards},
    )
    assert _step(loop, clock, 1.0, released) == []
    assert loop.state.phase is HandDreamsPhase.LIVE
    assert loop.state.pipeline.calibrations == 1
    assert loop.state.poses.peek(0)[0]["source_frame_id"] == 4
    np.testing.assert_allclose(
        loop.state.controller_calibration.neutral[0].position, released.left.position
    )
    assert len(_step(loop, clock, 1.1, _virtual_frame(5))) == 1
    loop.close()


@pytest.mark.parametrize("side", ["left", "right"])
def test_physical_controller_single_trigger_never_skips_alignment(clock, side):
    loop = _session().model_loop
    for index in range(6):
        _step(loop, clock, index / 4, _frame(index, **{side: {"trigger": 1.0}}))
    _step(loop, clock, 1.5, _frame(6))
    assert loop.state.phase is HandDreamsPhase.CALIBRATING
    assert loop.state.pipeline.calibrations == 0
    assert loop.state.calibration_progress == 0.0
    loop.close()


@pytest.mark.parametrize("batched", [False, True])
def test_virtual_hold_cannot_combine_different_controller_triggers(clock, batched):
    loop = _session().model_loop
    _step(loop, clock, 0.0, _virtual_frame(0, left={"trigger": 1.0}))
    _step(loop, clock, 0.3, _virtual_frame(1, left={"trigger": 1.0}))
    switched = _virtual_frame(2, right={"trigger": 1.0})
    if not batched:
        _step(loop, clock, 0.4, switched)
    _step(
        loop,
        clock,
        0.5,
        *([switched] if batched else []),
        _virtual_frame(3, right={"trigger": 1.0}),
    )
    assert loop.state.calibration_progress == 0.0
    assert loop.state.calibration_since == 0.5
    _step(loop, clock, 0.75, _virtual_frame(4, right={"trigger": 1.0}))
    _step(loop, clock, 1.0, _virtual_frame(5, right={"trigger": 1.0}))
    assert loop.state.calibration_progress == 1.0
    _step(loop, clock, 1.1, _virtual_frame(6))
    assert loop.state.phase is HandDreamsPhase.LIVE
    loop.close()


@pytest.mark.parametrize("interruption", ["tracking", "disconnect", "silence"])
@pytest.mark.parametrize("completed", [False, True])
def test_virtual_hold_and_release_latch_cancel_on_input_loss(
    clock, interruption, completed
):
    loop = _session().model_loop
    _step(loop, clock, 0.0, _virtual_frame(0, left={"trigger": 1.0}))
    _step(loop, clock, 0.25, _virtual_frame(1, left={"trigger": 1.0}))
    if completed:
        _step(loop, clock, 0.5, _virtual_frame(2, left={"trigger": 1.0}))
    now = 0.75
    if interruption == "tracking":
        events = [_virtual_frame(3, right={"tracked": False}), _virtual_frame(4)]
    elif interruption == "disconnect":
        events = [
            ControllerFrameEvent(timestamp=np.uint64(3), action="disconnected"),
            _virtual_frame(4),
        ]
    else:
        now = 1.1
        events = [_virtual_frame(4)]
    _step(loop, clock, now, *events)
    assert loop.state.phase is HandDreamsPhase.CALIBRATING
    assert loop.state.calibration_progress == 0.0
    assert loop.state.calibration_hand is None
    assert loop.state.pipeline.calibrations == 0
    loop.close()


def test_virtual_release_requires_both_triggers_open_and_latest_pose(clock):
    loop = _session().model_loop
    for index, now in enumerate((0.0, 0.25, 0.5)):
        _step(loop, clock, now, _virtual_frame(index, left={"trigger": 1.0}))
    _step(loop, clock, 0.6, _virtual_frame(3, right={"trigger": 0.10001}))
    assert loop.state.phase is HandDreamsPhase.CALIBRATING
    assert loop.state.pipeline.calibrations == 0
    _step(loop, clock, 0.7, _virtual_frame(4), _virtual_frame(5))
    assert loop.state.phase is HandDreamsPhase.LIVE
    assert loop.state.poses.peek(0)[0]["source_frame_id"] == 5
    loop.close()


def test_virtual_duplicate_frames_and_reset_cannot_reuse_trigger_hold(clock):
    loop = _session().model_loop
    held = _virtual_frame(0, left={"trigger": 1.0})
    _step(loop, clock, 0.0, held)
    _step(loop, clock, 0.25, held)
    _step(loop, clock, 0.5, held)
    assert loop.state.calibration_progress == 0.0
    _step(loop, clock, 0.51, _virtual_frame(1))
    assert loop.state.pipeline.calibrations == 0
    for index, now in enumerate((0.6, 0.85, 1.1), start=2):
        _step(loop, clock, now, _virtual_frame(index, left={"trigger": 1.0}))
    assert loop.state.calibration_progress == 1.0
    loop.reset()
    assert loop.state.input_is_emulated is None
    assert loop.state.calibration_hand is None
    _step(loop, clock, 1.2)
    _step(loop, clock, 1.3, _virtual_frame(0))
    assert loop.state.phase is HandDreamsPhase.CALIBRATING
    assert loop.state.calibration_progress == 0.0
    loop.close()


def test_physical_source_cannot_inherit_virtual_live_calibration(clock):
    loop = _session().model_loop
    for index, now in enumerate((0.0, 0.25, 0.5)):
        _step(loop, clock, now, _virtual_frame(index, left={"trigger": 1.0}))
    _step(loop, clock, 0.6, _virtual_frame(3))
    assert loop.state.phase is HandDreamsPhase.LIVE
    _step(loop, clock, 0.7, _virtual_frame(4), _frame(5))
    assert loop.state.phase is HandDreamsPhase.CALIBRATING
    assert loop.state.input_is_emulated is False
    assert loop.state.controller_calibration is None
    assert not loop.state.poses.ready(0)
    assert loop.state.last_input_frame_id == 5
    loop.close()


def test_missing_scene_geometry_has_actionable_error_and_auto_start_escape(clock):
    scene = _scene()
    scene.wrist_poses = None
    with pytest.raises(ValueError, match="--auto-start"):
        _session(scene=scene)
    loop = _session(scene=scene, auto_start=True).model_loop
    _step(loop, clock, 0.0, _frame(0))
    assert loop.state.phase is HandDreamsPhase.LIVE
    loop.close()
