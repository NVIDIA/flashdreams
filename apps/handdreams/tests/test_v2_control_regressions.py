# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-cadence, disconnect ordering, and recenter transaction regressions."""

from __future__ import annotations

import copy
import queue
import threading
from dataclasses import replace

import numpy as np
import pytest
from flashdreams_controller_input.events import ControllerFrameEvent
from handdreams.config import (
    HandDreamsLaunchConfig,
    HanddreamsScenario,
)
from handdreams.controllers.isaac_teleop_control import (
    ControllerControlConfig,
    ControllerControlShaper,
)
from handdreams.controllers.sampler import ControlWindowSampler
from handdreams.session import (
    HandDreamsModelLoop,
    HandDreamsModelState,
)
from handdreams.ui import HandDreamsPhase

from apps.handdreams.tests.helpers import (
    _event,
    _FakeInferencePipeline,
    _FakePipeline,
    _FakeRolloutBuilder,
    _pipeline_config,
    _raw_frame,
    _scene,
)
from apps.handdreams.tests.test_teleop_pipeline import (
    _Bridge,
    _grippers,
    _pipeline,
)
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.user_input_events import UserInputEvents

pytestmark = pytest.mark.ci_cpu


def _live_loop(*, control_fps: int | None = 36, stale_ms: float | None = 500.0):
    state = HandDreamsModelState(
        builder=_FakeRolloutBuilder(
            pipeline=_FakeInferencePipeline(),
            scene=_scene(),
        ),
        config=HandDreamsLaunchConfig(
            scenario=HanddreamsScenario(sample="scene-a"),
            pipeline_config=_pipeline_config(),
            control_fps=control_fps,
            stale_ms=stale_ms,
            auto_start=True,
        ),
        session_desc=SessionDesc(),
        pipeline=_FakePipeline(),
        phase=HandDreamsPhase.LIVE,
    )
    state.builder.prepare_scene()
    loop = HandDreamsModelLoop()
    loop.register_session_loop_objects(
        state=state,
        frequency=36,
        shutdown_event=threading.Event(),
        failure_queue=queue.Queue(),
    )
    return loop


def _source_frame(index: int, *, hz: int = 72):
    frame = _raw_frame(index)
    frame["client_capture_ns"] = 1 + round(index * 1e9 / hz)
    return frame


def test_batch_ending_in_disconnect_does_not_revive_pre_disconnect_poses() -> None:
    loop = _live_loop()
    events = [
        _event(1),
        ControllerFrameEvent(timestamp=np.uint64(2), action="disconnected"),
    ]

    loop._consume_live_events(UserInputEvents(events), now=1.0)

    assert len(loop.state.poses) == 0
    assert loop.state.last_wrist_observed_at is None
    assert loop.state.last_input_frame_id == -1
    assert loop.state.last_input_capture_ns == -1


def test_disconnect_keeps_only_reconnected_suffix_from_one_batch() -> None:
    loop = _live_loop()
    events = [
        replace(_event(1), timestamp=np.uint64(0)),
        ControllerFrameEvent(timestamp=np.uint64(0), action="disconnected"),
        replace(_event(3), timestamp=np.uint64(0)),
    ]

    loop._consume_live_events(UserInputEvents(events), now=1.0)

    assert loop.state.poses.peek(0)[0]["source_frame_id"] == 3
    assert loop.state.last_input_frame_id == 3
    assert loop.state.last_input_capture_ns == _event(3).client_capture_ns


def test_discarded_prefix_does_not_advance_reconnected_source_identity() -> None:
    loop = _live_loop()
    loop._consume_live_events(UserInputEvents([_event(5)]), now=1.0)
    events = [
        replace(_event(20), timestamp=np.uint64(10)),
        ControllerFrameEvent(timestamp=np.uint64(11), action="disconnected"),
        replace(_event(6), timestamp=np.uint64(12)),
    ]

    loop._consume_live_events(UserInputEvents(events), now=2.0)

    assert loop.state.poses.peek(0)[0]["source_frame_id"] == 6
    assert loop.state.last_input_frame_id == 6
    assert loop.state.last_input_capture_ns == _event(6).client_capture_ns


def test_controller_frames_follow_source_order_and_reject_reused_identity(
    monkeypatch,
) -> None:
    loop = _live_loop()
    transformed = []

    def transform(frame):
        transformed.append(frame["source_frame_id"])
        return frame

    monkeypatch.setattr(loop.state.pipeline, "transform", transform)
    events = [replace(_event(index), timestamp=np.uint64(0)) for index in (3, 1, 2, 2)]
    loop._consume_live_events(UserInputEvents(events), now=1.0)
    assert transformed == [1, 2, 3]

    events = [
        replace(_event(3), client_capture_ns=5_000_001),
        replace(_event(4), client_capture_ns=_event(3).client_capture_ns),
        replace(_event(5), client_capture_ns=_event(2).client_capture_ns),
        _event(6),
    ]
    loop._consume_live_events(UserInputEvents(events), now=2.0)
    assert transformed == [1, 2, 3, 6]
    assert loop.state.last_input_frame_id == 6
    assert loop.state.last_input_capture_ns == _event(6).client_capture_ns


def test_disconnect_preserves_source_identity_until_session_reset() -> None:
    loop = _live_loop()
    loop._consume_live_events(UserInputEvents([_event(5)]), now=1.0)
    loop._consume_live_events(
        UserInputEvents(
            [ControllerFrameEvent(timestamp=np.uint64(6), action="disconnected")]
        ),
        now=2.0,
    )
    loop._consume_live_events(UserInputEvents([_event(5)]), now=3.0)
    assert len(loop.state.poses) == 0
    assert loop.state.last_input_frame_id == 5

    loop.reset()
    assert loop.state.last_input_frame_id == -1
    assert loop.state.last_input_capture_ns == -1
    loop._consume_live_events(UserInputEvents([_event(0)]), now=4.0)
    assert loop.state.poses.peek(0)[0]["source_frame_id"] == 0
    assert loop.state.last_input_frame_id == 0
    loop.close()
    assert loop.state.last_input_frame_id == -1
    assert loop.state.last_input_capture_ns == -1


@pytest.mark.parametrize("source_hz", [72, 120])
@pytest.mark.parametrize("ui_hz", [15, 36, 72])
def test_live_cadence_uses_source_time_not_ui_batch_rate(
    source_hz: int, ui_hz: int
) -> None:
    loop = _live_loop()
    batch = []
    next_ui_ns = round(1e9 / ui_hz)
    for index in range(source_hz + 1):
        capture_ns = 1 + round(index * 1e9 / source_hz)
        batch.append(
            replace(
                _event(index),
                timestamp=np.uint64(capture_ns // 1000),
                client_capture_ns=capture_ns,
            )
        )
        if capture_ns >= next_ui_ns or index == source_hz:
            loop._consume_live_events(UserInputEvents(batch), now=capture_ns / 1e9)
            batch.clear()
            next_ui_ns += round(1e9 / ui_hz)

    frames = loop.state.poses.peek(36)
    assert len(frames) == 37  # 37 source poses -> 36 action rows.
    assert [frame["source_frame_id"] for frame in frames] == [
        round(index * source_hz / 36) for index in range(37)
    ]
    assert [frame["client_capture_ns"] for frame in frames] == [
        1 + round(frame["source_frame_id"] * 1e9 / source_hz) for frame in frames
    ]
    loop.state.poses.consume(36)
    assert len(loop.state.poses) == 1


def test_resampler_preserves_jittered_original_poses_and_source_timestamps() -> None:
    loop = _live_loop()
    source = []
    for index in range(73):
        frame = _source_frame(index)
        if 0 < index < 72:
            frame["client_capture_ns"] += (index % 3 - 1) * 700_000
        source.append(frame)
        loop._append_live_pose(frame)

    selected = loop.state.poses.peek(36)
    interval_ns = int(1e9 / 36)
    expected = [
        min(
            source,
            key=lambda frame: abs(
                frame["client_capture_ns"] - (1 + index * interval_ns)
            ),
        )
        for index in range(37)
    ]
    assert selected == expected
    assert all(frame in source for frame in selected)


def test_low_control_rate_does_not_require_retaining_a_whole_source_second() -> None:
    loop = _live_loop(control_fps=1)
    for index in range(121):
        loop._append_live_pose(_source_frame(index, hz=120))

    assert [frame["source_frame_id"] for frame in loop.state.poses.peek(1)] == [0, 120]
    assert len(loop.state.control_sampler) <= 16


@pytest.mark.parametrize(
    ("scene_fps", "control_fps", "expected_rate"),
    [(30, None, 30), (24, None, 24), (30, 15, 15)],
)
def test_live_cadence_defaults_to_scene_fps_and_preserves_explicit_overrides(
    scene_fps: int, control_fps: int | None, expected_rate: int
) -> None:
    loop = _live_loop(control_fps=control_fps)
    loop.state.scene.fps = scene_fps
    assert loop.state.target_frames is None
    loop.state.config = replace(loop.state.config, duration_seconds=10.0)
    assert loop.state.target_frames == 10 * expected_rate
    for index in range(121):
        loop._append_live_pose(_source_frame(index, hz=120))

    assert loop.state.control_sampler.interval_ns == int(1e9 / expected_rate)
    frames = loop.state.poses.peek(expected_rate)
    assert len(frames) == expected_rate + 1
    assert [frame["source_frame_id"] for frame in frames] == [
        index * 120 // expected_rate for index in range(expected_rate + 1)
    ]
    loop.state.poses.consume(expected_rate)
    assert len(loop.state.poses) == 1


@pytest.mark.parametrize("stale_ms", [500.0, None])
def test_stale_gap_reseeds_without_interpolated_or_catchup_actions(stale_ms) -> None:
    loop = _live_loop(stale_ms=stale_ms)
    for index in range(4):
        loop._append_live_pose(_source_frame(index))
    resumed = _source_frame(10000)

    loop._append_live_pose(resumed)

    assert len(loop.state.poses) == 1
    assert loop.state.last_control_ns == resumed["client_capture_ns"]
    assert loop.state.next_control_ns > resumed["client_capture_ns"]
    loop._append_live_pose(_source_frame(10002))
    assert loop.state.poses.peek(1) == [resumed, _source_frame(10002)]


def test_tracking_loss_clears_sampler_and_overlap_endpoint() -> None:
    loop = _live_loop()
    loop._append_live_pose(_source_frame(0))
    loop._append_live_pose(_source_frame(2))
    loop._handle_tracking_loss()
    assert len(loop.state.poses) == len(loop.state.control_sampler) == 0
    assert loop.state.next_control_ns is None
    loop._append_live_pose(_source_frame(3))
    assert len(loop.state.poses) == 1


def test_source_silence_expires_tracking_without_an_explicit_disconnect() -> None:
    loop = _live_loop()
    loop._append_live_pose(_source_frame(0))
    loop.state.last_wrist_observed_at = 0.0

    loop._expire_stale_input(now=0.6)

    assert len(loop.state.poses) == len(loop.state.control_sampler) == 0
    assert loop.state.next_control_ns is None
    assert loop.state.last_wrist_observed_at is None


def test_incremental_sampler_reuses_nearest_neighbour_with_no_extrapolation() -> None:
    sampler = ControlWindowSampler(rate_hz=1)
    first, second = _source_frame(0, hz=1), _source_frame(2, hz=1)
    assert sampler.interval_ns == 1_000_000_000
    assert sampler.sample_at(0) is None
    sampler.add(first)
    sampler.add(second)
    assert not sampler.add(first)
    assert not sampler.add(second)
    assert sampler.sample_at(first["client_capture_ns"]) is first
    midpoint = (first["client_capture_ns"] + second["client_capture_ns"]) // 2
    assert sampler.sample_at(midpoint - 1) is first
    assert sampler.sample_at(midpoint) is second
    assert sampler.sample_at(second["client_capture_ns"] - 1) is second
    assert sampler.sample_at(second["client_capture_ns"] + 1) is None


def test_incremental_sampler_retains_sixteen_original_observations() -> None:
    sampler = ControlWindowSampler(rate_hz=36)
    frames = [_source_frame(index) for index in range(18)]
    for frame in frames:
        assert sampler.add(frame)
    assert len(sampler) == 16
    assert sampler.sample_at(frames[1]["client_capture_ns"]) is None
    assert sampler.sample_at(frames[2]["client_capture_ns"]) is frames[2]


def test_recenter_preserves_committed_gripper_but_session_reset_reseeds() -> None:
    pipeline, wrist = _pipeline(slew=0.25)
    frames = [copy.deepcopy(wrist) for _ in range(3)]
    first = pipeline.prepare_window(frames, command_sequence=1)
    pipeline.commit(first)

    def calibrate():
        pipeline.calibrate(
            wrist,
            scene_gripper_open={"left": 1.0, "right": 1.0},
            position_gain=1.2,
            grip_threshold=0.0,
            stale_ms=500.0,
            gripper_slew_per_action=0.25,
        )

    calibrate()
    control = _Bridge.instance.controls[-1]
    assert control is not None
    assert control["scene_gripper_open"] == {
        "left": 0.5,
        "right": 0.5,
    }
    recentered = pipeline.prepare_window(frames, command_sequence=2)
    assert _grippers(recentered) == pytest.approx([0.5, 0.25, 0.0])
    pipeline.abort(recentered)
    pipeline.reset()
    calibrate()
    control = _Bridge.instance.controls[-1]
    assert control is not None
    assert control["scene_gripper_open"] == {
        "left": 1.0,
        "right": 1.0,
    }
    reset = pipeline.prepare_window(frames, command_sequence=3)
    assert _grippers(reset) == pytest.approx([1.0, 0.75, 0.5])


def test_recenter_held_trigger_stays_closed_until_release_in_actual_shaper() -> None:
    pipeline, wrist = _pipeline(slew=0.25)
    closed = pipeline.prepare_window(
        [copy.deepcopy(wrist) for _ in range(5)], command_sequence=1
    )
    pipeline.commit(closed)
    neutral = copy.deepcopy(wrist)
    for side in ("left", "right"):
        neutral[side]["trigger"] = 1.0
    pipeline.calibrate(
        neutral,
        scene_gripper_open={"left": 1.0, "right": 1.0},
        position_gain=1.2,
        grip_threshold=0.0,
        stale_ms=500.0,
        gripper_slew_per_action=0.25,
    )
    shaper = ControllerControlShaper(
        ControllerControlConfig.from_payload(_Bridge.instance.controls[-1])
    )
    frames = [copy.deepcopy(neutral) for _ in range(3)]
    for index, frame in enumerate(frames):
        frame["client_capture_ns"] = 1 + index * 30_000_000
    for side in ("left", "right"):
        frames[-1][side]["trigger"] = 0.0

    candidate = shaper.prepare(frames, copy.deepcopy(frames))
    shaper.commit(candidate)
    shaped = candidate.frames

    for side in ("left", "right"):
        assert [frame[side]["gripper_open"] for frame in shaped] == [0.0, 0.0, 1.0]
