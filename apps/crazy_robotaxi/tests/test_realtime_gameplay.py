# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Robotaxi presentation while independent gameplay is stationary."""

import threading
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from omnidreams_game_engine.engine import SimulationStep
from omnidreams_game_engine.input import DriverInput
from omnidreams_game_engine.types import TrajectoryChunk, VehicleState

from flashdreams.runtime_v2.user_input_events import UserInputEvents

pytestmark = pytest.mark.ci_cpu


def test_stationary_world_retains_image_and_ar_cursor_until_motion(monkeypatch):
    import queue
    from unittest.mock import Mock

    import torch
    from crazy_robotaxi.controls import ControlsConfig
    from crazy_robotaxi.free_roam import FreeRoamSnapshot
    from crazy_robotaxi.live_edit.config import LiveEditConfig
    from crazy_robotaxi.session import (
        CrazyRobotaxiModelLoop,
        ModelState,
        _stationary_world,
    )
    from omnidreams_game_engine.timeline import WorldSample, WorldTimeline

    from flashdreams.runtime_v2.session_desc import SessionDesc

    snapshot = FreeRoamSnapshot()
    vehicle = VehicleState(0, 0, 0, 0, 0, 0)
    path = TrajectoryChunk(
        np.array([0]), np.eye(4, dtype=np.float32)[None], (vehicle,), vehicle
    )
    timeline = WorldTimeline()
    for tick in range(33):
        timeline.append(
            WorldSample(
                SimulationStep(
                    replace(path, timestamps_us=np.array([tick * 8333])), (snapshot,)
                )
            )
        )
    rollout = SimpleNamespace(
        engine=SimpleNamespace(current_game_frame=snapshot),
        scene=SimpleNamespace(prompt="A road."),
        frame_count=lambda index: 8,
        step=Mock(
            side_effect=AssertionError("Advanced model while world was stationary")
        ),
    )
    monkeypatch.setattr(ModelState, "ensure_rollout", lambda self: rollout)
    published = []
    config = SimpleNamespace(
        controls=ControlsConfig(),
        debug_capture=False,
        profile_input_latency=False,
        live_edit=LiveEditConfig(),
        total_blocks=4,
        pipeline_profiling=False,
        no_ui=False,
    )
    state = ModelState(
        pipeline_factory=lambda: None,
        scene_factory=cast(Any, lambda *args: None),
        config=cast(Any, config),
        session_desc=SessionDesc(video_width=8, video_height=4),
        driver_input=DriverInput(),
        ui_loop=cast(Any, SimpleNamespace(_invoke_async=published.append)),
    )
    state.game_selected = True
    state.gameplay = cast(Any, SimpleNamespace(timeline=timeline, check=lambda: None))
    state.last_video = torch.ones(1, 3, 4, 8)
    state.last_hdmap = torch.zeros(1, 3, 4, 8)
    state.last_pose = np.eye(4, dtype=np.float32)
    state.blocks_generated = 3
    state.blocks_output = 3
    loop = CrazyRobotaxiModelLoop()
    loop.register_session_loop_objects(
        state=state,
        frequency=30,
        shutdown_event=threading.Event(),
        failure_queue=queue.Queue(),
    )
    result = loop.step(0, UserInputEvents([]))
    assert state.blocks_generated == 3 and not rollout.step.called
    assert state.blocks_output == 4 and state.finished
    assert torch.equal(
        result[0].read_output(), torch.ones_like(result[0].read_output())
    )
    assert result[0].metrics is not None
    assert result[0].metrics["reused_video"] == 1
    assert timeline.generation_tick > 0
    moving = replace(path, vehicle_states=(replace(vehicle, speed_mps=1),))
    assert not _stationary_world(moving, state.last_pose)
    shifted = path.rig_poses_world.copy()
    shifted[0, 0, 3] = 0.1
    assert not _stationary_world(
        replace(path, rig_poses_world=shifted), state.last_pose
    )
