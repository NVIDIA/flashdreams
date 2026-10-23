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

"""First output at production controller cadence without GPU inference."""

from dataclasses import replace

import numpy as np
import pytest
from flashdreams_controller_input.events import ControllerFrameEvent
from handdreams import session as session_module
from handdreams.ui import HandDreamsPhase

from apps.handdreams.tests.test_calibration_flow import (
    _frame,
    _session,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

pytestmark = pytest.mark.ci_cpu


@pytest.fixture
def live_loop(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(session_module.time, "perf_counter", lambda: clock[0])
    session = _session(auto_start=True)
    loop = session.model_loop
    loop.state.builder.pipeline.actions_per_step = 8

    def feed(index):
        clock[0] = index / 72.0
        event = replace(
            _frame(index),
            client_capture_ns=1 + round(index * 1_000_000_000 / 72.0),
        )
        return loop.step(index, UserInputEvents([event]))

    yield loop, feed
    loop.close()


def test_stationary_controllers_generate_full_chunks_and_log_startup_once(
    live_loop, capsys
):
    loop, feed = live_loop
    assert feed(0) == []
    assert loop.state.phase is HandDreamsPhase.LIVE
    for index in range(1, 16):
        assert feed(index) == []
    assert "preparing first" not in capsys.readouterr().out

    assert len(feed(16)) == 1
    assert loop.state.pipeline.windows == [list(range(0, 17, 2))]
    assert loop.state.inference_cache.actions[0].shape == (8, 64)
    startup = capsys.readouterr().out
    stages = (
        "preparing first 8 actions from 9 controller poses",
        "generating first video chunk",
        "first diffusion complete; decoding video",
        "first video chunk ready",
    )
    assert [startup.index(stage) for stage in stages] == sorted(
        startup.index(stage) for stage in stages
    )

    for index in range(17, 32):
        assert feed(index) == []
    assert len(feed(32)) == 1
    assert loop.state.pipeline.windows[-1] == list(range(16, 33, 2))
    assert capsys.readouterr().out == ""
    assert loop.state.rows_consumed == 16


def test_disconnect_discards_partial_first_chunk_and_stationary_input_recovers(
    live_loop,
):
    loop, feed = live_loop
    for index in range(5):
        assert feed(index) == []
    assert len(loop.state.poses) == 3
    disconnected = ControllerFrameEvent(timestamp=np.uint64(5), action="disconnected")
    assert loop.step(5, UserInputEvents([disconnected])) == []
    assert len(loop.state.poses) == 0
    for index in range(6, 22):
        assert feed(index) == []
    assert len(feed(22)) == 1
    assert loop.state.pipeline.windows == [list(range(6, 23, 2))]
    assert loop.state.pipeline.calibrations == 1


def test_reset_restarts_first_chunk_with_fresh_source_identity(live_loop, capsys):
    loop, feed = live_loop
    for index in range(17):
        feed(index)
    assert loop.state.steps_generated == 1
    loop.reset()
    assert loop.step(0, UserInputEvents([])) == []
    capsys.readouterr()
    for index in range(16):
        assert feed(index) == []
    assert len(feed(16)) == 1
    assert loop.state.pipeline.windows == [list(range(0, 17, 2))] * 2
    assert loop.state.steps_generated == 1
    assert loop.state.pipeline.calibrations == 2
    assert "first video chunk ready" in capsys.readouterr().out


def test_failed_first_inference_does_not_report_ready_output(live_loop, capsys):
    loop, feed = live_loop
    loop.state.builder.pipeline.reject_before_callback = True
    for index in range(16):
        assert feed(index) == []
    capsys.readouterr()
    with pytest.raises(RuntimeError, match="Cosmos rejected action"):
        feed(16)
    startup = capsys.readouterr().out
    assert "generating first video chunk" in startup
    assert "diffusion complete" not in startup
    assert "video chunk ready" not in startup
    assert loop.state.pipeline.aborts == 1
