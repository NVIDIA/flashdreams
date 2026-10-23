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

"""CPU checks for the live stopwatch's timing, placement, and reset lifecycle."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from flashdreams_controller_input.events import ControllerFrameEvent
from handdreams import ui
from handdreams.scenes import ScenePreview

from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (-1.0, "00:00"),
        (0.0, "00:00"),
        (59.99, "00:59"),
        (65.9, "01:05"),
        (3599.0, "59:59"),
        (3661.0, "01:01:01"),
    ],
)
def test_stopwatch_formats_elapsed_time(seconds: float, expected: str) -> None:
    assert ui._format_stopwatch(seconds) == expected


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.float16, torch.float32])
def test_stopwatch_preserves_pixels_outside_bottom_left_and_alpha(
    channels: int, dtype: torch.dtype
) -> None:
    frame = torch.full(
        (channels, 360, 640), 0.5 if dtype.is_floating_point else 128, dtype=dtype
    )
    original = frame.clone()
    ui._draw_stopwatch(frame, 3661.0)
    assert frame.dtype is dtype
    assert frame.shape == original.shape
    assert not torch.equal(frame[:3, 312:352, 8:124], original[:3, 312:352, 8:124])
    assert torch.equal(frame[:, :312], original[:, :312])
    assert torch.equal(frame[:, 352:], original[:, 352:])
    assert torch.equal(frame[:, :, :8], original[:, :, :8])
    assert torch.equal(frame[:, :, 124:], original[:, :, 124:])
    if channels == 4:
        assert torch.equal(frame[3], original[3])
    if dtype.is_floating_point:
        assert -1 <= frame.min() <= frame.max() <= 1


@pytest.mark.parametrize("height,width", [(1, 1), (32, 64), (199, 640), (360, 319)])
def test_stopwatch_omits_panels_that_cannot_fit(height: int, width: int) -> None:
    frame = torch.zeros((3, height, width), dtype=torch.uint8)
    ui._draw_stopwatch(frame, 65)
    assert not frame.any()


def test_hour_readout_leaves_room_for_xr_hud_on_narrow_video() -> None:
    frame = torch.zeros((3, 200, 320), dtype=torch.uint8)
    ui._draw_stopwatch(frame, 3661)
    assert frame[:, 138:192, 8:132].any()
    assert not frame[:, :, 132:].any()


def _loop() -> ui.HandDreamsUILoop:
    loop = ui.HandDreamsUILoop()
    loop.state = ui.HandDreamsUIState(
        status=ui.HandDreamsUIStatus(
            phase=ui.HandDreamsPhase.LIVE,
            current_scene="scene-a",
            scene_catalog=("scene-a", "scene-b"),
            left_tracked=True,
            right_tracked=True,
        ),
        scene_previews=(
            ScenePreview("scene-a", "Scene A"),
            ScenePreview("scene-b", "Scene B"),
        ),
    )
    loop.session_desc = SessionDesc(video_width=640, video_height=468)
    loop.output_layout = VideoTensorLayout.tchw
    loop._ui_loop_requests = None
    return loop


def test_stopwatch_starts_at_first_video_and_advances_during_tracking_loss(
    monkeypatch,
) -> None:
    loop = _loop()
    clock = [100.0]
    current = [None]
    elapsed = []
    draw_stopwatch = ui._draw_stopwatch

    def draw(frame: torch.Tensor, seconds: float) -> None:
        elapsed.append(seconds)
        draw_stopwatch(frame, seconds)

    monkeypatch.setattr(ui.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(ui, "_draw_stopwatch", draw)
    monkeypatch.setattr(loop, "presented_model_frame", lambda: current[0])
    events = UserInputEvents([])
    loop.step(0, events)
    assert loop.state.live_started_at is None
    assert not elapsed

    clock[0] = 300.0
    source = torch.full((3, 360, 640), 128, dtype=torch.uint8)
    current[0] = source
    (first,) = loop.step(1, events)
    assert loop.state.live_started_at == 300.0
    assert elapsed == [0.0]
    assert torch.all(source == 128)

    clock[0] = 365.9
    disconnected = ControllerFrameEvent(timestamp=np.uint64(1), action="disconnected")
    (later,) = loop.step(2, UserInputEvents([disconnected]))
    assert not loop.state.status.left_tracked
    assert loop.state.live_started_at == 300.0
    assert elapsed == pytest.approx([0.0, 65.9])
    first_frame, later_frame = first.read_output()[0], later.read_output()[0]
    assert not torch.equal(
        first_frame[:, 298:352, 8:140], later_frame[:, 298:352, 8:140]
    )
    assert torch.equal(first_frame[:, 360:], later_frame[:, 360:])
    assert torch.equal(first_frame[:, 18:298], later_frame[:, 18:298])
    assert torch.equal(first_frame[:, 18:, 180:], later_frame[:, 18:, 180:])


@pytest.mark.parametrize("action", ["reset", "next_scene", "scene_picker"])
def test_stopwatch_resets_with_session_and_scene_changes(
    monkeypatch, action: str
) -> None:
    loop = _loop()
    loop.state.live_started_at = 100.0
    if action == "reset":
        loop.reset()
    else:
        loop._request_scene_name("scene-b" if action == "next_scene" else None)
    assert loop.state.live_started_at is None

    loop.state.status = replace(loop.state.status, phase=ui.HandDreamsPhase.LIVE)
    monkeypatch.setattr(ui.time, "monotonic", lambda: 500.0)
    monkeypatch.setattr(loop, "presented_model_frame", lambda: None)
    loop.step(0, UserInputEvents([]))
    assert loop.state.live_started_at is None
    monkeypatch.setattr(
        loop,
        "presented_model_frame",
        lambda: torch.zeros((3, 360, 640), dtype=torch.uint8),
    )
    loop.step(1, UserInputEvents([]))
    assert loop.state.live_started_at == 500.0


@pytest.mark.parametrize(
    "phase",
    [phase for phase in ui.HandDreamsPhase if phase is not ui.HandDreamsPhase.LIVE],
)
def test_stopwatch_is_hidden_outside_live_video(
    monkeypatch, phase: ui.HandDreamsPhase
) -> None:
    loop = _loop()
    loop.state.status = replace(loop.state.status, phase=phase)
    elapsed = []
    monkeypatch.setattr(
        ui, "_draw_stopwatch", lambda frame, seconds: elapsed.append(seconds)
    )
    monkeypatch.setattr(ui.TutorialVideo, "frame", lambda *args: None)
    monkeypatch.setattr(
        loop,
        "presented_model_frame",
        lambda: torch.zeros((3, 360, 640), dtype=torch.uint8),
    )
    loop.step(0, UserInputEvents([]))
    assert loop.state.live_started_at is None
    assert not elapsed
