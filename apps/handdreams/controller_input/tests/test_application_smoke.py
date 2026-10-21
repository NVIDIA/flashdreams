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

"""Controller input in a model-free V2 application with CPU-only transports."""

import time
from typing import cast

import pytest
import torch
from flashdreams_controller_input.controller_source_v2_window import (
    ControllerSourceWindow,
)
from flashdreams_controller_input.controller_types import Observation
from flashdreams_controller_input.events import ControllerFrameEvent
from flashdreams_controller_input.isaac_teleop_source import IsaacTeleopSource
from flashdreams_controller_input.isaac_teleop_v2_window import IsaacTeleopWebRTCWindow
from flashdreams_controller_input.pose import WRIST_PROXY_POSE_CONVENTION

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.loop import IModelLoop
from flashdreams.api_v2.session import ISession
from flashdreams.runtime_v2.application_runner import ApplicationRunner
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout
from flashdreams.runtime_v2.webrtc_client_window import WebRTCClientWindow

pytestmark = pytest.mark.ci_cpu


class _Source:
    name = "cpu-test"

    def __init__(self):
        self.sequence = 0
        self.stopped = False

    def start(self):
        pass

    def poll(self):
        self.sequence += 1
        hand = {
            "position": [0.0, 1.0, -0.5],
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "primary_button": True,
        }
        captured = time.perf_counter_ns()
        return Observation(
            frame={
                "left": hand,
                "right": hand,
                "client_capture_ns": captured,
                "source_frame_index": self.sequence,
            },
            capture_ns=captured,
            source="cpu-test",
        )

    def stop(self):
        self.stopped = True


class _Output:
    def __init__(self):
        self.frames = []
        self.closed = False

    def open(self, session_desc):
        pass

    def get_user_input_events(self):
        return UserInputEvents([])

    def write(self, result):
        self.frames.append(result.read_output())

    def close(self):
        self.closed = True


class _Model(IModelLoop[list]):
    def step(self, step_index, events):
        frames = [
            event
            for event in events.get_events()
            if isinstance(event, ControllerFrameEvent) and event.action == "state"
        ]
        if not frames:
            return []
        self.state.extend(frames)
        return [
            StepResult(
                step_index=step_index,
                output=torch.zeros((1, 3, 2, 2), dtype=torch.uint8),
                frame_count=1,
                output_layout=VideoTensorLayout.tchw,
            )
        ]

    def is_finished(self):
        return bool(self.state)


class _Session(ISession):
    def __init__(self, description, frames):
        self.description = description
        self.frames = frames

    @property
    def session_desc(self):
        return self.description

    def init(self):
        self.register_model_loop(_Model, state=self.frames)


class _Application(IApplication):
    def __init__(self):
        self.frames = []
        self.closed = False

    def init(self, commandline_args):
        pass

    def create_session(self, session_desc):
        return _Session(session_desc, self.frames)

    def close(self):
        self.closed = True


@pytest.mark.parametrize("route", ["remote", "isaac"])
def test_device_window_runs_with_an_unrelated_v2_application(route):
    """Deliver paired events and video through the normal application runner."""
    source, output = _Source(), _Output()
    if route == "remote":
        window = ControllerSourceWindow(
            output_window=cast(WebRTCClientWindow, output),
            source=source,
            source_name="remote Isaac Teleop",
            pose_convention=WRIST_PROXY_POSE_CONVENTION,
            source_thread_name="test-controller-input",
            source_failure_message="remote Isaac input failed",
            client_capture_keys=("source_client_capture_ns", "client_capture_ns"),
        )
    else:
        window = IsaacTeleopWebRTCWindow(
            host="127.0.0.1",
            port=0,
            python_executable="unused",
            bridge_port=0,
            source_rate_hz=120,
            tracking_rate_hz=72,
            wrist_offset_m=0.05,
            startup_timeout_s=None,
            install_dir="unused",
            env_config=None,
            device_profile="Quest3",
            accept_eula=False,
            verbose=False,
            _isaac_source=cast(IsaacTeleopSource, source),
            _webrtc_window=cast(WebRTCClientWindow, output),
        )
    app = _Application()
    ApplicationRunner(app, window).run(
        SessionDesc(video_width=2, video_height=2),
        timeout_seconds=3,
    )
    assert app.frames and app.frames[0].left.primary_button
    assert output.frames and output.frames[-1].shape[-3:] == (3, 2, 2)
    assert source.stopped and output.closed and app.closed
    assert not window._source_thread.is_alive()
