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

"""CPU checks for scripted controls and paired headless observations."""

import argparse
import json
import socket
from functools import partial
from pathlib import Path

import pytest
from crazy_robotaxi.controls import GamepadControls, gamepad_driver_command
from crazy_robotaxi.debug import (
    DEBUG_METADATA_KEY,
    RobotaxiDebugMode,
    RobotaxiDebugWindow,
)
from omnidreams_game_engine.config import RasterConfig
from omnidreams_game_engine.debug_input import (
    DEBUG_CONTROL_KEY,
    HeadlessInput,
)
from omnidreams_game_engine.driving_script import load_driving_script
from omnidreams_game_engine.game_map import compile_game_map
from omnidreams_game_engine.input import DriverInput
from omnidreams_game_engine.scene import load_scene_bundle

from flashdreams.runtime_v2.session_desc import (
    SessionDesc,
)
from flashdreams.runtime_v2.user_input_event import (
    GamepadUserInputEvent,
)

pytestmark = pytest.mark.ci_cpu


def test_timestamped_gamepad_file_and_socket_reach_driver(tmp_path):
    packet = {
        "id": "gentle",
        "timestamp_us": 100_000,
        "type": "gamepad",
        "axes": [-0.25, 0, 0, 0],
        "buttons": [0] * 7 + [0.6],
    }
    source = tmp_path / "inputs.yaml"
    source.write_text(json.dumps({"events": [packet]}))
    assert load_driving_script(source).events == (packet,)
    now = [0]
    control = HeadlessInput(lambda: now[0])
    window = RobotaxiDebugWindow(tmp_path / "capture", clock_ns=lambda: now[0])
    window.open(
        SessionDesc(
            video_width=6,
            video_height=4,
            metadata={DEBUG_METADATA_KEY: {}, DEBUG_CONTROL_KEY: control},
        )
    )
    driver = DriverInput(
        gamepad_command=partial(gamepad_driver_command, GamepadControls())
    )
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as peer:
        peer.bind(str(tmp_path / "gamepad.sock"))
        peer.settimeout(1)
        peer.sendto(json.dumps(packet).encode(), str(window.directory / "control.sock"))
        assert window.get_user_input_events().get_events() == []
        assert json.loads(peer.recv(4096))["status"] == "accepted"
        control.start()
        now[0] = 99_000_000
        assert window.get_user_input_events().get_events() == []
        now[0] = 100_000_000
        events = window.get_user_input_events()
        assert isinstance(events.get_events()[0], GamepadUserInputEvent)
        driver.apply(events)
        command = driver.command()
        assert 0 < command.steer < 0.25 and 0 < command.throttle < 0.6
        assert command.brake == 0 and not command.handbrake
        assert json.loads(peer.recv(4096))["status"] == "dispatched"
        control.consumed(events, step=12)
        window.get_user_input_events()
        ack = json.loads(peer.recv(4096))
        assert ack["status"] == "consumed" and ack["model_step"] == 12
        assert ack["consumed_us"] == 100_000
    window.close()


def test_mode_rejects_bad_arguments_and_existing_captures(tmp_path: Path) -> None:
    mode = RobotaxiDebugMode()
    parser = argparse.ArgumentParser()
    mode.add_arguments(parser)
    with pytest.raises(ValueError, match="required"):
        mode.check_arguments(parser.parse_args([]))
    with pytest.raises(ValueError, match="fresh directory"):
        mode.check_arguments(parser.parse_args(["--debug-output-dir", str(tmp_path)]))
    with pytest.raises(ValueError, match="must be positive"):
        mode.check_arguments(
            parser.parse_args(
                [
                    "--debug-output-dir",
                    str(tmp_path / "new"),
                    "--debug-snapshot-every",
                    "0",
                ]
            )
        )


def test_debug_map_compiles_with_a_race_spawn_and_ground_mesh(tmp_path: Path) -> None:
    source = Path(__file__).parent / "maps/debug_flat.robotaxi.yaml"
    compiled = compile_game_map(source, cache_root=tmp_path)
    assert compiled.archive_path.is_file()
    scene = load_scene_bundle(
        compiled.archive_path,
        "camera_front_wide_120fov",
        RasterConfig(width=64, height=32),
    )
    assert scene.game_map is not None
    assert scene.ground_mesh_vertices is not None
    assert scene.ground_mesh_faces is not None
    assert scene.game_map.race_courses[0].course_id == "debug-lap"
    assert scene.ground_mesh_vertices.shape[1] == 3
    assert scene.ground_mesh_faces.shape[1] == 3
    assert scene.initial_rig_to_world[2, 3] == pytest.approx(0)


@pytest.mark.parametrize("value", ["nan", "inf", "-1"])
def test_generation_delay_rejects_invalid_values(tmp_path, value) -> None:
    mode = RobotaxiDebugMode()
    parser = argparse.ArgumentParser()
    mode.add_arguments(parser)
    args = parser.parse_args(
        [
            "--debug-output-dir",
            str(tmp_path / "new"),
            "--no-debug-capture-images",
            "--debug-generation-delay-ms",
            value,
        ]
    )
    with pytest.raises(ValueError, match="finite and nonnegative"):
        mode.check_arguments(args)


@pytest.mark.parametrize("chunks,events", [(1, 256), (128, 4096)])
def test_capture_buffer_budgets_reach_recorder_and_manifest(
    tmp_path, monkeypatch, chunks, events
):
    from omnidreams_game_engine.debug_capture import CaptureRecorder

    received = []

    def record_budget(*args, **kwargs):
        received.append((kwargs["image_slots"], kwargs["queue_size"]))
        return CaptureRecorder(*args, **kwargs)

    monkeypatch.setattr(
        "omnidreams_game_engine.debug_window.CaptureRecorder", record_budget
    )
    mode = RobotaxiDebugMode()
    parser = argparse.ArgumentParser()
    mode.add_arguments(parser)
    args = parser.parse_args(
        [
            "--debug-output-dir",
            str(tmp_path / "capture"),
            "--debug-image-buffer-chunks",
            str(chunks),
            "--debug-record-buffer-events",
            str(events),
        ]
    )
    window = mode.create(args)
    window.open(
        SessionDesc(video_width=6, video_height=4, metadata={DEBUG_METADATA_KEY: {}})
    )
    window.close()
    assert received == [(chunks, events)]
    manifest = json.loads((tmp_path / "capture" / "manifest.json").read_text())
    assert manifest["image_buffer_chunks"] == chunks
    assert manifest["record_buffer_events"] == events


@pytest.mark.parametrize("chunks", [0, -1])
@pytest.mark.parametrize("option", ["image-buffer-chunks", "record-buffer-events"])
def test_capture_buffer_budget_rejects_nonpositive_values(tmp_path, chunks, option):
    mode = RobotaxiDebugMode()
    parser = argparse.ArgumentParser()
    mode.add_arguments(parser)
    args = parser.parse_args(
        [
            "--debug-output-dir",
            str(tmp_path / "capture"),
            f"--debug-{option}",
            str(chunks),
        ]
    )
    with pytest.raises(ValueError, match=f"--debug-{option} must be positive"):
        mode.check_arguments(args)
