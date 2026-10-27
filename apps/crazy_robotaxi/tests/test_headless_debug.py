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
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
from crazy_robotaxi.debug import (
    DEBUG_METADATA_KEY,
    DebugFrameResult,
    RobotaxiDebugMode,
    RobotaxiDebugWindow,
)
from crazy_robotaxi.debug_input import DEBUG_CONTROL_KEY, HeadlessInput, control_event
from crazy_robotaxi.driving_script import load_driving_script
from crazy_robotaxi.physics import TaxiPhysicsWorld
from ludus_renderer import BodyState, RigidBodyModel
from omnidreams_game_engine.config import RasterConfig
from omnidreams_game_engine.game_map import compile_game_map
from omnidreams_game_engine.scene import load_scene_bundle
from omnidreams_game_engine.simulation.ground_snap import GroundSnapper
from omnidreams_game_engine.types import PhysicsDebugFrame, VehicleState
from PIL import Image

from flashdreams.runtime_v2.session_desc import (
    PresentationMode,
    SessionDesc,
)
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    CloseUserInputEvent,
    KeyboardInputState,
    KeyboardUserInputEvent,
)
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu


def test_live_headless_input_queues_rejects_late_and_acknowledges_consumption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("crazy_robotaxi.debug.Mp4Encoder", RecordingEncoder)
    now = [10_000_000_000]
    control = HeadlessInput(lambda: now[0])
    window = RobotaxiDebugWindow(
        tmp_path / "live", snapshot_every=1, cpu_sample_ms=1, clock_ns=lambda: now[0]
    )
    window.open(
        SessionDesc(
            video_width=6,
            video_height=4,
            metadata={
                DEBUG_METADATA_KEY: {},
                DEBUG_CONTROL_KEY: control,
            },
        )
    )
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as controller:
        controller.bind(str(tmp_path / "agent.sock"))
        controller.settimeout(1)

        def send(command):
            controller.sendto(
                json.dumps(command).encode(), str(window.directory / "control.sock")
            )

        def reply():
            return json.loads(controller.recv(4096))

        send({"id": "clock", "type": "status"})
        assert window.get_user_input_events().get_events() == []
        assert reply()["status"] == "loading"
        # Accept out of order while loading; dispatch strictly by deadline.
        send(
            {
                "id": "release",
                "timestamp_us": 200_000,
                "type": "keyboard",
                "key": "w",
                "pressed": False,
            }
        )
        send(
            {
                "id": "press",
                "timestamp_us": 100_000,
                "type": "keyboard",
                "key": "w",
                "pressed": True,
            }
        )
        assert window.get_user_input_events().get_events() == []
        assert reply()["status"] == reply()["status"] == "accepted"
        control.start()
        now[0] += 50_000_000
        assert window.get_user_input_events().get_events() == []
        now[0] += 75_000_000
        events = window.get_user_input_events()
        (pressed,) = events.get_events()
        assert isinstance(pressed, KeyboardUserInputEvent)
        assert (
            pressed.state is KeyboardInputState.PRESSED
            and int(pressed.timestamp) == 100_000
        )
        assert reply()["status"] == "dispatched"
        control.consumed(events, step=7)
        window.get_user_input_events()
        consumed = reply()
        assert consumed["status"] == "consumed" and consumed["consumed_us"] == 125_000
        send({"id": "late", "timestamp_us": 124_999, "type": "close"})
        assert window.get_user_input_events().get_events() == []
        assert reply()["reason"] == "late"
        send({"id": "press", "timestamp_us": 300_000, "type": "close"})
        window.get_user_input_events()
        assert reply()["reason"] == "duplicate_id"
        # UI stall must not invalidate a command that arrived on time.
        now[0] += 400_000_000
        events = window.get_user_input_events()
        (released,) = events.get_events()
        assert isinstance(released, KeyboardUserInputEvent)
        assert (
            released.state is KeyboardInputState.RELEASED
            and int(released.timestamp) == 200_000
        )
        assert reply()["status"] == "dispatched"
        control.consumed(events, step=8)
        window.write(
            StepResult(
                step_index=7,
                output=torch.ones(1, 3, 4, 6),
                frame_count=1,
                output_layout=VideoTensorLayout.tchw,
            )
        )
        window.close()
        assert reply()["status"] == "consumed"
    inputs = [
        json.loads(line)
        for line in (window.directory / "input_events.jsonl").read_text().splitlines()
    ]
    assert [event["id"] for event in inputs if event["status"] == "consumed"] == [
        "press",
        "release",
    ]
    assert not (window.directory / "control.sock").exists()
    assert (window.directory / "latest.png").is_file()
    samples = [
        json.loads(line)
        for line in (window.directory / "cpu_samples.jsonl").read_text().splitlines()
    ]
    assert samples and samples[0]["native_thread_id"] == control.native_thread_id
    assert any(
        frame["function"].startswith("test_live_headless")
        for frame in samples[0]["stack"]
    )
    record = json.loads((window.directory / "telemetry.jsonl").read_text())
    assert record["ui_step"] == 7 and record["elapsed_us"] == 525_000
    manifest = json.loads((window.directory / "manifest.json").read_text())
    assert manifest["views"] == ["presentation"] and manifest["frames_written"] == 1


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"type": "keyboard", "key": "w", "pressed": 1},
        {"type": "keyboard", "key": "", "pressed": True},
        {"type": "close", "extra": 1},
    ],
)
def test_live_headless_input_rejects_malformed_packets(payload: object) -> None:
    with pytest.raises(ValueError):
        control_event(payload)


def test_input_deadline_boundary_order_and_shutdown() -> None:
    now = [0]
    control = HeadlessInput(lambda: now[0])
    control.start()
    for identifier in ("first", "second"):
        control.accept(
            {"id": identifier, "timestamp_us": 0, "type": "focus", "focused": False}
        )
    control.accept({"id": "later", "timestamp_us": 10, "type": "close"})
    assert len(control.due()) == 2
    control.close()
    messages = []
    while not control.messages.empty():
        messages.append(control.messages.get_nowait()[1])
    assert [m["id"] for m in messages if m["status"] == "dispatched"] == [
        "first",
        "second",
    ]
    assert {m["id"] for m in messages if m["status"] == "cancelled"} == {
        "first",
        "second",
        "later",
    }
    # Invalid identifiers must not break JSON acknowledgments (including NaN).
    control.accept({"id": float("nan"), "type": "status"})
    rejected = control.messages.get_nowait()[1]
    assert rejected["status"] == "rejected" and rejected["id"] is None
    json.dumps(rejected, allow_nan=False)


def test_observation_preserves_native_pose_before_game_policy() -> None:
    world = object.__new__(TaxiPhysicsWorld)
    world._ego_model = RigidBodyModel(mass_kg=1500, half_extents_m=(2, 1, 0.5))
    world._world = SimpleNamespace(
        collider_state_arrays=lambda: (
            (),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 4), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
        )
    )
    world._debug_barrier_segments = np.empty((0, 2, 2), dtype=np.float32)
    world._debug_barrier_thicknesses = np.empty(0, dtype=np.float32)
    world._debug_barrier_heights = np.empty(0, dtype=np.float32)
    world._debug_barrier_ids = ()
    world.last_step_actor_collision = False
    world.last_step_static_barrier_collision = True
    world._last_contact_resolved_state = VehicleState(10, 20, 0, 0, 8, 0)
    roll = 0.3
    body = BodyState(
        position_m=np.asarray([10, 20, 1.5], dtype=np.float32),
        orientation_xyzw=np.asarray(
            [np.sin(roll / 2), 0, 0, np.cos(roll / 2)], dtype=np.float32
        ),
        linear_velocity_mps=np.asarray([8, 0, 2], dtype=np.float32),
        angular_velocity_radps=np.asarray([1, 0, 0], dtype=np.float32),
    )
    world._last_debug_ego = body
    displayed = VehicleState(10, 20, 0, 0, 8, 0)
    debug = world.debug_frame(displayed)
    assert debug.static_barrier_collision
    assert debug.vehicle_state.roll_rad == pytest.approx(roll)
    assert debug.vehicle_state.z_m == pytest.approx(1.5 - np.cos(roll) * 0.5)
    np.testing.assert_array_equal(debug.ego_orientation_xyzw, body.orientation_xyzw)
    np.testing.assert_array_equal(debug.ego_linear_velocity_mps, [8, 0, 2])
    assert displayed.z_m == displayed.roll_rad == 0
    assert world._last_contact_resolved_state.z_m == 0
    body.position_m[2] = 99
    assert debug.ego_position_m[2] == 1.5


def test_ground_measurement_is_read_only_and_selects_nearest_surface() -> None:
    vertices = np.asarray([[0, 0, 0], [2, 0, 2], [0, 2, 0]], dtype=np.float32)
    stacked = np.concatenate([vertices, vertices + [0, 0, 10]])
    ground = GroundSnapper(
        stacked, np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.float32)
    )
    assert ground.ground_height_at(0.5, 0.5, 0) == pytest.approx(0.5)
    assert ground.ground_height_at(0.5, 0.5, 10) == pytest.approx(10.5)
    assert ground.ground_height_at(-1, -1, 0) is None
    assert ground._anchor_offset_m is None


def _debug_record(frame: int, *, contact: bool = False) -> dict:
    vehicle = VehicleState(frame, 0, 0, 0, 10, 0)
    colliders = PhysicsDebugFrame(
        ego_position_m=np.asarray([frame, 0, 0.8], dtype=np.float32),
        ego_orientation_xyzw=np.asarray([0, 0, 0, 1], dtype=np.float32),
        ego_dimensions_lwh=np.asarray([4.8, 2, 1.6], dtype=np.float32),
        actor_positions_m=np.empty((0, 3), dtype=np.float32),
        actor_orientations_xyzw=np.empty((0, 4), dtype=np.float32),
        actor_dimensions_lwh=np.empty((0, 3), dtype=np.float32),
        barrier_segments_xy_m=np.asarray(
            [[[frame - 10, -1.2], [frame + 10, -1.2]]], dtype=np.float32
        ),
        barrier_thicknesses_m=np.asarray([0.3], dtype=np.float32),
        barrier_heights_m=np.asarray([3], dtype=np.float32),
        ego_chassis_offset_m=np.asarray([0, 0, -0.04], dtype=np.float32),
        ego_chassis_dimensions_lwh=np.asarray([4.096, 1.44, 1.36], dtype=np.float32),
    )
    return {
        "frame": frame,
        "rig_pose_world": np.eye(4, dtype=np.float32),
        "physics_vehicle": asdict(vehicle),
        "actor_collision": False,
        "static_barrier_collision": contact,
        "physics_colliders": asdict(colliders),
    }


def test_script_uses_the_live_command_schema_and_rejects_duplicates(
    tmp_path: Path,
) -> None:
    source = tmp_path / "drive.yaml"
    source.write_text("events: [{id: end, timestamp_us: 1000000, type: close}]")
    script = load_driving_script(source)
    assert script.events == ({"id": "end", "timestamp_us": 1000000, "type": "close"},)
    source.write_text(json.dumps({"events": list(script.events) * 2}))
    with pytest.raises(ValueError, match="duplicate"):
        load_driving_script(source)


@pytest.mark.parametrize(
    "text",
    [
        "steps: [{frames: 1}]",
        "events: []",
        "events: null",
        "events: [{id: a, timestamp_us: -1, type: close}]",
        "events: [{id: a, timestamp_us: true, type: close}]",
        "events: [{id: a, timestamp_us: 1.5, type: close}]",
        "events: [{id: a, timestamp_us: 0, type: keyboard, key: w, pressed: 1}]",
        "events: [{id: a, timestamp_us: 0, type: close, extra: 1}]",
        "events: [{id: a, timestamp_us: 0, type: close}]\nextra: yes",
    ],
)
def test_invalid_scripts_fail_before_game_setup(tmp_path: Path, text: str) -> None:
    source = tmp_path / "bad.yaml"
    source.write_text(text)
    with pytest.raises(ValueError):
        load_driving_script(source)


class RecordingEncoder:
    """In-memory encoder used to inspect pixels without invoking ffmpeg."""

    def __init__(self, *args, **kwargs) -> None:
        self.frames = []
        self.closed = False

    def write(self, frames) -> None:
        self.frames.extend(frames.copy())

    def close(self) -> None:
        self.closed = True


def test_sink_preserves_both_views_and_rejects_repeated_simulation_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("crazy_robotaxi.debug.Mp4Encoder", RecordingEncoder)
    desc = SessionDesc(
        video_width=6,
        video_height=4,
        presentation_mode=PresentationMode.ON_DEMAND,
        metadata={DEBUG_METADATA_KEY: {"paired_views": True}},
    )
    window = RobotaxiDebugWindow(tmp_path / "capture", snapshot_every=30)
    window.open(desc)
    output = torch.cat((torch.ones(1, 3, 4, 6), -torch.ones(1, 3, 4, 6)), dim=-1)
    result = DebugFrameResult(
        step_index=0,
        output=output,
        frame_count=1,
        output_layout=VideoTensorLayout.tchw,
        record=_debug_record(0),
    )
    window.write(result)
    assert np.all(
        cast(RecordingEncoder, window._encoders["generated"]).frames[0] == 255
    )
    assert np.all(cast(RecordingEncoder, window._encoders["hdmap"]).frames[0] == 0)
    with Image.open(window.directory / "frames/generated/000000.png") as image:
        assert image.size == (6, 4)
        assert np.all(np.asarray(image, dtype=np.float32) == 255)
    with pytest.raises(ValueError, match="once, in simulation order"):
        window.write(result)
    window.close()
    assert all(
        cast(RecordingEncoder, encoder).closed for encoder in window._encoders.values()
    )
    manifest = json.loads((window.directory / "manifest.json").read_text())
    assert manifest["frames_written"] == 1 and manifest["finalized"]
    records = [
        json.loads(line)
        for line in (window.directory / "telemetry.jsonl").read_text().splitlines()
    ]
    assert len(records) == 1 and records[0]["rig_pose_world"] == np.eye(4).tolist()
    with pytest.raises(FileExistsError):
        RobotaxiDebugWindow(window.directory).open(desc)


@pytest.mark.parametrize("written,requested", [(0, 3), (1, 3), (2, 3), (2, 2)])
def test_close_saves_last_written_views_without_rewriting_snapshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    written: int,
    requested: int,
) -> None:
    monkeypatch.setattr("crazy_robotaxi.debug.Mp4Encoder", RecordingEncoder)
    saved = []
    original_save = Image.Image.save

    def save(image, path, *args, **kwargs):
        saved.append(Path(path))
        return original_save(image, path, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "save", save)
    window = RobotaxiDebugWindow(tmp_path / "capture", snapshot_every=30)
    window.open(
        SessionDesc(
            video_width=6,
            video_height=4,
            presentation_mode=PresentationMode.ON_DEMAND,
            metadata={DEBUG_METADATA_KEY: {"paired_views": True}},
        )
    )
    for frame in range(written):
        window.write(
            DebugFrameResult(
                step_index=frame,
                output=torch.full((1, 3, 4, 12), (frame + 1) / 5),
                frame_count=1,
                output_layout=VideoTensorLayout.tchw,
                record=_debug_record(frame),
            )
        )
    before_close = set(saved)
    if written == 2 and requested == 3:
        assert len(before_close) == 3  # Only frame zero was sampled.
    window.close()
    window.close()
    assert len(saved) == len(set(saved))
    expected = before_close.copy()
    for view, encoder in window._encoders.items():
        if written:
            path = window.directory / "frames" / view / f"{written - 1:06d}.png"
            expected.add(path)
            with Image.open(path) as image:
                np.testing.assert_array_equal(
                    np.asarray(image), cast(RecordingEncoder, encoder).frames[-1]
                )
    assert set(saved) == expected
    manifest = json.loads((window.directory / "manifest.json").read_text())
    assert manifest["finalized"]
    assert manifest["frames_written"] == written


def test_physics_video_shows_bounds_and_saves_contact_between_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("crazy_robotaxi.debug.Mp4Encoder", RecordingEncoder)
    desc = SessionDesc(
        video_width=320,
        video_height=320,
        presentation_mode=PresentationMode.ON_DEMAND,
        metadata={DEBUG_METADATA_KEY: {"paired_views": True}},
    )
    window = RobotaxiDebugWindow(tmp_path / "capture", snapshot_every=30)
    window.open(desc)
    for frame in range(3):
        window.write(
            DebugFrameResult(
                step_index=frame,
                output=torch.zeros(1, 3, 320, 640),
                frame_count=1,
                output_layout=VideoTensorLayout.tchw,
                record=_debug_record(frame, contact=frame == 1),
            )
        )
    window.close()
    pixels = cast(RecordingEncoder, window._encoders["physics"]).frames[1]
    np.testing.assert_array_equal(pixels[172, 80], [0, 220, 235])
    np.testing.assert_array_equal(pixels[170, 160], [245, 245, 245])
    np.testing.assert_array_equal(pixels[167, 160], [70, 150, 255])
    assert all(
        len(cast(RecordingEncoder, encoder).frames) == 3
        for encoder in window._encoders.values()
    )
    manifest = json.loads((window.directory / "manifest.json").read_text())
    assert manifest["finalized"] and manifest["contact_frames"] == [1]
    for view in manifest["views"]:
        assert (window.directory / "frames" / view / "000001.png").is_file()


def test_sink_requires_capture_metadata_and_tchw(tmp_path: Path) -> None:
    window = RobotaxiDebugWindow(tmp_path / "out")
    desc = SessionDesc(
        metadata={DEBUG_METADATA_KEY: {}}, output_layout=VideoTensorLayout.btchw
    )
    with pytest.raises(ValueError, match="tchw"):
        window.open(desc)
    with pytest.raises(ValueError, match="requires Crazy Robotaxi"):
        window.open(replace(desc, metadata={}))


def test_debug_window_delivers_runtime_events_once_in_timestamp_order(
    tmp_path: Path,
) -> None:
    window = RobotaxiDebugWindow(tmp_path / "capture")
    earlier = CloseUserInputEvent(timestamp=np.uint64(1))
    later = CloseUserInputEvent(timestamp=np.uint64(2))
    window.put_user_input_event(later)
    window.put_user_input_event(earlier)
    assert window.get_user_input_events().get_events() == [earlier, later]
    assert window.get_user_input_events().get_events() == []


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
