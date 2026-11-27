# SPDX-FileCopyrightText: Copyright (c) 2026 FlashDreams contributors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the ROS admission gate on the real V2 model loop with a CPU backend."""

from types import SimpleNamespace

import numpy as np
import pytest

from flashdreams.runtime_v2.event_buffer import EventBuffer
from flashdreams.runtime_v2.user_input_event import GameWheelUserInputEvent
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from interactive_drive import (
    DriveInputState,
    InteractiveDriveApplication,
    InteractiveDriveApplicationDefaults,
)
from interactive_drive import core
from interactive_drive.config import AppConfig, VehicleConfig
from interactive_drive.ros_drive import driver_command
from interactive_drive.simulation.ego_vehicle_kinematics import integrate_vehicle
from interactive_drive.types import (
    CameraCalibration,
    FrameChunk,
    PresentedFrame,
    VehicleState,
)

pytestmark = pytest.mark.ci_cpu


def test_manual_ros_control_matches_web_wheel_motion_in_every_gear(tmp_path):
    # A steering stop tighter than the nominal full manual wheel also applies.
    config = AppConfig(scene_path=tmp_path, vehicle=VehicleConfig(max_steer_rad=0.25))
    web_input = DriveInputState()
    vehicle = VehicleState(x_m=0, y_m=0, z_m=0, yaw_rad=0, speed_mps=0, steer_rad=0)
    cases = [
        # gear, throttle, brake, resulting signed speed after 0.1 sim seconds
        ("D", 1.0, 0.005, 0.2),  # Below the pedal deadzone, RT still works.
        ("D", 1.0, 1.0, 0.0),  # A / handbrake while still holding throttle.
        ("R", 1.0, 0.0, -0.2),
        ("N", 1.0, 0.0, -0.15),
        ("P", 1.0, 0.0, 0.0),
    ]
    for gear, throttle, brake, speed in cases:
        pedals = dict(throttle=throttle, brake=brake, reverse=gear == "R")
        web_input.apply(
            UserInputEvents(
                [
                    GameWheelUserInputEvent(
                        timestamp=np.uint64(0),
                        steering=-1.0,
                        gear=gear,
                        **pedals,
                    )
                ]
            )
        )
        web_state = integrate_vehicle(vehicle, web_input.command(), 0.1, config.vehicle)
        cmd = driver_command(
            dict(manual_gear=gear, steering_angle_rad=0.4, **pedals), config
        )
        vehicle = integrate_vehicle(vehicle, cmd, 0.1, config.vehicle)
        assert cmd.manual_control and cmd.gear == gear
        assert vehicle == web_state
        assert vehicle.steer_rad == pytest.approx(0.25)
        assert vehicle.speed_mps == pytest.approx(speed)


@pytest.mark.parametrize("late_images", [False, True])
def test_ros_blocks_on_controls_and_publishes_the_matching_states(
    tmp_path, monkeypatch, late_images
):
    reset = {"type": "reset", "episode_id": "test"}
    step = {
        "type": "step",
        "episode_id": "test",
        "expected_step": 0,
        "steering_angle_rad": 0.2,
        "throttle": 1.0,
        "brake": 0.0,
        "reverse": False,
    }
    requests = iter(
        [
            reset,
            dict(step, steering_angle_rad=20.0),
            dict(step, front_condition=True),
            dict(step, expected_step=1, throttle=1.0, brake=0.0, manual_gear="P"),
        ]
    )
    messages = []

    def send(header, images=()):
        messages.append((header, [bytes(image) for image in images]))

    endpoint = SimpleNamespace(
        receive=lambda stopping: next(requests, None),
        send=send,
        close=lambda: None,
    )
    monkeypatch.setattr(core, "DriveEndpoint", lambda path, *, front_condition: endpoint)
    pixel = np.zeros((2, 3, 3), dtype=np.uint8)
    camera = CameraCalibration(
        "front", "front", 3, 2, 1.5, 1.0, np.ones(6), False, np.ones(3), np.eye(4)
    )
    scene = SimpleNamespace(
        cameras=(camera,),
        metadata={
            "scene_geometry": {"root": str(tmp_path), "manifest_sha256": "a" * 64},
            "rig_ground_clearance_m": 0.85,
        },
        initial_timestamp_us=1000,
        initial_rig_to_world=np.eye(4),
        initial_rgbs=(pixel,),
        initial_speed_mps=0.0,
        initial_yaw_rad=0.0,
        ground_mesh_faces=None,
        ground_mesh_vertices=None,
    )
    rendered = []

    def render(trajectory):
        rendered.append(trajectory)
        frames = tuple(
            PresentedFrame(
                int(t) + int(late_images),
                (pixel + 99,),
                None,
                view_model_rgb_host_uint8=(pixel + i + 1,),
                scene_time_us=min(int(t), 201000),
            )
            for i, t in enumerate(trajectory.timestamps_us)
        )
        return FrameChunk(frames, trajectory.boundary_state_after_chunk, "stub")

    backend = SimpleNamespace(
        initial_chunk_frames=3,
        chunk_frames=2,
        render_first_chunk=render,
        render_next_chunk=render,
        warmup_model=lambda: None,
        load_scene=lambda _: None,
        set_postprocess_enabled=lambda _: None,
        finalize=lambda: {},
        close=lambda: None,
    )
    app = InteractiveDriveApplication(
        defaults=InteractiveDriveApplicationDefaults(
            backend_factory=lambda _: backend, physics=False
        ),
        scene_loader=lambda *a, **kw: scene,
    )
    app.init(
        [
            "--scene",
            str(tmp_path),
            "--ros-socket",
            str(tmp_path / "drive.sock"),
            "--fps",
            "10",
            "--camera",
            "front",
            "--total-blocks",
            "0",
        ]
    )
    session = app.create_session(app.session_desc())
    session.init()
    loop = session.model_loop
    completed = []
    loop._run_model_loop(
        event_buffer=EventBuffer(),
        reader_id=0,
        publish=lambda *unused: None,
        steps_run_out=completed,
    )
    headers = [header for header, _ in messages]
    assert headers[0]["step"] == -1 and headers[0]["timestamp_us"] == 1000
    assert headers[0]["calibration"][0]["sensor_to_rig_flu"] == np.eye(4).tolist()
    assert headers[0]["scene_geometry"] == scene.metadata["scene_geometry"]
    assert headers[0]["ego_dimensions_lwh"] == (4.8, 2.0, 1.6)
    assert headers[0]["rig_ground_clearance_m"] == 0.85
    assert headers[1]["next_step"] == 0
    assert headers[2]["code"] == "INVALID_REQUEST"
    if late_images:
        assert headers[3]["type"] == "error" and headers[3]["fatal"]
        assert completed == [0] and len(rendered) == 1
        assert "timestamp order" in str(session._failure_queue.get_nowait())
        return
    assert session._failure_queue.empty()
    assert completed == [2] and len(rendered) == 2 and loop.state.blocks_generated == 2
    frames = [(h, payloads[0]) for h, payloads in messages if h["type"] == "frame"]
    assert [h["timestamp_us"] for h, _ in frames] == [
        1000,
        101000,
        201000,
        301000,
        401000,
    ]
    assert [pixels[0] for _, pixels in frames] == [0, 2, 3, 1, 2]
    for h, payloads in messages:
        if h["type"] != "frame":
            continue
        if h["step"] == 0:
            assert h["images"][-1] == {
                "name": "condition/front", "role": "condition",
                "width": 3, "height": 2, "nbytes": 18,
            }
            assert payloads[1] == bytes([99]) * 18
        else:
            assert len(h["images"]) == len(payloads) == 1
    assert [h["scene_time_us"] for h, _ in frames] == [
        1000,
        101000,
        201000,
        201000,
        201000,
    ]
    assert all(
        "scene_geometry" not in h and "ego_dimensions_lwh" not in h
        for h, _ in frames[1:]
    )
    # The first sample of the P block still reflects the preceding D interval.
    assert [
        (h["vehicle"]["gear"], h["vehicle"]["throttle"], h["vehicle"]["brake"])
        for h, _ in frames
    ] == [
        ("D", 0.0, 0.0),
        ("D", 1.0, 0.0),
        ("D", 1.0, 0.0),
        ("D", 1.0, 0.0),
        ("P", 0.0, 1.0),
    ]
    assert [h["vehicle"]["speed_mps"] for h, _ in frames[1:]] == [
        state.speed_mps
        for state in (*rendered[0].vehicle_states[1:], *rendered[1].vehicle_states)
    ]
    done = [h for h in headers if h["type"] == "done"]
    assert [h["next_apply_us"] for h in done] == [1000, 301000, 501000]
    assert [h["frame_count"] for h in done] == [1, 2, 2]
    assert done[1]["control"]["steering_angle_rad"] == pytest.approx(0.2)
    assert "manual_gear" not in done[1]
    assert done[2]["manual_gear"] == "P"
    assert done[2]["control"]["brake"] == 1.0 and done[2]["control"]["throttle"] == 0.0
