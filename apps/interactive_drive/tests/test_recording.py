# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check synchronized recording, finalization, and simulation playback."""

import json
import math
from dataclasses import replace
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest
import torch
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.user_input_event import RecordingUserInputEvent
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from interactive_drive.app import InteractiveDriveUIState
from interactive_drive.comparison import (
    ComparisonRenderer,
    asset,
    future_edges,
    load_source,
    nearest,
    project_raw_camera,
)
from interactive_drive.config import AppConfig, ChunkConfig, VehicleConfig
from interactive_drive.core import (
    InteractiveDriveConfig,
    InteractiveDriveModelLoop,
    InteractiveDriveModelState,
)
from interactive_drive.recording import DriveRecording
from interactive_drive.simulation.ego_vehicle_kinematics import sample_chunk_trajectory
from interactive_drive.types import (
    DriverCommand,
    DynamicActorTrajectory,
    FrameChunk,
    PresentedFrame,
    VehicleState,
)
from interactive_drive.vehicle_mesh import chase_projection, render_vehicle
from PIL import Image

pytestmark = pytest.mark.ci_cpu


def test_recording_keeps_every_camera_and_the_state_at_each_frame(tmp_path):
    config = AppConfig(
        scene_path=tmp_path / "scene",
        camera_names=("front", "rear"),
        chunk=ChunkConfig(fps=10),
    )
    scene = SimpleNamespace(scene_id="stub", scene_path=config.scene_path, cameras=())
    config.scene_path.mkdir()
    (config.scene_path / "map.json").write_text(
        '{"lines": [{"vertices": [[[0, 0, 0], [10, 0, 0]]]}]}'
    )
    command = DriverCommand(throttle=1, steer=0.2, steer_is_direct=True)
    trajectory = sample_chunk_trajectory(
        VehicleState(0, 0, 0, 0, 0, 0),
        1_000_000,
        command,
        3,
        config.chunk,
        VehicleConfig(),
        None,
    )
    trajectory = replace(
        trajectory,
        dynamic_actors=(
            DynamicActorTrajectory(
                "car-1",
                "vehicle",
                trajectory.timestamps_us,
                np.array([[10 + i, 2, 0] for i in range(3)]),
                np.array([[0, 0, 0, 1]] * 3),
                np.array([4, 2, 1.5]),
            ),
        ),
    )
    frames = tuple(
        PresentedFrame(
            timestamp_us=int(t),
            view_rgb_host_uint8=tuple(
                np.full((8, 12, 3), 20 + i * 40 + j, np.uint8) for i in range(2)
            ),
            depth_host_f32=None,
            bev_host_uint8=np.full((6, 6, 3), j, np.uint8),
        )
        for j, t in enumerate(trajectory.timestamps_us)
    )
    recording = DriveRecording(tmp_path / "exports", config, scene)
    recording.append(
        FrameChunk(frames, trajectory.boundary_state_after_chunk, "stub"),
        trajectory,
        [command],
        [0.0],
    )
    assert not recording.path.with_suffix(".zip").exists()
    status = recording.close()
    assert status["state"] == "ready" and status["frames"] == 3
    ui_state = SimpleNamespace(export_dir=tmp_path / "exports", recording_archive=None)
    InteractiveDriveUIState.set_recording_status(ui_state, status)
    InteractiveDriveUIState.set_recording_status(
        ui_state, {"state": "recording", "id": "next-session"}
    )
    assert InteractiveDriveUIState.recording_files(ui_state) == {
        recording.id: recording.path.with_suffix(".zip")
    }
    with ZipFile(recording.path.with_suffix(".zip")) as archive:
        records = [
            json.loads(line) for line in archive.read("frames.jsonl").splitlines()
        ]
        assert [r["timestamp_us"] for r in records] == ["1000000", "1100000", "1200000"]
        speeds = [r["vehicle"]["speed_mps"] for r in records]
        assert (
            speeds[0]
            == 0
            < speeds[1]
            < speeds[2]
            < trajectory.boundary_state_after_chunk.speed_mps
        )
        assert all(set(r["images"]) == {"front", "rear"} for r in records)
        for r in records:
            with archive.open(r["images"]["rear"]) as stream:
                assert Image.open(stream).size == (12, 8)
        assert json.loads(archive.read("manifest.json"))["frame_count"] == 3
        assert records[1]["actors"][0]["translation_world"] == [11, 2, 0]
        assert json.loads(archive.read("scene/map.json"))["lines"][0]["vertices"][0][
            1
        ] == [10, 0, 0]
        assert "scene_geometry.json" in archive.namelist()
    source = load_source(recording.path, tmp_path)
    assert source["times"].tolist() == [1_000_000, 1_100_000, 1_200_000]
    assert source["frames"][0]["speed"] == 0
    assert source["frames"][1]["actors"][0]["translation_world"] == [11, 2, 0]
    with pytest.raises(ValueError, match="escapes"):
        asset(recording.path, "../outside")
    assert nearest(np.array([0, 10, 20]), 15) == 1


def test_model_loop_starts_and_stops_recording_without_changing_output(
    tmp_path, monkeypatch
):
    config = AppConfig(
        scene_path=tmp_path / "scene",
        export_dir=tmp_path / "exports",
        camera_names=("front",),
        chunk=ChunkConfig(fps=10),
        world_model_device="cpu",
    )
    scene = SimpleNamespace(scene_id="stub", scene_path=config.scene_path, cameras=())

    def render(trajectory):
        return FrameChunk(
            tuple(
                PresentedFrame(
                    timestamp_us=int(t),
                    view_rgb_host_uint8=(np.zeros((4, 4, 3), np.uint8),),
                    depth_host_f32=None,
                )
                for t in trajectory.timestamps_us
            ),
            trajectory.boundary_state_after_chunk,
            "stub",
        )

    backend = SimpleNamespace(
        initial_chunk_frames=3,
        chunk_frames=3,
        render_first_chunk=render,
        render_next_chunk=render,
        finalize=dict,
        close=lambda: None,
    )
    loop = InteractiveDriveModelLoop()
    loop.state = InteractiveDriveModelState(
        backend_factory=lambda _: backend,
        config=InteractiveDriveConfig(config, 0, "rgb"),
        desc=SessionDesc(),
        scene_loader=lambda *_: scene,
        scene=scene,
        vehicle=VehicleState(0, 0, 0, 0, 0, 0),
        backend=backend,
    )
    monkeypatch.setattr(
        loop,
        "_frame_chunk_tensors",
        lambda chunk: [torch.zeros((len(chunk.frames), 3, 4, 4))],
    )
    loop.step(
        0,
        UserInputEvents(
            [RecordingUserInputEvent(timestamp=np.uint64(0), action="start")]
        ),
    )
    recording = loop.state.recording
    assert recording is not None and recording.frames == 3
    loop.step(
        1,
        UserInputEvents(
            [RecordingUserInputEvent(timestamp=np.uint64(1), action="stop")]
        ),
    )
    assert loop.state.recording is None
    assert (
        json.loads((recording.path / "manifest.json").read_text())["frame_count"] == 3
    )
    loop.close()


def test_failed_recording_retains_partial_frames(tmp_path):
    config = AppConfig(scene_path=tmp_path, camera_names=("front",))
    recording = DriveRecording(
        tmp_path / "exports",
        config,
        SimpleNamespace(scene_id="stub", scene_path=tmp_path, cameras=()),
    )
    status = recording.close(error="disk write failed")
    assert status["state"] == "error"
    assert (recording.path / "manifest.json").exists()
    assert not recording.path.with_suffix(".zip").exists()
    with pytest.raises(ValueError, match="completed"):
        load_source(recording.path, tmp_path)


def test_delayed_frames_keep_their_original_vehicle_samples(tmp_path):
    config = AppConfig(
        scene_path=tmp_path, camera_names=("front",), chunk=ChunkConfig(fps=10)
    )
    recording = DriveRecording(
        tmp_path / "exports",
        config,
        SimpleNamespace(scene_id="stub", scene_path=tmp_path, cameras=()),
    )
    command = DriverCommand(throttle=1)
    first = sample_chunk_trajectory(
        VehicleState(0, 0, 0, 0, 0, 0),
        1_000_000,
        command,
        2,
        config.chunk,
        config.vehicle,
        None,
    )
    recording.append(
        FrameChunk((), first.boundary_state_after_chunk, "stub"),
        first,
        [command],
        [0.0],
    )
    second = sample_chunk_trajectory(
        first.boundary_state_after_chunk,
        1_200_000,
        command,
        2,
        config.chunk,
        config.vehicle,
        None,
    )
    frames = tuple(
        PresentedFrame(
            timestamp_us=int(t),
            view_rgb_host_uint8=(np.zeros((4, 4, 3), np.uint8),),
            depth_host_f32=None,
        )
        for t in (*first.timestamps_us, *second.timestamps_us)
    )
    recording.append(
        FrameChunk(frames, second.boundary_state_after_chunk, "stub"),
        second,
        [command],
        [0.0],
    )
    recording.close()
    records = [
        json.loads(line)
        for line in (recording.path / "frames.jsonl").read_text().splitlines()
    ]
    assert len(records) == 4
    assert records[0]["vehicle"]["speed_mps"] == 0
    assert records[2]["vehicle"]["speed_mps"] == second.vehicle_states[0].speed_mps
    assert (
        json.loads((recording.path / "manifest.json").read_text())[
            "unemitted_frame_count"
        ]
        == 0
    )


def test_vehicle_mesh_depth_buffer_keeps_the_near_surface(tmp_path):
    far = np.array([[-0.8, -0.8, 0], [0.8, -0.8, 0], [0, 0.8, 0]], dtype=np.float32)
    camera = np.array([-14, 0, 12.8])
    near = camera + (far - camera) * 0.9
    model = tmp_path / "mesh.npz"
    for faces in ([[0, 1, 2], [3, 4, 5]], [[3, 4, 5], [0, 1, 2]]):
        np.savez(
            model,
            vertices=np.concatenate((far, near)),
            faces=np.array(faces),
            colors=np.array(
                [[0, 0, 1] if face[0] == 0 else [1, 0, 0] for face in faces]
            ),
            alpha=np.ones(2),
        )
        image = np.asarray(render_vehicle(model, (604, 330)))
        solid = image[..., 3] == 255
        assert solid.sum() > 30
        assert image[..., 0][solid].mean() > 100
        assert image[..., 2][solid].mean() < 1


def test_reference_projection_respects_mount_distortion_and_ribbon_height():
    mount = np.eye(4)
    mount[2, 3] = 2
    calibration = {
        "sensor_to_ego_matrix": mount,
        "K": [[100, 0, 50], [0, 200, 40], [0, 0, 1]],
        "width": 100,
        "height": 80,
        "D": [0] * 8,
    }
    points = np.array([[1, 2, 12], [0, 0, 1]], dtype=float)
    pixels, depths = project_raw_camera(points, np.eye(4), calibration)
    np.testing.assert_allclose(pixels[0], [0.6, 1.0])
    np.testing.assert_allclose(depths, [10, -1])
    calibration["D"] = [0.1, 0.2, 0.01, 0.02, 0.3, 0.4, 0.5, 0.6]
    pixels, _ = project_raw_camera(points[:1], np.eye(4), calibration)
    radial = (1 + 0.1 * 0.05 + 0.2 * 0.05**2 + 0.3 * 0.05**3) / (
        1 + 0.4 * 0.05 + 0.5 * 0.05**2 + 0.6 * 0.05**3
    )
    distorted = [0.1 * radial + 0.0004 + 0.0014, 0.2 * radial + 0.0013 + 0.0008]
    np.testing.assert_allclose(
        pixels[0], [distorted[0] + 0.5, distorted[1] * 2.5 + 0.5]
    )
    poses = [np.eye(4), np.eye(4)]
    poses[1][:2, :2] = [[0, -1], [1, 0]]
    source = {
        "times": np.array([0, 100_000]),
        "frames": [{"pose": p} for p in poses],
        "floor_offset": 0.8,
        "ribbon_half_width": 1,
    }
    edges = future_edges(source, 0)
    np.testing.assert_allclose(edges[:, :, 2], -0.8)
    np.testing.assert_allclose(edges[0, :, :2], [[0, 1], [0, -1]])
    np.testing.assert_allclose(edges[1, :, :2], [[-1, 0], [1, 0]])


def test_chase_boxes_keep_visible_walls_at_every_heading():
    renderer = ComparisonRenderer.__new__(ComparisonRenderer)
    renderer.lines, renderer.ego_sprite = [], None
    source = {
        "times": np.array([0]),
        "frames": [{"pose": np.eye(4), "timestamp_us": 0}],
        "floor_offset": 0.8,
        "ribbon_half_width": 1,
    }
    camera = np.array([-14, 0, 12])
    for y in (-8, 8):
        center = np.array([12, y, 0.0])
        for heading in (0, math.pi / 2, math.pi, -math.pi / 2):
            renderer.tracks = [
                {
                    "timestamps_us": [0],
                    "translations": [center],
                    "scale": [4.5, 1.8, 1.6],
                    "quaternions_xyzw": [
                        [0, 0, math.sin(heading / 2), math.cos(heading / 2)]
                    ],
                }
            ]
            image = np.asarray(renderer.map_view(source, 0, (604, 330), "#39c7aa"))
            for normal, half in (
                (np.array([math.cos(heading), math.sin(heading), 0]), 2.25),
                (np.array([-math.sin(heading), math.cos(heading), 0]), 0.9),
            ):
                normal *= 1 if normal @ (camera - center) > 0 else -1
                point = center + normal * half - [0, 0, 0.4]
                pixel, _ = chase_projection(
                    np.array([[-point[1], point[0], point[2]]]), (604, 330)
                )
                x, yy = np.round(pixel[0]).astype(int)
                assert image[yy, x, 2] > image[yy, x, 0], (y, heading, normal)
