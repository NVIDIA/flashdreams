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

from __future__ import annotations

import io
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import grpc
import numpy as np
import pytest
from omnidreams.grpc.protos import (
    camera_pb2,
    common_pb2,
    video_model_pb2,
    video_model_pb2_grpc,
)
from PIL import Image

pytestmark = pytest.mark.ci_gpu

REPO_ROOT = Path(__file__).resolve().parents[3]


def _make_initial_frame_png_bytes(height: int, width: int) -> bytes:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[..., 0] = 64
    frame[..., 1] = 128
    frame[..., 2] = 192
    image = Image.fromarray(frame, mode="RGB")
    image_buffer = io.BytesIO()
    image.save(image_buffer, format="PNG")
    return image_buffer.getvalue()


def _build_camera_spec(
    logical_id: str, height: int, width: int
) -> camera_pb2.CameraSpec:
    focal = float(width) / (2.0 * np.deg2rad(60.0))
    return camera_pb2.CameraSpec(
        logical_id=logical_id,
        resolution_h=height,
        resolution_w=width,
        shutter_type=camera_pb2.ShutterType.GLOBAL,
        ftheta_param=camera_pb2.FthetaCameraParam(
            principal_point_x=width / 2.0,
            principal_point_y=height / 2.0,
            reference_poly=camera_pb2.FthetaCameraParam.ANGLE_TO_PIXELDIST,
            angle_to_pixeldist_poly=[focal],
            linear_cde=camera_pb2.LinearCde(linear_c=1.0, linear_d=0.0, linear_e=0.0),
        ),
        rig_to_camera=common_pb2.Pose(
            vec=common_pb2.Vec3(x=0.0, y=0.0, z=0.0),
            quat=common_pb2.Quat(w=1.0, x=0.0, y=0.0, z=0.0),
        ),
    )


def _build_rig_trajectory(
    num_poses: int, start_timestamp_us: int = 1_000_000
) -> common_pb2.Trajectory:
    poses: list[common_pb2.PoseAtTime] = []
    for i in range(num_poses):
        poses.append(
            common_pb2.PoseAtTime(
                timestamp_us=start_timestamp_us + i * 33_333,
                pose=common_pb2.Pose(
                    vec=common_pb2.Vec3(x=0.5 * i, y=0.0, z=0.0),
                    quat=common_pb2.Quat(w=1.0, x=0.0, y=0.0, z=0.0),
                ),
            )
        )
    return common_pb2.Trajectory(poses=poses)


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_server_ready(
    target: str,
    process: subprocess.Popen[str],
    startup_timeout_s: int,
    log_path: Path,
) -> None:
    deadline = time.monotonic() + startup_timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            logs = log_path.read_text(encoding="utf-8", errors="replace")
            raise RuntimeError(
                f"Server exited early with code {process.returncode}.\n=== server logs ===\n{logs}"
            )

        channel = grpc.insecure_channel(target)
        try:
            grpc.channel_ready_future(channel).result(timeout=5.0)
            return
        except grpc.FutureTimeoutError:
            pass
        finally:
            channel.close()

    logs = log_path.read_text(encoding="utf-8", errors="replace")
    raise TimeoutError(
        f"Timed out waiting for server on {target} after {startup_timeout_s}s.\n"
        f"=== server logs ===\n{logs}"
    )


def _stop_server(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return

    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _initial_chunk_size(num_frames_per_block: int) -> int:
    if num_frames_per_block % 4 != 0:
        raise ValueError("num_frames_per_block must be divisible by 4")
    len_t = num_frames_per_block // 4
    return 1 + (len_t - 1) * 4


def test_grpc_server_start_render_close_roundtrip(
    tmp_path: Path, example_scene_zip_bytes: bytes
) -> None:
    resolution_h = 704
    resolution_w = 1280
    num_frames_per_block = 8
    expected_initial_chunk_size = _initial_chunk_size(num_frames_per_block)
    camera_name = "camera_front_wide_120fov"

    port = _pick_free_port()
    target = f"127.0.0.1:{port}"
    log_path = tmp_path / "grpc_server.log"

    server_cmd = [
        sys.executable,
        "-m",
        "integrations.omnidreams.omnidreams.grpc.server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--output_format",
        "jpeg",
        # Single-view chunk2 (len_t=2 -> num_frames_per_block=8), local
        # attention window 6, denoising_timesteps=[1000, 450].
        "--pipeline_config_name",
        "omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae",
    ]

    server_env = os.environ.copy()
    server_env["PYTHONUNBUFFERED"] = "1"
    pythonpath_entries = [
        str(REPO_ROOT / "flashdreams"),
        str(REPO_ROOT / "integrations" / "omnidreams"),
        str(REPO_ROOT / "integrations" / "omnidreams" / "ludus-renderer"),
    ]
    existing_pythonpath = server_env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    server_env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            server_cmd,
            cwd=str(REPO_ROOT),
            env=server_env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

    try:
        _wait_for_server_ready(
            target=target,
            process=process,
            startup_timeout_s=600,
            log_path=log_path,
        )

        channel = grpc.insecure_channel(
            target,
            options=[
                ("grpc.max_send_message_length", 100 * 1024 * 1024),
                ("grpc.max_receive_message_length", 100 * 1024 * 1024),
            ],
        )
        stub = video_model_pb2_grpc.WorldModelServiceStub(channel)

        try:
            start_request = video_model_pb2.SessionRequest(
                static_world_map=video_model_pb2.StaticWorldMap(
                    hdmap_parquets=example_scene_zip_bytes
                ),
                text_prompt=video_model_pb2.TextPrompt(
                    positive=(
                        "Driving scene from a front-facing car camera. Urban environment with roads, "
                        "vehicles, pedestrians, traffic signs, and buildings."
                    ),
                    negative="",
                ),
                debug_options=video_model_pb2.DebugOptions(
                    return_hdmap_frames=True,
                    skip_video_generation=False,
                ),
                camera_specs=[
                    _build_camera_spec(
                        logical_id=camera_name,
                        height=resolution_h,
                        width=resolution_w,
                    )
                ],
                initial_frames=[
                    video_model_pb2.Image(
                        data=_make_initial_frame_png_bytes(
                            height=resolution_h,
                            width=resolution_w,
                        ),
                        format=video_model_pb2.ImageFormat.PNG,
                    )
                ],
                random_seed=42,
            )

            start_response = stub.start_session(start_request, timeout=180)
            assert start_response.session_id, "start_session returned empty session_id"

            render_request = video_model_pb2.VideoChunkRequest(
                session_id=video_model_pb2.SessionId(
                    session_id=start_response.session_id
                ),
                rig_trajectory=_build_rig_trajectory(expected_initial_chunk_size),
                dynamic_state=video_model_pb2.DynamicWorldState(),
            )
            render_response = stub.render_video_chunk(render_request, timeout=600)

            assert len(render_response.camera_outputs) == 1
            camera_output = render_response.camera_outputs[0]
            assert camera_output.camera_logical_id == camera_name
            assert len(camera_output.rgb_frames) > 0, (
                "No RGB frames returned from render_video_chunk"
            )
            assert all(frame.data for frame in camera_output.rgb_frames)
            assert all(
                frame.format == video_model_pb2.ImageFormat.JPEG
                for frame in camera_output.rgb_frames
            )
            assert len(camera_output.hdmap_condition_frames) > 0, (
                "No HDMap condition frames returned despite return_hdmap_frames=True"
            )
            assert (
                len(render_response.poses_and_timestamps_of_frames.poses)
                == expected_initial_chunk_size
            )

            close_response = stub.close_session(
                video_model_pb2.SessionCloseRequest(
                    session_id=start_response.session_id
                ),
                timeout=60,
            )
            assert close_response is not None
        finally:
            channel.close()
    finally:
        _stop_server(process)
