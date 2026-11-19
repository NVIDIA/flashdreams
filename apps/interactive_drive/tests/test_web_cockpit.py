# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise browser transport and driving physics with CPU image fixtures."""

import asyncio
import base64
import json
from dataclasses import replace
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from interactive_drive.app import InteractiveDriveApplication, InteractiveDriveWebUILoop
from interactive_drive.core import InteractiveDriveApplicationDefaults
from interactive_drive.types import FrameChunk, PresentedFrame, VehicleState
from PIL import Image

pytestmark = pytest.mark.ci_cpu

CAMERAS = (
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_rear_right_70fov",
    "camera_rear_tele_30fov",
    "camera_rear_left_70fov",
    "camera_cross_left_120fov",
    "camera_front_tele_30fov",
)


@pytest.mark.asyncio
async def test_web_cockpit_video_controls_and_native_bev_transport(tmp_path):
    aiohttp = pytest.importorskip("aiohttp")
    rtc = pytest.importorskip("aiortc")
    from flashdreams.runtime_v2.webrtc_client_window import WebRTCClientWindow

    scene = tmp_path / "scene.usdz"
    scene.touch()
    app = InteractiveDriveApplication(
        defaults=InteractiveDriveApplicationDefaults(
            physics=False,
            reverse=True,
            bev=True,
        )
    )
    app.init(
        [
            "--scene",
            str(scene),
            "--web-ui",
            "--export-dir",
            str(tmp_path / "exports"),
            "--world-model-device",
            "cpu",
            "--camera",
            ",".join(CAMERAS),
            "--present-camera",
            CAMERAS[0],
            "--width",
            "32",
            "--height",
            "32",
            "--total-blocks",
            "0",
        ]
    )
    desc = app.session_desc()
    assert (desc.video_width, desc.video_height) == (32, 32)
    session = app.create_session(desc)
    session.init()
    model, ui = session.model_loop, session.ui_loop
    assert isinstance(ui, InteractiveDriveWebUILoop)
    assert model.state.config.app.bev.enabled
    model.state.scene = SimpleNamespace(
        scene_id="cpu-test", scene_path=scene, cameras=()
    )
    model.state.vehicle = VehicleState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def render(trajectory):
        pixels = tuple(np.full((32, 32, 3), 30 + i * 20, np.uint8) for i in range(7))
        return FrameChunk(
            (
                PresentedFrame(
                    timestamp_us=int(trajectory.timestamps_us[-1]),
                    view_rgb_host_uint8=pixels,
                    view_model_rgb_host_uint8=pixels,
                    depth_host_f32=None,
                    # A stand-in for the backend's native WorldScenario render, passed through unchanged.
                    bev_host_uint8=np.full((16, 16, 3), 77, np.uint8),
                ),
            ),
            trajectory.boundary_state_after_chunk,
            "cpu-test",
        )

    model.state.backend = SimpleNamespace(
        initial_chunk_frames=1,
        chunk_frames=1,
        render_first_chunk=render,
        render_next_chunk=render,
        finalize=dict,
    )
    window = WebRTCClientWindow()
    peer = rtc.RTCPeerConnection(rtc.RTCConfiguration(iceServers=[]))
    received = []
    track_future = asyncio.get_running_loop().create_future()
    opened = asyncio.Event()
    channel = peer.createDataChannel("controls")
    channel.on("open", opened.set)
    channel.on("message", lambda message: received.append(json.loads(message)))
    peer.on("track", lambda track: track_future.set_result(track))
    peer.addTransceiver("video", direction="recvonly")

    async def until(predicate):
        async with asyncio.timeout(8):
            while not predicate():
                await asyncio.sleep(0.01)

    async def send_events(*messages):
        for message in messages:
            channel.send(json.dumps(message))
        events = []
        async with asyncio.timeout(5):
            while len(events) < len(messages):
                events.extend(window.get_user_input_events().get_events())
                await asyncio.sleep(0.01)
        return UserInputEvents(events)

    async def step(index, events):
        outputs = model.step(index, events)
        ui._run_message_batch()
        session._presentation_manager.publish(0, outputs)
        session._presentation_manager.advance(0, now=1.0 + index)
        results = ui.step(index, UserInputEvents([]))
        assert len(results) == 1
        window.write(results[0])
        await until(
            lambda: (
                window.metrics_snapshot()["webrtc_sender_handed_off_count"] >= index + 1
            )
        )
        return results[0]

    try:
        window.open(desc)
        await peer.setLocalDescription(await peer.createOffer())
        async with (
            aiohttp.ClientSession() as client,
            client.post(
                f"{window.server.url}api/webrtc/offer",
                json={
                    "type": peer.localDescription.type,
                    "sdp": peer.localDescription.sdp,
                },
            ) as response,
        ):
            assert response.status == 200
            answer = await response.json()
        await peer.setRemoteDescription(rtc.RTCSessionDescription(**answer))
        await asyncio.wait_for(opened.wait(), 5)
        window.get_user_input_events()  # discard connection metadata
        original_config = model.state.config
        result = await step(
            0,
            await send_events(
                {"type": "recording", "action": "start"},
                {"type": "select_camera", "camera": CAMERAS[2], "view_index": 2},
                {
                    "type": "game_wheel",
                    "action": "state",
                    "steering": 1,
                    "throttle": 1,
                    "gear": "D",
                },
            ),
        )
        assert original_config.app.present_camera == CAMERAS[0]
        assert model.state.config.app.present_camera == CAMERAS[2]
        assert model.state.vehicle.speed_mps > 0
        assert model.state.vehicle.steer_rad < 0  # D/right rotates clockwise
        assert result.read_output()[0, :, 0, 0].tolist() == [70, 70, 70]
        speed = model.state.vehicle.speed_mps
        await step(
            1,
            await send_events(
                {"type": "recording", "action": "stop"},
                {
                    "type": "game_wheel",
                    "action": "state",
                    "steering": 0,
                    "throttle": 0,
                },
            ),
        )
        assert model.state.vehicle.steer_rad == 0
        assert model.state.last_command.throttle == 0
        assert model.state.vehicle.speed_mps <= speed
        assert model.state.recording is None
        archives = list((tmp_path / "exports").glob("*.zip"))
        assert len(archives) == 1
        await until(lambda: any(p.get("type") == "download_ready" for p in received))
        download = next(p for p in received if p.get("type") == "download_ready")
        async with aiohttp.ClientSession() as client:
            async with client.get(
                window.server.url + download["url"].lstrip("/")
            ) as response:
                assert response.status == 200
                assert response.headers["Content-Disposition"].startswith("attachment;")
                assert await response.read() == archives[0].read_bytes()
            async with client.get(
                window.server.url + download["url"].lstrip("/"),
                headers={"Range": "bytes=0-63"},
            ) as response:
                assert response.status == 206 and len(await response.read()) == 64
            async with client.get(
                window.server.url + "api/downloads/unknown"
            ) as response:
                assert response.status == 404
            async with client.get(window.server.url) as response:
                assert 'id="recording-start"' in await response.text()
        track = await asyncio.wait_for(track_future, 5)
        frame = await asyncio.wait_for(track.recv(), 5)
        assert abs(frame.to_ndarray(format="rgb24").mean() - 70) <= 3
        await until(lambda: any(t.get("type") == "telemetry" for t in received))
        telemetry = next(t for t in received if t.get("type") == "telemetry")
        assert telemetry["throttle"] == 1 and telemetry["steer"] == -1
        assert telemetry["speed_kmh"] > 0
        assert telemetry["gear"] == "D"
        await until(
            lambda: len([p for p in received if p.get("type") == "camera_preview"]) >= 7
        )
        previews = [p for p in received if p.get("type") == "camera_preview"][:7]
        assert [p["camera"] for p in previews] == list(CAMERAS)
        for i, preview in enumerate(previews):
            image = Image.open(
                BytesIO(base64.b64decode(preview["image"].split(",")[1]))
            )
            assert abs(np.asarray(image).mean() - (30 + i * 20)) <= 1
            assert len(json.dumps(preview)) < 65_536
        image = Image.open(
            BytesIO(base64.b64decode(telemetry["bev_image"].split(",")[1]))
        )
        assert image.size == (16, 16) and np.asarray(image).mean() == 77
        assert telemetry["active_view"] == CAMERAS[2]
        model.state.vehicle.speed_mps = 0
        await step(
            2,
            await send_events(
                {
                    "type": "game_wheel",
                    "action": "state",
                    "gear": "R",
                    "throttle": 1,
                }
            ),
        )
        assert model.state.vehicle.speed_mps < 0
        await step(3, await send_events({"type": "focus", "focused": False}))
        assert model.state.drive_input.controller_command is None
        assert model.state.last_command.throttle == 0
        for index, gear in enumerate(("N", "P", "D", "R"), start=4):
            result = await step(
                index,
                await send_events(
                    {
                        "type": "game_wheel",
                        "action": "state",
                        "gear": gear,
                        "throttle": 1,
                    }
                ),
            )
            command = model.state.last_command
            assert result.client_data[0]["gear"] == gear
            assert command.gear == gear
            assert command.throttle == (0 if gear in ("P", "N") else 1)
            assert command.brake == (1 if gear == "P" else 0)
            assert command.reverse == (gear == "R")
        # Camera selection changes the main video while previews retain all seven views.
        await step(
            8, await send_events({"type": "select_camera", "camera": CAMERAS[5]})
        )
        assert model.state.config.app.present_camera == CAMERAS[5]
        assert len(ui.client_data()) == 9
        model.state.config = replace(
            model.state.config, app=replace(model.state.config.app, reverse=False)
        )
        result = await step(
            9,
            await send_events(
                {"type": "game_wheel", "action": "state", "gear": "R", "throttle": 1}
            ),
        )
        assert model.state.last_command.throttle == 0
        assert model.state.last_command.brake == 1
        assert result.client_data[0]["gear"] == "D"
    finally:
        server_peer = window.server._peer_connection
        if server_peer is not None:
            await asyncio.wrap_future(
                asyncio.run_coroutine_threadsafe(
                    window.server._release_peer_connection(server_peer),
                    window.server._loop,
                )
            )
        await peer.close()
        window.close()
        session.close()


def test_preview_jpeg_fits_sctp_message_limit():
    from interactive_drive.core import _jpeg_data_url

    pixels = np.random.default_rng(17).integers(0, 256, (1024, 1024, 3), dtype=np.uint8)
    url = _jpeg_data_url(pixels, (512, 512))
    assert len(json.dumps({"type": "telemetry", "bev_image": url})) < 65_536
    image = Image.open(BytesIO(base64.b64decode(url.split(",")[1])))
    assert image.width == image.height and image.width <= 512
