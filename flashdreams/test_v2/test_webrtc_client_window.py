# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the v2 WebRTC client window."""

import asyncio
import json

import pytest
import torch

pytestmark = pytest.mark.ci_cpu

pytest.importorskip("aiohttp")
pytest.importorskip("aiortc")

from aiohttp import ClientSession
from aiortc import (
    MediaStreamTrack,
    RTCDataChannel,
    RTCPeerConnection,
    RTCSessionDescription,
)
from av import VideoFrame

from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    FocusUserInputEventData,
    GamepadUserInputEventData,
    GameWheelUserInputEventData,
    KeyboardInputState,
    KeyboardUserInputEventData,
    MouseUserInputEventData,
    TouchUserInputEventData,
    XRControllerUserInputEventData,
)
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout
from flashdreams.runtime_v2.webrtc_client_window import WebRTCClientWindow


def _session_desc() -> SessionDesc:
    return SessionDesc(
        output_layout=VideoTensorLayout.tchw,
        frames_per_second_for_ui=30,
        frames_per_second_for_step=30,
        video_width=16,
        video_height=16,
    )


async def _connect_browser(
    window: WebRTCClientWindow,
) -> tuple[RTCPeerConnection, RTCDataChannel, asyncio.Future[MediaStreamTrack]]:
    peer = RTCPeerConnection()
    channel = peer.createDataChannel("controls")
    peer.addTransceiver("video", direction="recvonly")
    channel_opened = asyncio.Event()
    video_track: asyncio.Future[MediaStreamTrack] = (
        asyncio.get_running_loop().create_future()
    )

    @channel.on("open")
    def on_open() -> None:
        channel_opened.set()

    @peer.on("track")
    def on_track(track: MediaStreamTrack) -> None:
        if not video_track.done():
            video_track.set_result(track)

    await peer.setLocalDescription(await peer.createOffer())
    async with ClientSession() as client:
        async with client.post(
            f"{window.server.url}api/webrtc/offer",
            json={
                "sdp": peer.localDescription.sdp,
                "type": peer.localDescription.type,
            },
        ) as response:
            assert response.status == 200
            answer = await response.json()
    await peer.setRemoteDescription(
        RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
    )
    await asyncio.wait_for(channel_opened.wait(), timeout=5)
    return peer, channel, video_track


@pytest.mark.asyncio
async def test_window_buffers_browser_events_until_drained() -> None:
    window = WebRTCClientWindow()
    peer: RTCPeerConnection | None = None
    try:
        async with ClientSession() as client:
            async with client.get(f"{window.server.url}healthz") as response:
                assert response.status == 200
                assert await response.json() == {
                    "open": False,
                    "client_connected": False,
                }
            async with client.get(window.server.url) as response:
                browser_page = await response.text()
                assert response.status == 200
                assert 'id="activate"' not in browser_page
                assert 'id="reset"' not in browser_page
                assert '<video id="video" autoplay muted playsinline>' in browser_page
                assert 'id="status"' in browser_page
                assert '<script src="/app.js"></script>' in browser_page
            async with client.get(f"{window.server.url}app.js") as response:
                browser_script = await response.text()
                assert response.status == 200
                assert "activationPressed" not in browser_script
                assert 'type: "reset"' not in browser_script
                assert "waitForIceGatheringComplete" in browser_script
                assert 'peer.iceGatheringState === "complete"' in browser_script
                assert "Unable to start WebRTC" in browser_script
                assert "renderedVideoBounds" in browser_script
                assert "pressedKeys" in browser_script
                assert "pointercancel" in browser_script
                assert "navigator.getGamepads" in browser_script
                assert 'type: "touch"' in browser_script

        window.open(_session_desc())
        peer, channel, _ = await _connect_browser(window)
        channel.send(json.dumps({"type": "keyboard", "key": "w", "pressed": True}))
        channel.send(json.dumps({"type": "keyboard", "key": "w", "pressed": False}))
        channel.send(
            json.dumps({"type": "mouse", "action": "move", "x": 0.25, "y": 0.75})
        )
        channel.send(json.dumps({"type": "focus", "focused": True}))
        channel.send(
            json.dumps(
                {
                    "type": "touch",
                    "action": "move",
                    "touch_id": 3,
                    "x": 0.4,
                    "y": 0.6,
                    "pressure": 0.75,
                    "primary": True,
                }
            )
        )
        channel.send(
            json.dumps(
                {
                    "type": "gamepad",
                    "action": "state",
                    "index": 1,
                    "controller_id": "standard pad",
                    "mapping": "standard",
                    "axes": [-0.5, 0.25],
                    "buttons": [0.0, 1.0],
                    "pressed": [False, True],
                }
            )
        )
        channel.send(
            json.dumps(
                {
                    "type": "game_wheel",
                    "action": "state",
                    "index": 2,
                    "id": "wheel",
                    "steering": -0.25,
                    "throttle": 0.8,
                    "brake": 0.1,
                }
            )
        )
        channel.send(
            json.dumps(
                {
                    "type": "xr_controller",
                    "action": "state",
                    "handedness": "right",
                    "position": [1, 2, 3],
                    "orientation": [0, 0, 0, 1],
                }
            )
        )

        events = []
        for _ in range(100):
            events.extend(window.get_user_input_events().get_events())
            if len(events) == 8:
                break
            await asyncio.sleep(0.01)

        assert len(events) == 8
        keyboard_events = [
            data
            for event in events
            if isinstance(data := event.get_event_data(), KeyboardUserInputEventData)
        ]
        assert [(event.key, event.state) for event in keyboard_events] == [
            ("w", KeyboardInputState.PRESSED),
            ("w", KeyboardInputState.RELEASED),
        ]
        assert events[0].get_timestamp() <= events[1].get_timestamp()
        mouse = next(
            data
            for event in events
            if isinstance(data := event.get_event_data(), MouseUserInputEventData)
        )
        assert (mouse.action, mouse.x, mouse.y) == ("move", 0.25, 0.75)
        focus = next(
            data
            for event in events
            if isinstance(data := event.get_event_data(), FocusUserInputEventData)
        )
        assert focus.focused
        touch = next(
            data
            for event in events
            if isinstance(data := event.get_event_data(), TouchUserInputEventData)
        )
        assert (touch.touch_id, touch.x, touch.y, touch.pressure, touch.primary) == (
            3,
            0.4,
            0.6,
            0.75,
            True,
        )
        gamepad = next(
            data
            for event in events
            if isinstance(data := event.get_event_data(), GamepadUserInputEventData)
        )
        assert gamepad.axes == (-0.5, 0.25)
        assert gamepad.buttons == (0.0, 1.0)
        assert gamepad.pressed == (False, True)
        wheel = next(
            data
            for event in events
            if isinstance(data := event.get_event_data(), GameWheelUserInputEventData)
        )
        assert (wheel.steering, wheel.throttle, wheel.brake) == (-0.25, 0.8, 0.1)
        xr = next(
            data
            for event in events
            if isinstance(
                data := event.get_event_data(), XRControllerUserInputEventData
            )
        )
        assert xr.handedness == "right"
        assert xr.position == (1.0, 2.0, 3.0)
        assert xr.orientation == (0.0, 0.0, 0.0, 1.0)
        assert window.get_user_input_events().get_events() == []
    finally:
        if peer is not None:
            await peer.close()
        window.close()


@pytest.mark.asyncio
async def test_write_delivers_a_video_frame_to_the_browser() -> None:
    window = WebRTCClientWindow()
    peer: RTCPeerConnection | None = None
    try:
        window.open(_session_desc())
        peer, _, video_track = await _connect_browser(window)
        track = await asyncio.wait_for(video_track, timeout=5)

        window.write(
            StepResult(
                step_index=0,
                output=torch.full((2, 3, 16, 16), 17, dtype=torch.uint8),
                frame_count=2,
                output_layout=VideoTensorLayout.tchw,
                metrics={},
            )
        )

        frame = await asyncio.wait_for(track.recv(), timeout=5)
        assert isinstance(frame, VideoFrame)
        pixels = frame.to_ndarray(format="rgb24")
        assert pixels.shape == (16, 16, 3)
        assert abs(float(pixels.mean()) - 17.0) <= 2.0
    finally:
        if peer is not None:
            await peer.close()
        window.close()
