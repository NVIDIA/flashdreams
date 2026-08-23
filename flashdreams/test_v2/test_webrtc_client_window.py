# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the v2 WebRTC client window."""

import asyncio
import json
from urllib.parse import urlsplit

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.ci_cpu

pytest.importorskip("aiohttp")
pytest.importorskip("aiortc")

from aiohttp import ClientSession
from aiortc import (
    MediaStreamTrack,
    RTCDataChannel,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from av import VideoFrame
from flashdreams.runtime_v2.serving.webrtc_server import (
    WebRTCServerConfig,
    _closed_connection_status,
    _VideoTrack,
)
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    ActivationUserInputEventData,
    CloseUserInputEventData,
    KeyboardUserInputEventData,
    MouseUserInputEventData,
    UserInputEvent,
)
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout
from flashdreams.runtime_v2.webrtc_client_window import WebRTCClientWindow


@pytest.mark.parametrize(
    ("connection_state", "client_connected", "expected"),
    (("failed", False, "failed"), ("closed", False, "failed"), ("closed", True, "disconnected")),
)
def test_closed_connection_status(
    connection_state: str, client_connected: bool, expected: str
) -> None:
    assert (
        _closed_connection_status(
            connection_state=connection_state, client_connected=client_connected
        )
        == expected
    )


def _session_desc() -> SessionDesc:
    return SessionDesc(
        output_layout=VideoTensorLayout.tchw,
        frames_per_second_for_ui=30,
        frames_per_second_for_step=30,
        video_width=16,
        video_height=16,
    )


def _origin(window: WebRTCClientWindow) -> str:
    parsed = urlsplit(window.server.url)
    return f"{parsed.scheme}://{parsed.netloc}"


async def _connect_browser(
    window: WebRTCClientWindow,
    *,
    client_id: str = "test-browser-client",
) -> tuple[RTCPeerConnection, RTCDataChannel, asyncio.Future[MediaStreamTrack]]:
    peer = RTCPeerConnection()
    channel = peer.createDataChannel("controls")
    peer.addTransceiver("video", direction="recvonly")
    channel_opened = asyncio.Event()
    video_track: asyncio.Future[MediaStreamTrack] = asyncio.get_running_loop().create_future()

    @channel.on("open")
    def on_open() -> None:
        channel_opened.set()

    @peer.on("track")
    def on_track(track: MediaStreamTrack) -> None:
        if not video_track.done():
            video_track.set_result(track)

    await peer.setLocalDescription(await peer.createOffer())
    async with (
        ClientSession() as client,
        client.post(
            f"{window.server.url}api/webrtc/offer",
            headers={"Origin": _origin(window)},
            json={
                "sdp": peer.localDescription.sdp,
                "type": peer.localDescription.type,
                "client_id": client_id,
            },
        ) as response,
    ):
        assert response.status == 200
        answer = await response.json()
    await peer.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))
    await asyncio.wait_for(channel_opened.wait(), timeout=5)
    return peer, channel, video_track


async def _drain_until(window: WebRTCClientWindow, expected: int) -> list[UserInputEvent]:
    events = []
    for _ in range(200):
        events.extend(window.get_user_input_events().get_events())
        if len(events) >= expected:
            return events
        await asyncio.sleep(0.01)
    return events


@pytest.mark.asyncio
async def test_window_secures_url_and_buffers_browser_events_until_drained() -> None:
    window = WebRTCClientWindow(
        config=WebRTCServerConfig(
            capability_path="secret-path",
            max_pending_frames=2,
            allow_reconnect=True,
            pointer_lock_controls=True,
        )
    )
    statuses: list[tuple[str, dict[str, str]]] = []
    window.server.register_connection_status_callback(
        lambda status, fields: statuses.append((status, fields))
    )
    peer: RTCPeerConnection | None = None
    replacement_peer: RTCPeerConnection | None = None
    try:
        async with ClientSession() as client:
            async with client.get(f"{_origin(window)}/") as response:
                assert response.status == 404
            async with client.get(f"{window.server.url}healthz") as response:
                assert response.status == 200
                assert await response.json() == {
                    "open": False,
                    "client_connected": False,
                }
            async with client.get(window.server.url) as response:
                browser_page = await response.text()
                assert response.status == 200
                assert 'id="activate"' in browser_page
                assert '<script src="app.js"></script>' in browser_page
                assert "default-src 'none'" in response.headers["Content-Security-Policy"]
            async with client.get(f"{window.server.url}app.js") as response:
                browser_script = await response.text()
                assert response.status == 200
                assert 'type: "activation"' in browser_script
                assert 'type: "mouse"' in browser_script
                assert "client_id: clientId" in browser_script
            async with client.get(f"{window.server.url}api/webrtc/config") as response:
                assert response.status == 200
                assert await response.json() == {
                    "iceServers": [],
                    "iceTransportPolicy": "all",
                    "allowReconnect": True,
                    "pointerLockControls": True,
                }

        window.open(_session_desc())
        async with (
            ClientSession() as client,
            client.post(f"{window.server.url}api/webrtc/offer", json={}) as response,
        ):
            assert response.status == 403

        peer, channel, _ = await _connect_browser(window)
        channel.send(json.dumps({"type": "activation", "active": True}))
        channel.send(json.dumps({"type": "keyboard", "key": "w", "pressed": True}))
        channel.send(json.dumps({"type": "mouse", "movement_x": 3.5, "movement_y": -2}))
        channel.send(json.dumps({"type": "keyboard", "key": "w", "pressed": False}))

        events = await _drain_until(window, 4)
        assert len(events) == 4
        event_data = [event.get_event_data() for event in events]
        assert isinstance(event_data[0], ActivationUserInputEventData)
        assert event_data[0].active
        assert isinstance(event_data[1], KeyboardUserInputEventData)
        assert (event_data[1].key, event_data[1].pressed) == ("w", True)
        assert isinstance(event_data[2], MouseUserInputEventData)
        assert (event_data[2].movement_x, event_data[2].movement_y) == (3.5, -2.0)
        assert isinstance(event_data[3], KeyboardUserInputEventData)
        assert (event_data[3].key, event_data[3].pressed) == ("w", False)
        assert events[0].get_timestamp() <= events[-1].get_timestamp()
        assert window.get_user_input_events().get_events() == []

        await peer.close()
        peer = None
        disconnect_events = await _drain_until(window, 1)
        assert len(disconnect_events) == 1
        disconnected = disconnect_events[0].get_event_data()
        assert isinstance(disconnected, ActivationUserInputEventData)
        assert not disconnected.active
        assert any(
            status == "connected" and fields["remote_address"] == "127.0.0.1"
            for status, fields in statuses
        )
        assert statuses[-1][0] == "disconnected"

        replacement_peer, replacement_channel, _ = await _connect_browser(window)
        replacement_channel.send(json.dumps({"type": "activation", "active": True}))
        replacement_events = await _drain_until(window, 1)
        assert len(replacement_events) == 1
        reconnected = replacement_events[0].get_event_data()
        assert isinstance(reconnected, ActivationUserInputEventData)
        assert reconnected.active
    finally:
        if peer is not None:
            await peer.close()
        if replacement_peer is not None:
            await replacement_peer.close()
        window.close()


@pytest.mark.asyncio
async def test_window_serves_turn_configuration_without_caching_credentials() -> None:
    window = WebRTCClientWindow(
        config=WebRTCServerConfig(
            capability_path="secret-path",
            ice_servers=(
                RTCIceServer(
                    urls="turn:127.0.0.1:3478?transport=tcp",
                    username="test-user",
                    credential="secret",
                ),
            ),
            browser_ice_transport_policy="relay",
        )
    )
    try:
        async with (
            ClientSession() as client,
            client.get(f"{window.server.url}api/webrtc/config") as response,
        ):
            assert response.status == 200
            assert response.headers["Cache-Control"] == "no-store"
            assert await response.json() == {
                "iceServers": [
                    {
                        "urls": "turn:127.0.0.1:3478?transport=tcp",
                        "username": "test-user",
                        "credential": "secret",
                    }
                ],
                "iceTransportPolicy": "relay",
                "allowReconnect": False,
                "pointerLockControls": False,
            }
    finally:
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
        await peer.close()
        peer = None
        disconnect_events = await _drain_until(window, 1)
        assert len(disconnect_events) == 1
        assert isinstance(disconnect_events[0].get_event_data(), CloseUserInputEventData)

    finally:
        if peer is not None:
            await peer.close()
        window.close()


@pytest.mark.asyncio
async def test_video_track_keeps_only_the_latest_pending_frames() -> None:
    track = _VideoTrack(frames_per_second=1000, max_pending_frames=2)
    try:
        await track.enqueue(tuple(np.full((2, 2, 3), value, dtype=np.uint8) for value in (1, 2)))
        await track.enqueue(tuple(np.full((2, 2, 3), value, dtype=np.uint8) for value in (3, 4)))

        first = await track.recv()
        second = await track.recv()
        assert int(first.to_ndarray(format="rgb24")[0, 0, 0]) == 3
        assert int(second.to_ndarray(format="rgb24")[0, 0, 0]) == 4
    finally:
        await track.close()


@pytest.mark.asyncio
async def test_video_track_preserves_all_frames_without_an_explicit_bound() -> None:
    track = _VideoTrack(frames_per_second=1000, max_pending_frames=None)
    try:
        await track.enqueue(tuple(np.full((2, 2, 3), value, dtype=np.uint8) for value in (1, 2)))
        await track.enqueue(tuple(np.full((2, 2, 3), value, dtype=np.uint8) for value in (3, 4)))

        frames = [await track.recv() for _ in range(4)]
        assert [int(frame.to_ndarray(format="rgb24")[0, 0, 0]) for frame in frames] == [1, 2, 3, 4]
    finally:
        await track.close()


@pytest.mark.asyncio
async def test_same_browser_replaces_its_stale_peer_without_displacing_another() -> None:
    window = WebRTCClientWindow(
        config=WebRTCServerConfig(
            capability_path="secret-path",
            allow_reconnect=True,
        )
    )
    first_peer: RTCPeerConnection | None = None
    replacement_peer: RTCPeerConnection | None = None
    try:
        window.open(_session_desc())
        first_peer, _, _ = await _connect_browser(
            window,
            client_id="same-browser-client",
        )
        async with (
            ClientSession() as client,
            client.post(
                f"{window.server.url}api/webrtc/offer",
                headers={"Origin": _origin(window)},
                json={
                    "sdp": "",
                    "type": "offer",
                    "client_id": "different-browser-client",
                },
            ) as response,
        ):
            assert response.status == 409

        replacement_peer, _, _ = await _connect_browser(
            window,
            client_id="same-browser-client",
        )
        events = await _drain_until(window, 1)
        assert len(events) == 1
        disconnected = events[0].get_event_data()
        assert isinstance(disconnected, ActivationUserInputEventData)
        assert not disconnected.active
    finally:
        if first_peer is not None:
            await first_peer.close()
        if replacement_peer is not None:
            await replacement_peer.close()
        window.close()
