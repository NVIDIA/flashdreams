# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the v2 WebRTC client window."""

import asyncio
import json
import threading
from dataclasses import replace

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
    CloseUserInputEventData,
    KeyboardUserInputEventData,
    NewSessionUserInputEventData,
)
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout
from flashdreams.runtime_v2.webrtc_client_window import WebRTCClientWindow


def _session_desc() -> SessionDesc:
    return SessionDesc(
        output_layout=VideoTensorLayout.tchw,
        frames_per_second_for_ui=60,
        frames_per_second_for_step=20,
        video_width=16,
        video_height=16,
    )


async def _connect_browser(
    window: WebRTCClientWindow,
) -> tuple[
    RTCPeerConnection,
    RTCDataChannel,
    asyncio.Future[MediaStreamTrack],
]:
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
    deadline = asyncio.get_running_loop().time() + 5
    async with ClientSession() as client:
        while True:
            async with client.post(
                f"{window.url}api/webrtc/offer",
                json={
                    "sdp": peer.localDescription.sdp,
                    "type": peer.localDescription.type,
                },
            ) as response:
                if response.status != 409:
                    assert response.status == 200
                    answer = await response.json()
                    break
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("The previous browser did not disconnect.")
            await asyncio.sleep(0.01)
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
        assert window.keeps_open_between_sessions is False
        async with ClientSession() as client:
            async with client.get(f"{window.url}healthz") as response:
                assert response.status == 200
                assert await response.json() == {
                    "open": False,
                    "client_connected": False,
                }
            async with client.get(window.url) as response:
                browser_page = await response.text()
                assert response.status == 200
                assert 'id="activate"' in browser_page
                assert 'id="prompt"' in browser_page
                assert 'id="new-session" type="button">' in browser_page
                assert (
                    'id="new-session" type="button">Opening...</button>' in browser_page
                )
                assert '<script src="/app.js"></script>' in browser_page
            async with client.get(f"{window.url}app.js") as response:
                browser_script = await response.text()
                assert response.status == 200
                assert 'key: "r", pressed: activationPressed' in browser_script
                assert 'type: "new_session"' in browser_script
                assert "metadata: {prompt: promptInput.value}" in browser_script
                assert "pendingNewSession = request" in browser_script
                assert 'newSessionButton.textContent = "Opening..."' in browser_script
                assert "event.target === promptInput" in browser_script
                assert "response.status !== 409" in browser_script

        window.open(_session_desc())
        peer, channel, _ = await _connect_browser(window)
        channel.send(json.dumps({"type": "keyboard", "key": "w", "pressed": True}))
        channel.send(json.dumps({"type": "keyboard", "key": "w", "pressed": False}))
        channel.send(
            json.dumps(
                {
                    "type": "new_session",
                    "metadata": {"prompt": "A dog snowboarding"},
                }
            )
        )

        events = []
        for _ in range(100):
            events.extend(window.get_user_input_events().get_events())
            if len(events) == 3:
                break
            await asyncio.sleep(0.01)

        assert len(events) == 3
        keyboard_events = [
            data
            for event in events
            if isinstance(data := event.get_event_data(), KeyboardUserInputEventData)
        ]
        assert [(event.key, event.pressed) for event in keyboard_events] == [
            ("w", True),
            ("w", False),
        ]
        new_session_events = [
            data
            for event in events
            if isinstance(data := event.get_event_data(), NewSessionUserInputEventData)
        ]
        assert [event.metadata for event in new_session_events] == [
            {"prompt": "A dog snowboarding"}
        ]
        assert events[0].get_timestamp() <= events[1].get_timestamp()
        assert window.get_user_input_events().get_events() == []

        # Make the old timestamp epoch observably older than a fresh offer, then
        # reopen between the old browser's close and the refreshed request. The
        # drained FIFO order must survive that session boundary.
        await asyncio.sleep(1)
        await peer.close()
        peer = None
        async with ClientSession() as client:
            deadline = asyncio.get_running_loop().time() + 5
            while True:
                async with client.get(f"{window.url}healthz") as response:
                    health = await response.json()
                if not health["client_connected"]:
                    break
                if asyncio.get_running_loop().time() >= deadline:
                    raise TimeoutError("The closed browser was not released.")
                await asyncio.sleep(0.01)
        window.open(replace(_session_desc(), metadata={"prompt": "next"}))
        peer, channel, _ = await _connect_browser(window)
        channel.send(
            json.dumps(
                {
                    "type": "new_session",
                    "metadata": {"prompt": "A fox in a forest"},
                }
            )
        )
        refreshed_events = []
        for _ in range(100):
            refreshed_events.extend(window.get_user_input_events().get_events())
            if any(
                isinstance(event.get_event_data(), NewSessionUserInputEventData)
                for event in refreshed_events
            ):
                break
            await asyncio.sleep(0.01)
        assert [
            event_data.metadata
            for event in refreshed_events
            if isinstance(
                event_data := event.get_event_data(),
                NewSessionUserInputEventData,
            )
        ] == [{"prompt": "A fox in a forest"}]
        lifecycle_types = [
            type(event.get_event_data())
            for event in refreshed_events
            if isinstance(
                event.get_event_data(),
                (CloseUserInputEventData, NewSessionUserInputEventData),
            )
        ]
        assert lifecycle_types == [
            CloseUserInputEventData,
            NewSessionUserInputEventData,
        ]
    finally:
        if peer is not None:
            await peer.close()
            await asyncio.sleep(0.05)
        window.close()


@pytest.mark.asyncio
async def test_overlapping_offer_connects_after_active_browser_releases() -> None:
    window = WebRTCClientWindow()
    first = RTCPeerConnection()
    second = RTCPeerConnection()
    try:
        window.open(_session_desc())
        for peer in (first, second):
            peer.createDataChannel("controls")
            peer.addTransceiver("video", direction="recvonly")
            await peer.setLocalDescription(await peer.createOffer())

        async with ClientSession() as client:

            async def offer(
                peer: RTCPeerConnection,
            ) -> tuple[int, dict[str, str] | None]:
                async with client.post(
                    f"{window.url}api/webrtc/offer",
                    json={
                        "sdp": peer.localDescription.sdp,
                        "type": peer.localDescription.type,
                    },
                ) as response:
                    answer = await response.json() if response.status == 200 else None
                    return response.status, answer

            responses = await asyncio.gather(offer(first), offer(second))

            assert sorted(status for status, _ in responses) == [200, 409]
            admitted_index = next(
                index for index, (status, _) in enumerate(responses) if status == 200
            )
            rejected_index = 1 - admitted_index
            peers = (first, second)
            admitted = peers[admitted_index]
            rejected = peers[rejected_index]
            answer = responses[admitted_index][1]
            assert answer is not None
            await admitted.setRemoteDescription(
                RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
            )
            deadline = asyncio.get_running_loop().time() + 5
            while admitted.connectionState != "connected":
                if asyncio.get_running_loop().time() >= deadline:
                    raise TimeoutError("The admitted peer did not connect.")
                await asyncio.sleep(0.01)

            # The refreshed browser overlaps the old one and must be rejected
            # while that old peer remains active.
            status, _ = await offer(rejected)
            assert status == 409

            # Once the old page releases its peer, retrying that same pending
            # offer must succeed within a bound.
            await admitted.close()
            deadline = asyncio.get_running_loop().time() + 5
            replacement_answer = None
            while replacement_answer is None:
                status, replacement_answer = await offer(rejected)
                if status != 409:
                    assert status == 200
                    break
                if asyncio.get_running_loop().time() >= deadline:
                    raise TimeoutError("The refreshed browser did not connect.")
                await asyncio.sleep(0.01)
            assert replacement_answer is not None
            await rejected.setRemoteDescription(
                RTCSessionDescription(
                    sdp=replacement_answer["sdp"],
                    type=replacement_answer["type"],
                )
            )
            deadline = asyncio.get_running_loop().time() + 5
            while rejected.connectionState != "connected":
                if asyncio.get_running_loop().time() >= deadline:
                    raise TimeoutError("The refreshed browser did not connect.")
                await asyncio.sleep(0.01)
    finally:
        await first.close()
        await second.close()
        await asyncio.sleep(0.05)
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

        window.open(replace(_session_desc(), metadata={"prompt": "replacement"}))
        window.write(
            StepResult(
                step_index=0,
                output=torch.full((2, 3, 16, 16), 211, dtype=torch.uint8),
                frame_count=2,
                output_layout=VideoTensorLayout.tchw,
                metrics={},
            )
        )
        # One frame may already be inside the encoder when the replacement
        # starts. The track discards the rest of the old source queue, so the
        # replacement must arrive immediately after that in-flight frame.
        replacement_pixels = None
        for _ in range(2):
            replacement_frame = await asyncio.wait_for(track.recv(), timeout=5)
            assert isinstance(replacement_frame, VideoFrame)
            pixels = replacement_frame.to_ndarray(format="rgb24")
            if float(pixels.mean()) > 100:
                replacement_pixels = pixels
                break
        assert replacement_pixels is not None
        assert abs(float(replacement_pixels.mean()) - 211.0) <= 2.0
        assert peer.connectionState == "connected"
    finally:
        if peer is not None:
            await peer.close()
            await asyncio.sleep(0.05)
        window.close()


def test_an_open_peer_keeps_its_media_format_across_sessions() -> None:
    window = WebRTCClientWindow()
    try:
        session_desc = _session_desc()
        window.open(session_desc)

        # UI polling is a runtime concern and does not change the media track.
        window.open(replace(session_desc, frames_per_second_for_ui=30))
        with pytest.raises(ValueError, match="original output"):
            window.open(replace(session_desc, frames_per_second_for_step=10))
    finally:
        window.close()


def test_closing_a_window_stops_its_server_thread() -> None:
    window = WebRTCClientWindow()

    window.close()
    window.close()

    assert not any(
        thread.name == "flashdreams-webrtc" for thread in threading.enumerate()
    )
