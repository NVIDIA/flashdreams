# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the v2 WebRTC client window."""

import asyncio
import json
from typing import Any, cast
from unittest.mock import ANY, Mock, call

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

from flashdreams.runtime_v2.serving import webrtc_server
from flashdreams.runtime_v2.serving.webrtc_server import (
    _FramePacer,
    _PendingRGBFrame,
    _VideoTrack,
)
from flashdreams.runtime_v2.session_desc import PresentationMode, SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    FocusUserInputEventData,
    KeyboardInputState,
    KeyboardUserInputEventData,
    MouseUserInputEventData,
)
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout
from flashdreams.runtime_v2.webrtc_client_window import WebRTCClientWindow


def _session_desc() -> SessionDesc:
    return SessionDesc(
        output_layout=VideoTensorLayout.tchw,
        presentation_mode=PresentationMode.ONLY_PRESENT_NEW,
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
async def test_window_buffers_browser_events_until_drained(monkeypatch: Any) -> None:
    logger = Mock()
    monkeypatch.setattr(webrtc_server, "logger", logger)
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

        window.open(_session_desc())
        peer, channel, _ = await _connect_browser(window)
        channel.send(json.dumps({"type": "keyboard", "key": "w", "pressed": True}))
        channel.send(json.dumps({"type": "keyboard", "key": "w", "pressed": False}))
        channel.send(
            json.dumps({"type": "mouse", "action": "move", "x": 0.25, "y": 0.75})
        )
        channel.send(json.dumps({"type": "focus", "focused": True}))

        events = []
        for _ in range(100):
            events.extend(window.get_user_input_events().get_events())
            if len(events) == 4:
                break
            await asyncio.sleep(0.01)

        assert len(events) == 4
        keyboard_events = [
            data
            for event in events
            if isinstance(data := event.get_event_data(), KeyboardUserInputEventData)
        ]
        assert [(event.key, event.state) for event in keyboard_events] == [
            ("w", KeyboardInputState.PRESSED),
            ("w", KeyboardInputState.RELEASED),
        ]
        assert (
            call(
                "WebRTC received keyboard event key={} state={} timestamp_us={}",
                "w",
                "Pressed",
                ANY,
            )
            in logger.info.call_args_list
        )
        assert (
            call(
                "WebRTC received keyboard event key={} state={} timestamp_us={}",
                "w",
                "Released",
                ANY,
            )
            in logger.info.call_args_list
        )
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


@pytest.mark.asyncio
async def test_video_track_resolves_pending_cuda_transfer_without_blocking() -> None:
    class FakeCUDAEvent:
        def __init__(self) -> None:
            self.queries = 0

        def query(self) -> bool:
            self.queries += 1
            return self.queries >= 3

    ready_event = FakeCUDAEvent()
    pending = _PendingRGBFrame(
        host_frames=torch.full((1, 16, 16, 3), 23, dtype=torch.uint8),
        frame_index=0,
        ready_event=cast(Any, ready_event),
    )
    track = _VideoTrack(frames_per_second=30)
    try:
        await track.enqueue((pending,))
        frame = await asyncio.wait_for(track.recv(), timeout=1)

        assert ready_event.queries == 3
        pixels = frame.to_ndarray(format="rgb24")
        assert pixels.shape == (16, 16, 3)
        assert abs(float(pixels.mean()) - 23.0) <= 2.0
    finally:
        await track.close()


@pytest.mark.asyncio
async def test_video_track_does_not_burst_to_catch_up_after_a_stall() -> None:
    track = _VideoTrack(frames_per_second=30)
    frame = torch.zeros((16, 16, 3), dtype=torch.uint8).numpy()
    try:
        await track.enqueue((frame, frame, frame))
        await track.recv()
        await asyncio.sleep(0.1)

        await track.recv()
        resumed_at = asyncio.get_running_loop().time()
        await track.recv()
        next_frame_at = asyncio.get_running_loop().time()

        assert next_frame_at - resumed_at >= 0.02
    finally:
        await track.close()


def test_video_track_pacer_does_not_accumulate_wakeup_delay() -> None:
    """Keep repeated scheduler overshoot out of subsequent frame deadlines."""
    pacer = _FramePacer(frames_per_second=60)
    now = 0.0
    sent_at: list[float] = []
    overshoot = 0.001

    for frame_index in range(240):
        delay = pacer.delay_seconds(
            now=now,
            source_at=frame_index / 120.0,
        )
        now += delay
        if delay > 0.0:
            now += overshoot
        sent_at.append(now)

    ideal_last_frame_at = (len(sent_at) - 1) / 60.0
    assert sent_at[-1] - ideal_last_frame_at == pytest.approx(overshoot)


@pytest.mark.asyncio
async def test_video_track_timestamps_sparse_frames_at_their_source_cadence() -> None:
    track = _VideoTrack(frames_per_second=30)
    frame = torch.zeros((16, 16, 3), dtype=torch.uint8).numpy()
    try:
        await track.enqueue((frame,))
        first = await track.recv()
        await asyncio.sleep(0.1)
        await track.enqueue((frame,))
        second = await track.recv()

        assert first.pts == 0
        assert second.pts is not None
        assert second.pts >= 2
    finally:
        await track.close()


@pytest.mark.asyncio
async def test_drop_oldest_video_track_keeps_the_latest_ui_frame() -> None:
    track = _VideoTrack(frames_per_second=60, drop_oldest=True)
    frames = tuple(
        torch.full((16, 16, 3), value, dtype=torch.uint8).numpy()
        for value in (10, 20, 30)
    )
    try:
        for frame in frames:
            await track.enqueue((frame,))

        assert track.qsize() == 1
        assert track.dropped_for_lag == 2
        latest = await track.recv()
        assert abs(float(latest.to_ndarray(format="rgb24").mean()) - 30.0) <= 2.0
    finally:
        await track.close()
