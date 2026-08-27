# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the v2 WebRTC client window."""

import asyncio
import json
from contextlib import suppress
from typing import Any, cast
from unittest.mock import ANY, AsyncMock, Mock, call

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
from aiortc.mediastreams import MediaStreamError
from av import VideoFrame

from flashdreams.runtime_v2.serving import webrtc_server
from flashdreams.runtime_v2.serving.webrtc_server import _VideoTrack
from flashdreams.runtime_v2.session_desc import PresentationMode, SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    FocusUserInputEvent,
    KeyboardInputState,
    KeyboardUserInputEvent,
    MouseUserInputEvent,
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


def _video_frame(value: int) -> VideoFrame:
    """Return one independently owned RGB frame filled with ``value``."""
    pixels = torch.full((16, 16, 3), value, dtype=torch.uint8).numpy()
    return VideoFrame.from_ndarray(pixels, format="rgb24")


def _frame_mean(frame: VideoFrame) -> float:
    """Return the mean RGB value of one video frame."""
    return float(frame.to_ndarray(format="rgb24").mean())


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
            event for event in events if isinstance(event, KeyboardUserInputEvent)
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
            event for event in events if isinstance(event, MouseUserInputEvent)
        )
        assert (mouse.action, mouse.x, mouse.y) == ("move", 0.25, 0.75)
        focus = next(
            event for event in events if isinstance(event, FocusUserInputEvent)
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
                output=torch.full((1, 3, 16, 16), 17, dtype=torch.uint8),
                frame_count=1,
                output_layout=VideoTensorLayout.tchw,
                metrics={},
            )
        )
        # WebRTC receivers may hold one decoded timestamp until the next RTP
        # timestamp arrives. Use a distinct second source frame, then verify
        # that the first was delivered rather than replaced.
        window.write(
            StepResult(
                step_index=1,
                output=torch.full((1, 3, 16, 16), 29, dtype=torch.uint8),
                frame_count=1,
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


def test_cpu_materialization_returns_an_independently_owned_video_frame() -> None:
    source = torch.full((3, 16, 16), 23, dtype=torch.uint8)

    materialized = webrtc_server._prepare_cpu_video_frame(source)
    source.fill_(99)

    assert _frame_mean(materialized) == 23.0


def test_window_write_materializes_before_synchronous_sender_admission() -> None:
    window = WebRTCClientWindow()
    captured: list[VideoFrame] = []
    track = Mock()
    track.enqueue.side_effect = lambda frame: captured.append(frame) or True
    source = torch.full((1, 3, 16, 16), 31, dtype=torch.uint8)
    try:
        window.open(_session_desc())
        window.server._video_track = cast(Any, track)
        window.server._media_connected.set()
        window.write(
            StepResult(
                step_index=0,
                output=source,
                frame_count=1,
                output_layout=VideoTensorLayout.tchw,
            )
        )

        source.fill_(99)
        track.enqueue.assert_called_once()
        assert [_frame_mean(frame) for frame in captured] == [31.0]
    finally:
        window.server._video_track = None
        window.server._media_connected.clear()
        window.close()


def test_window_write_queues_during_media_negotiation() -> None:
    window = WebRTCClientWindow()
    captured: list[VideoFrame] = []
    track = Mock()
    track.enqueue.side_effect = lambda frame: captured.append(frame) or True
    try:
        window.open(_session_desc())
        window.server._video_track = cast(Any, track)

        window.write(
            StepResult(
                step_index=0,
                output=torch.zeros((1, 3, 16, 16), dtype=torch.uint8),
                frame_count=1,
                output_layout=VideoTensorLayout.tchw,
            )
        )

        track.enqueue.assert_called_once()
        assert [_frame_mean(frame) for frame in captured] == [0.0]
    finally:
        window.server._video_track = None
        window.close()


def test_window_write_rejects_a_multi_frame_ui_result() -> None:
    window = WebRTCClientWindow()
    try:
        window.open(_session_desc())
        with pytest.raises(ValueError, match="exactly one UI-composited frame"):
            window.write(
                StepResult(
                    step_index=0,
                    output=torch.zeros((2, 3, 16, 16), dtype=torch.uint8),
                    frame_count=2,
                    output_layout=VideoTensorLayout.tchw,
                )
            )
    finally:
        window.close()


@pytest.mark.asyncio
async def test_video_track_waits_for_a_new_frame_instead_of_repeating_latest() -> None:
    track = _VideoTrack(frames_per_second=60)
    waiting: asyncio.Task[VideoFrame] | None = None
    try:
        track.enqueue(_video_frame(31))
        first = await asyncio.wait_for(track.recv(), timeout=1)
        assert track.qsize() == 0

        waiting = asyncio.create_task(track.recv())
        await asyncio.sleep(0)
        assert not waiting.done()
        track.enqueue(_video_frame(47))
        second = await asyncio.wait_for(waiting, timeout=1)

        assert first.pts == 0
        assert second.pts is not None and second.pts >= 1
        assert [_frame_mean(first), _frame_mean(second)] == [31.0, 47.0]
    finally:
        await track.close()
        if waiting is not None:
            with suppress(MediaStreamError):
                await waiting


@pytest.mark.asyncio
async def test_video_track_delivers_two_queued_frames_in_fifo_order() -> None:
    track = _VideoTrack(frames_per_second=60)
    try:
        track.enqueue(_video_frame(10))
        track.enqueue(_video_frame(20))

        assert track.qsize() == 2
        frames = [await track.recv(), await track.recv()]

        assert track.qsize() == 0
        assert [_frame_mean(frame) for frame in frames] == [10.0, 20.0]
    finally:
        await track.close()


@pytest.mark.asyncio
async def test_video_track_overflow_retains_the_two_newest_queued_frames() -> None:
    track = _VideoTrack(frames_per_second=60)
    try:
        track.enqueue(_video_frame(10))
        track.enqueue(_video_frame(20))
        track.enqueue(_video_frame(30))

        assert track.qsize() == 2
        retained = [await track.recv(), await track.recv()]
        assert [_frame_mean(frame) for frame in retained] == [20.0, 30.0]
    finally:
        await track.close()


@pytest.mark.asyncio
async def test_video_track_does_not_overwrite_a_dequeued_frame(
    monkeypatch: Any,
) -> None:
    track = _VideoTrack(frames_per_second=60)
    dequeued = asyncio.Event()
    continue_recv = asyncio.Event()
    original_next = track._next_queued_frame

    async def pause_after_dequeue() -> Any:
        queued = await original_next()
        dequeued.set()
        await continue_recv.wait()
        return queued

    monkeypatch.setattr(track, "_next_queued_frame", pause_after_dequeue)
    recv_task: asyncio.Task[VideoFrame] | None = None
    try:
        track.enqueue(_video_frame(10))
        recv_task = asyncio.create_task(track.recv())
        await asyncio.wait_for(dequeued.wait(), timeout=1)

        track.enqueue(_video_frame(20))
        track.enqueue(_video_frame(30))
        continue_recv.set()

        committed = await asyncio.wait_for(recv_task, timeout=1)
        retained = [await track.recv(), await track.recv()]
        assert [_frame_mean(frame) for frame in (committed, *retained)] == [
            10.0,
            20.0,
            30.0,
        ]
    finally:
        continue_recv.set()
        await track.close()
        if recv_task is not None:
            with suppress(MediaStreamError):
                await recv_task


@pytest.mark.asyncio
async def test_video_track_recv_has_no_sender_side_pacing(monkeypatch: Any) -> None:
    track = _VideoTrack(frames_per_second=1)
    sender_sleep = AsyncMock(
        side_effect=AssertionError("recv must not pace send-ready frames")
    )
    try:
        track.enqueue(_video_frame(10))
        track.enqueue(_video_frame(20))
        with monkeypatch.context() as patch:
            patch.setattr(webrtc_server.asyncio, "sleep", sender_sleep)
            frames = [await track.recv(), await track.recv()]

        sender_sleep.assert_not_awaited()
        assert [_frame_mean(frame) for frame in frames] == [10.0, 20.0]
    finally:
        await track.close()


@pytest.mark.asyncio
async def test_video_track_pts_follow_source_time_and_remain_monotonic(
    monkeypatch: Any,
) -> None:
    track = _VideoTrack(frames_per_second=30)
    source_time = 100.0
    try:
        with monkeypatch.context() as patch:
            patch.setattr(webrtc_server.time, "monotonic", lambda: source_time)
            track.enqueue(_video_frame(10))
            track.enqueue(_video_frame(20))
        first = await track.recv()
        second = await track.recv()

        source_time += 0.1
        with monkeypatch.context() as patch:
            patch.setattr(webrtc_server.time, "monotonic", lambda: source_time)
            track.enqueue(_video_frame(30))
        third = await track.recv()

        assert [first.pts, second.pts, third.pts] == [0, 1, 3]
    finally:
        await track.close()


@pytest.mark.asyncio
async def test_video_track_drain_signal_follows_the_next_sender_request() -> None:
    track = _VideoTrack(frames_per_second=60)
    waiting: asyncio.Task[VideoFrame] | None = None
    try:
        track.enqueue(_video_frame(10))
        assert not track.wait_until_drained(0.0)
        await track.recv()
        assert not track.wait_until_drained(0.0)

        waiting = asyncio.create_task(track.recv())
        await asyncio.sleep(0)
        assert track.wait_until_drained(0.0)
    finally:
        await track.close()
        if waiting is not None:
            with suppress(MediaStreamError):
                await waiting


@pytest.mark.asyncio
async def test_video_track_close_wakes_a_waiting_receiver() -> None:
    track = _VideoTrack(frames_per_second=60)
    receiver = asyncio.create_task(track.recv())
    await asyncio.sleep(0)

    await track.close()

    with pytest.raises(MediaStreamError):
        await receiver


def test_server_close_drains_without_inventing_a_frame() -> None:
    window = WebRTCClientWindow()
    track = Mock()
    track.wait_until_drained.return_value = True
    track.close = AsyncMock()
    window.server._video_track = cast(Any, track)
    window.server._media_connected.set()

    window.close()

    track.wait_until_drained.assert_called_once_with(
        webrtc_server._SHUTDOWN_DRAIN_TIMEOUT_SECONDS
    )
    track.enqueue.assert_not_called()
    track.close.assert_awaited_once()


def test_server_close_quarantines_storage_after_a_failed_cuda_drain() -> None:
    window = WebRTCClientWindow()
    stream = Mock()
    stream.synchronize.side_effect = RuntimeError("CUDA drain failed")
    buffer = window.server._materialization_buffer
    buffer._frame = torch.empty((16, 16, 3), dtype=torch.uint8)
    window.server._transfer_streams[0] = cast(Any, stream)

    try:
        with pytest.raises(RuntimeError, match="CUDA drain failed"):
            window.close()

        with webrtc_server._QUARANTINED_CUDA_TRANSFERS_LOCK:
            retained_streams, retained_buffer = (
                webrtc_server._QUARANTINED_CUDA_TRANSFERS.pop()
            )
        assert retained_streams == (stream,)
        assert retained_buffer is buffer
        assert buffer._frame is not None
    finally:
        buffer.close()
