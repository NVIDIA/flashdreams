# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the v2 WebRTC client window."""

# ruff: noqa: E402 - optional WebRTC imports must follow importorskip.

import asyncio
import json
import time
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
import torch
from numpy import uint64

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
from flashdreams.runtime_v2.step_result import InputEventTrace, StepResult
from flashdreams.runtime_v2.user_input_event import (
    CloseUserInputEvent,
    FocusUserInputEvent,
    KeyboardInputState,
    KeyboardUserInputEvent,
    MouseUserInputEvent,
)
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout
from flashdreams.runtime_v2.webrtc_client_window import WebRTCClientWindow


def _session_desc(
    presentation_mode: PresentationMode = PresentationMode.ON_DEMAND,
) -> SessionDesc:
    return SessionDesc(
        output_layout=VideoTensorLayout.tchw,
        presentation_mode=presentation_mode,
        frames_per_second_for_ui=30,
        frames_per_second_for_step=30,
        video_width=16,
        video_height=16,
    )


def _video_frame(value: int = 0) -> VideoFrame:
    """Return one independently owned RGB frame filled with ``value``."""
    pixels = torch.full((16, 16, 3), value, dtype=torch.uint8).numpy()
    return VideoFrame.from_ndarray(pixels, format="rgb24")


def _frame_mean(frame: VideoFrame) -> float:
    """Return the mean RGB value of one video frame."""
    return float(frame.to_ndarray(format="rgb24").mean())


class _BufferedControlChannel:
    """Small data-channel stand-in for sender-backlog tests."""

    readyState = "open"

    def __init__(self, buffered_amount: int) -> None:
        self.bufferedAmount = buffered_amount
        self.sent: list[str] = []

    def send(self, message: str) -> None:
        self.sent.append(message)


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


def test_window_coalesces_consecutive_pointer_moves_before_polling() -> None:
    window = WebRTCClientWindow()
    callback = window.server._input_callback
    assert callback is not None
    try:
        callback(MouseUserInputEvent(timestamp=uint64(1), x=0.1, y=0.2))
        callback(MouseUserInputEvent(timestamp=uint64(2), x=0.7, y=0.8))

        events = window.get_user_input_events().get_events()

        assert len(events) == 1
        assert isinstance(events[0], MouseUserInputEvent)
        assert (events[0].x, events[0].y) == (0.7, 0.8)
    finally:
        window.close()


def test_window_preserves_traced_pointer_moves_before_polling() -> None:
    window = WebRTCClientWindow()
    callback = window.server._input_callback
    assert callback is not None
    try:
        callback(
            MouseUserInputEvent(timestamp=uint64(1), x=0.1, y=0.2, event_id="page:1")
        )
        callback(
            MouseUserInputEvent(timestamp=uint64(2), x=0.7, y=0.8, event_id="page:2")
        )

        events = window.get_user_input_events().get_events()

        assert [event.event_id for event in events] == ["page:1", "page:2"]
    finally:
        window.close()


@pytest.mark.asyncio
async def test_window_buffers_browser_events_until_drained() -> None:
    window = WebRTCClientWindow()
    assert window.metrics_snapshot() == {
        "webrtc_sender_queue_depth_count": 0,
        "webrtc_sender_queue_capacity_count": 2,
        "webrtc_sender_enqueued_count": 0,
        "webrtc_sender_handed_off_count": 0,
        "webrtc_sender_dropped_for_lag_count": 0,
        "webrtc_sender_discarded_on_close_count": 0,
        "webrtc_sender_oldest_queue_age_s": 0.0,
        "webrtc_sender_materialized_count": 0,
    }
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
                assert 'id="input-latency"' not in browser_page
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
                assert "event_id" in browser_script
                assert "browser_event_at_ms" not in browser_script
                assert "requestVideoFrameCallback" in browser_script
                assert "expectedDisplayTime" in browser_script
                assert "input_frame" in browser_script
                assert "input_frame_dropped" in browser_script
                assert "input_trace_reset" in browser_script
                assert "failPendingInputs" in browser_script
                assert "pressedKeys.has(keyId)" in browser_script
                assert "pressedKeys.set(keyId, event.key)" in browser_script
                assert "MAX_NONCRITICAL_BUFFER_BYTES" in browser_script
                assert 'createDataChannel("pointer-controls")' in browser_script
                assert "channel: pointerControls" in browser_script
                assert "requestAnimationFrame" in browser_script
                assert "latency_report" not in browser_script
                assert "latencyPanel" not in browser_script
                assert "renderLatencyPanel" not in browser_script
                connection_handler = browser_script.split(
                    'peer.addEventListener("connectionstatechange"',
                    maxsplit=1,
                )[1].split('window.addEventListener("keydown"', maxsplit=1)[0]
                assert "status.hidden = true" in connection_handler
                assert '"disconnected"' not in connection_handler
                assert '["failed", "closed"]' in connection_handler

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
async def test_peer_state_distinguishes_recovery_from_terminal_close(
    monkeypatch: Any,
) -> None:
    handlers: dict[str, Any] = {}

    def capture_handler(event_name: str) -> Any:
        def register(callback: Any) -> Any:
            handlers[event_name] = callback
            return callback

        return register

    peer = Mock()
    peer.connectionState = "new"
    peer.localDescription = RTCSessionDescription(sdp="v=0\r\n", type="answer")
    peer.on.side_effect = capture_handler
    peer.setRemoteDescription = AsyncMock()
    peer.createAnswer = AsyncMock(return_value=peer.localDescription)
    peer.setLocalDescription = AsyncMock()
    peer.close = AsyncMock()
    monkeypatch.setattr(webrtc_server, "RTCPeerConnection", Mock(return_value=peer))
    window = WebRTCClientWindow()
    try:
        window.open(_session_desc())
        async with ClientSession() as client:
            async with client.post(
                f"{window.server.url}api/webrtc/offer",
                json={"sdp": "v=0\r\n", "type": "offer"},
            ) as response:
                assert response.status == 200

        window.server._client_connected = True

        async def transition(connection_state: str) -> None:
            peer.connectionState = connection_state
            await handlers["connectionstatechange"]()

        server_loop = window.server._loop
        assert server_loop is not None

        for connection_state in ("connected", "disconnected", "connected"):
            future = asyncio.run_coroutine_threadsafe(
                transition(connection_state),
                server_loop,
            )
            await asyncio.wrap_future(future)

        assert window.server._media_connected.is_set()
        assert window.server._client_connected
        assert window.get_user_input_events().get_events() == []

        for connection_state in ("failed", "closed"):
            future = asyncio.run_coroutine_threadsafe(
                transition(connection_state),
                server_loop,
            )
            await asyncio.wrap_future(future)

        assert not window.server._media_connected.is_set()
        assert not window.server._client_connected
        events = window.get_user_input_events().get_events()
        assert len(events) == 1
        assert isinstance(events[0], CloseUserInputEvent)
    finally:
        window.close()


@pytest.mark.asyncio
async def test_browser_keyboard_trace_id_survives_the_data_channel() -> None:
    window = WebRTCClientWindow()
    peer: RTCPeerConnection | None = None
    try:
        window.open(_session_desc())
        peer, channel, _ = await _connect_browser(window)
        channel.send(
            json.dumps(
                {
                    "type": "keyboard",
                    "key": "w",
                    "pressed": True,
                    "event_id": "keyboard-w-1",
                }
            )
        )

        events = []
        for _ in range(100):
            events.extend(window.get_user_input_events().get_events())
            if events:
                break
            await asyncio.sleep(0.01)
        assert len(events) == 1
        event = events[0]
        assert event.event_id == "keyboard-w-1"
        assert isinstance(event, KeyboardUserInputEvent)
        assert (event.key, event.state) == (
            "w",
            KeyboardInputState.PRESSED,
        )
    finally:
        if peer is not None:
            await peer.close()
        window.close()


@pytest.mark.asyncio
async def test_video_track_emits_markers_only_for_traced_frames() -> None:
    markers: list[dict[str, object]] = []
    track = _VideoTrack(frames_per_second=30, on_frame_marker=markers.append)
    traces = (
        InputEventTrace(
            event_id="keyboard-w-1",
            frame_index=0,
        ),
        InputEventTrace(
            event_id="keyboard-d-2",
            frame_index=0,
        ),
    )
    try:
        track.enqueue(_video_frame())
        first = await track.recv()
        track.enqueue(_video_frame(), traces)
        second = await track.recv()
        track.enqueue(_video_frame())
        third = await track.recv()
        presented = [first, second, third]

        assert [item.pts for item in presented] == [0, 1, 2]
        assert len(markers) == 1
        assert markers[0]["type"] == "input_frame"
        assert markers[0]["frame_id"] == 1
        assert markers[0]["frame_pts"] == 1
        assert markers[0]["source_rtp_timestamp"] == 3000

        traced = cast(list[dict[str, object]], markers[0]["traces"])
        assert [trace["event_id"] for trace in traced] == [
            "keyboard-w-1",
            "keyboard-d-2",
        ]
        assert traced == [
            {"event_id": "keyboard-w-1"},
            {"event_id": "keyboard-d-2"},
        ]
    finally:
        await track.close()


@pytest.mark.asyncio
async def test_traced_window_write_sends_an_input_frame_marker() -> None:
    window = WebRTCClientWindow()
    peer: RTCPeerConnection | None = None
    marker_messages: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    try:
        window.open(_session_desc())
        peer, channel, _ = await _connect_browser(window)

        @channel.on("message")
        def on_message(message: object) -> None:
            if not isinstance(message, str):
                return
            payload = json.loads(message)
            if payload.get("type") == "input_frame":
                marker_messages.put_nowait(payload)

        trace = InputEventTrace(
            event_id="keyboard-w-1",
            frame_index=0,
        )
        window.write(
            StepResult(
                step_index=0,
                output=torch.full((1, 3, 16, 16), 17, dtype=torch.uint8),
                frame_count=1,
                output_layout=VideoTensorLayout.tchw,
                input_event_traces=(trace,),
            )
        )

        marker = await asyncio.wait_for(marker_messages.get(), timeout=5)
        assert marker["frame_id"] == 0
        assert marker["frame_pts"] == 0
        assert marker["source_rtp_timestamp"] == 0
        traces = cast(list[dict[str, object]], marker["traces"])
        assert traces == [{"event_id": "keyboard-w-1"}]
    finally:
        if peer is not None:
            await peer.close()
        window.close()


def test_browser_event_ids_must_be_nonempty_and_unique() -> None:
    window = WebRTCClientWindow()
    try:
        window.open(_session_desc())
        with pytest.raises(ValueError, match="event_id"):
            window.server._buffer_browser_message(
                json.dumps(
                    {
                        "type": "keyboard",
                        "key": "w",
                        "pressed": True,
                        "event_id": "",
                    }
                )
            )
        tagged = {
            "type": "keyboard",
            "key": "w",
            "pressed": True,
            "event_id": "keyboard-w-1",
        }
        window.server._buffer_browser_message(json.dumps(tagged))
        with pytest.raises(ValueError, match="unique"):
            window.server._buffer_browser_message(json.dumps(tagged))
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
                output=torch.full((1, 3, 16, 16), 17, dtype=torch.uint8),
                frame_count=1,
                output_layout=VideoTensorLayout.tchw,
                metrics={},
            )
        )
        deadline = time.monotonic() + 5
        while window.metrics_snapshot()["webrtc_sender_handed_off_count"] < 1:
            assert time.monotonic() < deadline
            await asyncio.sleep(0.01)
        # Some codecs retain their first input until a subsequent RTP timestamp
        # arrives. Use a distinct second source frame, then verify the first was
        # delivered rather than replaced.
        window.write(
            StepResult(
                step_index=1,
                output=torch.full((1, 3, 16, 16), 29, dtype=torch.uint8),
                frame_count=1,
                output_layout=VideoTensorLayout.tchw,
                metrics={},
            )
        )
        sender_metrics = window.metrics_snapshot()
        assert sender_metrics["webrtc_sender_queue_depth_count"] >= 0
        assert sender_metrics["webrtc_sender_queue_capacity_count"] == 2
        assert sender_metrics["webrtc_sender_enqueued_count"] == 2
        assert sender_metrics["webrtc_sender_handed_off_count"] >= 1
        assert sender_metrics["webrtc_sender_dropped_for_lag_count"] == 0
        assert sender_metrics["webrtc_sender_oldest_queue_age_s"] >= 0.0

        frame = await asyncio.wait_for(track.recv(), timeout=5)
        assert isinstance(frame, VideoFrame)
        pixels = frame.to_ndarray(format="rgb24")
        assert pixels.shape == (16, 16, 3)
        assert abs(float(pixels.mean()) - 17.0) <= 2.0
    finally:
        if peer is not None:
            await peer.close()
        window.close()


@pytest.mark.parametrize("presentation_mode", list(PresentationMode))
@pytest.mark.asyncio
async def test_webrtc_always_configures_a_bounded_two_frame_sender_queue(
    presentation_mode: PresentationMode,
) -> None:
    window = WebRTCClientWindow()
    peer: RTCPeerConnection | None = None
    try:
        window.open(_session_desc(presentation_mode))
        peer, _, _ = await _connect_browser(window)

        track = window.server._video_track
        assert track is not None
        assert track.metrics_snapshot()["webrtc_sender_queue_capacity_count"] == 2
    finally:
        if peer is not None:
            await peer.close()
        window.close()


def test_cpu_materialization_returns_an_independently_owned_video_frame() -> None:
    source = torch.full((3, 16, 16), 10, dtype=torch.uint8)

    materialized = webrtc_server._prepare_cpu_video_frame(source)
    source.fill_(99)

    assert _frame_mean(materialized) == 10.0


def test_window_write_materializes_before_synchronous_sender_admission(
    monkeypatch: Any,
) -> None:
    window = WebRTCClientWindow()
    captured: list[VideoFrame] = []
    track = Mock()
    track.enqueue.side_effect = lambda frame, _traces=(): captured.append(frame) or True
    source = torch.full((1, 3, 16, 16), 31, dtype=torch.uint8)
    try:
        window.open(_session_desc())
        window.server._video_track = cast(Any, track)
        window.server._media_connected.set()
        with monkeypatch.context() as patch:
            patch.setattr(
                webrtc_server.asyncio,
                "run_coroutine_threadsafe",
                Mock(side_effect=AssertionError("write must not wait on WebRTC")),
            )
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
        assert len(captured) == 1
        assert _frame_mean(captured[0]) == 31.0
    finally:
        window.server._video_track = None
        window.server._media_connected.clear()
        window.close()


def test_window_write_queues_during_media_negotiation() -> None:
    window = WebRTCClientWindow()
    captured: list[VideoFrame] = []
    track = Mock()
    track.enqueue.side_effect = lambda frame, _traces=(): captured.append(frame) or True
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


def test_window_write_reports_a_trace_rejected_during_track_close(
    monkeypatch: Any,
) -> None:
    window = WebRTCClientWindow()
    track = Mock()
    track.enqueue.return_value = False
    report_discarded = Mock()
    try:
        window.open(_session_desc())
        window.server._video_track = cast(Any, track)
        monkeypatch.setattr(
            window.server,
            "report_discarded_input_events",
            report_discarded,
        )

        window.write(
            StepResult(
                step_index=0,
                output=torch.zeros((1, 3, 16, 16), dtype=torch.uint8),
                frame_count=1,
                output_layout=VideoTensorLayout.tchw,
                input_event_traces=(InputEventTrace(event_id="page:1", frame_index=0),),
            )
        )

        report_discarded.assert_called_once_with(("page:1",))
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
        assert track.metrics_snapshot()["webrtc_sender_queue_depth_count"] == 0

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
        if waiting is not None and not waiting.done():
            with pytest.raises(MediaStreamError):
                await waiting


@pytest.mark.asyncio
async def test_video_track_delivers_two_queued_frames_in_fifo_order() -> None:
    track = _VideoTrack(frames_per_second=60)
    try:
        track.enqueue(_video_frame(10))
        track.enqueue(_video_frame(20))

        assert track.metrics_snapshot()["webrtc_sender_queue_depth_count"] == 2
        first = await track.recv()
        second = await track.recv()

        assert [_frame_mean(first), _frame_mean(second)] == [10.0, 20.0]
        metrics = track.metrics_snapshot()
        assert metrics["webrtc_sender_queue_depth_count"] == 0
        assert metrics["webrtc_sender_dropped_for_lag_count"] == 0
    finally:
        await track.close()


@pytest.mark.asyncio
async def test_video_track_overflow_retains_the_two_newest_frames() -> None:
    track = _VideoTrack(frames_per_second=60)
    try:
        track.enqueue(_video_frame(10))
        track.enqueue(_video_frame(20))
        track.enqueue(_video_frame(30))

        pending = track.metrics_snapshot()
        assert pending["webrtc_sender_queue_depth_count"] == 2
        assert pending["webrtc_sender_dropped_for_lag_count"] == 1
        retained = [await track.recv(), await track.recv()]

        assert [_frame_mean(frame) for frame in retained] == [20.0, 30.0]
        metrics = track.metrics_snapshot()
        assert metrics["webrtc_sender_enqueued_count"] == 3
        assert metrics["webrtc_sender_handed_off_count"] == 2
        assert metrics["webrtc_sender_dropped_for_lag_count"] == 1
    finally:
        await track.close()


@pytest.mark.asyncio
async def test_video_track_commits_a_frame_once_dequeued(monkeypatch: Any) -> None:
    track = _VideoTrack(frames_per_second=60)
    dequeued = asyncio.Event()
    continue_recv = asyncio.Event()
    original_next = track._next_queued_frame

    async def pause_after_dequeue() -> Any:
        presented = await original_next()
        dequeued.set()
        await continue_recv.wait()
        return presented

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
        assert track.metrics_snapshot()["webrtc_sender_dropped_for_lag_count"] == 0
    finally:
        continue_recv.set()
        if recv_task is not None and not recv_task.done():
            recv_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await recv_task
        await track.close()


@pytest.mark.asyncio
async def test_video_track_recv_has_no_sender_side_pacing(
    monkeypatch: Any,
) -> None:
    track = _VideoTrack(frames_per_second=1)
    sender_sleep = AsyncMock(
        side_effect=AssertionError("recv must not pace send-ready frames")
    )
    try:
        track.enqueue(_video_frame(10))
        track.enqueue(_video_frame(20))
        with monkeypatch.context() as patch:
            patch.setattr(webrtc_server.asyncio, "sleep", sender_sleep)
            first = await track.recv()
            second = await track.recv()

        sender_sleep.assert_not_awaited()
        assert [_frame_mean(first), _frame_mean(second)] == [10.0, 20.0]
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
async def test_video_track_overflow_emits_markers_and_drop_metrics() -> None:
    markers: list[dict[str, object]] = []
    track = _VideoTrack(frames_per_second=60, on_frame_marker=markers.append)

    def trace(event_id: str) -> InputEventTrace:
        return InputEventTrace(
            event_id=event_id,
            frame_index=0,
        )

    try:
        track.enqueue(_video_frame(10), (trace("keyboard-a-1"),))
        track.enqueue(_video_frame(20), (trace("keyboard-b-2"),))
        track.enqueue(_video_frame(30), (trace("keyboard-c-3"),))
        await asyncio.sleep(0)

        assert markers == [
            {
                "type": "input_frame_dropped",
                "event_ids": ["keyboard-a-1"],
            }
        ]
        await track.recv()
        await track.recv()

        assert [marker["type"] for marker in markers] == [
            "input_frame_dropped",
            "input_frame",
            "input_frame",
        ]
        assert [
            cast(list[dict[str, object]], marker["traces"])[0]["event_id"]
            for marker in markers[1:]
        ] == ["keyboard-b-2", "keyboard-c-3"]
        metrics = track.metrics_snapshot()
        assert metrics["webrtc_sender_queue_depth_count"] == 0
        assert metrics["webrtc_sender_queue_capacity_count"] == 2
        assert metrics["webrtc_sender_enqueued_count"] == 3
        assert metrics["webrtc_sender_handed_off_count"] == 2
        assert metrics["webrtc_sender_dropped_for_lag_count"] == 1
        assert metrics["webrtc_sender_discarded_on_close_count"] == 0
    finally:
        await track.close()


@pytest.mark.asyncio
async def test_video_track_coalesces_wakeups_and_bounds_drop_markers() -> None:
    markers: list[dict[str, object]] = []
    sender_loop = Mock()
    track = _VideoTrack(frames_per_second=60, on_frame_marker=markers.append)
    track._sender_loop = cast(Any, sender_loop)

    for frame_index in range(300):
        trace = InputEventTrace(
            event_id=f"keyboard-{frame_index}",
            frame_index=0,
        )
        track.enqueue(_video_frame(frame_index % 256), (trace,))

    sender_loop.call_soon_threadsafe.assert_called_once()
    pending = track.metrics_snapshot()
    assert pending["webrtc_sender_queue_depth_count"] == 2
    assert pending["webrtc_sender_dropped_for_lag_count"] == 298
    deferred_wake = sender_loop.call_soon_threadsafe.call_args.args[0]

    await track.close()

    # Overflow expires all older browser trace state explicitly, then close
    # reports the two still-queued frames individually.
    assert markers == [
        {"type": "input_trace_reset"},
        {"type": "input_frame_dropped", "event_ids": ["keyboard-298"]},
        {"type": "input_frame_dropped", "event_ids": ["keyboard-299"]},
    ]
    assert track.metrics_snapshot()["webrtc_sender_discarded_on_close_count"] == 2

    deferred_wake()
    assert len(markers) == 3


@pytest.mark.asyncio
async def test_video_track_drain_signal_follows_the_next_sender_request() -> None:
    track = _VideoTrack(frames_per_second=60)
    assert track.wait_until_drained(0.0)

    track.enqueue(_video_frame(10))
    assert not track.wait_until_drained(0.0)
    await track.recv()
    assert not track.wait_until_drained(0.0)

    next_request = asyncio.create_task(track.recv())
    drained = asyncio.create_task(asyncio.to_thread(track.wait_until_drained, 1.0))
    assert await drained
    assert not next_request.done()

    track.enqueue(_video_frame(20))
    assert _frame_mean(await next_request) == 20.0
    assert not track.wait_until_drained(0.0)
    closed = asyncio.create_task(asyncio.to_thread(track.wait_until_drained, 1.0))
    await track.close()
    assert await closed


@pytest.mark.asyncio
async def test_video_track_close_wakes_a_waiting_receiver() -> None:
    track = _VideoTrack(frames_per_second=60)
    receiver = asyncio.create_task(track.recv())
    await asyncio.sleep(0)

    await track.close()

    with pytest.raises(MediaStreamError):
        await receiver


def test_frame_markers_wait_for_control_channel_backlog_to_clear() -> None:
    window = WebRTCClientWindow()
    channel = _BufferedControlChannel(
        webrtc_server._MAX_CONTROL_CHANNEL_BUFFER_BYTES + 1
    )
    window.server._control_channel = channel
    try:
        markers: list[dict[str, object]] = [
            {"type": "input_frame_dropped", "event_ids": ["page:1"]},
            {"type": "input_frame_dropped", "event_ids": ["page:2"]},
        ]
        for marker in markers:
            window.server._send_frame_marker(marker)

        assert channel.sent == []
        assert list(window.server._pending_frame_markers) == markers

        channel.bufferedAmount = 0
        window.server._flush_frame_markers()

        assert [json.loads(message) for message in channel.sent] == markers
        assert not window.server._pending_frame_markers
    finally:
        window.close()


def test_window_reports_frames_discarded_before_webrtc_admission() -> None:
    window = WebRTCClientWindow()
    channel = _BufferedControlChannel(0)
    window.server._control_channel = channel
    try:
        window.discard_input_event_ids(("page:1", "page:1", "page:2"))
        deadline = time.monotonic() + 1.0
        while not channel.sent:
            assert time.monotonic() < deadline
            time.sleep(0.01)

        assert json.loads(channel.sent[0]) == {
            "type": "input_frame_dropped",
            "event_ids": ["page:1", "page:2"],
        }
    finally:
        window.close()


def test_server_stops_after_an_async_cleanup_failure() -> None:
    window = WebRTCClientWindow()
    server = window.server
    runner = server._runner
    assert runner is not None

    class FailingRunner:
        async def cleanup(self) -> None:
            await runner.cleanup()
            raise RuntimeError("cleanup failed")

    server._runner = cast(Any, FailingRunner())
    with pytest.raises(RuntimeError, match="cleanup failed"):
        window.close()

    assert not server._thread.is_alive()
    assert server._loop is None


def test_server_close_drains_without_inventing_a_frame() -> None:
    window = WebRTCClientWindow()
    track = Mock()
    track_metrics = {
        "webrtc_sender_queue_depth_count": 0,
        "webrtc_sender_queue_capacity_count": 2,
        "webrtc_sender_enqueued_count": 3,
        "webrtc_sender_handed_off_count": 3,
        "webrtc_sender_dropped_for_lag_count": 0,
        "webrtc_sender_discarded_on_close_count": 0,
        "webrtc_sender_oldest_queue_age_s": 0.0,
    }
    track.metrics_snapshot.return_value = track_metrics
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
    assert window.metrics_snapshot() == {
        **track_metrics,
        "webrtc_sender_materialized_count": 0,
    }


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


@pytest.mark.asyncio
async def test_video_track_close_reports_both_discarded_queued_frames() -> None:
    markers: list[dict[str, object]] = []
    track = _VideoTrack(frames_per_second=60, on_frame_marker=markers.append)
    traces = tuple(
        InputEventTrace(
            event_id=event_id,
            frame_index=0,
        )
        for event_id in ("keyboard-a-1", "keyboard-b-2")
    )

    track.enqueue(_video_frame(10), (traces[0],))
    track.enqueue(_video_frame(20), (traces[1],))
    assert track.metrics_snapshot()["webrtc_sender_queue_depth_count"] == 2

    await track.close()

    assert track.metrics_snapshot()["webrtc_sender_queue_depth_count"] == 0
    metrics = track.metrics_snapshot()
    assert metrics["webrtc_sender_discarded_on_close_count"] == 2
    assert metrics["webrtc_sender_dropped_for_lag_count"] == 0
    assert markers == [
        {
            "type": "input_frame_dropped",
            "event_ids": ["keyboard-a-1"],
        },
        {
            "type": "input_frame_dropped",
            "event_ids": ["keyboard-b-2"],
        },
    ]
    with pytest.raises(MediaStreamError):
        await track.recv()
