# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone WebRTC server used by the v2 client window."""

from __future__ import annotations

import asyncio
import json
import math
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from importlib.resources import files
from typing import Any, TypeAlias

import numpy as np
import torch
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError
from av import VideoFrame
from loguru import logger

from flashdreams.runtime_v2.session_desc import PresentationMode, SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    CloseUserInputEventData,
    FocusUserInputEventData,
    KeyboardInputState,
    KeyboardUserInputEventData,
    MouseUserInputEventData,
    ResetUserInputEventData,
    UserInputEvent,
)
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

_WEB_RESOURCES = files("flashdreams.runtime_v2.serving").joinpath("web")
_BROWSER_PAGE = _WEB_RESOURCES.joinpath("index.html").read_text(encoding="utf-8")
_BROWSER_SCRIPT = _WEB_RESOURCES.joinpath("app.js").read_text(encoding="utf-8")

_CUDA_EVENT_POLL_SECONDS = 0.001
"""Polling interval that keeps CUDA waits off the WebRTC event-loop thread."""

_INTERACTIVE_FRAME_QUEUE_SIZE = 1
"""Pending sender frames retained for latest-frame presentation."""

_RGBArray: TypeAlias = np.ndarray[Any, np.dtype[np.uint8]]


@dataclass(frozen=True, slots=True)
class _PendingRGBFrame:
    """Pinned host frame whose asynchronous CUDA transfer is in flight."""

    host_frames: torch.Tensor
    """Pinned ``[T, H, W, C]`` uint8 storage shared by one result."""

    frame_index: int
    """Frame selected from ``host_frames`` after the transfer completes."""

    ready_event: torch.cuda.Event
    """Event recorded after the device-to-host transfer."""

    async def resolve(self) -> _RGBArray:
        """Wait without blocking the event loop and return the host array."""
        while not self.ready_event.query():
            await asyncio.sleep(_CUDA_EVENT_POLL_SECONDS)
        return np.asarray(self.host_frames[self.frame_index].numpy())


_QueuedRGBFrame: TypeAlias = _RGBArray | _PendingRGBFrame


@dataclass(frozen=True, slots=True)
class _PresentedRGBFrame:
    """One prepared frame with its io-thread presentation time."""

    frame: _QueuedRGBFrame
    """RGB pixels, possibly awaiting an asynchronous CUDA transfer."""

    presented_at: float
    """Event-loop timestamp at which the io-thread submitted this frame."""


class _FramePacer:
    """Pace source frames against drift-free absolute deadlines."""

    def __init__(self, frames_per_second: int) -> None:
        self._minimum_interval = 1.0 / frames_per_second
        self._last_source_at: float | None = None
        self._next_frame_at: float | None = None

    def delay_seconds(self, *, now: float, source_at: float) -> float:
        """Return the delay before presenting one source frame.

        Small scheduling overruns are recovered by the next absolute deadline
        instead of accumulating. A stall of at least one frame interval
        reanchors the schedule so queued frames are not emitted in a burst.
        """
        last_source_at = self._last_source_at
        next_frame_at = self._next_frame_at
        self._last_source_at = source_at
        if last_source_at is None or next_frame_at is None:
            self._next_frame_at = now
            return 0.0

        source_interval = max(
            self._minimum_interval,
            source_at - last_source_at,
        )
        next_frame_at += source_interval
        if now - next_frame_at >= self._minimum_interval:
            next_frame_at = now
        self._next_frame_at = next_frame_at
        return max(0.0, next_frame_at - now)


class _VideoTrack(MediaStreamTrack):
    """Video track whose frames are supplied by the server."""

    kind = "video"

    def __init__(self, frames_per_second: int, *, drop_oldest: bool = False) -> None:
        """Configure frame pacing and optional latest-frame delivery.

        Args:
            frames_per_second: RTP clock and maximum delivery rate.
            drop_oldest: Whether a newly presented frame replaces a queued one.
        """
        super().__init__()
        self._frames_per_second = frames_per_second
        self._frame_interval = 1.0 / frames_per_second
        self._time_base = Fraction(1, frames_per_second)
        self._drop_oldest = drop_oldest
        self._frames: asyncio.Queue[_PresentedRGBFrame | None] = asyncio.Queue(
            maxsize=_INTERACTIVE_FRAME_QUEUE_SIZE if drop_oldest else 0
        )
        self._retired_pending_frames: list[_PendingRGBFrame] = []
        self._dropped_for_lag = 0
        self._pacer = _FramePacer(frames_per_second)
        self._presentation_started_at: float | None = None
        self._next_pts = 0
        self._closed = False

    @property
    def dropped_for_lag(self) -> int:
        """Return the number of stale sender frames replaced before encoding."""
        return self._dropped_for_lag

    def qsize(self) -> int:
        """Return the number of frames waiting for the WebRTC sender."""
        return self._frames.qsize()

    async def enqueue(self, frames: tuple[_QueuedRGBFrame, ...]) -> None:
        """Append generated RGB frames for the WebRTC sender."""
        if self._closed:
            return
        self._reap_retired_pending_frames()
        presented_at = asyncio.get_running_loop().time()
        for frame_index, frame in enumerate(frames):
            presented_frame = _PresentedRGBFrame(
                frame=frame,
                presented_at=presented_at + frame_index * self._frame_interval,
            )
            if self._drop_oldest:
                self._drop_queued_frame()
                self._frames.put_nowait(presented_frame)
            else:
                await self._frames.put(presented_frame)

    async def recv(self) -> VideoFrame:
        """Return the next generated frame when aiortc requests one."""
        if self._closed:
            raise MediaStreamError
        presented_frame = await self._frames.get()
        if presented_frame is None:
            raise MediaStreamError
        self._reap_retired_pending_frames()
        queued_frame = presented_frame.frame
        frame = (
            await queued_frame.resolve()
            if isinstance(queued_frame, _PendingRGBFrame)
            else queued_frame
        )

        loop = asyncio.get_running_loop()
        now = loop.time()
        wait_seconds = self._pacer.delay_seconds(
            now=now,
            source_at=presented_frame.presented_at,
        )
        if wait_seconds > 0:
            await asyncio.sleep(wait_seconds)

        if self._presentation_started_at is None:
            self._presentation_started_at = presented_frame.presented_at
        elapsed = presented_frame.presented_at - self._presentation_started_at
        pts = max(self._next_pts, round(elapsed * self._frames_per_second))

        video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = self._time_base
        self._next_pts = pts + 1
        return video_frame

    async def close(self) -> None:
        """Stop the track and release a pending receiver."""
        if self._closed:
            return
        self._closed = True
        while True:
            try:
                queued = self._frames.get_nowait()
            except asyncio.QueueEmpty:
                break
            if queued is not None:
                self._retire_pending_frame(queued.frame)
        if self._retired_pending_frames:
            await asyncio.gather(
                *(frame.resolve() for frame in self._retired_pending_frames)
            )
            self._retired_pending_frames.clear()
        self._frames.put_nowait(None)
        self.stop()

    def _drop_queued_frame(self) -> None:
        if not self._frames.full():
            return
        try:
            stale = self._frames.get_nowait()
        except asyncio.QueueEmpty:
            return
        if stale is not None:
            self._retire_pending_frame(stale.frame)
            self._dropped_for_lag += 1

    def _retire_pending_frame(self, frame: _QueuedRGBFrame) -> None:
        if isinstance(frame, _PendingRGBFrame) and not frame.ready_event.query():
            self._retired_pending_frames.append(frame)

    def _reap_retired_pending_frames(self) -> None:
        self._retired_pending_frames = [
            frame
            for frame in self._retired_pending_frames
            if not frame.ready_event.query()
        ]


class WebRTCServer:
    """Own the HTTP, signaling, input buffering, and media transport."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        startup_timeout_seconds: float = 10.0,
    ) -> None:
        """
        Args:
            host: Interface on which the HTTP server listens.
            port: Listening port. Zero asks the operating system to choose one.
            startup_timeout_seconds: Maximum time to wait for server startup.

        Raises:
            RuntimeError: The server cannot start.
            TimeoutError: The server does not start before the timeout.
        """
        if not host:
            raise ValueError("host must not be empty.")
        if port < 0 or port > 65535:
            raise ValueError("port must be between 0 and 65535.")
        if startup_timeout_seconds <= 0:
            raise ValueError("startup_timeout_seconds must be > 0.")

        self._host = host
        self._port = port
        self._startup_timeout_seconds = startup_timeout_seconds
        self._input_callback: Callable[[UserInputEvent], None] | None = None
        self._started = threading.Event()
        self._startup_error: BaseException | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._runner: web.AppRunner | None = None
        self._peer_connection: RTCPeerConnection | None = None
        self._video_track: _VideoTrack | None = None
        self._session_desc: SessionDesc | None = None
        self._session_start_ns: int | None = None
        self._closed = False
        self._client_connected = False
        self._thread = threading.Thread(
            target=self._run_server,
            name="flashdreams-webrtc",
            daemon=True,
        )
        self._thread.start()
        if not self._started.wait(startup_timeout_seconds):
            raise TimeoutError("WebRTC server did not start before the timeout.")
        if self._startup_error is not None:
            raise RuntimeError(
                "WebRTC server failed to start."
            ) from self._startup_error

    @property
    def host(self) -> str:
        """Return the interface on which the server is listening."""
        return self._host

    @property
    def port(self) -> int:
        """Return the bound server port."""
        return self._port

    @property
    def url(self) -> str:
        """Return the browser URL for this server."""
        return f"http://{self._host}:{self._port}/"

    def open(self, session_desc: SessionDesc) -> None:
        """Configure the server for one session's generated video.

        Args:
            session_desc: Resolved dimensions, frame rate, and tensor layout.

        Raises:
            RuntimeError: The server is closed or already open.
        """
        if self._closed:
            raise RuntimeError("Cannot open a closed WebRTC server.")
        if self._session_desc is not None:
            raise RuntimeError("WebRTC server is already open.")
        if self._input_callback is None:
            raise RuntimeError("Register an input callback before opening WebRTC.")
        self._session_desc = session_desc
        self._session_start_ns = time.monotonic_ns()

    def register_input_callback(
        self, callback: Callable[[UserInputEvent], None]
    ) -> None:
        """Register the function called for each received browser event.

        Args:
            callback: Function that accepts one validated, timestamped event.

        Raises:
            RuntimeError: A callback has already been registered.
        """
        if self._input_callback is not None:
            raise RuntimeError("An input callback is already registered.")
        self._input_callback = callback

    def write(self, result: StepResult) -> None:
        """Deliver one generated result to the browser's video track.

        Args:
            result: Generated frames matching the description passed to
                :meth:`open`.

        Raises:
            RuntimeError: The server is not open or has been closed.
            ValueError: The result shape or layout does not match the session.
        """
        if self._closed:
            raise RuntimeError("Cannot write to a closed WebRTC server.")
        session_desc = self._session_desc
        if session_desc is None:
            raise RuntimeError("Open the WebRTC server before writing.")
        frames = _validated_result_frames(result, session_desc)
        if self._video_track is None:
            return
        queued_frames = _prepare_rgb_frames(frames)
        loop = self._loop
        if loop is None:
            raise RuntimeError("WebRTC server is not running.")
        future = asyncio.run_coroutine_threadsafe(
            self._enqueue_frames(queued_frames), loop
        )
        future.result()

    def close(self) -> None:
        """Close the peer connection and stop the WebRTC server."""
        if self._closed:
            return
        self._closed = True
        loop = self._loop
        if loop is None:
            return
        future = asyncio.run_coroutine_threadsafe(self._shutdown(), loop)
        future.result(timeout=self._startup_timeout_seconds)
        loop.call_soon_threadsafe(loop.stop)
        self._thread.join(timeout=self._startup_timeout_seconds)
        if self._thread.is_alive():
            raise TimeoutError("WebRTC server did not stop before the timeout.")

    def _run_server(self) -> None:
        """Own the WebRTC asyncio loop for the lifetime of the server."""
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._start_server())
        except BaseException as error:
            self._startup_error = error
            self._started.set()
            loop.close()
            return
        self._started.set()
        try:
            loop.run_forever()
        finally:
            loop.close()

    async def _start_server(self) -> None:
        """Create and bind the standalone aiohttp application."""
        app = web.Application()
        app.router.add_get("/", self._serve_browser)
        app.router.add_get("/app.js", self._serve_browser_script)
        app.router.add_get("/healthz", self._health)
        app.router.add_post("/api/webrtc/offer", self._offer)
        runner = web.AppRunner(app)
        await runner.setup()
        address_family = socket.AF_INET6 if ":" in self._host else socket.AF_INET
        server_socket = socket.socket(address_family, socket.SOCK_STREAM)
        try:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self._host, self._port))
            server_socket.setblocking(False)
            server_socket.listen(128)
            self._port = int(server_socket.getsockname()[1])
            site = web.SockSite(runner, server_socket)
            await site.start()
        except Exception:
            server_socket.close()
            await runner.cleanup()
            raise
        self._runner = runner

    async def _serve_browser(self, _: web.Request) -> web.Response:
        """Return the minimal browser client."""
        return web.Response(text=_BROWSER_PAGE, content_type="text/html")

    async def _serve_browser_script(self, _: web.Request) -> web.Response:
        """Return the browser client's JavaScript."""
        return web.Response(text=_BROWSER_SCRIPT, content_type="text/javascript")

    async def _health(self, _: web.Request) -> web.Response:
        """Report whether the server has an open session and client."""
        return web.json_response(
            {
                "open": self._session_desc is not None,
                "client_connected": self._client_connected,
            }
        )

    async def _offer(self, request: web.Request) -> web.Response:
        """Negotiate one browser peer connection."""
        if self._closed:
            raise web.HTTPServiceUnavailable(reason="WebRTC server is closed.")
        session_desc = self._session_desc
        if session_desc is None:
            raise web.HTTPConflict(reason="WebRTC server is not open.")
        if self._peer_connection is not None:
            raise web.HTTPConflict(reason="A WebRTC client is already connected.")

        try:
            payload = await request.json()
        except (json.JSONDecodeError, web.HTTPException) as error:
            raise web.HTTPBadRequest(reason="Expected a JSON WebRTC offer.") from error
        if not isinstance(payload, dict):
            raise web.HTTPBadRequest(reason="WebRTC offer must be an object.")
        sdp = payload.get("sdp")
        offer_type = payload.get("type")
        if not isinstance(sdp, str) or not isinstance(offer_type, str):
            raise web.HTTPBadRequest(
                reason="WebRTC offer requires string sdp and type."
            )

        peer_connection = RTCPeerConnection()
        video_track = _VideoTrack(
            session_desc.frames_per_second_for_ui,
            drop_oldest=(
                session_desc.presentation_mode is PresentationMode.ONLY_PRESENT_NEWEST
            ),
        )
        peer_connection.addTrack(video_track)
        self._peer_connection = peer_connection
        self._video_track = video_track

        @peer_connection.on("datachannel")
        def on_datachannel(channel: Any) -> None:
            self._client_connected = True

            @channel.on("message")
            def on_message(message: Any) -> None:
                try:
                    self._buffer_browser_message(message)
                except ValueError as error:
                    channel.send(json.dumps({"type": "error", "message": str(error)}))

            @channel.on("close")
            def on_close() -> None:
                self._record_client_disconnect()

        @peer_connection.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            if peer_connection.connectionState in {"failed", "disconnected", "closed"}:
                self._record_client_disconnect()

        try:
            await peer_connection.setRemoteDescription(
                RTCSessionDescription(sdp=sdp, type=offer_type)
            )
            await peer_connection.setLocalDescription(
                await peer_connection.createAnswer()
            )
        except Exception:
            self._peer_connection = None
            self._video_track = None
            await video_track.close()
            await peer_connection.close()
            raise

        local_description = peer_connection.localDescription
        if local_description is None:
            raise web.HTTPInternalServerError(
                reason="WebRTC peer did not create an answer."
            )
        return web.json_response(
            {"sdp": local_description.sdp, "type": local_description.type}
        )

    def _buffer_browser_message(self, raw_message: object) -> None:
        """Validate and append one data-channel message."""
        if not isinstance(raw_message, str):
            raise ValueError("Browser event must be a JSON string.")
        try:
            payload = json.loads(raw_message)
        except json.JSONDecodeError as error:
            raise ValueError("Browser event must contain valid JSON.") from error
        if not isinstance(payload, dict):
            raise ValueError("Browser event must be a JSON object.")

        event_type = payload.get("type")
        if event_type == "keyboard":
            key = payload.get("key")
            pressed = payload.get("pressed")
            if not isinstance(key, str) or not key:
                raise ValueError("Keyboard event requires a non-empty key.")
            if not isinstance(pressed, bool):
                raise ValueError("Keyboard event requires a boolean pressed value.")
            event_data = KeyboardUserInputEventData(
                key=key,
                state=(
                    KeyboardInputState.PRESSED
                    if pressed
                    else KeyboardInputState.RELEASED
                ),
            )
        elif event_type == "mouse":
            action = payload.get("action")
            if action not in {"move", "button", "wheel"}:
                raise ValueError(
                    "Mouse event action must be 'move', 'button', or 'wheel'."
                )
            x = _normalized_coordinate(payload.get("x"), label="Mouse x")
            y = _normalized_coordinate(payload.get("y"), label="Mouse y")
            button = payload.get("button", 0)
            pressed = payload.get("pressed", False)
            wheel_x = _finite_number(payload.get("wheel_x", 0.0), label="wheel_x")
            wheel_y = _finite_number(payload.get("wheel_y", 0.0), label="wheel_y")
            if isinstance(button, bool) or not isinstance(button, int) or button < 0:
                raise ValueError("Mouse button must be a non-negative integer.")
            if not isinstance(pressed, bool):
                raise ValueError("Mouse pressed must be a boolean.")
            event_data = MouseUserInputEventData(
                action=action,
                x=x,
                y=y,
                button=button,
                pressed=pressed,
                wheel_x=wheel_x,
                wheel_y=wheel_y,
            )
        elif event_type == "focus":
            focused = payload.get("focused")
            if not isinstance(focused, bool):
                raise ValueError("Focus event requires a boolean focused value.")
            event_data = FocusUserInputEventData(focused=focused)
        elif event_type == "reset":
            event_data = ResetUserInputEventData()
        elif event_type == "close":
            event_data = CloseUserInputEventData()
        else:
            raise ValueError("Unsupported browser event type.")
        self._append_event(event_data)

    def _append_event(
        self,
        event_data: (
            KeyboardUserInputEventData
            | MouseUserInputEventData
            | FocusUserInputEventData
            | ResetUserInputEventData
            | CloseUserInputEventData
        ),
    ) -> None:
        """Timestamp and buffer one validated browser event."""
        session_start_ns = self._session_start_ns
        if session_start_ns is None:
            return
        timestamp_us = np.uint64((time.monotonic_ns() - session_start_ns) // 1_000)
        event = UserInputEvent(timestamp=timestamp_us, event_data=event_data)
        if isinstance(event_data, KeyboardUserInputEventData):
            logger.info(
                "WebRTC received keyboard event key={} state={} timestamp_us={}",
                event_data.key,
                event_data.state.value,
                int(timestamp_us),
            )
        callback = self._input_callback
        if callback is None:
            raise RuntimeError("WebRTC input callback is not registered.")
        #  Pass that UserInputEvent to the callback.
        #  The callback stores it in WebRTCClientWindow’s thread-safe queue.
        callback(event)

    def _record_client_disconnect(self) -> None:
        """Buffer one close event when the active browser disconnects."""
        if not self._client_connected:
            return
        self._client_connected = False
        if not self._closed:
            self._append_event(CloseUserInputEventData())

    async def _enqueue_frames(self, frames: tuple[_QueuedRGBFrame, ...]) -> None:
        """Append frames to the active media track, if connected."""
        track = self._video_track
        if track is not None:
            await track.enqueue(frames)

    async def _shutdown(self) -> None:
        """Release async server resources on their owning loop."""
        peer_connection = self._peer_connection
        self._peer_connection = None
        track = self._video_track
        self._video_track = None
        if track is not None:
            await track.close()
        if peer_connection is not None:
            await peer_connection.close()
        runner = self._runner
        self._runner = None
        if runner is not None:
            await runner.cleanup()


def _finite_number(value: object, *, label: str) -> float:
    """Return a finite browser-input number."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite.")
    return result


def _normalized_coordinate(value: object, *, label: str) -> float:
    """Return a normalized browser pointer coordinate."""
    result = _finite_number(value, label=label)
    if result < 0.0 or result > 1.0:
        raise ValueError(f"{label} must be between 0 and 1.")
    return result


def _validated_result_frames(
    result: StepResult, session_desc: SessionDesc
) -> torch.Tensor:
    """Return validated time-major frames without materializing them on the host."""
    output = result.output.detach()
    if result.output_layout == VideoTensorLayout.tchw:
        frames = output
    elif result.output_layout == VideoTensorLayout.btchw:
        if output.ndim != 5 or output.shape[0] != 1:
            raise ValueError("btchw WebRTC output requires a batch size of one.")
        frames = output[0]
    elif result.output_layout == VideoTensorLayout.bcthw:
        if output.ndim != 5 or output.shape[0] != 1:
            raise ValueError("bcthw WebRTC output requires a batch size of one.")
        frames = output[0].permute(1, 0, 2, 3)
    elif result.output_layout == VideoTensorLayout.bvtchw:
        if output.ndim != 6 or output.shape[:2] != (1, 1):
            raise ValueError(
                "bvtchw WebRTC output requires one batch and one video view."
            )
        frames = output[0, 0]
    else:
        raise ValueError(f"Unsupported WebRTC output layout: {result.output_layout}.")

    if frames.ndim != 4:
        raise ValueError("WebRTC output must resolve to a tchw tensor.")
    if frames.shape[0] != result.frame_count:
        raise ValueError("StepResult.frame_count does not match its output tensor.")
    if frames.shape[1] not in (1, 3):
        raise ValueError("WebRTC output must have one or three color channels.")
    if frames.shape[2:] != (session_desc.video_height, session_desc.video_width):
        raise ValueError("WebRTC output dimensions do not match SessionDesc.")
    if result.output_layout != session_desc.output_layout:
        raise ValueError("StepResult.output_layout does not match SessionDesc.")

    return frames


def _rgb_uint8_thwc(frames: torch.Tensor) -> torch.Tensor:
    """Convert validated frames to contiguous ``[T, H, W, C]`` uint8 storage."""

    if frames.shape[1] == 1:
        frames = frames.repeat(1, 3, 1, 1)
    if frames.is_floating_point():
        frames = ((frames.to(torch.float32).clamp(-1.0, 1.0) + 1.0) * 127.5).round()
    frames = frames.clamp(0, 255).to(torch.uint8)
    return frames.permute(0, 2, 3, 1).contiguous()


def _prepare_rgb_frames(frames: torch.Tensor) -> tuple[_QueuedRGBFrame, ...]:
    """Prepare RGB frames without synchronizing the calling thread on CUDA work."""
    frames = _rgb_uint8_thwc(frames)
    if not frames.is_cuda:
        return tuple(np.asarray(frame.numpy()) for frame in frames.cpu())

    host_frames = torch.empty(
        frames.shape,
        dtype=torch.uint8,
        device="cpu",
        pin_memory=True,
    )
    host_frames.copy_(frames, non_blocking=True)
    ready_event = torch.cuda.Event()
    ready_event.record(torch.cuda.current_stream(frames.device))
    return tuple(
        _PendingRGBFrame(
            host_frames=host_frames,
            frame_index=frame_index,
            ready_event=ready_event,
        )
        for frame_index in range(frames.shape[0])
    )
