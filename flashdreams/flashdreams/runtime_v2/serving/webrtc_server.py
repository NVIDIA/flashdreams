# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone WebRTC server used by the v2 client window."""

from __future__ import annotations

import asyncio
import json
import math
import socket
import string
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from importlib.resources import files
from typing import Any, Literal

import numpy as np
import torch
from aiohttp import web
from aiortc import (
    MediaStreamTrack,
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.mediastreams import MediaStreamError
from av import VideoFrame

from flashdreams.api_v2.user_input_event_data import UserInputEventData
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    ActivationUserInputEventData,
    CloseUserInputEventData,
    KeyboardUserInputEventData,
    MouseUserInputEventData,
    ResetUserInputEventData,
    UserInputEvent,
)
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

_WEB_RESOURCES = files("flashdreams.runtime_v2.serving").joinpath("web")
_BROWSER_PAGE = _WEB_RESOURCES.joinpath("index.html").read_text(encoding="utf-8")
_BROWSER_SCRIPT = _WEB_RESOURCES.joinpath("app.js").read_text(encoding="utf-8")


@dataclass(frozen=True, slots=True)
class WebRTCServerConfig:
    """Optional WebRTC behavior beyond the backward-compatible browser path."""

    capability_path: str = ""
    """Optional unguessable URL path segment protecting every endpoint."""
    max_pending_frames: int | None = None
    """Maximum unsent frames, or ``None`` to preserve every generated frame."""
    ice_servers: tuple[RTCIceServer, ...] = ()
    """STUN or TURN servers used by both WebRTC peers."""
    browser_ice_transport_policy: Literal["all", "relay"] = "all"
    """Browser ICE transport policy."""
    allow_reconnect: bool = False
    """Keep the application alive when its browser disconnects."""
    pointer_lock_controls: bool = False
    """Enable relative mouse input and explicit control activation."""

    def __post_init__(self) -> None:
        """Validate one immutable server configuration."""
        allowed_path_characters = string.ascii_letters + string.digits + "-_"
        if self.capability_path and any(
            character not in allowed_path_characters for character in self.capability_path
        ):
            raise ValueError("capability_path must be one URL-safe path segment.")
        if self.max_pending_frames is not None and self.max_pending_frames <= 0:
            raise ValueError("max_pending_frames must be > 0 when provided.")
        if self.browser_ice_transport_policy not in {"all", "relay"}:
            raise ValueError("browser_ice_transport_policy must be all or relay.")


def _closed_connection_status(*, connection_state: str, client_connected: bool) -> str:
    """Classify a closed peer without hiding failed initial negotiation."""
    if connection_state == "failed" or not client_connected:
        return "failed"
    return "disconnected"


class _VideoTrack(MediaStreamTrack):
    """Video track whose frames are supplied by the server."""

    kind = "video"

    def __init__(self, frames_per_second: int, max_pending_frames: int | None) -> None:
        super().__init__()
        self._frames_per_second = frames_per_second
        self._time_base = Fraction(1, frames_per_second)
        self._frames: asyncio.Queue[np.ndarray[Any, np.dtype[np.uint8]] | None] = asyncio.Queue(
            maxsize=max_pending_frames or 0
        )
        self._next_frame_time: float | None = None
        self._pts = 0
        self._closed = False

    async def enqueue(self, frames: tuple[np.ndarray[Any, np.dtype[np.uint8]], ...]) -> None:
        """Append generated RGB frames for the WebRTC sender."""
        if self._closed:
            return
        for frame in frames:
            while self._frames.full():
                self._frames.get_nowait()
            self._frames.put_nowait(frame)

    async def recv(self) -> VideoFrame:
        """Return the next generated frame when aiortc requests one."""
        if self._closed:
            raise MediaStreamError
        frame = await self._frames.get()
        if frame is None:
            raise MediaStreamError

        loop = asyncio.get_running_loop()
        now = loop.time()
        if self._next_frame_time is None:
            self._next_frame_time = now
        else:
            self._next_frame_time += 1.0 / self._frames_per_second
            await asyncio.sleep(max(0.0, self._next_frame_time - now))

        video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        video_frame.pts = self._pts
        video_frame.time_base = self._time_base
        self._pts += 1
        return video_frame

    async def close(self) -> None:
        """Stop the track and release a pending receiver."""
        if self._closed:
            return
        self._closed = True
        while not self._frames.empty():
            self._frames.get_nowait()
        self._frames.put_nowait(None)
        self.stop()


class WebRTCServer:
    """Own the HTTP, signaling, input buffering, and media transport."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        startup_timeout_seconds: float = 10.0,
        config: WebRTCServerConfig | None = None,
    ) -> None:
        """
        Args:
            host: Interface on which the HTTP server listens.
            port: Listening port. Zero asks the operating system to choose one.
            startup_timeout_seconds: Maximum time to wait for server startup.
            config: Optional security, transport, and interactive behavior.

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
        self._config = config or WebRTCServerConfig()
        self._input_callback: Callable[[UserInputEvent], None] | None = None
        self._connection_status_callback: Callable[[str, dict[str, str]], None] | None = None
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
        self._client_id: str | None = None
        self._client_remote_address: str | None = None
        self._last_connection_failure: str | None = None
        self._disconnect_tasks: set[asyncio.Task[None]] = set()
        self._thread = threading.Thread(
            target=self._run_server,
            name="flashdreams-webrtc",
            daemon=True,
        )
        self._thread.start()
        if not self._started.wait(startup_timeout_seconds):
            raise TimeoutError("WebRTC server did not start before the timeout.")
        if self._startup_error is not None:
            raise RuntimeError("WebRTC server failed to start.") from self._startup_error

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
        prefix = f"{self._config.capability_path}/" if self._config.capability_path else ""
        return f"http://{self._host}:{self._port}/{prefix}"

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

    def register_input_callback(self, callback: Callable[[UserInputEvent], None]) -> None:
        """Register the function called for each received browser event.

        Args:
            callback: Function that accepts one validated, timestamped event.

        Raises:
            RuntimeError: A callback has already been registered.
        """
        if self._input_callback is not None:
            raise RuntimeError("An input callback is already registered.")
        self._input_callback = callback

    def register_connection_status_callback(
        self,
        callback: Callable[[str, dict[str, str]], None],
    ) -> None:
        """Register one observer for browser signaling and transport state.

        The observer may be called from the server thread. Status reporting is
        separate from user input so applications can distinguish a runtime that
        is ready from a browser that is connected.

        Args:
            callback: Function accepting a stable status name and string fields.

        Raises:
            RuntimeError: A callback has already been registered.
        """
        if self._connection_status_callback is not None:
            raise RuntimeError("A connection status callback is already registered.")
        self._connection_status_callback = callback

    def _report_connection_status(self, status: str, **fields: str) -> None:
        """Report one browser connection transition to the registered observer."""
        callback = self._connection_status_callback
        if callback is not None:
            callback(status, fields)

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
        frames = _result_to_rgb_frames(result, session_desc)
        loop = self._loop
        if loop is None:
            raise RuntimeError("WebRTC server is not running.")
        future = asyncio.run_coroutine_threadsafe(self._enqueue_frames(frames), loop)
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
        except Exception as error:  # noqa: BLE001 - report startup thread failures
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
        prefix = f"/{self._config.capability_path}" if self._config.capability_path else ""
        app.router.add_get(f"{prefix}/", self._serve_browser)
        app.router.add_get(f"{prefix}/app.js", self._serve_browser_script)
        app.router.add_get(f"{prefix}/healthz", self._health)
        app.router.add_get(f"{prefix}/api/webrtc/config", self._configuration)
        app.router.add_post(f"{prefix}/api/webrtc/offer", self._offer)
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
        """Return the minimal browser client with restrictive headers."""
        response = web.Response(text=_BROWSER_PAGE, content_type="text/html")
        response.headers.update(
            {
                "Cache-Control": "no-store",
                "Content-Security-Policy": (
                    "default-src 'none'; connect-src 'self'; media-src blob:; "
                    "script-src 'self'; style-src 'unsafe-inline'; base-uri 'none'; "
                    "frame-ancestors 'none'"
                ),
                "Referrer-Policy": "no-referrer",
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "DENY",
            }
        )
        return response

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

    async def _configuration(self, _: web.Request) -> web.Response:
        """Return the browser's authenticated ICE configuration."""
        ice_servers = []
        for server in self._config.ice_servers:
            document: dict[str, object] = {"urls": server.urls}
            if server.username is not None:
                document["username"] = server.username
            if server.credential is not None:
                document["credential"] = server.credential
            ice_servers.append(document)
        return web.json_response(
            {
                "iceServers": ice_servers,
                "iceTransportPolicy": self._config.browser_ice_transport_policy,
                "allowReconnect": self._config.allow_reconnect,
                "pointerLockControls": self._config.pointer_lock_controls,
            },
            headers={"Cache-Control": "no-store"},
        )

    async def _offer(self, request: web.Request) -> web.Response:
        """Negotiate one browser peer connection."""
        expected_origin = f"{request.scheme}://{request.host}"
        if request.headers.get("Origin") != expected_origin:
            raise web.HTTPForbidden(reason="WebRTC offer origin is invalid.")
        if self._closed:
            raise web.HTTPServiceUnavailable(reason="WebRTC server is closed.")
        session_desc = self._session_desc
        if session_desc is None:
            raise web.HTTPConflict(reason="WebRTC server is not open.")
        try:
            payload = await request.json()
        except (json.JSONDecodeError, web.HTTPException) as error:
            raise web.HTTPBadRequest(reason="Expected a JSON WebRTC offer.") from error
        if not isinstance(payload, dict):
            raise web.HTTPBadRequest(reason="WebRTC offer must be an object.")
        sdp = payload.get("sdp")
        offer_type = payload.get("type")
        if not isinstance(sdp, str) or not isinstance(offer_type, str):
            raise web.HTTPBadRequest(reason="WebRTC offer requires string sdp and type.")
        client_id = payload.get("client_id")
        if (
            not isinstance(client_id, str)
            or not 16 <= len(client_id) <= 128
            or any(
                character not in string.ascii_letters + string.digits + "-_"
                for character in client_id
            )
        ):
            raise web.HTTPBadRequest(reason="WebRTC offer client_id is invalid.")

        active_peer = self._peer_connection
        if active_peer is not None:
            if not self._config.allow_reconnect or self._client_id != client_id:
                self._report_connection_status(
                    "rejected",
                    remote_address=request.remote or "unknown",
                    reason="another_client_is_active",
                )
                raise web.HTTPConflict(reason="A WebRTC client is already connected.")
            await self._disconnect_peer(active_peer)

        peer_connection = RTCPeerConnection(
            RTCConfiguration(iceServers=list(self._config.ice_servers) or None)
        )
        video_track = _VideoTrack(
            session_desc.frames_per_second_for_ui, self._config.max_pending_frames
        )
        peer_connection.addTrack(video_track)
        self._peer_connection = peer_connection
        self._video_track = video_track
        self._client_id = client_id
        self._client_remote_address = request.remote or "unknown"
        connecting_fields = self._connection_fields(peer_connection)
        if self._last_connection_failure is not None:
            connecting_fields["last_failure"] = self._last_connection_failure
        self._report_connection_status("signaling", **connecting_fields)

        @peer_connection.on("datachannel")
        def on_datachannel(channel: Any) -> None:
            self._client_connected = True
            self._report_connection_status(
                "connected",
                **self._connection_fields(peer_connection),
            )
            self._last_connection_failure = None

            @channel.on("message")
            def on_message(message: Any) -> None:
                try:
                    self._buffer_browser_message(message)
                except ValueError as error:
                    channel.send(json.dumps({"type": "error", "message": str(error)}))

            @channel.on("close")
            def on_close() -> None:
                self._schedule_disconnect(peer_connection)

        @peer_connection.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            if self._peer_connection is not peer_connection:
                return
            state = peer_connection.connectionState
            if state in {"new", "connecting"}:
                self._report_connection_status(
                    "connecting",
                    **self._connection_fields(peer_connection),
                )
                return
            if state == "connected" and not self._client_connected:
                self._report_connection_status(
                    "transport_connected",
                    **self._connection_fields(peer_connection),
                )
                return
            if state in {"failed", "disconnected", "closed"}:
                terminal_status = _closed_connection_status(
                    connection_state=state,
                    client_connected=self._client_connected,
                )
                if terminal_status == "failed":
                    self._last_connection_failure = (
                        f"connection={state}, ice={peer_connection.iceConnectionState}"
                    )
                self._report_connection_status(
                    terminal_status,
                    **self._connection_fields(peer_connection),
                )
                await self._disconnect_peer(peer_connection, report_status=False)

        @peer_connection.on("iceconnectionstatechange")
        def on_iceconnectionstatechange() -> None:
            if self._peer_connection is not peer_connection:
                return
            ice_state = peer_connection.iceConnectionState
            if ice_state in {"new", "checking"}:
                status = "ice_checking"
            elif ice_state in {"connected", "completed"}:
                status = "ice_connected"
            elif ice_state == "failed":
                status = "ice_failed"
                self._last_connection_failure = (
                    f"connection={peer_connection.connectionState}, ice={ice_state}"
                )
            else:
                return
            self._report_connection_status(
                status,
                **self._connection_fields(peer_connection),
            )

        try:
            await peer_connection.setRemoteDescription(
                RTCSessionDescription(sdp=sdp, type=offer_type)
            )
            await peer_connection.setLocalDescription(await peer_connection.createAnswer())
        except Exception:
            self._last_connection_failure = "signaling_failed"
            self._report_connection_status(
                "failed",
                **self._connection_fields(peer_connection),
            )
            await self._disconnect_peer(
                peer_connection,
                notify=False,
                report_status=False,
            )
            raise

        local_description = peer_connection.localDescription
        if local_description is None:
            raise web.HTTPInternalServerError(reason="WebRTC peer did not create an answer.")
        return web.json_response({"sdp": local_description.sdp, "type": local_description.type})

    def _buffer_browser_message(self, raw_message: object) -> None:
        """Validate and append one data-channel message."""
        if not isinstance(raw_message, str):
            raise TypeError("Browser event must be a JSON string.")
        try:
            payload = json.loads(raw_message)
        except json.JSONDecodeError as error:
            raise ValueError("Browser event must contain valid JSON.") from error
        if not isinstance(payload, dict):
            raise TypeError("Browser event must be a JSON object.")

        event_type = payload.get("type")
        if event_type == "activation":
            if not self._config.pointer_lock_controls:
                raise ValueError("Activation events require pointer-lock controls.")
            active = payload.get("active")
            if not isinstance(active, bool):
                raise ValueError("Activation event requires a boolean active value.")
            event_data: UserInputEventData = ActivationUserInputEventData(active=active)
        elif event_type == "keyboard":
            key = payload.get("key")
            pressed = payload.get("pressed")
            if not isinstance(key, str) or not key:
                raise ValueError("Keyboard event requires a non-empty key.")
            if not isinstance(pressed, bool):
                raise ValueError("Keyboard event requires a boolean pressed value.")
            event_data = KeyboardUserInputEventData(key=key, pressed=pressed)
        elif event_type == "mouse":
            if not self._config.pointer_lock_controls:
                raise ValueError("Mouse events require pointer-lock controls.")
            movement_x = payload.get("movement_x")
            movement_y = payload.get("movement_y")
            coordinates = (movement_x, movement_y)
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in coordinates
            ):
                raise ValueError("Mouse movement must contain two finite numbers.")
            event_data = MouseUserInputEventData(
                movement_x=float(movement_x),
                movement_y=float(movement_y),
            )
        elif event_type == "reset":
            event_data = ResetUserInputEventData()
        elif event_type == "close":
            event_data = CloseUserInputEventData()
        else:
            raise ValueError(
                "Browser event type must be activation, keyboard, mouse, reset, or close."
            )
        self._append_event(event_data)

    def _append_event(self, event_data: UserInputEventData) -> None:
        """Timestamp and buffer one validated browser event."""
        session_start_ns = self._session_start_ns
        if session_start_ns is None:
            return
        timestamp_us = np.uint64((time.monotonic_ns() - session_start_ns) // 1_000)
        event = UserInputEvent(timestamp=timestamp_us, event_data=event_data)
        callback = self._input_callback
        if callback is None:
            raise RuntimeError("WebRTC input callback is not registered.")
        #  Pass that UserInputEvent to the callback.
        #  The callback stores it in WebRTCClientWindow’s thread-safe queue.
        callback(event)

    def _schedule_disconnect(self, peer_connection: RTCPeerConnection) -> None:
        """Schedule peer cleanup and retain it until the server can await it."""
        task = asyncio.create_task(self._disconnect_peer(peer_connection))
        self._disconnect_tasks.add(task)
        task.add_done_callback(self._disconnect_tasks.discard)

    async def _disconnect_peer(
        self,
        peer_connection: RTCPeerConnection,
        *,
        notify: bool = True,
        report_status: bool = True,
    ) -> None:
        """Release one browser peer and apply the configured disconnect behavior."""
        if self._peer_connection is not peer_connection:
            return
        self._peer_connection = None
        track = self._video_track
        self._video_track = None
        was_connected = self._client_connected
        self._client_connected = False
        self._client_id = None
        if report_status:
            status = _closed_connection_status(
                connection_state=peer_connection.connectionState,
                client_connected=was_connected,
            )
            if not was_connected:
                self._last_connection_failure = (
                    "connection="
                    f"{peer_connection.connectionState}, "
                    f"ice={peer_connection.iceConnectionState}"
                )
            self._report_connection_status(
                status,
                **self._connection_fields(peer_connection),
            )
        self._client_remote_address = None
        if notify and was_connected and not self._closed:
            if self._config.allow_reconnect:
                self._append_event(ActivationUserInputEventData(active=False))
            else:
                self._append_event(CloseUserInputEventData())
        if track is not None:
            await track.close()
        if peer_connection.connectionState != "closed":
            await peer_connection.close()

    def _connection_fields(self, peer_connection: RTCPeerConnection) -> dict[str, str]:
        """Return a stable, non-secret snapshot of one browser transport."""
        return {
            "remote_address": self._client_remote_address or "unknown",
            "connection_state": peer_connection.connectionState,
            "ice_state": peer_connection.iceConnectionState,
        }

    async def _enqueue_frames(
        self, frames: tuple[np.ndarray[Any, np.dtype[np.uint8]], ...]
    ) -> None:
        """Append frames to the active media track, if connected."""
        track = self._video_track
        if track is not None:
            await track.enqueue(frames)

    async def _shutdown(self) -> None:
        """Release async server resources on their owning loop."""
        peer_connection = self._peer_connection
        if peer_connection is not None:
            await self._disconnect_peer(peer_connection, notify=False)
        if self._disconnect_tasks:
            await asyncio.gather(*tuple(self._disconnect_tasks), return_exceptions=True)
        runner = self._runner
        self._runner = None
        if runner is not None:
            await runner.cleanup()


def _result_to_rgb_frames(
    result: StepResult, session_desc: SessionDesc
) -> tuple[np.ndarray[Any, np.dtype[np.uint8]], ...]:
    """Convert one result to time-major RGB uint8 frames."""
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
            raise ValueError("bvtchw WebRTC output requires one batch and one video view.")
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

    if frames.shape[1] == 1:
        frames = frames.repeat(1, 3, 1, 1)
    if frames.is_floating_point():
        frames = ((frames.to(torch.float32).clamp(-1.0, 1.0) + 1.0) * 127.5).round()
    frames = frames.clamp(0, 255).to(torch.uint8)
    frames = frames.permute(0, 2, 3, 1).contiguous().cpu()
    return tuple(np.asarray(frame.numpy()) for frame in frames)
