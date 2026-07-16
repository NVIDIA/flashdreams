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

"""Headless end-to-end client for the session-scoped WebRTC API."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from time import monotonic
from typing import Any

from aiohttp import ClientSession, ClientTimeout
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError

from flashdreams.serving.webrtc.warmup import wait_for_ice_gathering_complete


@dataclass(frozen=True, slots=True)
class WebRTCClientConfig:
    """Connection and success criteria for one WebRTC smoke session."""

    base_url: str
    """HTTP base URL of ``flashdreams-serve`` without an endpoint path."""

    model: str
    """Model slug selected from ``GET /v1/models``."""

    key: str = "w"
    """Keyboard control sent to trigger the first generated chunk."""

    timeout_s: float = 600.0
    """Maximum time for negotiation and first-chunk generation."""

    channel_open_timeout_s: float = 30.0
    """Maximum time for the ordered control data channel to open."""

    ice_gathering_timeout_s: float = 30.0
    """Maximum time for local ICE gathering to complete."""

    lease_seconds: float = 900.0
    """Server-side idle session lease."""

    parameters: dict[str, Any] = field(default_factory=dict)
    """Model-specific payload forwarded to ``POST /v1/sessions``."""


@dataclass(frozen=True, slots=True)
class WebRTCClientResult:
    """Successful end-to-end WebRTC test result."""

    model: str
    """Model slug reported by the test."""

    session_id: str
    """Allocated server session identifier."""

    frames_received: int
    """Number of decoded video frames received before completion."""

    connection_state: str
    """Final peer-connection state observed before teardown."""

    elapsed_s: float
    """Total elapsed test time in seconds."""

    chunk: dict[str, Any]
    """First ``chunk_done`` control payload returned by the server."""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable result payload."""
        return asdict(self)


@dataclass(slots=True)
class _ClientState:
    """Mutable negotiation, control, and media observations."""

    channel_open: asyncio.Event = field(default_factory=asyncio.Event)
    """Set when the ordered control channel opens."""

    chunk_received: asyncio.Event = field(default_factory=asyncio.Event)
    """Set after the server reports the first completed chunk."""

    media_complete: asyncio.Event = field(default_factory=asyncio.Event)
    """Set after all enqueued frames for the reported chunk arrive."""

    failed: asyncio.Event = field(default_factory=asyncio.Event)
    """Set when control, media, or peer-connection handling fails."""

    frames_received: int = 0
    """Number of decoded video frames received."""

    expected_frames: int | None = None
    """Frames expected from the first chunk; ``None`` before ``chunk_done``."""

    chunk: dict[str, Any] | None = None
    """First ``chunk_done`` payload; ``None`` until received."""

    error: str | None = None
    """Failure message; ``None`` while healthy."""

    def record_frame(self) -> None:
        """Record one decoded video frame and update completion state."""
        self.frames_received += 1
        self._update_media_complete()

    def record_control_message(self, message: Any) -> None:
        """Parse one server data-channel message and update test state."""
        if isinstance(message, bytes):
            message = message.decode("utf-8", errors="replace")
        if not isinstance(message, str):
            self.fail("Server returned a non-text control message.")
            return
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            self.fail(f"Server returned invalid control JSON: {message!r}")
            return
        if not isinstance(payload, dict):
            self.fail("Server control payload is not an object.")
            return
        message_type = payload.get("type")
        if message_type == "error":
            self.fail(str(payload.get("message", "Unknown server error.")))
            return
        if message_type != "chunk_done" or self.chunk is not None:
            return
        self.chunk = payload
        reported_frames = payload.get("enqueued_frames", payload.get("num_frames", 1))
        self.expected_frames = (
            max(1, reported_frames) if isinstance(reported_frames, int) else 1
        )
        self.chunk_received.set()
        self._update_media_complete()

    def fail(self, message: str) -> None:
        """Record the first failure and wake the result waiter."""
        if self.error is None:
            self.error = message
            self.failed.set()

    def _update_media_complete(self) -> None:
        if (
            self.expected_frames is not None
            and self.frames_received >= self.expected_frames
        ):
            self.media_complete.set()


async def run_webrtc_test(config: WebRTCClientConfig) -> WebRTCClientResult:
    """Exercise discovery, session lifecycle, signaling, control, and video media.

    Args:
        config: Server location, model slug, timeouts, and session parameters.

    Returns:
        Result containing the session, first chunk metrics, and received frame count.

    Raises:
        RuntimeError: Discovery, signaling, control, media, or cleanup fails.
        TimeoutError: Negotiation or first-chunk generation exceeds a timeout.
    """
    started_at = monotonic()
    base_url = config.base_url.rstrip("/")
    state = _ClientState()
    session_id: str | None = None
    peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    control_channel = peer.createDataChannel("controls", ordered=True)
    peer.addTransceiver("video", direction="recvonly")
    media_tasks: set[asyncio.Task[None]] = set()
    heartbeat_task: asyncio.Task[None] | None = None

    @control_channel.on("open")
    def on_open() -> None:
        state.channel_open.set()

    @control_channel.on("message")
    def on_message(message: Any) -> None:
        state.record_control_message(message)

    @peer.on("connectionstatechange")
    def on_connectionstatechange() -> None:
        if peer.connectionState in {"failed", "closed", "disconnected"}:
            if not state.media_complete.is_set():
                state.fail(f"Peer connection entered {peer.connectionState!r} state.")

    @peer.on("track")
    def on_track(track: Any) -> None:
        if track.kind == "video":
            media_tasks.add(asyncio.create_task(_consume_video(track, state)))

    async with ClientSession(timeout=ClientTimeout(total=config.timeout_s)) as http:
        try:
            models_payload = await _request_json(http, "GET", f"{base_url}/v1/models")
            _validate_model(models_payload, config.model)
            session_payload = await _request_json(
                http,
                "POST",
                f"{base_url}/v1/sessions",
                body={
                    "model": config.model,
                    "parameters": config.parameters,
                    "lease_seconds": config.lease_seconds,
                },
            )
            session_id = _require_string(session_payload, "id")
            await _wait_for_session_ready(http, base_url, session_payload)

            offer = await peer.createOffer()
            await peer.setLocalDescription(offer)
            await wait_for_ice_gathering_complete(
                peer, timeout_s=config.ice_gathering_timeout_s
            )
            local_description = peer.localDescription
            if local_description is None:
                raise RuntimeError("Client peer did not produce a local description.")
            answer = await _request_json(
                http,
                "POST",
                f"{base_url}/v1/sessions/{session_id}/webrtc/offer",
                body={
                    "sdp": local_description.sdp,
                    "type": local_description.type,
                },
            )
            await peer.setRemoteDescription(
                RTCSessionDescription(
                    sdp=_require_string(answer, "sdp"),
                    type=_require_string(answer, "type"),
                )
            )

            await asyncio.wait_for(
                state.channel_open.wait(), timeout=config.channel_open_timeout_s
            )
            heartbeat_task = asyncio.create_task(_send_heartbeats(control_channel))
            control_channel.send(
                json.dumps(
                    {
                        "type": "action",
                        "action": {"event": "keydown", "key": config.key},
                    }
                )
            )
            await asyncio.wait_for(_wait_for_result(state), timeout=config.timeout_s)
            assert state.chunk is not None
            return WebRTCClientResult(
                model=config.model,
                session_id=session_id,
                frames_received=state.frames_received,
                connection_state=peer.connectionState,
                elapsed_s=round(monotonic() - started_at, 3),
                chunk=state.chunk,
            )
        finally:
            if heartbeat_task is not None:
                heartbeat_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await heartbeat_task
            with contextlib.suppress(Exception):
                await peer.close()
            for task in media_tasks:
                task.cancel()
            for task in media_tasks:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            if session_id is not None:
                with contextlib.suppress(Exception):
                    await _request_json(
                        http,
                        "DELETE",
                        f"{base_url}/v1/sessions/{session_id}",
                    )


async def _consume_video(track: Any, state: _ClientState) -> None:
    try:
        while True:
            await track.recv()
            state.record_frame()
    except MediaStreamError:
        if not state.media_complete.is_set():
            state.fail("Video track ended before the first chunk was delivered.")
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - media task error boundary
        state.fail(f"Video receive failed: {exc}")


async def _send_heartbeats(control_channel: Any) -> None:
    while True:
        await asyncio.sleep(2.0)
        if control_channel.readyState == "open":
            control_channel.send(json.dumps({"type": "heartbeat"}))


async def _wait_for_result(state: _ClientState) -> None:
    success = asyncio.gather(state.chunk_received.wait(), state.media_complete.wait())
    failure = asyncio.create_task(state.failed.wait())
    try:
        done, _pending = await asyncio.wait(
            {success, failure}, return_when=asyncio.FIRST_COMPLETED
        )
        if failure in done and state.failed.is_set():
            raise RuntimeError(state.error or "WebRTC client failed.")
        await success
    finally:
        for task in (success, failure):
            if not task.done():
                task.cancel()
        for task in (success, failure):
            with contextlib.suppress(asyncio.CancelledError):
                await task


async def _wait_for_session_ready(
    http: ClientSession, base_url: str, session_payload: Mapping[str, Any]
) -> None:
    session_id = _require_string(session_payload, "id")
    payload = session_payload
    while True:
        status = payload.get("status")
        if status == "ready":
            return
        if status in {"failed", "closed", "expired"}:
            raise RuntimeError(
                f"Session {session_id!r} entered {status!r} state: "
                f"{payload.get('error')}"
            )
        await asyncio.sleep(0.5)
        payload = await _request_json(
            http, "GET", f"{base_url}/v1/sessions/{session_id}"
        )


async def _request_json(
    session: ClientSession,
    method: str,
    url: str,
    *,
    body: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    async with session.request(method, url, json=body) as response:
        text = await response.text()
        if response.status < 200 or response.status >= 300:
            raise RuntimeError(
                f"{method} {url} failed with HTTP {response.status}: {text}"
            )
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{method} {url} returned invalid JSON: {text}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"{method} {url} did not return a JSON object.")
        return payload


def _validate_model(payload: Mapping[str, Any], model: str) -> None:
    models = payload.get("data")
    if not isinstance(models, list):
        raise RuntimeError("GET /v1/models response is missing a data list.")
    for descriptor in models:
        if not isinstance(descriptor, dict) or descriptor.get("id") != model:
            continue
        capabilities = descriptor.get("capabilities", {})
        transports = (
            capabilities.get("transports", []) if isinstance(capabilities, dict) else []
        )
        if "webrtc" not in transports:
            raise RuntimeError(f"Model {model!r} does not advertise WebRTC support.")
        return
    raise RuntimeError(f"Model {model!r} was not returned by GET /v1/models.")


def _require_string(payload: Mapping[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"Response requires non-empty string field {field_name!r}.")
    return value


def build_parser() -> argparse.ArgumentParser:
    """Build the headless WebRTC client command-line parser."""
    parser = argparse.ArgumentParser(
        prog="flashdreams-webrtc-client",
        description=(
            "Create a serving session and verify WebRTC control plus video output."
        ),
    )
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", required=True)
    parser.add_argument("--key", default="w")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--channel-open-timeout", type=float, default=30.0)
    parser.add_argument("--ice-gathering-timeout", type=float, default=30.0)
    parser.add_argument("--lease-seconds", type=float, default=900.0)
    parser.add_argument("--parameters-json", default="{}")
    return parser


def entrypoint(argv: Sequence[str] | None = None) -> None:
    """Run the headless client and print its result as JSON."""
    args = build_parser().parse_args(argv)
    try:
        parameters = json.loads(args.parameters_json)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--parameters-json is invalid JSON: {exc}") from None
    if not isinstance(parameters, dict):
        raise SystemExit("--parameters-json must decode to a JSON object.")
    if (
        args.timeout <= 0
        or args.channel_open_timeout <= 0
        or args.ice_gathering_timeout <= 0
    ):
        raise SystemExit("Client timeouts must be positive.")
    if args.lease_seconds <= 0:
        raise SystemExit("--lease-seconds must be positive.")
    if not args.key.strip():
        raise SystemExit("--key must be non-empty.")
    config = WebRTCClientConfig(
        base_url=args.url,
        model=args.model,
        key=args.key,
        timeout_s=args.timeout,
        channel_open_timeout_s=args.channel_open_timeout,
        ice_gathering_timeout_s=args.ice_gathering_timeout,
        lease_seconds=args.lease_seconds,
        parameters=parameters,
    )
    try:
        result = asyncio.run(run_webrtc_test(config))
    except KeyboardInterrupt:
        print("WebRTC client stopped.", file=sys.stderr)
        return
    except Exception as exc:  # noqa: BLE001 - CLI error boundary
        raise SystemExit(f"WebRTC test failed: {exc}") from None
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    entrypoint()
