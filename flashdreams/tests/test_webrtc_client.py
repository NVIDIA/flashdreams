# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from fractions import Fraction
from typing import Any

import pytest
from aiohttp.test_utils import TestServer
from aiortc import (
    RTCConfiguration,
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
)
from av import VideoFrame

from flashdreams.serving.api import (
    ModelCapabilities,
    ModelDescriptor,
    StreamInput,
    StreamOutput,
)
from flashdreams.serving.backend import LocalWorkerScheduler, ModelWorker
from flashdreams.serving.service import SessionService
from flashdreams.serving.transport import WebRTCTransport
from flashdreams.serving.webrtc.client import (
    _ClientState,
    _validate_model,
    _wait_for_result,
    WebRTCClientConfig,
    build_parser,
    run_webrtc_test,
)
from flashdreams.serving.webrtc.warmup import wait_for_ice_gathering_complete

pytestmark = pytest.mark.ci_cpu


class _VideoTrack(VideoStreamTrack):
    async def recv(self) -> VideoFrame:
        pts, time_base = await self.next_timestamp()
        frame = VideoFrame(width=2, height=2, format="yuv420p")
        frame.pts = pts
        frame.time_base = time_base or Fraction(1, 90000)
        return frame


class _WebRTCWorker(ModelWorker):
    def __init__(self, worker_id: str) -> None:
        self.worker_id = worker_id
        self._descriptor = ModelDescriptor(
            id="fake-webrtc",
            capabilities=ModelCapabilities(inputs=("action",), outputs=("video",)),
        )
        self._peers: set[RTCPeerConnection] = set()

    @property
    def descriptor(self) -> ModelDescriptor:
        return self._descriptor

    async def start(self) -> None:
        return None

    async def create_session(
        self, session_id: str, parameters: Mapping[str, Any]
    ) -> None:
        del session_id, parameters

    async def stream(
        self, session_id: str, request: StreamInput
    ) -> AsyncIterator[StreamOutput]:
        del session_id, request
        yield StreamOutput(type="unused")

    async def create_webrtc_answer(
        self, session_id: str, offer: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        del session_id
        peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        self._peers.add(peer)
        peer.addTrack(_VideoTrack())

        @peer.on("datachannel")
        def on_datachannel(channel: Any) -> None:
            @channel.on("message")
            def on_message(message: Any) -> None:
                if not isinstance(message, str):
                    return
                payload = json.loads(message)
                if payload.get("type") == "action":
                    channel.send(
                        json.dumps(
                            {
                                "type": "chunk_done",
                                "chunk_index": 0,
                                "num_frames": 1,
                                "enqueued_frames": 1,
                            }
                        )
                    )

        await peer.setRemoteDescription(
            RTCSessionDescription(
                sdp=str(offer["sdp"]),
                type=str(offer["type"]),
            )
        )
        answer = await peer.createAnswer()
        await peer.setLocalDescription(answer)
        await wait_for_ice_gathering_complete(peer)
        assert peer.localDescription is not None
        return {
            "sdp": peer.localDescription.sdp,
            "type": peer.localDescription.type,
        }

    async def close_session(self, session_id: str) -> None:
        del session_id
        await self._close_peers()

    async def close(self) -> None:
        await self._close_peers()

    async def _close_peers(self) -> None:
        peers = list(self._peers)
        self._peers.clear()
        for peer in peers:
            await peer.close()


def test_client_state_waits_for_every_enqueued_video_frame() -> None:
    state = _ClientState()
    state.record_frame()
    state.record_control_message(
        json.dumps(
            {
                "type": "chunk_done",
                "chunk_index": 0,
                "num_frames": 3,
                "enqueued_frames": 2,
            }
        )
    )

    assert state.chunk_received.is_set()
    assert not state.media_complete.is_set()
    state.record_frame()
    assert state.media_complete.is_set()


def test_client_state_surfaces_server_error() -> None:
    state = _ClientState()
    state.record_control_message(json.dumps({"type": "error", "message": "boom"}))

    assert state.failed.is_set()
    assert state.error == "boom"


@pytest.mark.asyncio
async def test_wait_for_result_raises_recorded_failure() -> None:
    state = _ClientState()
    state.fail("negotiation failed")

    with pytest.raises(RuntimeError, match="negotiation failed"):
        await _wait_for_result(state)


def test_validate_model_requires_webrtc_capability() -> None:
    payload = {
        "data": [
            {
                "id": "fake",
                "capabilities": {"transports": ["websocket"]},
            }
        ]
    }

    with pytest.raises(RuntimeError, match="does not advertise WebRTC"):
        _validate_model(payload, "fake")


def test_client_parser_defaults_to_local_server() -> None:
    args = build_parser().parse_args(["--model", "fake"])

    assert args.url == "http://127.0.0.1:8080"
    assert args.key == "w"


@pytest.mark.asyncio
async def test_client_exercises_rest_signaling_control_and_video() -> None:
    descriptor = _WebRTCWorker("descriptor").descriptor
    scheduler = LocalWorkerScheduler({"fake-webrtc": _WebRTCWorker})
    service = SessionService({"fake-webrtc": descriptor}, scheduler)
    server = TestServer(WebRTCTransport(service).create_app())
    await server.start_server()
    try:
        result = await run_webrtc_test(
            WebRTCClientConfig(
                base_url=str(server.make_url("/")).rstrip("/"),
                model="fake-webrtc",
                timeout_s=10.0,
            )
        )
    finally:
        await server.close()

    assert result.model == "fake-webrtc"
    assert result.frames_received >= 1
    assert result.chunk["chunk_index"] == 0
    with pytest.raises(KeyError):
        await service.get_session(result.session_id)
