# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Protocol

from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError
from loguru import logger as loguru_logger


class CreateAnswerCallback(Protocol):
    async def __call__(self, *, offer_sdp: str, offer_type: str) -> dict[str, str]: ...


DEFAULT_WARMUP_ACTIONS: tuple[dict[str, Any], ...] = (
    {"type": "action", "action": {"event": "keydown", "key": "w"}},
    {"type": "action", "action": {"event": "keydown", "key": "d"}},
)


async def run_loopback_warmup_session(
    *,
    num_chunks: int,
    warmup_timeout_s: float,
    create_answer: CreateAnswerCallback,
    close_active_session: Callable[[], Awaitable[None]] | None = None,
    action_payloads: Sequence[dict[str, Any]] = DEFAULT_WARMUP_ACTIONS,
    label: str = "WebRTC",
    channel_open_timeout_s: float = 15.0,
    ice_gathering_timeout_s: float = 15.0,
    logger: Any = loguru_logger,
) -> None:
    if num_chunks < 0:
        raise ValueError("num_chunks must be >= 0")
    if num_chunks == 0:
        return

    logger.info("Starting {} loopback warmup with {} chunk(s).", label, num_chunks)
    client_peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    control_channel = client_peer.createDataChannel("controls", ordered=True)
    client_peer.addTransceiver("video", direction="recvonly")
    channel_open = asyncio.Event()
    warmup_done = asyncio.Event()
    received_chunks = 0
    drain_tasks: set[asyncio.Task[Any]] = set()
    heartbeat_task: asyncio.Task[Any] | None = None

    @control_channel.on("open")
    def on_open() -> None:
        channel_open.set()

    @control_channel.on("message")
    def on_message(message: Any) -> None:
        nonlocal received_chunks
        if not isinstance(message, str):
            return
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            return
        if not isinstance(payload, dict) or payload.get("type") != "chunk_done":
            return
        received_chunks += 1
        logger.info(
            "{} loopback warmup chunk done chunk={} num_frames={}",
            label,
            payload.get("chunk_index"),
            payload.get("num_frames"),
        )
        if received_chunks >= num_chunks:
            warmup_done.set()

    @client_peer.on("track")
    def on_track(track: Any) -> None:
        drain_tasks.add(
            asyncio.create_task(_drain_loopback_track(track, logger=logger))
        )

    try:
        offer = await client_peer.createOffer()
        await client_peer.setLocalDescription(offer)
        await wait_for_ice_gathering_complete(
            client_peer, timeout_s=ice_gathering_timeout_s
        )
        local_description = client_peer.localDescription
        if local_description is None:
            raise RuntimeError("Loopback peer did not produce local description.")

        answer_payload = await create_answer(
            offer_sdp=local_description.sdp,
            offer_type=local_description.type,
        )
        await client_peer.setRemoteDescription(
            RTCSessionDescription(
                sdp=answer_payload["sdp"],
                type=answer_payload["type"],
            )
        )

        await asyncio.wait_for(channel_open.wait(), timeout=channel_open_timeout_s)
        logger.info("{} loopback warmup data channel open; sending fake inputs.", label)
        heartbeat_task = asyncio.create_task(_send_loopback_heartbeats(control_channel))
        for action_payload in action_payloads:
            control_channel.send(json.dumps(action_payload))
        await asyncio.wait_for(warmup_done.wait(), timeout=warmup_timeout_s)
    finally:
        if heartbeat_task is not None:
            heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat_task
        await client_peer.close()
        for task in drain_tasks:
            task.cancel()
        for task in drain_tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if close_active_session is not None:
            await close_active_session()
    logger.info("{} loopback warmup complete.", label)


async def _send_loopback_heartbeats(control_channel: Any) -> None:
    while True:
        await asyncio.sleep(2.0)
        if getattr(control_channel, "readyState", None) == "open":
            control_channel.send(json.dumps({"type": "heartbeat"}))


async def wait_for_ice_gathering_complete(
    peer_connection: Any, *, timeout_s: float = 15.0
) -> None:
    if peer_connection.iceGatheringState == "complete":
        return
    ice_complete = asyncio.Event()

    @peer_connection.on("icegatheringstatechange")
    def on_icegatheringstatechange() -> None:
        if peer_connection.iceGatheringState == "complete":
            ice_complete.set()

    if peer_connection.iceGatheringState == "complete":
        ice_complete.set()
    await asyncio.wait_for(ice_complete.wait(), timeout=timeout_s)


async def _drain_loopback_track(track: Any, *, logger: Any = loguru_logger) -> None:
    try:
        while True:
            await track.recv()
    except MediaStreamError:
        return
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.opt(exception=True).debug("Loopback warmup video drain stopped.")
