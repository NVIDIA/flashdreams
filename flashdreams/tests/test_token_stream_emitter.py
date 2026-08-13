# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :class:`TokenFrameEmitter`.

Drives the emitter with :func:`asyncio.run` against a fake WebSocket that
records every sent frame. Covers the session-header control frame, per-frame
token framing, and ack-based flow control. No GPU is required.
"""

from __future__ import annotations

import asyncio
import json

import pytest
import torch

from flashdreams.serving.token_stream import framing
from flashdreams.serving.token_stream.codec import RawFloat16TokenCodecConfig
from flashdreams.serving.token_stream.emitter import TokenFrameEmitter

pytestmark = pytest.mark.ci_cpu


class FakeWebSocket:
    """Minimal stand-in that records every ``send_bytes`` payload."""

    def __init__(self) -> None:
        self.sent: list[bytes] = []

    async def send_bytes(self, data: bytes) -> None:
        self.sent.append(bytes(data))


def _make_emitter(ws: FakeWebSocket, *, fps: int = 16, flow_window_size: int = 4):
    codec = RawFloat16TokenCodecConfig().setup()
    return TokenFrameEmitter(
        ws, codec=codec, fps=fps, flow_window_size=flow_window_size
    )


def test_emit_chunk_sends_header_then_token_frames() -> None:
    t_frames, channels, height, width = 3, 2, 4, 4
    latent = torch.randn(1, 1, t_frames, channels, height, width)
    ws = FakeWebSocket()
    emitter = _make_emitter(ws, fps=16)

    frames_sent = asyncio.run(emitter.emit_chunk(latent, chunk_index=5))

    assert frames_sent == t_frames
    assert len(ws.sent) == 1 + t_frames

    control_header = framing.parse_header(ws.sent[0])
    assert control_header.is_control is True
    session = json.loads(ws.sent[0][framing.HEADER_SIZE :].decode())
    assert session["latent_shape"] == [channels, height, width]
    assert session["frames_per_chunk"] == t_frames
    assert session["fps"] == 16
    assert session["codec"]["id"] == "raw_f16"

    for frame_idx, raw in enumerate(ws.sent[1:]):
        header = framing.parse_header(raw)
        assert header.is_control is False
        assert header.chunk_id == 5
        assert header.frame_idx == frame_idx
        assert header.frame_total == t_frames
        assert header.is_last_in_chunk == (frame_idx == t_frames - 1)


def test_header_sent_only_once_across_chunks() -> None:
    latent = torch.randn(1, 1, 2, 2, 4, 4)
    ws = FakeWebSocket()
    emitter = _make_emitter(ws)

    async def run() -> None:
        await emitter.emit_chunk(latent, chunk_index=0)
        await emitter.emit_chunk(latent, chunk_index=1)

    asyncio.run(run())

    control_frames = [raw for raw in ws.sent if framing.parse_header(raw).is_control]
    assert len(control_frames) == 1


def test_flow_control_blocks_until_ack() -> None:
    latent = torch.randn(1, 1, 2, 2, 4, 4)
    ws = FakeWebSocket()
    emitter = _make_emitter(ws, flow_window_size=1)

    async def run() -> None:
        # First chunk consumes the single flow-window slot.
        await emitter.emit_chunk(latent, chunk_index=0)

        # A second chunk must block: no slot is free until an ack arrives.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                emitter.emit_chunk(latent, chunk_index=1), timeout=0.2
            )

        # Acknowledging the first chunk frees the slot; the retry proceeds.
        emitter.handle_ack(0)
        frames_sent = await asyncio.wait_for(
            emitter.emit_chunk(latent, chunk_index=1), timeout=1.0
        )
        assert frames_sent == int(latent.shape[2])

    asyncio.run(run())
