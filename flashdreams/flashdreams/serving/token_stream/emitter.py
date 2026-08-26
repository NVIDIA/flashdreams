# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Emit encoded latent chunks over a WebSocket with ack-based flow control."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Protocol

from loguru import logger

from flashdreams.serving.token_stream import framing

if TYPE_CHECKING:
    import torch

    from flashdreams.serving.token_stream.codec import TokenCodec


class _ByteSink(Protocol):
    """Minimal async byte sink the emitter writes frames to.

    Structurally satisfied by :class:`aiohttp.web.WebSocketResponse` (whose
    ``send_bytes`` takes an extra optional ``compress`` argument) and by test
    doubles, so the emitter does not depend on the concrete WebSocket type.
    """

    async def send_bytes(self, data: bytes) -> None: ...


class TokenFrameEmitter:
    """Serialize latent chunks into wire frames and send them over a WebSocket.

    The first :meth:`emit_chunk` call sends a control frame carrying the session
    header before any token frame. A bounded flow window limits how many chunks
    can be in flight before the client acknowledges them via :meth:`handle_ack`.
    """

    def __init__(
        self,
        ws: _ByteSink,
        *,
        codec: TokenCodec,
        fps: int,
        flow_window_size: int,
        extra_header: dict | None = None,
    ) -> None:
        self._ws = ws
        self._codec = codec
        self._fps = fps
        self._extra_header = extra_header or {}
        self._flow = asyncio.Semaphore(flow_window_size)
        self._inflight: set[int] = set()
        self._header_sent = False

    async def emit_chunk(
        self, latent: torch.Tensor, *, chunk_index: int, is_keyframe: bool = False
    ) -> int:
        """Encode and send one latent chunk, returning the number of frames sent.

        Blocks on the flow window until an ack frees a slot. The latent is
        reduced to ``[T, Cl, Hl, Wl]`` before per-frame encoding.
        """
        # Collapse any leading batch/view dimensions to the canonical
        # [T, Cl, Hl, Wl] layout expected by the per-frame codec.
        if latent.dim() == 6:
            latent = latent[0, 0]
        elif latent.dim() == 5:
            latent = latent[0]

        if not self._header_sent:
            await self._send_session_header(latent)
            self._header_sent = True

        await self._flow.acquire()
        self._inflight.add(chunk_index)

        total = int(latent.shape[0])
        for frame_idx in range(total):
            # Encode off the event loop. encode_frame is synchronous GPU work, and a
            # compiling codec (SAS pays a one-time multi-second Triton JIT on its first
            # call) blocks heartbeat handling long enough to trip the 10 s client
            # liveness watchdog, killing the session before its first frame lands.
            # Awaiting each frame in turn keeps the send order the framing relies on.
            result = await asyncio.to_thread(self._codec.encode_frame, latent[frame_idx])
            frame = framing.pack_frame(
                chunk_id=chunk_index,
                frame_idx=frame_idx,
                frame_total=total,
                payload=result.payload,
                codec_params=result.frame_params,
                is_keyframe=(is_keyframe and frame_idx == 0),
                is_last_in_chunk=(frame_idx == total - 1),
            )
            await self._ws.send_bytes(frame)
        return total

    def handle_ack(self, chunk_id: int) -> None:
        """Release a flow-window slot when the client acknowledges a chunk."""
        if chunk_id in self._inflight:
            self._inflight.discard(chunk_id)
            self._flow.release()
        else:
            logger.debug("ignoring ack for unknown token chunk {}", chunk_id)

    async def _send_session_header(self, latent: torch.Tensor) -> None:
        """Send the one-shot control frame describing the stream to the client."""
        header = {
            "protocol_version": framing.PROTOCOL_VERSION,
            "latent_shape": list(latent.shape[1:]),
            "frames_per_chunk": int(latent.shape[0]),
            "fps": self._fps,
            "codec": {
                "id": self._codec.codec_id,
                "version": 1,
                "static_params": self._codec.static_params,
            },
            **self._extra_header,
        }
        await self._ws.send_bytes(framing.pack_control(json.dumps(header).encode()))
