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
        # Fix #2 (eager pre-warm): set when the header is sent at attach from the
        # VAE descriptor shape, so the first real chunk can assert the generated
        # latent matches (a mismatch would corrupt SAS/raw decode).
        self._eager_shape: tuple[int, tuple[int, ...]] | None = None

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
        elif self._eager_shape is not None:
            # Fix #2: verify the eagerly-sent header matches the real latent, so a
            # shape drift surfaces loudly instead of silently corrupting decode.
            actual = (int(latent.shape[0]), tuple(int(x) for x in latent.shape[1:]))
            if actual != self._eager_shape:
                logger.error(
                    "token eager header shape {} != generated latent {}; "
                    "decode will corrupt",
                    self._eager_shape,
                    actual,
                )
            self._eager_shape = None  # check once

        await self._flow.acquire()
        self._inflight.add(chunk_index)

        total = int(latent.shape[0])
        for frame_idx in range(total):
            result = self._codec.encode_frame(latent[frame_idx])
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

    def _header_bytes(self, *, frames_per_chunk: int, latent_shape: list[int]) -> bytes:
        """Build the control-frame bytes for the session header."""
        header = {
            "protocol_version": framing.PROTOCOL_VERSION,
            "latent_shape": list(latent_shape),
            "frames_per_chunk": int(frames_per_chunk),
            "fps": self._fps,
            "codec": {
                "id": self._codec.codec_id,
                "version": 1,
                "static_params": self._codec.static_params,
            },
            **self._extra_header,
        }
        return framing.pack_control(json.dumps(header).encode())

    async def _send_session_header(self, latent: torch.Tensor) -> None:
        """Send the one-shot control frame describing the stream to the client."""
        await self._ws.send_bytes(
            self._header_bytes(
                frames_per_chunk=int(latent.shape[0]),
                latent_shape=list(latent.shape[1:]),
            )
        )

    async def send_session_header_eager(
        self, *, frames_per_chunk: int, latent_shape: list[int]
    ) -> None:
        """Fix #2 (eager pre-warm): send the session header at WS-attach time,
        before the first generated chunk, using the shape from the VAE
        descriptor. Lets the client download and compile the WebGPU VAE decoder
        during the idle window before the user drives, so token modes go live on
        the first control input instead of dropping the warmup chunks. Idempotent
        and byte-identical to the lazy path (the descriptor shape equals the
        generated latent's); the first real chunk asserts they match.
        """
        if self._header_sent:
            return
        await self._ws.send_bytes(
            self._header_bytes(
                frames_per_chunk=frames_per_chunk, latent_shape=latent_shape
            )
        )
        self._header_sent = True
        self._eager_shape = (int(frames_per_chunk), tuple(int(x) for x in latent_shape))
