# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np
import torch
from aiortc import MediaStreamTrack
from aiortc.mediastreams import MediaStreamError
from av import VideoFrame
from av.packet import Packet
from loguru import logger

if TYPE_CHECKING:
    from flashdreams.serving.webrtc.encode import EncodedVideoPacket

_STALL_THRESHOLD_MS = 1.0
_PACING_LAG_LOG_MS = 5.0


def tensor_chunk_to_rgb_frames(video_chunk: torch.Tensor) -> list[np.ndarray]:
    """Convert common model output tensor layouts to RGB uint8 frames."""
    if video_chunk.ndim == 4:
        frames = video_chunk.float().permute(0, 2, 3, 1).numpy()
        frames = ((frames + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
        return [np.ascontiguousarray(frame) for frame in frames]
    if video_chunk.ndim == 6:
        if video_chunk.shape[0] != 1 or video_chunk.shape[1] != 1:
            raise ValueError(
                "Expected single-batch single-view video chunk [1, 1, T, 3, H, W], "
                f"got {tuple(video_chunk.shape)}"
            )
        chunk = video_chunk[0, 0]
        if chunk.dtype == torch.uint8:
            frames = chunk.permute(0, 2, 3, 1).cpu().numpy()
        else:
            frames = chunk.float().permute(0, 2, 3, 1).cpu().numpy()
            frames = ((frames + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
        return [np.ascontiguousarray(frame) for frame in frames]
    raise ValueError(
        "Expected video chunk [T, C, H, W] or [1, 1, T, 3, H, W], "
        f"got {tuple(video_chunk.shape)}"
    )


class BufferedVideoTrack(MediaStreamTrack):
    """WebRTC video track with a bounded producer-side frame queue."""

    kind = "video"

    def __init__(
        self,
        *,
        fps: int,
        maxsize: int,
        frame_converter: Callable[[torch.Tensor], list[np.ndarray]] | None = None,
    ) -> None:
        super().__init__()
        if fps <= 0:
            raise ValueError("fps must be > 0")
        if maxsize <= 0:
            raise ValueError("maxsize must be > 0")
        self._fps = fps
        self._time_base = Fraction(1, fps)
        self._frame_interval_s = 1.0 / fps
        self._next_deadline_s: float | None = None
        self._pts = 0
        self._maxsize = maxsize
        self._frame_converter = frame_converter or tensor_chunk_to_rgb_frames
        self._frames: asyncio.Queue[np.ndarray | None] = asyncio.Queue(maxsize=maxsize)
        self._closed = False

    @property
    def fps(self) -> int:
        return self._fps

    @property
    def maxsize(self) -> int:
        return self._maxsize

    def qsize(self) -> int:
        return self._frames.qsize()

    async def enqueue_chunk(self, video_chunk: torch.Tensor) -> int:
        if self._closed:
            return 0
        frames = await asyncio.to_thread(self._frame_converter, video_chunk)
        for i, frame in enumerate(frames):
            if self._closed:
                return i
            await self._frames.put(frame)
        return len(frames)

    async def recv(self) -> VideoFrame:
        if self._closed:
            raise MediaStreamError

        loop = asyncio.get_running_loop()
        t_get_start = loop.time()
        frame_array = await self._frames.get()
        if frame_array is None:
            raise MediaStreamError
        get_wait_ms = (loop.time() - t_get_start) * 1000.0
        first_frame = self._next_deadline_s is None
        just_stalled = (not first_frame) and get_wait_ms > _STALL_THRESHOLD_MS
        if just_stalled:
            logger.debug(
                "Playback stall: pts={} waited {:.1f}ms for next frame; "
                "queue depth now {}.",
                self._pts,
                get_wait_ms,
                self._frames.qsize(),
            )

        now_s = loop.time()
        if first_frame or just_stalled:
            self._next_deadline_s = now_s
        else:
            proposed = self._next_deadline_s + self._frame_interval_s
            wait_s = proposed - now_s
            if wait_s > 0:
                await asyncio.sleep(wait_s)
                self._next_deadline_s = proposed
            else:
                if -wait_s * 1000.0 > _PACING_LAG_LOG_MS:
                    logger.debug(
                        "Pacing lag: pts={} deadline {:.1f}ms behind walltime; "
                        "re-anchoring to avoid burst (queue depth {}).",
                        self._pts,
                        -wait_s * 1000.0,
                        self._frames.qsize(),
                    )
                self._next_deadline_s = now_s

        frame = VideoFrame.from_ndarray(frame_array, format="rgb24")
        frame.pts = self._pts
        frame.time_base = self._time_base
        self._pts += 1
        return frame

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        while True:
            try:
                self._frames.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._frames.put_nowait(None)
        self.stop()


class EncodedPacketVideoTrack(MediaStreamTrack):
    """WebRTC video track that sends pre-encoded H.264 packets.

    Instead of raw frames, this track accepts :class:`EncodedVideoPacket`
    items and returns :class:`av.Packet` objects from :meth:`recv`. The
    aiortc sender bypasses its internal software encoder and passes the
    pre-encoded NAL bytes directly to the RTP packetizer.

    Requires H.264 to be the negotiated codec (see
    ``setCodecPreferences``); sending H.264 bytes through a VP8
    packetizer would produce corrupt output.
    """

    kind = "video"

    def __init__(
        self,
        *,
        fps: int,
        maxsize: int = 0,
    ) -> None:
        super().__init__()
        if fps <= 0:
            raise ValueError("fps must be > 0")
        if maxsize < 0:
            raise ValueError("maxsize must be >= 0")
        self._fps = fps
        self._time_base = Fraction(1, fps)
        self._frame_interval_s = 1.0 / fps
        self._next_deadline_s: float | None = None
        self._pts = 0
        self._maxsize = maxsize
        self._packets: asyncio.Queue[EncodedVideoPacket | None] = (
            asyncio.Queue(maxsize=maxsize) if maxsize > 0 else asyncio.Queue()
        )
        self._dropped_packets = 0
        self._closed = False

    @property
    def fps(self) -> int:
        return self._fps

    @property
    def maxsize(self) -> int:
        return self._maxsize

    def qsize(self) -> int:
        return self._packets.qsize()

    @property
    def dropped_packets(self) -> int:
        return self._dropped_packets

    async def enqueue_encoded_packets(
        self, packets: list[EncodedVideoPacket]
    ) -> int:
        """Enqueue pre-encoded H.264 packets for transmission.

        If the queue is bounded and full, the oldest packet is dropped
        to make room (drop-oldest overflow).

        Returns:
            Number of packets successfully enqueued.
        """
        if self._closed:
            return 0
        enqueued = 0
        for packet in packets:
            if self._closed:
                break
            if self._maxsize > 0 and self._packets.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    self._packets.get_nowait()
                    self._dropped_packets += 1
            await self._packets.put(packet)
            enqueued += 1
        return enqueued

    def enqueue_encoded_packet_nowait(self, packet: EncodedVideoPacket) -> bool:
        """Synchronously enqueue a single pre-encoded packet on the loop thread.

        Intended for the streaming-encode callback path: the encode worker
        thread schedules this via ``loop.call_soon_threadsafe`` so each packet
        becomes visible to ``recv`` as soon as it is produced, rather than
        waiting for the whole chunk to finish encoding. Applies the same
        drop-oldest overflow policy as :meth:`enqueue_encoded_packets`.

        Returns:
            True if the packet was enqueued, False if the track is closed.
        """
        if self._closed:
            return False
        if self._maxsize > 0 and self._packets.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                self._packets.get_nowait()
                self._dropped_packets += 1
        self._packets.put_nowait(packet)
        return True

    async def recv(self) -> Packet:
        """Return the next pre-encoded H.264 packet for RTP transmission.

        Paces delivery at the configured FPS using deadline-based
        sleeping, matching :class:`BufferedVideoTrack` behavior.
        """
        if self._closed:
            raise MediaStreamError

        loop = asyncio.get_running_loop()
        t_get_start = loop.time()
        item = await self._packets.get()
        if item is None:
            raise MediaStreamError
        get_wait_ms = (loop.time() - t_get_start) * 1000.0
        first_packet = self._next_deadline_s is None
        just_stalled = (not first_packet) and get_wait_ms > _STALL_THRESHOLD_MS
        if just_stalled:
            logger.debug(
                "Encoded track stall: pts={} waited {:.1f}ms for next packet; "
                "queue depth now {}.",
                self._pts,
                get_wait_ms,
                self._packets.qsize(),
            )

        now_s = loop.time()
        if first_packet or just_stalled:
            self._next_deadline_s = now_s
        else:
            proposed = self._next_deadline_s + self._frame_interval_s
            wait_s = proposed - now_s
            if wait_s > 0:
                await asyncio.sleep(wait_s)
                self._next_deadline_s = proposed
            else:
                if -wait_s * 1000.0 > _PACING_LAG_LOG_MS:
                    logger.debug(
                        "Encoded track pacing lag: pts={} deadline {:.1f}ms behind "
                        "walltime; re-anchoring (queue depth {}).",
                        self._pts,
                        -wait_s * 1000.0,
                        self._packets.qsize(),
                    )
                self._next_deadline_s = now_s

        packet = Packet(item.payload)
        packet.pts = self._pts
        packet.time_base = self._time_base
        self._pts += 1
        return packet

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        while True:
            try:
                self._packets.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._packets.put_nowait(None)
        self.stop()
