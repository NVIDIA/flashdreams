# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np
from aiortc import MediaStreamTrack
from aiortc.mediastreams import MediaStreamError
from av import VideoFrame
from av.packet import Packet
from loguru import logger

from flashdreams.serving.realtime.media import tensor_chunk_to_rgb_frames

if TYPE_CHECKING:
    import torch

_STALL_THRESHOLD_MS = 1.0
_PACING_LAG_LOG_MS = 5.0


def _default_frame_converter(video_chunk: torch.Tensor) -> list[np.ndarray]:
    return tensor_chunk_to_rgb_frames(video_chunk, sync_device=True)


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
        self._frame_converter = frame_converter or _default_frame_converter
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


class NVENCVideoTrack(MediaStreamTrack):
    """WebRTC video track that delivers pre-encoded H.264 packets.

    Paired with ``PyNvHardwareEncoder``: :meth:`recv` returns
    :class:`av.Packet` (not :class:`av.VideoFrame`), which aiortc's
    ``RTCRtpSender`` routes through ``H264Encoder.pack()`` for RTP
    fragmentation only. The encoder sets ``pts`` and ``time_base`` on
    each packet before enqueueing; this track only paces delivery to
    ``fps`` and applies a drop-oldest overflow policy so a slow
    consumer cannot stall the encode worker.
    """

    kind = "video"

    def __init__(self, *, fps: int, maxsize: int) -> None:
        super().__init__()
        if fps <= 0:
            raise ValueError("fps must be > 0")
        if maxsize <= 0:
            raise ValueError("maxsize must be > 0")
        self._fps = fps
        self._frame_interval_s = 1.0 / fps
        self._next_deadline_s: float | None = None
        self._maxsize = maxsize
        self._packets: asyncio.Queue[Packet | None] = asyncio.Queue(
            maxsize=maxsize,
        )
        self._closed = False
        self._dropped_packets = 0

    @property
    def fps(self) -> int:
        return self._fps

    @property
    def maxsize(self) -> int:
        return self._maxsize

    @property
    def dropped_packets(self) -> int:
        return self._dropped_packets

    def qsize(self) -> int:
        return self._packets.qsize()

    def enqueue_encoded_packet_nowait(self, packet: Packet) -> bool:
        """Synchronously enqueue one packet on the loop thread.

        Called from the encode worker via ``loop.call_soon_threadsafe`` so
        packets become visible to :meth:`recv` as soon as they are
        produced, without waiting for the whole chunk to finish encoding.
        Drops the oldest queued packet on overflow so real-time streaming
        does not stall behind a slow consumer.
        """
        if self._closed:
            return False
        if self._maxsize > 0 and self._packets.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                self._packets.get_nowait()
                self._dropped_packets += 1
                logger.debug(
                    "NVENCVideoTrack overflow: dropped oldest packet "
                    "(total dropped={})",
                    self._dropped_packets,
                )
        self._packets.put_nowait(packet)
        return True

    async def recv(self) -> Packet:
        if self._closed:
            raise MediaStreamError

        loop = asyncio.get_running_loop()
        t_get_start = loop.time()
        packet = await self._packets.get()
        if packet is None:
            raise MediaStreamError
        get_wait_ms = (loop.time() - t_get_start) * 1000.0
        first_packet = self._next_deadline_s is None
        just_stalled = (not first_packet) and get_wait_ms > _STALL_THRESHOLD_MS

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
                        "NVENCVideoTrack pacing lag: deadline {:.1f}ms "
                        "behind walltime; re-anchoring (queue depth {}).",
                        -wait_s * 1000.0,
                        self._packets.qsize(),
                    )
                self._next_deadline_s = now_s
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
