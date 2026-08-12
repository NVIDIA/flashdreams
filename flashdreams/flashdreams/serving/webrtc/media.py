# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Sequence
from fractions import Fraction
from typing import cast

import numpy as np
from aiortc import MediaStreamTrack
from aiortc.mediastreams import MediaStreamError
from av import VideoFrame
from av.packet import Packet
from loguru import logger

from flashdreams.runtime import StepResult
from flashdreams.serving.realtime.media import (
    FrameLayout,
    ValueRange,
    rgb_array_to_uint8_frames,
)
from flashdreams.serving.realtime.media import (
    tensor_chunk_to_rgb_frames as tensor_chunk_to_rgb_frames,
)

_STALL_THRESHOLD_MS = 1.0
_PACING_LAG_LOG_MS = 5.0


def _default_frame_converter(result: StepResult) -> list[np.ndarray]:
    video_chunk = result.video_chunk
    value_range: ValueRange = (
        "minus_one_one" if video_chunk.is_floating_point() else "uint8"
    )
    return rgb_array_to_uint8_frames(
        video_chunk,
        layout=cast(FrameLayout, result.layout),
        value_range=value_range,
        sync_device=True,
    )


class BufferedVideoTrack(MediaStreamTrack):
    """WebRTC video track with a bounded producer-side frame queue."""

    kind = "video"

    def __init__(
        self,
        *,
        fps: int,
        maxsize: int,
        frame_converter: Callable[[StepResult], list[np.ndarray]] | None = None,
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

    def prepare_result_frames(self, result: StepResult) -> tuple[np.ndarray, ...]:
        if self._closed:
            return ()
        return tuple(self._frame_converter(result))

    async def enqueue_frames(self, frames: Sequence[np.ndarray]) -> int:
        if self._closed:
            return 0
        for i, frame in enumerate(frames):
            if self._closed:
                return i
            await self._frames.put(frame)
        return len(frames)

    async def enqueue_result(self, result: StepResult) -> int:
        if self._closed:
            return 0
        frames = await asyncio.to_thread(self.prepare_result_frames, result)
        return await self.enqueue_frames(frames)

    async def flush(self) -> None:
        """Drop queued frames while keeping the RTP timestamp sequence alive."""
        if self._closed:
            return
        while True:
            try:
                self._frames.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._next_deadline_s = None

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
    ``fps``. The async enqueue path applies backpressure so the browser
    receives every frame in timestamp order instead of seeing silent
    server-side drops.
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

    async def enqueue_encoded_packet(self, packet: Packet) -> bool:
        """Enqueue one encoded packet, waiting for sender-side queue space."""
        if self._closed:
            return False
        await self._packets.put(packet)
        return True

    async def enqueue_encoded_packets(self, packets: Sequence[Packet]) -> int:
        """Enqueue encoded packets in order, applying backpressure if full."""
        for i, packet in enumerate(packets):
            if not await self.enqueue_encoded_packet(packet):
                return i
        return len(packets)

    def enqueue_encoded_packet_nowait(self, packet: Packet) -> bool:
        """Synchronously enqueue one packet on the loop thread.

        This compatibility helper is intentionally lossy on overflow.
        The production NVENC path uses :meth:`enqueue_encoded_packets`
        so playback preserves every frame; tests and diagnostic probes
        can still use this helper when they explicitly want nonblocking
        enqueue semantics.
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

    async def flush(self) -> None:
        """Drop queued encoded packets while preserving the open media track."""
        if self._closed:
            return
        while True:
            try:
                self._packets.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._next_deadline_s = None

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
