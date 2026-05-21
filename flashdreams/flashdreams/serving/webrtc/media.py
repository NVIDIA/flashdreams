# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from collections.abc import Callable
from fractions import Fraction

import numpy as np
import torch
from aiortc import MediaStreamTrack
from aiortc.mediastreams import MediaStreamError
from av import VideoFrame
from loguru import logger

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
