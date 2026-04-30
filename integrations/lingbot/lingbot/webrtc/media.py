from __future__ import annotations

import asyncio
from fractions import Fraction

import numpy as np
import torch
from aiortc import MediaStreamTrack
from aiortc.mediastreams import MediaStreamError
from av import VideoFrame


def tensor_chunk_to_rgb_frames(video_chunk: torch.Tensor) -> list[np.ndarray]:
    """Convert Lingbot output tensor [B, V, T, C, H, W] to RGB uint8 frames."""
    if video_chunk.ndim != 6:
        raise ValueError(
            f"Expected video chunk with 6 dimensions [B, V, T, C, H, W], got {video_chunk.shape}"
        )
    if video_chunk.shape[0] < 1 or video_chunk.shape[1] < 1:
        raise ValueError("Video chunk must contain at least one batch and one view.")

    frames = video_chunk[0, 0].float().permute(0, 2, 3, 1).numpy()
    frames = ((frames + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    return [np.ascontiguousarray(frame) for frame in frames]


class LingbotVideoTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, fps: int = 16) -> None:
        super().__init__()
        if fps <= 0:
            raise ValueError("fps must be > 0")
        self._fps = fps
        self._time_base = Fraction(1, fps)
        self._frame_interval_s = 1.0 / fps
        self._next_deadline_s: float | None = None
        self._pts = 0
        self._frames: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
        self._closed = False

    async def enqueue_chunk(self, video_chunk: torch.Tensor) -> int:
        frames = tensor_chunk_to_rgb_frames(video_chunk)
        for frame in frames:
            await self._frames.put(frame)
        return len(frames)

    async def recv(self) -> VideoFrame:
        if self._closed:
            raise MediaStreamError

        frame_array = await self._frames.get()
        if frame_array is None:
            raise MediaStreamError

        loop = asyncio.get_running_loop()
        now_s = loop.time()
        if self._next_deadline_s is None:
            self._next_deadline_s = now_s
        else:
            self._next_deadline_s += self._frame_interval_s
            wait_s = self._next_deadline_s - now_s
            if wait_s > 0:
                await asyncio.sleep(wait_s)

        frame = VideoFrame.from_ndarray(frame_array, format="rgb24")
        frame.pts = self._pts
        frame.time_base = self._time_base
        self._pts += 1
        return frame

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._frames.put(None)
        self.stop()
