# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import logging
from fractions import Fraction

import numpy as np
import torch
from aiortc import MediaStreamTrack
from aiortc.mediastreams import MediaStreamError
from av import VideoFrame

LOGGER = logging.getLogger(__name__)
"""Module logger; warns on playback stalls so we can correlate with
session-level chunk timing in :mod:`lingbot.webrtc.session`."""

_STALL_THRESHOLD_MS = 1.0
"""Minimum ``await get()`` wait in milliseconds that we treat as a stall.
In steady state the queue is non-empty when ``recv`` arrives so ``get``
returns instantly; anything above ~1ms means generation did not keep
ahead of playback for this frame."""

_PACING_LAG_LOG_MS = 5.0
"""Below this lag we re-anchor pacing silently. Above it the lag is
worth a one-line warning so bursts (which the browser jitter buffer
turns into visible playback speed-ups) are correlatable in the log."""


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
    """WebRTC video track with a bounded producer-side buffer.

    The buffer is a fixed-size :class:`asyncio.Queue`; ``enqueue_chunk``'s
    ``await put`` blocks once it is full, so the producer is throttled
    to the consumer's drain rate of one frame per ``1/fps`` seconds.

    ``maxsize`` must equal the runtime's *steady-state* per-chunk
    frame count so the queue holds *exactly* one chunk in flight in
    steady state. With a larger queue the producer would buffer extra
    latency on top of the design floor; with a smaller queue the
    producer would block mid-chunk and the consumer-paced backpressure
    model in :meth:`enqueue_chunk` breaks. The construction site is
    therefore responsible for asking the runtime for the
    steady-state ``num_frames`` and passing it in — there is no
    sensible default.

    Sizing to the steady-state count (rather than to AR step 0's
    output) matters: in the lingbot/Wan pipelines AR 0 produces fewer
    frames than every subsequent step due to causal first-frame
    padding, so a queue sized to AR 0 would over-backpressure every
    steady-state chunk and create a once-per-chunk playback stall.
    """

    kind = "video"

    def __init__(self, *, fps: int, maxsize: int) -> None:
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
        self._frames: asyncio.Queue[np.ndarray | None] = asyncio.Queue(maxsize=maxsize)
        self._closed = False

    @property
    def fps(self) -> int:
        """Configured playback frame rate."""
        return self._fps

    @property
    def maxsize(self) -> int:
        """Hard upper bound on buffered frames; the producer blocks past this."""
        return self._maxsize

    def qsize(self) -> int:
        """Return the number of pre-generated frames waiting to be sent.

        Cheap snapshot intended for diagnostics; the value can become
        stale the moment ``recv`` or ``enqueue_chunk`` runs again.
        """
        return self._frames.qsize()

    async def enqueue_chunk(self, video_chunk: torch.Tensor) -> int:
        """Push one generated chunk into the playback queue.

        Returns the number of frames successfully enqueued. If the track
        is closed mid-chunk, the partial count is returned so the caller
        can log accurately instead of dead-awaiting ``put`` on a queue
        that ``close`` is about to drain.
        """
        if self._closed:
            return 0
        # Offload the bfloat16->uint8 conversion to a worker thread:
        # on a 720p chunk it can dominate per-frame budgets if it runs
        # on the asyncio loop, which would starve ``recv``'s 1/fps
        # pacing and force the re-anchor branch to keep absorbing the
        # drift on every chunk boundary.
        frames = await asyncio.to_thread(tensor_chunk_to_rgb_frames, video_chunk)
        for i, frame in enumerate(frames):
            if self._closed:
                return i
            # Once the queue saturates, this ``await`` blocks until
            # ``recv`` has drained a slot; that is the sole rate-limiter
            # that paces the producer to ``fps``.
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
        # ``_next_deadline_s is None`` is the single source of truth for
        # "we haven't emitted any frame yet". The pre-first-frame wait
        # is the time aiortc spends calling ``recv`` before the producer
        # has generated anything; it is expected, not a stall, so we
        # neither warn about it nor re-anchor on it as if recovering.
        first_frame = self._next_deadline_s is None
        just_stalled = (not first_frame) and get_wait_ms > _STALL_THRESHOLD_MS
        if just_stalled:
            LOGGER.warning(
                "Playback stall: pts=%d waited %.1fms for next frame; queue depth now %d.",
                self._pts,
                get_wait_ms,
                self._frames.qsize(),
            )

        now_s = loop.time()
        if first_frame or just_stalled:
            # First frame, or recovering from a queue stall: anchor pacing
            # at ``now`` instead of adding ``frame_interval_s`` to a stale
            # absolute deadline. The catch-up behaviour (``wait_s`` deeply
            # negative for several consecutive recvs) burst-drains the
            # queue in microseconds, which (a) collapses the smooth 16fps
            # RTP cadence the browser jitter buffer expects and (b) makes
            # the *next* chunk look like another empty-queue stall, even
            # when generation outpaces playback. The result is the
            # sawtooth "stall on every chunk boundary" pattern visible in
            # the logs with otherwise-healthy ``gen_ms < play_ms``.
            self._next_deadline_s = now_s
        else:
            proposed = self._next_deadline_s + self._frame_interval_s
            wait_s = proposed - now_s
            if wait_s > 0:
                await asyncio.sleep(wait_s)
                self._next_deadline_s = proposed
            else:
                # The queue had a frame ready (no stall) but our deadline
                # is already in the past — typical causes are aiortc's
                # send loop lagging briefly, ``asyncio.sleep`` over-sleeping,
                # or another task hogging the loop. Without re-anchoring,
                # the next several recv()s would also see ``wait_s < 0``
                # and we'd emit every queued frame as fast as aiortc asks
                # for them; the browser jitter buffer responds by speeding
                # playback up to consume the burst, which the viewer sees
                # as "video suddenly plays very quickly". Anchor at
                # ``now_s`` so subsequent frames resume the smooth 1/fps
                # cadence; the cost is a single early frame per anchor.
                if -wait_s * 1000.0 > _PACING_LAG_LOG_MS:
                    LOGGER.warning(
                        "Pacing lag: pts=%d deadline %.1fms behind walltime; "
                        "re-anchoring to avoid burst (queue depth %d).",
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
        # Drain any buffered frames so ``put_nowait`` of the sentinel can
        # never raise ``QueueFull`` on a producer that was paced flat
        # against the bounded buffer. The session lifecycle (see
        # :meth:`_ManagedLingbotSession.close`) cancels the generation
        # worker before this runs, so no concurrent ``enqueue_chunk`` is
        # racing with the drain.
        while True:
            try:
                self._frames.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._frames.put_nowait(None)
        self.stop()
