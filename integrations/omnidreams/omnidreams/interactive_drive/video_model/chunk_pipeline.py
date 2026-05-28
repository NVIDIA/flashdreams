# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from omnidreams.interactive_drive.runtime.timing import ChunkTimes
from omnidreams.interactive_drive.types import (
    FrameChunk,
    PresentedFrame,
    SceneBundle,
    TrajectoryChunk,
)


class VideoModelBackend(Protocol):
    """Video-model interface called from the pipeline worker thread.

    Backends are *cold* after construction: ``warmup`` must be called before
    any ``render_chunk``. :class:`ChunkPipeline` is the only thing that calls
    ``warmup``, on its worker thread, before processing requests; callers
    outside the pipeline never see a cold backend.
    """

    def warmup(self, scene: SceneBundle) -> None: ...

    def render_chunk(self, trajectory: TrajectoryChunk) -> FrameChunk: ...

    def reset(self) -> None: ...


@dataclass(frozen=True)
class ChunkRequest:
    """A pose chunk plus its timing record, ready to submit to the pipeline.

    ``make_chunk_request`` in the loop builds these; ``ChunkPipeline.request_pose_chunk``
    consumes them. Keeping the pair as a single object avoids drift between
    the trajectory the worker renders and the timing record the loop will
    later index by.
    """

    trajectory: TrajectoryChunk
    chunk_times: ChunkTimes


@dataclass(frozen=True)
class QueuedFrame:
    frame: PresentedFrame
    chunk_times: ChunkTimes
    frame_index: int


# Worker commands are closures that take the backend and return ``True`` to
# keep running or ``False`` to exit. Renders, reset, and shutdown all flow
# through the same queue so ordering is FIFO without runtime type dispatch.
_WorkerCommand = Callable[["VideoModelBackend"], bool]


class ChunkPipeline:
    def __init__(self, backend: VideoModelBackend, scene: SceneBundle) -> None:
        self._backend = backend
        self._scene = scene
        # TODO: replace the loop's chunk-level ``chunks_outstanding`` gate with
        # frame-level in-flight tracking (frames requested - frames consumed,
        # alpasim style) and surface a hook here so callers gate at the
        # request site instead of the queue boundary. Until then the queue is
        # unbounded so ``put`` cannot deadlock the worker against shutdown.
        self._frame_queue: queue.Queue[QueuedFrame] = queue.Queue()
        self._command_queue: queue.Queue[_WorkerCommand] = queue.Queue()
        # Captures any exception raised on the worker thread (warmup, render,
        # backend.reset) so the next public method call surfaces it on the
        # caller's thread instead of silently leaking the worker.
        self._worker_error_lock = threading.Lock()
        self._worker_error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._worker,
            name="interactive_drive-chunk-pipeline",
            daemon=True,
        )
        self._thread.start()

    @property
    def frame_queue(self) -> "queue.Queue[QueuedFrame]":
        self._raise_worker_error_if_any()
        return self._frame_queue

    def request_pose_chunk(self, request: ChunkRequest) -> None:
        self._raise_worker_error_if_any()

        chunk_times = request.chunk_times
        trajectory = request.trajectory

        def render_command(backend: VideoModelBackend) -> bool:
            chunk_times.chunk_render_start_time = time.perf_counter()
            frame_chunk = backend.render_chunk(trajectory)
            chunk_times.chunk_ready_time = time.perf_counter()
            for frame_index, frame in enumerate(frame_chunk.frames):
                frame_times = chunk_times.frames[frame_index]
                frame_times.image_ready_time = time.perf_counter()
                self._frame_queue.put(
                    QueuedFrame(
                        frame=frame, chunk_times=chunk_times, frame_index=frame_index
                    )
                )
            return True

        self._command_queue.put(render_command)

    def reset(self) -> None:
        """Signal the worker to start a new rollout. Non-blocking.

        The worker handles in-flight renders FIFO before processing the
        reset, so a brief stretch of old-rollout frames may still be
        presented before new-rollout frames arrive. That is accepted; the
        alternative (blocking-and-draining) would freeze the display for
        the duration of the in-flight chunk's render, which is worse UX.
        """
        self._raise_worker_error_if_any()

        def reset_command(backend: VideoModelBackend) -> bool:
            backend.reset()
            return True

        self._command_queue.put(reset_command)

    def shutdown(self) -> None:
        self._command_queue.put(_shutdown_command)
        self._thread.join()
        self._raise_worker_error_if_any()

    def _worker(self) -> None:
        try:
            warmup_start = time.perf_counter()
            print("[chunk-pipeline] warmup start", flush=True)
            self._backend.warmup(self._scene)
            warmup_elapsed_ms = (time.perf_counter() - warmup_start) * 1000.0
            print(
                f"[chunk-pipeline] warmup done elapsed_ms={warmup_elapsed_ms:.1f}",
                flush=True,
            )
            while True:
                command = self._command_queue.get()
                if not command(self._backend):
                    return
        except BaseException as exc:
            with self._worker_error_lock:
                self._worker_error = exc

    def _raise_worker_error_if_any(self) -> None:
        with self._worker_error_lock:
            error = self._worker_error
        if error is not None:
            raise error


def _shutdown_command(backend: VideoModelBackend) -> bool:
    del backend
    return False
