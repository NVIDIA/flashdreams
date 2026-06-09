# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from loguru import logger
from omnidreams.interactive_drive.runtime.timing import ChunkTimes
from omnidreams.interactive_drive.types import (
    FrameChunk,
    PresentedFrame,
    SceneBundle,
    TrajectoryChunk,
)


class VideoModelBackend(Protocol):
    """Video-model interface called from the pipeline worker thread.

    Backends are *cold* after construction. :class:`ChunkPipeline` calls
    ``warmup_model`` once on its worker thread (model load/compile,
    scene-independent), then ``load_scene`` for each scene before any
    ``render_chunk`` against it. Callers outside the pipeline never see a
    cold backend, and switching scenes re-runs only ``load_scene`` -- the
    warmed model stays resident.
    """

    def warmup_model(self) -> None: ...

    def load_scene(self, scene: SceneBundle) -> None: ...

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
    # Pipeline generation this frame was rendered under. A reset / scene
    # switch bumps the generation; the loop drops frames whose generation
    # no longer matches so stale rollout/scene frames aren't presented.
    generation: int = 0


# Worker commands are closures that take the backend and return ``True`` to
# keep running or ``False`` to exit. Renders, reset, and shutdown all flow
# through the same queue so ordering is FIFO without runtime type dispatch.
_WorkerCommand = Callable[["VideoModelBackend"], bool]


class ChunkPipeline:
    def __init__(self, backend: VideoModelBackend) -> None:
        self._backend = backend
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
        # Set once ``warmup_model`` finishes on the worker thread (or fails).
        # Lets callers overlap the scene-selection wait with the model load
        # and show a "ready" affordance once the model is resident.
        self._model_ready = threading.Event()
        # Set once the worker queues its first generated chunk -- i.e. the
        # one-time first-chunk optimization is done. Never cleared; the model
        # stays optimized across resets and scene switches.
        self._first_chunk_produced = threading.Event()
        # Monotonic generation bumped on every reset / scene switch. Renders
        # submitted under an older generation are superseded: their frames
        # are dropped instead of presented, so a reset or scene load doesn't
        # first flash stale frames from the rollout it replaced. The worker
        # can't interrupt an in-flight torch generate(), but its output is
        # discarded -- the single-process analog of alpasim cancelling the
        # runtime stream and clearing its frame queues on reload.
        self._generation_lock = threading.Lock()
        self._generation = 0
        self._thread = threading.Thread(
            target=self._worker,
            name="interactive_drive-chunk-pipeline",
            daemon=True,
        )
        self._thread.start()

    @property
    def model_ready(self) -> threading.Event:
        """Event set when scene-independent model warmup has completed."""
        return self._model_ready

    @property
    def first_chunk_produced(self) -> threading.Event:
        """Event set once the worker has queued its first generated chunk."""
        return self._first_chunk_produced

    @property
    def current_generation(self) -> int:
        """Current generation token; frames tagged with an older value are stale."""
        with self._generation_lock:
            return self._generation

    def _bump_generation(self) -> int:
        with self._generation_lock:
            self._generation += 1
            return self._generation

    def _clear_frame_queue(self) -> int:
        """Drop already-produced frames superseded by a reset / scene switch."""
        cleared = 0
        while True:
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                return cleared
            cleared += 1

    @property
    def frame_queue(self) -> "queue.Queue[QueuedFrame]":
        self._raise_worker_error_if_any()
        return self._frame_queue

    def request_scene(self, scene: SceneBundle) -> None:
        """Bind ``scene`` on the worker thread. Non-blocking.

        Enqueued FIFO behind warmup and any in-flight renders, so a scene
        picked before warmup finishes simply waits for the model load. The
        worker runs ``backend.load_scene`` (geometry upload + rollout
        restart); the warmed model stays resident, so switching scenes
        never re-pays the warmup/compile cost.
        """
        self._raise_worker_error_if_any()
        # Supersede any in-flight / queued render so its frames are dropped
        # rather than briefly shown over the new scene's load. The generation
        # also guards queued scene-load commands themselves: a click can arrive
        # while a previous load is still sitting behind model warmup, and that
        # old load must not bind its prompt/seed after the newer selection wins.
        submit_generation = self._bump_generation()
        cleared = self._clear_frame_queue()
        if cleared:
            logger.info(
                "[chunk-pipeline] cleared stale frame queue "
                f"frames={cleared} generation={submit_generation}",
            )

        def load_scene_command(backend: VideoModelBackend) -> bool:
            if submit_generation != self.current_generation:
                logger.info(
                    "[chunk-pipeline] skip stale scene load "
                    f"scene={scene.scene_path.name!r} "
                    f"submit_generation={submit_generation} "
                    f"current_generation={self.current_generation}",
                )
                return True
            backend.load_scene(scene)
            return True

        self._command_queue.put(load_scene_command)

    def request_pose_chunk(self, request: ChunkRequest) -> None:
        self._raise_worker_error_if_any()

        chunk_times = request.chunk_times
        trajectory = request.trajectory
        submit_generation = self.current_generation

        def render_command(backend: VideoModelBackend) -> bool:
            chunk_times.chunk_render_start_time = time.perf_counter()
            if submit_generation != self.current_generation:
                chunk_times.chunk_ready_time = time.perf_counter()
                logger.info(
                    "[chunk-pipeline] skip stale render "
                    f"submit_generation={submit_generation} "
                    f"current_generation={self.current_generation}",
                )
                return True
            frame_chunk = backend.render_chunk(trajectory)
            chunk_times.chunk_ready_time = time.perf_counter()
            # Drop the output if a reset / scene switch superseded this chunk
            # while it was queued or rendering -- its frames belong to a
            # rollout the user has already moved on from.
            if submit_generation != self.current_generation:
                return True
            # Latch before enqueuing so a consumer can't dequeue and present
            # the first frame while first_chunk_produced() still reads False.
            if frame_chunk.frames:
                self._first_chunk_produced.set()
            for frame_index, frame in enumerate(frame_chunk.frames):
                frame_times = chunk_times.frames[frame_index]
                frame_times.image_ready_time = time.perf_counter()
                self._frame_queue.put(
                    QueuedFrame(
                        frame=frame,
                        chunk_times=chunk_times,
                        frame_index=frame_index,
                        generation=submit_generation,
                    )
                )
            return True

        self._command_queue.put(render_command)

    def reset(self) -> None:
        """Signal the worker to start a new rollout. Non-blocking.

        Bumps the generation so any in-flight / queued render is superseded:
        its frames are dropped rather than presented, so the reset doesn't
        first replay a stretch of old-rollout frames (the single-process
        analog of alpasim cancelling the runtime stream and clearing its
        frame queues). The in-flight torch generate() can't be interrupted,
        but its output is discarded; the worker still handles the reset
        FIFO so the next rollout starts from a clean cache.
        """
        self._raise_worker_error_if_any()
        generation = self._bump_generation()
        cleared = self._clear_frame_queue()
        if cleared:
            logger.info(
                "[chunk-pipeline] cleared stale frame queue "
                f"frames={cleared} generation={generation}",
            )

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
            logger.info("[chunk-pipeline] warmup start")
            self._backend.warmup_model()
            warmup_elapsed_ms = (time.perf_counter() - warmup_start) * 1000.0
            logger.info(
                f"[chunk-pipeline] warmup done elapsed_ms={warmup_elapsed_ms:.1f}",
            )
            self._model_ready.set()
            while True:
                command = self._command_queue.get()
                if not command(self._backend):
                    return
        except BaseException as exc:
            with self._worker_error_lock:
                self._worker_error = exc
            # Unblock anyone waiting on warmup; the error resurfaces on the
            # next public call via _raise_worker_error_if_any.
            self._model_ready.set()

    def _raise_worker_error_if_any(self) -> None:
        with self._worker_error_lock:
            error = self._worker_error
        if error is not None:
            raise error


def _shutdown_command(backend: VideoModelBackend) -> bool:
    del backend
    return False
