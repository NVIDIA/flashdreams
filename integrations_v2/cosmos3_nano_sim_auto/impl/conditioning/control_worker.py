# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Draw and encode the live controls in a process of their own.

The model thread spends a whole chunk on the CPU launching kernels, so a thread
that renders and encodes the next chunk's controls only shares its GIL: both
get slower by what the other needs, and the chunk time does not move. A process
has its own interpreter. It draws this rank's cameras with ludus, runs them
through the Wan encoder's streams, and hands back a few hundred kilobytes of
latents per chunk over a pipe, while the model thread never stops launching.
"""

from __future__ import annotations

import os
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing.connection import _ConnectionBase
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from torch import Tensor

__all__ = ["ControlWorker", "WorkerSpec", "worker_gpu"]


def worker_gpu(device: torch.device, offset: int) -> int:
    """The GPU a rank's worker draws on: the rank's own index plus ``offset``, modulo the node's count.

    Warns where that lands on another model rank's GPU, which is assumed to be
    local ranks ``0..LOCAL_WORLD_SIZE-1`` on GPUs of the same index.

    Raises:
        ValueError: ``offset`` is negative.
    """
    if offset < 0:
        raise ValueError(f"--prefetch-controls-gpu-offset must be >= 0, got {offset}.")
    mine = torch.cuda.current_device() if device.index is None else device.index
    count = torch.cuda.device_count()
    target = (mine + offset) % count
    local = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    if offset and count == 1:
        logger.warning(
            "--prefetch-controls-gpu-offset {} means nothing on a node with one GPU; "
            "the control worker shares it.",
            offset,
        )
    elif target != mine and target < local:
        logger.warning(
            "the control worker for GPU {} lands on GPU {}, which model rank {} of "
            "this node uses; an offset of at least LOCAL_WORLD_SIZE={} keeps them apart.",
            mine,
            target,
            target,
            local,
        )
    return target


@dataclass(frozen=True)
class WorkerSpec:
    """Everything the worker needs to open the clip and load the encoder, picklable."""

    clip: Path
    cameras: tuple[str, ...]
    owned: tuple[int, ...]
    device: int
    against: Path | None
    start_frame: int
    wanted: int | None
    dtype: torch.dtype
    compile_encode: bool
    requested_fps: float | None = None
    """Model frame rate to validate against a prepared scene's clock."""
    driven: bool = False
    """Draw along poses sent with each request instead of the clip's recorded
    track, and send the drawn frames back with the latents. A host application
    owns the car and presents the map, so both cross the pipe."""


class ControlWorker:
    """One range in flight at a time: :meth:`request` it early, :meth:`take` it later.

    Ranges must be taken in the order requested and each must carry on from the
    last, which is what the encoder's temporal state needs; :meth:`take` with
    nothing in flight asks and waits, so the opening window needs no request.
    """

    def __init__(self, conn: _ConnectionBase, *, device: torch.device) -> None:
        self._process: BaseProcess | None = None
        self._conn = conn
        self._device = device
        self._pending: tuple[int, int] | None = None
        self.last_drawn: Tensor | None = None
        """A driven worker's ``[len(owned)*T, H, W, 3]`` uint8 frames for the range last taken."""

    @classmethod
    def spawn(
        cls,
        spec: WorkerSpec,
        *,
        device: torch.device,
        target: Callable[[_ConnectionBase, WorkerSpec], None] | None = None,
    ) -> ControlWorker:
        """Start the process and wait until its scene and encoder are up.

        ``target`` replaces :func:`_serve`, for tests.
        """
        ctx = torch.multiprocessing.get_context("spawn")
        ours, theirs = ctx.Pipe()
        process = ctx.Process(target=target or _serve, args=(theirs, spec), daemon=True)
        process.start()
        # Only the child may hold its end, or a child that dies hard leaves the
        # pipe open and every recv here waits for ever, and the mesh with it.
        theirs.close()
        try:
            ready = ours.recv()
        except EOFError as died:
            raise RuntimeError(
                f"the control worker exited with code {process.exitcode} before it was ready"
            ) from died
        if isinstance(ready, BaseException):
            raise RuntimeError("the control worker failed to start") from ready
        worker = cls(ours, device=device)
        worker._process = process
        return worker

    def request(self, start: int, end: int, *, poses: np.ndarray | None = None) -> None:
        """Start drawing and encoding latent frames ``[start, end)`` now.

        ``poses`` is the anchor camera's ``[T, 4, 4]`` path over the range's pixel
        frames, which a driven worker draws along.

        Raises:
            ValueError: a different range is already in flight.
        """
        if self._pending is not None:
            if self._pending == (start, end):
                return
            raise ValueError(
                f"[{self._pending[0]}, {self._pending[1]}) is already in flight; "
                f"[{start}, {end}) cannot be until it is taken."
            )
        self._conn.send((start, end) if poses is None else (start, end, poses))
        self._pending = (start, end)

    def take(self, start: int, end: int) -> tuple[list[Tensor], int]:
        """This rank's encoded cameras for ``[start, end)``, and the pixel frames they came from.

        Raises:
            ValueError: a different range is in flight.
            RuntimeError: the worker failed; its traceback is the message.
        """
        self.request(start, end)
        assert self._pending is not None
        if self._pending != (start, end):
            raise ValueError(
                f"[{start}, {end}) was asked for but [{self._pending[0]}, "
                f"{self._pending[1]}) is in flight; the encoder consumed that span."
            )
        try:
            reply = self._conn.recv()
        except EOFError as died:
            process = getattr(self, "_process", None)
            code = None if process is None else process.exitcode
            raise RuntimeError(f"the control worker exited (code {code})") from died
        self._pending = None
        if isinstance(reply, str):
            raise RuntimeError(f"the control worker failed:\n{reply}")
        latents, pixel_frames, *drawn = reply
        self.last_drawn = drawn[0] if drawn else None
        return [
            latent.to(self._device, non_blocking=True) for latent in latents
        ], pixel_frames

    def close(self) -> None:
        """Stop the process: the pipe closing asks, and a terminate makes sure."""
        self._conn.close()
        process = getattr(self, "_process", None)
        if process is not None:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)


def _serve(conn: _ConnectionBase, spec: WorkerSpec) -> None:
    """The worker's whole life: open, then draw and encode what the pipe asks for."""
    try:
        from cosmos3_nano_sim_auto.impl.conditioning import ludus_controls
        from cosmos3_nano_sim_auto.impl.conditioning.control_provider import (
            pixel_span_for_latents,
            rendered_clips,
        )
        from cosmos3_nano_sim_auto.inputs import make_control_encode_streams

        torch.cuda.set_device(spec.device)
        device = torch.device("cuda", spec.device)
        given = _GivenPoses() if spec.driven else None
        drawn = ludus_controls.open_clip(
            spec.clip,
            list(spec.cameras),
            device=device,
            against=spec.against,
            start_frame=spec.start_frame,
            wanted=spec.wanted,
            drive=given,
            note=logger.info,
            # A driven run draws once a chunk for as long as the drive lasts.
            span_note=(lambda _: None) if given is not None else None,
            requested_fps=spec.requested_fps,
        )
        # A rank past the last camera owns none and only moves the car.
        streams = (
            make_control_encode_streams(
                len(spec.owned),
                dtype=spec.dtype,
                device=device,
                compile=spec.compile_encode,
            )
            if spec.owned
            else []
        )
    except BaseException:  # noqa: BLE001 - reported to the parent, which raises
        _send(conn, RuntimeError(traceback.format_exc()))
        return
    if not _send(conn, "ready"):
        return
    owned = list(spec.owned)
    while True:
        try:
            start, end, *poses = conn.recv()
        except (EOFError, OSError):
            return
        try:
            if given is not None:
                given.next = poses[0]
            first, last = pixel_span_for_latents(start, end)
            frames = drawn.render(first, last, views=owned)
            clips = rendered_clips(frames, pixel_frames=last - first, count=len(owned))
            latents = [
                stream(clip).to("cpu")
                for stream, clip in zip(streams, clips, strict=True)
            ]
            shown = (frames[..., :3].to("cpu"),) if given is not None else ()
            reply: Any = (latents, last - first, *shown)
        except BaseException:  # noqa: BLE001 - reported to the parent, which raises
            reply = traceback.format_exc()
        if not _send(conn, reply):
            return


def _send(conn: _ConnectionBase, message: Any) -> bool:
    """Send, or report that the parent has gone, which a restart does on purpose."""
    try:
        conn.send(message)
    except (BrokenPipeError, EOFError, ConnectionResetError):
        return False
    return True


class _GivenPoses:
    """The one ``PoseSource`` a driven worker needs: whatever came with the request."""

    next: np.ndarray | None = None

    def poses(self, first: int, last: int) -> np.ndarray:
        poses, self.next = self.next, None
        if poses is None or len(poses) != last - first:
            raise ValueError(f"no poses were sent for pixel frames [{first}, {last}).")
        return poses
