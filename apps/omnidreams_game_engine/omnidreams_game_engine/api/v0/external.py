# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An external engine as the rollout's engine.

:class:`ExternalEngineSession` implements the ``RolloutEngine`` protocol that
:class:`~omnidreams_game_engine.model.WorldModelRollout` drives, exactly as
the in-process :class:`~omnidreams_game_engine.engine.GameEngine` does. Its
``step`` does not simulate: it takes the chunk of states the engine sent and
renders the condition from it. Everything here runs on the model thread.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from omnidreams_game_engine.api.v0.assembly import ExternalChunk
from omnidreams_game_engine.api.v0.messages import Frame, Hello, Message
from omnidreams_game_engine.api.v0.state import FrameState
from omnidreams_game_engine.api.v0.timeline import ChunkTimeline
from omnidreams_game_engine.contracts import ConditionRenderer
from omnidreams_game_engine.engine import EngineStep
from omnidreams_game_engine.types import DriverCommand

CONNECTED = "connected"
START = "start"
SEED = "seed"
RESTART = "restart"
HOLD = "hold"
RELEASE = "release"
END = "end"
LOST = "lost"
ERROR = "error"

EVENT_KINDS = (CONNECTED, START, SEED, RESTART, HOLD, RELEASE, END, LOST, ERROR)
"""Kinds of :class:`SourceEvent`."""

CONTROL_EVENTS = frozenset({CONNECTED, START, SEED, RESTART, END, LOST})
"""Events that change the epoch or the session: chunk waits return early."""


@dataclass(frozen=True, slots=True)
class SeedFrame:
    """The image and state that start an epoch."""

    epoch: int
    tick: int
    state: FrameState
    image: bytes
    encoding: str | None
    width: int | None
    height: int | None
    message: Frame


@dataclass(frozen=True, slots=True)
class SourceEvent:
    """Something the engine did, delivered on the model thread."""

    kind: str
    message: Message | None = None
    hello: Hello | None = None
    """For ``connected``: the engine's hello."""
    reconnect: bool = False
    """For ``connected``: the same engine session connected again."""
    seed: SeedFrame | None = None
    reason: str | None = None


class EngineSource(Protocol):
    """Where an external engine's states come from and its frames go.

    Implementations: :class:`~.link_tcp.TcpEngineLink` (a live engine),
    :class:`~.replay.PacketLogSource` (a recorded session), and recorded-capture
    readers in model hosts. Every method is called on the model thread.
    """

    def poll_events(self, timeout_s: float = 0.0) -> list[SourceEvent]:
        """Return the events received since the last call, waiting up to ``timeout_s``."""
        ...

    def begin_epoch(self, epoch: int, timeline: ChunkTimeline) -> None:
        """Assemble frames of ``epoch`` from now on; drop all others."""
        ...

    def wait_chunk(
        self, epoch: int, ar_index: int, timeout_s: float
    ) -> ExternalChunk | None:
        """Return the complete chunk, or ``None`` on timeout or a pending control event."""
        ...

    def send(self, message: Message, payload: bytes = b"") -> None:
        """Send one message to the engine (dropped when no engine is connected)."""
        ...

    def send_frames(self, frames: Sequence[tuple[Message, bytes]]) -> None:
        """Send the generated frames of one chunk, in order."""
        ...

    def close(self) -> None: ...


class ChunkTimeoutError(TimeoutError):
    """The engine did not complete a chunk in time."""


class ExternalEngineSession:
    """``RolloutEngine`` whose states come from an external engine.

    One instance serves one epoch. ``step`` uses a chunk handed over with
    :meth:`supply` (a driver that already waited for it) or waits on the
    source itself, up to ``chunk_timeout_s``. The commands passed by the
    rollout only fix the frame count: the engine applies its own inputs and
    reports them per frame. The condition renderer is shared across epochs
    and owned by the caller; :meth:`close` leaves it open.
    """

    def __init__(
        self,
        *,
        source: EngineSource,
        epoch: int,
        condition_renderer: ConditionRenderer,
        timeline: ChunkTimeline,
        base_timestamp_us: int,
        initial_frame: FrameState | None = None,
        chunk_timeout_s: float = 5.0,
    ) -> None:
        self.source = source
        self.epoch = epoch
        self.condition_renderer = condition_renderer
        self.timeline = timeline
        self.base_timestamp_us = int(base_timestamp_us)
        self.chunk_timeout_s = float(chunk_timeout_s)
        self._ar_index = 0
        self._supplied: ExternalChunk | None = None
        self._last_frame = initial_frame
        self._closed = False

    @property
    def is_running(self) -> bool:
        return not self._closed

    @property
    def current_game_frame(self) -> FrameState | None:
        """The newest engine state this session consumed."""
        return self._last_frame

    @property
    def next_ar_index(self) -> int:
        return self._ar_index

    def supply(self, chunk: ExternalChunk) -> None:
        """Hand over the chunk the next ``step`` generates."""
        if chunk.epoch != self.epoch or chunk.ar_index != self._ar_index:
            raise ValueError(
                f"Chunk {chunk.epoch}/{chunk.ar_index} does not follow "
                f"{self.epoch}/{self._ar_index}"
            )
        self._supplied = chunk

    def submit_text(self, value: str) -> object:
        """External engines take text through their own UI."""
        del value
        return None

    def step(self, commands: tuple[DriverCommand, ...]) -> EngineStep:
        """Render the condition of the next chunk the engine sent."""
        if self._closed:
            raise RuntimeError("ExternalEngineSession is closed")
        if not commands:
            raise ValueError("A game-engine step requires at least one command")
        wall_started = time.perf_counter()
        cpu_started = time.thread_time()
        chunk = self._supplied
        self._supplied = None
        if chunk is None:
            chunk = self.source.wait_chunk(
                self.epoch, self._ar_index, self.chunk_timeout_s
            )
            if chunk is None:
                raise ChunkTimeoutError(
                    f"Epoch {self.epoch} chunk {self._ar_index} incomplete after "
                    f"{self.chunk_timeout_s:.1f} s"
                )
        waited_ms = (time.perf_counter() - wall_started) * 1000.0
        if len(chunk.frames) != len(commands):
            raise ValueError(
                f"Chunk {chunk.ar_index} has {len(chunk.frames)} frames; "
                f"the rollout expects {len(commands)}"
            )
        build_started = time.perf_counter()
        trajectory = chunk.to_trajectory(
            self.timeline, base_timestamp_us=self.base_timestamp_us
        )
        build_ms = (time.perf_counter() - build_started) * 1000.0
        conditioning_started = time.perf_counter()
        conditioning_cpu_started = time.thread_time()
        condition = self.condition_renderer.render(trajectory)
        conditioning_wall_ms = (time.perf_counter() - conditioning_started) * 1000.0
        conditioning_cpu_ms = (time.thread_time() - conditioning_cpu_started) * 1000.0
        if int(condition.hdmap_bvtchw.shape[2]) != len(commands):
            raise ValueError("Condition frames must align with the engine's frames")
        self._ar_index += 1
        self._last_frame = chunk.frames[-1]
        return EngineStep(
            trajectory=trajectory,
            game_frames=chunk.frames,
            condition=condition,
            metrics={
                "external_wait_ms": waited_ms,
                "external_build_ms": build_ms,
                "conditioning_wall_ms": conditioning_wall_ms,
                "conditioning_cpu_ms": conditioning_cpu_ms,
                "engine_step_wall_ms": (time.perf_counter() - wall_started) * 1000.0,
                "engine_step_cpu_ms": (time.thread_time() - cpu_started) * 1000.0,
            },
        )

    def close(self) -> None:
        """End this epoch's session; the shared renderer stays open."""
        self._closed = True
        self._supplied = None


__all__ = [
    "CONNECTED",
    "CONTROL_EVENTS",
    "END",
    "ERROR",
    "EVENT_KINDS",
    "HOLD",
    "LOST",
    "RELEASE",
    "RESTART",
    "SEED",
    "START",
    "ChunkTimeoutError",
    "EngineSource",
    "ExternalEngineSession",
    "SeedFrame",
    "SourceEvent",
]
