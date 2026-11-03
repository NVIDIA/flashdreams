# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-thread half of a message-based :class:`~.external.EngineSource`.

A transport (the TCP link, a packet-log replay) only fetches raw messages and
transmits outgoing ones. Parsing, state decoding, chunk assembly, event
generation and recording of consumed messages happen here, on the thread
that calls :meth:`poll_events` and :meth:`wait_chunk`: the model thread. A
transport's own threads never touch the assembler or the events.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from omnidreams_game_engine.api.v0.assembly import ChunkAssembler, ExternalChunk
from omnidreams_game_engine.api.v0.external import (
    CONNECTED,
    CONTROL_EVENTS,
    END,
    ERROR,
    HOLD,
    LOST,
    RELEASE,
    RESTART,
    SEED,
    START,
    SeedFrame,
    SourceEvent,
)
from omnidreams_game_engine.api.v0.messages import (
    End,
    Error,
    Frame,
    Hello,
    Hold,
    Message,
    ProtocolError,
    Release,
    Restart,
    Start,
    Status,
    from_header,
)
from omnidreams_game_engine.api.v0.recorder import RX, PacketRecorder
from omnidreams_game_engine.api.v0.state import StateDecoder
from omnidreams_game_engine.api.v0.timeline import ChunkTimeline

_LOGGER = logging.getLogger(__name__)

LOST_TYPE = "_lost"
"""Header type of the synthetic message a transport queues when the engine is gone."""


@dataclass(frozen=True, slots=True)
class Inbound:
    """One received message, as a transport queues it."""

    header: dict[str, Any]
    payload: bytes = b""
    t_arrive_us: int = 0
    meta: dict[str, Any] = field(default_factory=dict)
    """Transport facts about the message, such as ``reconnect`` on a hello."""


@dataclass(slots=True)
class SourceStats:
    consumed: int = 0
    frames: int = 0
    seeds: int = 0
    stale_frames: int = 0
    duplicate_frames: int = 0
    bad_messages: int = 0
    bad_states: int = 0
    unknown_messages: int = 0
    sent: int = 0
    dropped_sends: int = 0
    frames_out: int = 0


class MessageEngineSource:
    """Shared ``EngineSource`` logic over a stream of protocol messages."""

    _fetch_blocks = True
    """Whether ``_fetch`` with a timeout waits for new messages."""

    def __init__(
        self,
        decoder: StateDecoder,
        *,
        recorder: PacketRecorder | None = None,
    ) -> None:
        self.decoder = decoder
        self.recorder = recorder
        self.hello: Hello | None = None
        self.last_status: Status | None = None
        self.stats = SourceStats()
        self._assembler = ChunkAssembler(ChunkTimeline())
        self._events: deque[SourceEvent] = deque()
        self._sends = 0

    # ------------------------------------------------------------------ hooks

    def _fetch(self, timeout_s: float) -> list[Inbound]:
        """Return the messages available now, waiting up to ``timeout_s`` for one."""
        raise NotImplementedError

    def _transmit(self, message: Message, payload: bytes) -> bool:
        """Send one message; return whether it went to a connected engine."""
        raise NotImplementedError

    # ------------------------------------------------------------ EngineSource

    @property
    def frames_out(self) -> int:
        """Generated frames handed to :meth:`send_frames` so far."""
        return self.stats.frames_out

    @property
    def sends(self) -> int:
        """Messages the host has sent so far, delivered or not."""
        return self._sends

    @property
    def assembler(self) -> ChunkAssembler:
        return self._assembler

    def poll_events(self, timeout_s: float = 0.0) -> list[SourceEvent]:
        self._consume(self._fetch(0.0 if self._events else max(0.0, timeout_s)))
        events = list(self._events)
        self._events.clear()
        return events

    def begin_epoch(self, epoch: int, timeline: ChunkTimeline) -> None:
        self._assembler.begin_epoch(epoch, timeline)

    def wait_chunk(
        self, epoch: int, ar_index: int, timeout_s: float
    ) -> ExternalChunk | None:
        deadline = time.monotonic() + max(0.0, timeout_s)
        while True:
            self._consume(self._fetch(0.0))
            if any(event.kind in CONTROL_EVENTS for event in self._events):
                return None
            if self._assembler.epoch == epoch:
                chunk = self._assembler.take(ar_index)
                if chunk is not None:
                    return chunk
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return None
            received = self._fetch(remaining)
            if not received and not self._fetch_blocks:
                return None
            self._consume(received)

    def send(self, message: Message, payload: bytes = b"") -> None:
        self._sends += 1
        if self._transmit(message, payload):
            self.stats.sent += 1
        else:
            self.stats.dropped_sends += 1

    def send_frames(self, frames: Sequence[tuple[Message, bytes]]) -> None:
        for message, payload in frames:
            self.send(message, payload)
        self.stats.frames_out += len(frames)

    def close(self) -> None:
        if self.recorder is not None:
            self.recorder.flush()

    # -------------------------------------------------------------- internals

    def _consume(self, inbound: Sequence[Inbound]) -> None:
        for item in inbound:
            self.stats.consumed += 1
            if self.recorder is not None:
                self.recorder.record(
                    RX,
                    item.header,
                    item.payload,
                    t_arrive_us=item.t_arrive_us,
                    host_sends=self._sends,
                    meta=item.meta,
                )
            if item.header.get("type") == LOST_TYPE:
                self._events.append(
                    SourceEvent(LOST, reason=str(item.header.get("reason", "")))
                )
                continue
            try:
                message = from_header(item.header)
            except ProtocolError as error:
                self.stats.bad_messages += 1
                _LOGGER.warning("Dropped a malformed engine message: %s", error)
                continue
            self._dispatch(message, item)

    def _dispatch(self, message: Message, item: Inbound) -> None:
        if isinstance(message, Frame):
            self._on_frame(message, item.payload)
        elif isinstance(message, Hello):
            self.hello = message
            self._events.append(
                SourceEvent(
                    CONNECTED,
                    message=message,
                    hello=message,
                    reconnect=bool(item.meta.get("reconnect", False)),
                )
            )
        elif isinstance(message, Start):
            self._events.append(
                SourceEvent(START, message=message, reason=message.reason)
            )
        elif isinstance(message, Restart):
            self._events.append(
                SourceEvent(RESTART, message=message, reason=message.reason)
            )
        elif isinstance(message, Hold):
            self._events.append(
                SourceEvent(HOLD, message=message, reason=message.reason)
            )
        elif isinstance(message, Release):
            self._events.append(
                SourceEvent(RELEASE, message=message, reason=message.reason)
            )
        elif isinstance(message, End):
            self._events.append(
                SourceEvent(END, message=message, reason=message.reason)
            )
        elif isinstance(message, Error):
            self._events.append(
                SourceEvent(ERROR, message=message, reason=message.code)
            )
        elif isinstance(message, Status):
            self.last_status = message
        elif message.TYPE in ("pong", "ping"):
            pass
        else:
            self.stats.unknown_messages += 1

    def _on_frame(self, frame: Frame, payload: bytes) -> None:
        if frame.state is None:
            self.stats.bad_states += 1
            _LOGGER.warning("Frame %d/%d carries no state", frame.epoch, frame.g)
            return
        try:
            state = self.decoder.decode(frame.state, g=frame.g, tick=frame.tick)
        except ValueError as error:
            self.stats.bad_states += 1
            _LOGGER.warning("Frame %d/%d: %s", frame.epoch, frame.g, error)
            return
        if frame.seed and frame.g != 0:
            self.stats.bad_messages += 1
            _LOGGER.warning("Seed frame with g=%d dropped", frame.g)
            return
        outcome = self._assembler.add(frame.epoch, state)
        if outcome == "stale":
            self.stats.stale_frames += 1
            return
        if outcome == "duplicate":
            self.stats.duplicate_frames += 1
            return
        self.stats.frames += 1
        if frame.seed:
            self.stats.seeds += 1
            self._events.append(
                SourceEvent(
                    SEED,
                    message=frame,
                    seed=SeedFrame(
                        epoch=frame.epoch,
                        tick=frame.tick,
                        state=state,
                        image=payload,
                        encoding=frame.encoding,
                        width=frame.width,
                        height=frame.height,
                        message=frame,
                    ),
                )
            )


__all__ = ["LOST_TYPE", "Inbound", "MessageEngineSource", "SourceStats"]
