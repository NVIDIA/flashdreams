# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-thread half of a message-based :class:`~.external.EngineSource`."""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from omnidreams_game_engine.api.v0.assembly import ChunkAssembler, ExternalChunk
from omnidreams_game_engine.api.v0.external import (
    CONNECTED,
    CONTROL_EVENTS,
    END,
    ERROR,
    HOLD,
    INVALID,
    LOST,
    RELEASE,
    RESTART,
    SEED,
    START,
    VIEWS,
    SeedFrame,
    SourceEvent,
    SourceEventKind,
)
from omnidreams_game_engine.api.v0.messages import (
    End,
    Error,
    Frame,
    Hello,
    Hold,
    Input,
    Message,
    ProtocolError,
    Release,
    Restart,
    Start,
    Views,
    from_header,
)
from omnidreams_game_engine.api.v0.recorder import RX, PacketRecorder
from omnidreams_game_engine.api.v0.state import StateDecoder
from omnidreams_game_engine.api.v0.timeline import ChunkTimeline

_LOGGER = logging.getLogger(__name__)

LOST_TYPE = "_lost"
"""Header type of the synthetic message a transport queues when the engine is gone."""

_EVENT_KINDS: dict[type[Message], SourceEventKind] = {
    Hello: CONNECTED,
    Start: START,
    Restart: RESTART,
    Hold: HOLD,
    Release: RELEASE,
    End: END,
    Error: ERROR,
    Views: VIEWS,
}
"""Event kind of every message type other than ``frame`` that the host handles."""


@dataclass(frozen=True, slots=True)
class Inbound:
    """One received message, as a transport queues it."""

    header: dict[str, Any]
    payload: bytes = b""
    t_arrive_us: int = 0


@dataclass(slots=True)
class SourceStats:
    """Engine messages the source dropped."""

    bad_messages: int = 0
    """Messages that are not valid version-0 messages."""
    bad_states: int = 0
    """Frames whose state the decoder cannot read."""


class MessageEngineSource:
    """Shared ``EngineSource`` logic over a stream of protocol messages.

    A transport only fetches and transmits messages. Parsing, decoding,
    chunk assembly, events and recording run here, on the thread that calls
    :meth:`poll_events` and :meth:`wait_chunk`.
    """

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
        self.stats = SourceStats()
        self._assembler = ChunkAssembler(ChunkTimeline())
        self._events: deque[SourceEvent] = deque()
        self._sends = 0

    ## Hooks

    def _fetch(self, timeout_s: float) -> list[Inbound]:
        """Return the messages available now, waiting up to ``timeout_s`` for one."""
        raise NotImplementedError

    def _transmit(self, message: Message, payload: bytes) -> None:
        """Send one message, or drop it when no engine is connected."""
        raise NotImplementedError

    ## EngineSource

    @property
    def sends(self) -> int:
        """Messages the host has sent in reaction to the engine, delivered or not.

        Every message but ``input`` counts; a replay delivers each recorded
        engine message at the same count.
        """
        return self._sends

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
        if not isinstance(message, Input):
            self._sends += 1
        self._transmit(message, payload)

    def send_frames(self, frames: Sequence[tuple[Message, bytes]]) -> None:
        for message, payload in frames:
            self.send(message, payload)

    def close(self) -> None:
        if self.recorder is not None:
            self.recorder.flush()

    ## Internals

    def _consume(self, inbound: Sequence[Inbound]) -> None:
        for item in inbound:
            if self.recorder is not None:
                self.recorder.record(
                    RX,
                    item.header,
                    item.payload,
                    t_arrive_us=item.t_arrive_us,
                    host_sends=self._sends,
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
                self._reject(str(error), dropped=item.header.get("type") == "frame")
                continue
            if isinstance(message, Frame):
                self._on_frame(message, item.payload)
            elif type(message) in _EVENT_KINDS:
                self._events.append(
                    SourceEvent(
                        _EVENT_KINDS[type(message)],
                        message=message,
                        reason=getattr(message, "reason", None),
                    )
                )

    def _on_frame(self, frame: Frame, payload: bytes) -> None:
        try:
            state = self.decoder.decode(frame.state, g=frame.g, tick=frame.tick)
        except (KeyError, TypeError, ValueError) as error:
            self.stats.bad_states += 1
            self._reject(f"Frame {frame.epoch}/{frame.g}: {error}", frame=frame)
            return
        if frame.seed and frame.g != 0:
            self.stats.bad_messages += 1
            self._reject(f"Seed frame {frame.epoch}/{frame.g} is not g=0", frame=frame)
            return
        if self._assembler.add(frame.epoch, state) == "added" and frame.seed:
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
                    ),
                )
            )

    def _reject(
        self, reason: str, *, dropped: bool = True, frame: Frame | None = None
    ) -> None:
        """Answer an unusable engine message with ``error{code: "protocol"}``.

        A dropped frame leaves its chunk incomplete, so it also becomes an
        ``invalid`` event, on which the driver re-anchors the epoch.
        """
        _LOGGER.warning("Dropped an engine message: %s", reason)
        self.send(Error(code="protocol", message=reason))
        if dropped:
            self._events.append(SourceEvent(INVALID, message=frame, reason=reason))


__all__ = ["LOST_TYPE", "Inbound", "MessageEngineSource", "SourceStats"]
