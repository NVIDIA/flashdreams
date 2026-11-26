# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay of a recorded session without the engine."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

from omnidreams_game_engine.api.v0.messages import Message
from omnidreams_game_engine.api.v0.recorder import PacketLogReader, PacketRecorder
from omnidreams_game_engine.api.v0.source import Inbound, MessageEngineSource
from omnidreams_game_engine.api.v0.state import StateDecoder


class PacketLogSource(MessageEngineSource):
    """An :class:`~.external.EngineSource` that replays a packet log.

    A recorded engine message is delivered once the host has sent as many
    messages as the live host had when it consumed it (``host_sends``), so
    the host generates the same chunks of the same epochs. A log with a
    host-initiated restart (``request_refresh`` or a model failure) is refused,
    because a replay cannot reproduce it.
    """

    _fetch_blocks = False

    def __init__(
        self,
        log: str | Path | PacketLogReader,
        decoder: StateDecoder,
        *,
        recorder: PacketRecorder | None = None,
    ) -> None:
        super().__init__(decoder, recorder=recorder)
        self.log = log if isinstance(log, PacketLogReader) else PacketLogReader(log)
        if any(record.header.get("type") == "restart" for record in self.log.sent()):
            raise ValueError(
                f"{self.log.directory} has a host-initiated restart "
                "(request_refresh or a model failure), which a replay cannot "
                "reproduce"
            )
        self._pending = deque(self.log.received())
        self.sent: list[tuple[dict[str, Any], int]] = []
        """Every header the host sent, with its payload length."""

    @property
    def exhausted(self) -> bool:
        """Whether every recorded engine message has been delivered."""
        return not self._pending

    def _fetch(self, timeout_s: float) -> list[Inbound]:
        del timeout_s
        delivered: list[Inbound] = []
        while self._pending:
            record = self._pending[0]
            if record.host_sends is not None and record.host_sends > self.sends:
                break
            self._pending.popleft()
            delivered.append(
                Inbound(
                    header=record.header,
                    payload=record.payload,
                    t_arrive_us=record.t_arrive_us or record.t_us,
                )
            )
        return delivered

    def _transmit(self, message: Message, payload: bytes) -> None:
        self.sent.append((message.to_header(), len(payload)))


__all__ = ["PacketLogSource"]
