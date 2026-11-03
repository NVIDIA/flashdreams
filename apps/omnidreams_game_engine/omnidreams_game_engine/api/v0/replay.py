# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay a recorded session without the engine.

:class:`PacketLogSource` feeds the engine's recorded messages to the host
through the same :class:`~.external.EngineSource` interface as the live link.
A message is delivered once the replaying host has sent as many messages as
the live host had when it consumed the message (``host_sends`` in the log):
a seed after its ``seed_req``, a frame after its grant, a restart or an end
between the same chunks. The host's behavior depends only on what it has
received, so the replay generates the same chunks of the same epochs. What the host sends is
kept in :attr:`PacketLogSource.sent` for comparison with the recording.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

from omnidreams_game_engine.api.v0.messages import Message
from omnidreams_game_engine.api.v0.recorder import PacketLogReader, PacketRecorder
from omnidreams_game_engine.api.v0.source import Inbound, MessageEngineSource
from omnidreams_game_engine.api.v0.state import StateDecoder


class PacketLogSource(MessageEngineSource):
    """An :class:`~.external.EngineSource` that replays a packet log."""

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
