# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packet log of an external-engine session.

A session folder holds ``packets.jsonl`` (one JSON line per message: its
direction, stamps and header) and ``packets.bin`` (the payloads, back to
back; each line names its offset and length). The first line is a ``meta``
record. Messages from the engine are written when the model thread consumes
them, with ``host_sends``, the number of messages the host had sent by then;
:class:`~.replay.PacketLogSource` delivers them at the same points, so a
replay reproduces the session's chunks and epochs exactly.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omnidreams_game_engine.api.v0.messages import PROTOCOL

RX = "rx"
"""Engine to host."""
TX = "tx"
"""Host to engine."""

INDEX_NAME = "packets.jsonl"
PAYLOAD_NAME = "packets.bin"
LOG_FORMAT = "omnidreams-packet-log/0"


def now_us() -> int:
    """Wall-clock microseconds, the time base of every stamp on the wire."""
    return time.time_ns() // 1000


class PacketRecorder:
    """Write every message of a session in both directions. Thread-safe."""

    def __init__(
        self,
        directory: str | Path,
        *,
        record_tx_payloads: bool = False,
        meta: Mapping[str, Any] | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.record_tx_payloads = record_tx_payloads
        self._lock = threading.Lock()
        self._index = (self.directory / INDEX_NAME).open("w", encoding="utf-8")
        self._payloads = (self.directory / PAYLOAD_NAME).open("wb")
        self._offset = 0
        self._seq = 0
        self._closed = False
        self._write_line(
            {
                "kind": "meta",
                "format": LOG_FORMAT,
                "protocol": PROTOCOL,
                "created_us": now_us(),
                **dict(meta or {}),
            }
        )

    @property
    def records(self) -> int:
        return self._seq

    def record(
        self,
        direction: str,
        header: Mapping[str, Any],
        payload: bytes = b"",
        *,
        t_arrive_us: int | None = None,
        host_sends: int | None = None,
    ) -> None:
        """Append one message; ``host_sends`` marks when the host consumed it."""
        if direction not in (RX, TX):
            raise ValueError(f"Unknown direction {direction!r}")
        keep_payload = direction == RX or self.record_tx_payloads
        with self._lock:
            if self._closed:
                return
            offset = self._offset
            length = 0
            if payload and keep_payload:
                self._payloads.write(payload)
                length = len(payload)
                self._offset += length
            record: dict[str, Any] = {
                "seq": self._seq,
                "dir": direction,
                "t_us": now_us(),
                "header": dict(header),
                "offset": offset,
                "bytes": length,
            }
            if payload and not keep_payload:
                record["payload_omitted"] = len(payload)
            if t_arrive_us is not None:
                record["t_arrive_us"] = t_arrive_us
            if host_sends is not None:
                record["host_sends"] = host_sends
            self._seq += 1
            self._write_line(record)

    def flush(self) -> None:
        with self._lock:
            if not self._closed:
                self._index.flush()
                self._payloads.flush()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._index.close()
            self._payloads.close()

    def _write_line(self, record: Mapping[str, Any]) -> None:
        self._index.write(json.dumps(record, separators=(",", ":")) + "\n")


@dataclass(frozen=True, slots=True)
class PacketRecord:
    seq: int
    direction: str
    header: dict[str, Any]
    payload: bytes
    t_us: int
    t_arrive_us: int | None
    host_sends: int | None
    """Messages the host had sent when it consumed this one (engine messages)."""


class PacketLogReader:
    """Read a session folder written by :class:`PacketRecorder`."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        index = self.directory / INDEX_NAME
        lines = index.read_text(encoding="utf-8").splitlines()
        if not lines:
            raise ValueError(f"{index} is empty")
        meta = json.loads(lines[0])
        if meta.get("kind") != "meta" or meta.get("format") != LOG_FORMAT:
            raise ValueError(f"{index} is not a {LOG_FORMAT} packet log")
        self.meta: dict[str, Any] = meta
        self._lines = [json.loads(line) for line in lines[1:] if line.strip()]
        payload_path = self.directory / PAYLOAD_NAME
        self._payloads = payload_path.read_bytes() if payload_path.exists() else b""

    def __iter__(self) -> Iterator[PacketRecord]:
        for line in self._lines:
            offset = int(line.get("offset", 0))
            length = int(line.get("bytes", 0))
            yield PacketRecord(
                seq=int(line["seq"]),
                direction=str(line["dir"]),
                header=dict(line["header"]),
                payload=self._payloads[offset : offset + length],
                t_us=int(line["t_us"]),
                t_arrive_us=line.get("t_arrive_us"),
                host_sends=line.get("host_sends"),
            )

    def received(self) -> list[PacketRecord]:
        """Engine-to-host records in the order the host consumed them."""
        return [record for record in self if record.direction == RX]

    def sent(self) -> list[PacketRecord]:
        return [record for record in self if record.direction == TX]


__all__ = [
    "INDEX_NAME",
    "LOG_FORMAT",
    "PAYLOAD_NAME",
    "RX",
    "TX",
    "PacketLogReader",
    "PacketRecord",
    "PacketRecorder",
    "now_us",
]
