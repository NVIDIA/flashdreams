# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packet log of an external-engine session (``packets.jsonl``, ``packets.bin``)."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omnidreams_game_engine.api.v0.messages import PROTOCOL

_LOGGER = logging.getLogger(__name__)

RX = "rx"
"""Engine to host."""
TX = "tx"
"""Host to engine."""

INDEX_NAME = "packets.jsonl"
"""One JSON line per message: a ``meta`` line, then direction, stamps and header."""
PAYLOAD_NAME = "packets.bin"
"""The payloads back to back; each index line names its offset and length."""
LOG_FORMAT = "omnidreams-packet-log/0"


def now_us() -> int:
    """Return wall-clock microseconds, the time base of every stamp on the wire."""
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
        # Line-buffered, and each payload is flushed before the line that points
        # into it, so a host that exits without close() leaves a readable log.
        self._index = (self.directory / INDEX_NAME).open(
            "w", encoding="utf-8", buffering=1
        )
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
        """Messages recorded so far."""
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
                self._payloads.flush()
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
        """Write buffered data to the files."""
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
    """One recorded message."""

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
        self._lines: list[dict[str, Any]] = []
        for number, line in enumerate(lines[1:], start=2):
            try:
                self._lines.append(json.loads(line))
            except json.JSONDecodeError:
                if number < len(lines):
                    raise
                _LOGGER.warning("%s: skipped the unfinished last line", index)
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
        """Return the engine-to-host records in the order the host consumed them."""
        return [record for record in self if record.direction == RX]

    def sent(self) -> list[PacketRecord]:
        """Return the host-to-engine records in the order they were sent."""
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
