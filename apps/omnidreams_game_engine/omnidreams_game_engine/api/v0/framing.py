# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire framing of the external-engine protocol.

Every message is a 4-byte big-endian header length, a UTF-8 JSON object with
a string ``type`` and an integer ``bytes``, then ``bytes`` bytes of binary
payload (0 when absent). The module imports neither torch nor a renderer.
"""

from __future__ import annotations

import json
import socket
import struct
from collections.abc import Mapping
from typing import Any

HEADER_LENGTH = struct.Struct(">I")
"""Length prefix of the JSON header."""

MAX_HEADER_BYTES = 1 << 20
"""Largest JSON header a receiver accepts (1 MiB)."""

MAX_PAYLOAD_BYTES = 16 << 20
"""Largest binary payload a receiver accepts (16 MiB)."""


class FramingError(ValueError):
    """A malformed or oversized message on the wire."""


def encode_message(header: Mapping[str, Any], payload: bytes = b"") -> bytes:
    """Return one framed message; ``bytes`` in ``header`` is set from the payload."""
    message_type = header.get("type")
    if not isinstance(message_type, str) or not message_type:
        raise FramingError("A message header needs a non-empty string 'type'")
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise FramingError(f"Payload of {len(payload)} bytes exceeds the limit")
    body = dict(header)
    body["bytes"] = len(payload)
    data = json.dumps(
        body, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    if len(data) > MAX_HEADER_BYTES:
        raise FramingError(f"Header of {len(data)} bytes exceeds the limit")
    return HEADER_LENGTH.pack(len(data)) + data + bytes(payload)


def decode_header(data: bytes) -> dict[str, Any]:
    """Parse and check one JSON header."""
    try:
        header = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FramingError(f"Unreadable header: {error}") from error
    if not isinstance(header, dict):
        raise FramingError("A header must be a JSON object")
    message_type = header.get("type")
    if not isinstance(message_type, str) or not message_type:
        raise FramingError("A header needs a non-empty string 'type'")
    length = header.get("bytes", 0)
    if isinstance(length, bool) or not isinstance(length, int) or length < 0:
        raise FramingError(f"Invalid payload length {length!r}")
    if length > MAX_PAYLOAD_BYTES:
        raise FramingError(f"Payload of {length} bytes exceeds the limit")
    return header


class MessageDecoder:
    """Incremental decoder: feed received bytes, take complete messages."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._header: dict[str, Any] | None = None

    def feed(self, data: bytes) -> list[tuple[dict[str, Any], bytes]]:
        """Append ``data`` and return every message it completes."""
        self._buffer.extend(data)
        messages: list[tuple[dict[str, Any], bytes]] = []
        while True:
            if self._header is None:
                if len(self._buffer) < HEADER_LENGTH.size:
                    return messages
                (size,) = HEADER_LENGTH.unpack_from(self._buffer)
                if size > MAX_HEADER_BYTES:
                    raise FramingError(f"Header of {size} bytes exceeds the limit")
                if len(self._buffer) < HEADER_LENGTH.size + size:
                    return messages
                start = HEADER_LENGTH.size
                self._header = decode_header(bytes(self._buffer[start : start + size]))
                del self._buffer[: start + size]
            length = int(self._header.get("bytes", 0))
            if len(self._buffer) < length:
                return messages
            payload = bytes(self._buffer[:length])
            del self._buffer[:length]
            messages.append((self._header, payload))
            self._header = None


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    buffer = bytearray(size)
    view = memoryview(buffer)
    received = 0
    while received < size:
        count = sock.recv_into(view[received:], size - received)
        if count == 0:
            raise ConnectionError("The peer closed the connection")
        received += count
    return bytes(buffer)


def recv_message(sock: socket.socket) -> tuple[dict[str, Any], bytes]:
    """Block until one complete message is read from ``sock``."""
    (size,) = HEADER_LENGTH.unpack(_recv_exact(sock, HEADER_LENGTH.size))
    if size > MAX_HEADER_BYTES:
        raise FramingError(f"Header of {size} bytes exceeds the limit")
    header = decode_header(_recv_exact(sock, size))
    length = int(header.get("bytes", 0))
    payload = _recv_exact(sock, length) if length else b""
    return header, payload


def send_message(
    sock: socket.socket, header: Mapping[str, Any], payload: bytes = b""
) -> int:
    """Frame and send one message; return the number of bytes sent."""
    data = encode_message(header, payload)
    sock.sendall(data)
    return len(data)


__all__ = [
    "HEADER_LENGTH",
    "MAX_HEADER_BYTES",
    "MAX_PAYLOAD_BYTES",
    "FramingError",
    "MessageDecoder",
    "decode_header",
    "encode_message",
    "recv_message",
    "send_message",
]
