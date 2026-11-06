# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the wire framing."""

from __future__ import annotations

import json
import socket
import struct
import threading

import numpy as np
import pytest
from omnidreams_game_engine.api.v0 import framing

pytestmark = pytest.mark.ci_cpu


def test_framing_is_a_length_prefixed_json_header_and_payload() -> None:
    data = framing.encode_message({"type": "ping", "t_host_us": 5}, b"xy")
    (size,) = struct.unpack(">I", data[:4])
    header = json.loads(data[4 : 4 + size])
    assert header == {"type": "ping", "t_host_us": 5, "bytes": 2}
    assert data[4 + size :] == b"xy"


def test_decoder_handles_byte_by_byte_partial_reads() -> None:
    stream = framing.encode_message(
        {"type": "a"}, b"\x00\x01\x02"
    ) + framing.encode_message({"type": "b", "n": 1})
    decoder = framing.MessageDecoder()
    messages = []
    for index in range(len(stream)):
        messages.extend(decoder.feed(stream[index : index + 1]))
    assert messages == [
        ({"type": "a", "bytes": 3}, b"\x00\x01\x02"),
        ({"type": "b", "n": 1, "bytes": 0}, b""),
    ]


def test_large_payload_round_trips_through_a_socket() -> None:
    payload = bytes(range(256)) * (8 << 10)  # 2 MiB, larger than a socket buffer
    left, right = socket.socketpair()
    sent: list[int] = []
    sender = threading.Thread(
        target=lambda: sent.append(
            framing.send_message(left, {"type": "gen_frame"}, payload)
        ),
        daemon=True,
    )
    try:
        left.settimeout(10.0)
        right.settimeout(10.0)
        sender.start()
        header, received = framing.recv_message(right)
        sender.join(timeout=10.0)
    finally:
        left.close()
        right.close()
    assert sent == [len(framing.encode_message({"type": "gen_frame"}, payload))]
    assert header["bytes"] == len(payload)
    assert received == payload


def test_size_limits_are_enforced() -> None:
    with pytest.raises(framing.FramingError):
        framing.encode_message({"bytes": 0})
    with pytest.raises(framing.FramingError):
        framing.encode_message({"type": "x"}, b"\x00" * (framing.MAX_PAYLOAD_BYTES + 1))
    with pytest.raises(framing.FramingError):
        framing.encode_message({"type": "x", "pad": "a" * framing.MAX_HEADER_BYTES})
    oversized_header = struct.pack(">I", framing.MAX_HEADER_BYTES + 1)
    with pytest.raises(framing.FramingError):
        framing.MessageDecoder().feed(oversized_header)
    left, right = socket.socketpair()
    with left, right:
        left.sendall(oversized_header)
        with pytest.raises(framing.FramingError):
            framing.recv_message(right)
    oversized_payload = json.dumps(
        {"type": "x", "bytes": framing.MAX_PAYLOAD_BYTES + 1}
    ).encode()
    with pytest.raises(framing.FramingError):
        framing.MessageDecoder().feed(
            struct.pack(">I", len(oversized_payload)) + oversized_payload
        )


@pytest.mark.parametrize("value", [float("nan"), np.float32(1.0)])
def test_an_unencodable_header_is_a_framing_error(value: object) -> None:
    with pytest.raises(framing.FramingError, match="Unencodable"):
        framing.encode_message({"type": "status", "value": value})


@pytest.mark.parametrize(
    "raw",
    [
        b"not json",
        b"[1, 2]",
        b'{"bytes": 0}',
        b'{"type": ""}',
        b'{"type": "x", "bytes": -1}',
    ],
)
def test_malformed_headers_are_rejected(raw: bytes) -> None:
    with pytest.raises(framing.FramingError):
        framing.MessageDecoder().feed(struct.pack(">I", len(raw)) + raw)


def test_closed_peer_raises_connection_error() -> None:
    left, right = socket.socketpair()
    left.sendall(struct.pack(">I", 10) + b"{")
    left.close()
    try:
        right.settimeout(5.0)
        with pytest.raises(ConnectionError):
            framing.recv_message(right)
    finally:
        right.close()
