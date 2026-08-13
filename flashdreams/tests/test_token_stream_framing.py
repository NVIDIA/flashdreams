# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the token stream binary wire format.

Verifies that :func:`pack_frame` and :func:`pack_control` round-trip through
:func:`parse_header`, that flags and payload slicing are exact, and that
:func:`parse_header` rejects short buffers and bad magic. This mirrors the
byte-for-byte contract the browser client depends on.
"""

from __future__ import annotations

import json

import pytest

from flashdreams.serving.token_stream import framing

pytestmark = pytest.mark.ci_cpu


def test_pack_frame_round_trips_header_fields() -> None:
    payload = b"\x01\x02\x03\x04\x05"
    codec_params = b"\xaa\xbb"
    frame = framing.pack_frame(
        chunk_id=42,
        frame_idx=3,
        frame_total=7,
        payload=payload,
        codec_params=codec_params,
        is_keyframe=True,
        is_last_in_chunk=True,
    )

    header = framing.parse_header(frame)
    assert header.version == framing.PROTOCOL_VERSION
    assert header.chunk_id == 42
    assert header.frame_idx == 3
    assert header.frame_total == 7
    assert header.codec_params_len == len(codec_params)
    assert header.payload_len == len(payload)


def test_pack_frame_payload_and_codec_params_slice_back() -> None:
    payload = bytes(range(20))
    codec_params = b"params-bytes"
    frame = framing.pack_frame(
        chunk_id=1,
        frame_idx=0,
        frame_total=1,
        payload=payload,
        codec_params=codec_params,
    )

    header = framing.parse_header(frame)
    params_start = framing.HEADER_SIZE
    payload_start = params_start + header.codec_params_len
    assert frame[params_start:payload_start] == codec_params
    assert frame[payload_start : payload_start + header.payload_len] == payload
    assert len(frame) == payload_start + header.payload_len


def test_flags_map_correctly() -> None:
    keyframe = framing.parse_header(
        framing.pack_frame(
            chunk_id=0,
            frame_idx=0,
            frame_total=2,
            payload=b"x",
            is_keyframe=True,
            is_last_in_chunk=False,
        )
    )
    assert keyframe.is_keyframe is True
    assert keyframe.is_last_in_chunk is False
    assert keyframe.is_control is False
    assert keyframe.flags == framing.FLAG_KEYFRAME

    last = framing.parse_header(
        framing.pack_frame(
            chunk_id=0,
            frame_idx=1,
            frame_total=2,
            payload=b"x",
            is_keyframe=False,
            is_last_in_chunk=True,
        )
    )
    assert last.is_keyframe is False
    assert last.is_last_in_chunk is True
    assert last.is_control is False
    assert last.flags == framing.FLAG_LAST_IN_CHUNK


def test_parse_header_rejects_short_buffer() -> None:
    with pytest.raises(ValueError):
        framing.parse_header(b"\x00" * (framing.HEADER_SIZE - 1))


def test_parse_header_rejects_bad_magic() -> None:
    frame = bytearray(
        framing.pack_frame(chunk_id=0, frame_idx=0, frame_total=1, payload=b"x")
    )
    frame[0] ^= 0xFF
    with pytest.raises(ValueError):
        framing.parse_header(bytes(frame))


def test_pack_control_sets_flag_sentinel_and_round_trips_json() -> None:
    session_header = {
        "protocol_version": framing.PROTOCOL_VERSION,
        "latent_shape": [2, 4, 4],
        "frames_per_chunk": 3,
        "fps": 16,
        "codec": {"id": "raw_f16", "version": 1, "static_params": {}},
    }
    encoded = json.dumps(session_header).encode()
    frame = framing.pack_control(encoded)

    header = framing.parse_header(frame)
    assert header.is_control is True
    assert header.flags == framing.FLAG_CONTROL
    assert header.chunk_id == framing.CONTROL_CHUNK_ID
    assert header.frame_idx == 0
    assert header.frame_total == 0
    assert header.codec_params_len == 0
    assert header.payload_len == len(encoded)

    body = frame[framing.HEADER_SIZE :]
    assert json.loads(body.decode()) == session_header
