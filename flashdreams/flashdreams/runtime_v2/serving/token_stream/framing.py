# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Binary wire format for the video token stream.

This module is the single source of truth for the on-the-wire framing. The
browser client mirrors it byte-for-byte in ``framing.js``; any change here must
be reflected there. All multi-byte fields are little-endian.

Each binary frame is an 18-byte header optionally followed by codec parameters
and then the payload::

    offset 0  : MAGIC             4 bytes  = 0xFD 'T' 'O' 'K'
    offset 4  : VERSION           u8       = PROTOCOL_VERSION
    offset 5  : FLAGS             u8       bit0 KEYFRAME, bit1 LAST_IN_CHUNK, bit7 CONTROL
    offset 6  : CHUNK_ID          u32
    offset 10 : FRAME_IDX         u8
    offset 11 : FRAME_TOTAL       u8
    offset 12 : CODEC_PARAMS_LEN  u16
    offset 14 : PAYLOAD_LEN       u32
    offset 18 : CODEC_PARAMS (CODEC_PARAMS_LEN bytes), then PAYLOAD (PAYLOAD_LEN bytes)

A control frame carries the session header as a UTF-8 JSON payload. It sets the
CONTROL flag, uses the sentinel chunk id, and carries no codec parameters.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

MAGIC = b"\xfdTOK"
"""Four-byte frame prefix identifying the token stream wire format."""

PROTOCOL_VERSION = 1
"""Version stamped into every frame header and the session header JSON."""

FLAG_KEYFRAME = 0x01
"""Set on the first frame of a keyframe chunk."""

FLAG_LAST_IN_CHUNK = 0x02
"""Set on the final frame of a chunk."""

FLAG_CONTROL = 0x80
"""Set on control frames (session header JSON) rather than token frames."""

CONTROL_CHUNK_ID = 0xFFFFFFFF
"""Sentinel chunk id used by control frames."""

_HEADER = struct.Struct("<4sBBIBBHI")
"""Packed header layout shared with the client framing implementation."""

HEADER_SIZE = _HEADER.size
"""Size of the fixed frame header in bytes."""

assert HEADER_SIZE == 18


@dataclass(frozen=True, slots=True)
class FrameHeader:
    """Parsed view of a frame header."""

    version: int
    """Protocol version stamped into the frame."""

    flags: int
    """Bitfield combining the ``FLAG_*`` constants."""

    chunk_id: int
    """Chunk identifier, or :data:`CONTROL_CHUNK_ID` for control frames."""

    frame_idx: int
    """Index of this frame within its chunk."""

    frame_total: int
    """Total number of frames in this chunk."""

    codec_params_len: int
    """Length in bytes of the codec parameters that follow the header."""

    payload_len: int
    """Length in bytes of the payload that follows the codec parameters."""

    @property
    def is_control(self) -> bool:
        """Return whether the CONTROL flag is set."""
        return bool(self.flags & FLAG_CONTROL)

    @property
    def is_keyframe(self) -> bool:
        """Return whether the KEYFRAME flag is set."""
        return bool(self.flags & FLAG_KEYFRAME)

    @property
    def is_last_in_chunk(self) -> bool:
        """Return whether the LAST_IN_CHUNK flag is set."""
        return bool(self.flags & FLAG_LAST_IN_CHUNK)


def pack_frame(
    *,
    chunk_id: int,
    frame_idx: int,
    frame_total: int,
    payload: bytes,
    codec_params: bytes = b"",
    is_keyframe: bool = False,
    is_last_in_chunk: bool = False,
) -> bytes:
    """Pack one token frame into its wire representation."""
    flags = 0
    if is_keyframe:
        flags |= FLAG_KEYFRAME
    if is_last_in_chunk:
        flags |= FLAG_LAST_IN_CHUNK
    header = _HEADER.pack(
        MAGIC,
        PROTOCOL_VERSION,
        flags,
        chunk_id,
        frame_idx,
        frame_total,
        len(codec_params),
        len(payload),
    )
    return header + codec_params + payload


def pack_control(payload: bytes) -> bytes:
    """Pack a control frame carrying the session header JSON payload."""
    header = _HEADER.pack(
        MAGIC,
        PROTOCOL_VERSION,
        FLAG_CONTROL,
        CONTROL_CHUNK_ID,
        0,
        0,
        0,
        len(payload),
    )
    return header + payload


def parse_header(buffer: bytes) -> FrameHeader:
    """Parse a frame header from the first :data:`HEADER_SIZE` bytes.

    Raises ``ValueError`` if the buffer is too short or the magic prefix does
    not match.
    """
    if len(buffer) < HEADER_SIZE:
        raise ValueError(
            f"buffer too short for frame header: {len(buffer)} < {HEADER_SIZE}"
        )
    (
        magic,
        version,
        flags,
        chunk_id,
        frame_idx,
        frame_total,
        codec_params_len,
        payload_len,
    ) = _HEADER.unpack_from(buffer)
    if magic != MAGIC:
        raise ValueError(f"bad frame magic: {magic!r}")
    return FrameHeader(
        version=version,
        flags=flags,
        chunk_id=chunk_id,
        frame_idx=frame_idx,
        frame_total=frame_total,
        codec_params_len=codec_params_len,
        payload_len=payload_len,
    )
