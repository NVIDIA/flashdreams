# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Server-side video token streaming over WebSocket."""

from flashdreams.serving.token_stream import framing
from flashdreams.serving.token_stream.codec import (
    RawFloat16TokenCodec,
    RawFloat16TokenCodecConfig,
    TokenCodec,
    TokenCodecConfig,
    TokenCodecEncodeResult,
)
from flashdreams.serving.token_stream.config import TokenStreamConfig
from flashdreams.serving.token_stream.emitter import TokenFrameEmitter
from flashdreams.serving.token_stream.framing import (
    FLAG_CONTROL,
    FLAG_KEYFRAME,
    FLAG_LAST_IN_CHUNK,
    HEADER_SIZE,
    MAGIC,
    PROTOCOL_VERSION,
    FrameHeader,
    pack_control,
    pack_frame,
    parse_header,
)

__all__ = [
    "TokenFrameEmitter",
    "TokenStreamConfig",
    "TokenCodec",
    "TokenCodecConfig",
    "TokenCodecEncodeResult",
    "RawFloat16TokenCodec",
    "RawFloat16TokenCodecConfig",
    "framing",
    "FrameHeader",
    "MAGIC",
    "PROTOCOL_VERSION",
    "HEADER_SIZE",
    "FLAG_KEYFRAME",
    "FLAG_LAST_IN_CHUNK",
    "FLAG_CONTROL",
    "pack_frame",
    "pack_control",
    "parse_header",
]
