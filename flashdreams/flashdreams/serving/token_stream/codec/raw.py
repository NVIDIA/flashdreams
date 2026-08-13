# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Uncompressed float16 token codec.

Serializes each latent frame as raw little-endian float16 values with no
compression. This is the reference codec: it is lossy only in the float32 to
float16 narrowing and carries no per-frame parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from flashdreams.serving.token_stream.codec.base import (
    TokenCodec,
    TokenCodecConfig,
    TokenCodecEncodeResult,
)


@dataclass(kw_only=True)
class RawFloat16TokenCodecConfig(TokenCodecConfig):
    """Config for the uncompressed float16 token codec."""

    _target: type[TokenCodec] = field(default_factory=lambda: RawFloat16TokenCodec)


class RawFloat16TokenCodec(TokenCodec[RawFloat16TokenCodecConfig]):
    """Encode latent frames as raw contiguous float16 bytes."""

    @property
    def codec_id(self) -> str:
        """Identifier advertised in the session header."""
        return "raw_f16"

    def encode_frame(self, latent: torch.Tensor) -> TokenCodecEncodeResult:
        """Narrow to float16 and serialize the frame as raw little-endian bytes."""
        payload = latent.to(torch.float16).contiguous().cpu().numpy().tobytes()
        return TokenCodecEncodeResult(payload=payload, frame_params=b"")
