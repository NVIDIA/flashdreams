# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the video token stream."""

from __future__ import annotations

from dataclasses import dataclass, field

from flashdreams.serving.token_stream.codec import (
    RawFloat16TokenCodecConfig,
    TokenCodecConfig,
)


@dataclass
class TokenStreamConfig:
    """Settings for streaming latent tokens alongside the rendered video."""

    enabled: bool = False
    """Whether the token stream is emitted for a session."""

    codec: TokenCodecConfig = field(default_factory=RawFloat16TokenCodecConfig)
    """Codec used to serialize each latent frame."""

    flow_window_size: int = 4
    """Maximum number of chunks in flight before waiting on client acks."""
