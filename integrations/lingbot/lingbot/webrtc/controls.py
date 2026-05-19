# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility exports for shared FlashDreams WebRTC controls."""

from __future__ import annotations

from flashdreams.serving.webrtc.controls import (
    DEFAULT_SUPPORTED_KEYS as SUPPORTED_KEYS,
)
from flashdreams.serving.webrtc.controls import (
    KEY_ALIASES,
    CameraPoseIntegrator,
    KeyboardResampler,
    KeyboardState,
    PoseSegment,
    normalize_key,
)

__all__ = [
    "SUPPORTED_KEYS",
    "KEY_ALIASES",
    "CameraPoseIntegrator",
    "KeyboardResampler",
    "KeyboardState",
    "PoseSegment",
    "normalize_key",
]
