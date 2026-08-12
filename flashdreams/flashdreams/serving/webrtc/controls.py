# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WebRTC adapters for shared realtime input helpers."""

from __future__ import annotations

from flashdreams.runtime.keyboard import (
    DEFAULT_SUPPORTED_KEYS,
    KEY_ALIASES,
    WSAD_SUPPORTED_KEYS,
    ImageRequest,
    KeyboardState,
    PromptRequest,
    ResetRequest,
    SparseInputSnapshot,
    normalize_key,
)

__all__ = [
    "DEFAULT_SUPPORTED_KEYS",
    "KEY_ALIASES",
    "WSAD_SUPPORTED_KEYS",
    "ImageRequest",
    "KeyboardState",
    "PromptRequest",
    "ResetRequest",
    "SparseInputSnapshot",
    "normalize_key",
]
