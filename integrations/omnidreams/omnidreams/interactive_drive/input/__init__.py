# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compatibility re-exports; the input layer now lives in the app package."""

from interactive_drive_app.input.backend import InputBackend, SampledInput
from interactive_drive_app.input.keyboard import (
    KeyboardInputBackend,
    KeyboardState,
    command_from_snapshot,
    normalize_key,
)

__all__ = [
    "InputBackend",
    "KeyboardInputBackend",
    "KeyboardState",
    "SampledInput",
    "command_from_snapshot",
    "normalize_key",
]
