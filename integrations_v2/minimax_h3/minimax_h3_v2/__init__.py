# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native MiniMax H3 applications for the FlashDreams V2 API."""

from minimax_h3_v2.app import (
    MiniMaxH3Application,
    create_app,
    create_fl2va_app,
    create_ref2va_app,
    create_t2va_app,
)

__all__ = [
    "MiniMaxH3Application",
    "create_app",
    "create_fl2va_app",
    "create_ref2va_app",
    "create_t2va_app",
]
