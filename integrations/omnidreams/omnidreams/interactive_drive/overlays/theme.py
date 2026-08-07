# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared colour palette for the interactive-drive chrome overlays."""

from __future__ import annotations

NVIDIA_GREEN: tuple[int, int, int] = (118, 185, 0)
BG_COLOR: tuple[int, int, int] = (20, 20, 30)
PANEL_BG: tuple[int, int, int] = (25, 25, 35)
TEXT_COLOR: tuple[int, int, int] = (220, 220, 230)
LABEL_COLOR: tuple[int, int, int] = (150, 150, 170)
HEADER_BG: tuple[int, int, int] = (35, 35, 50)
HOVER_BG: tuple[int, int, int] = (50, 60, 80)
ACTIVE_BG: tuple[int, int, int] = (30, 80, 30)
ACCENT_AMBER: tuple[int, int, int] = (200, 150, 50)
GMAPS_LAND_RGB: tuple[int, int, int] = (234, 226, 209)

__all__ = [
    "ACCENT_AMBER",
    "ACTIVE_BG",
    "BG_COLOR",
    "GMAPS_LAND_RGB",
    "HEADER_BG",
    "HOVER_BG",
    "LABEL_COLOR",
    "NVIDIA_GREEN",
    "PANEL_BG",
    "TEXT_COLOR",
]
