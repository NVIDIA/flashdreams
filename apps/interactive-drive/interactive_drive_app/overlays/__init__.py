# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Driving chrome as stackable overlays over the shared local-window presenter.

Each widget from the legacy HUD lands here as its own
:class:`~flashdreams.serving.presentation.HudOverlay`, so a demo composes the
chrome it wants instead of inheriting one class that owns every widget.
"""

from interactive_drive_app.overlays.bev import BEV_OVERLAY_KEY, BevOverlay
from interactive_drive_app.overlays.controls import (
    PedalsOverlay,
    WheelOverlay,
)
from interactive_drive_app.overlays.header import SceneHeaderOverlay
from interactive_drive_app.overlays.panel import (
    PANEL_WIDTH,
    DrivingPanelOverlay,
)
from interactive_drive_app.overlays.speed import SpeedOverlay
from interactive_drive_app.overlays.theme import (
    ACCENT_AMBER,
    BG_COLOR,
    LABEL_COLOR,
    NVIDIA_GREEN,
    PANEL_BG,
    TEXT_COLOR,
)

__all__ = [
    "ACCENT_AMBER",
    "BEV_OVERLAY_KEY",
    "BevOverlay",
    "BG_COLOR",
    "LABEL_COLOR",
    "NVIDIA_GREEN",
    "PANEL_BG",
    "PANEL_WIDTH",
    "DrivingPanelOverlay",
    "PedalsOverlay",
    "SceneHeaderOverlay",
    "SpeedOverlay",
    "WheelOverlay",
    "TEXT_COLOR",
]
