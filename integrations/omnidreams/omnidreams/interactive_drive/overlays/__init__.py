# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Driving chrome as stackable overlays over the shared local-window presenter.

Each widget from the legacy HUD lands here as its own
:class:`~flashdreams.serving.presentation.HudOverlay`, so a demo composes the
chrome it wants instead of inheriting one class that owns every widget.
"""

from omnidreams.interactive_drive.overlays.bev import BEV_OVERLAY_KEY, BevOverlay
from omnidreams.interactive_drive.overlays.controls import (
    PedalsOverlay,
    WheelOverlay,
)
from omnidreams.interactive_drive.overlays.header import SceneHeaderOverlay
from omnidreams.interactive_drive.overlays.panel import (
    PANEL_WIDTH,
    DrivingPanelOverlay,
)
from omnidreams.interactive_drive.overlays.speed import SpeedOverlay
from omnidreams.interactive_drive.overlays.theme import (
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
