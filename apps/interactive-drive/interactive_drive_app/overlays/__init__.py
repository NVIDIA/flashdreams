# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Driving chrome for the shared local-window presenter.

Each widget from the legacy HUD lands here as its own
:class:`~flashdreams.serving.presentation.PanelWidget`, stacked by a
:class:`~flashdreams.serving.presentation.PanelOverlay`. A demo composes the
chrome it wants instead of inheriting one class that owns every widget.
"""

from interactive_drive_app.overlays.bev import BEV_OVERLAY_KEY, BevWidget
from interactive_drive_app.overlays.composition import build_driving_overlay
from interactive_drive_app.overlays.controls import PedalsWidget, WheelWidget
from interactive_drive_app.overlays.header import SceneHeaderWidget
from interactive_drive_app.overlays.panel import (
    MIN_CAMERA_WIDTH,
    PANEL_WIDTH,
    TitleWidget,
)
from interactive_drive_app.overlays.speed import SpeedWidget
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
    "BG_COLOR",
    "LABEL_COLOR",
    "MIN_CAMERA_WIDTH",
    "NVIDIA_GREEN",
    "PANEL_BG",
    "PANEL_WIDTH",
    "TEXT_COLOR",
    "BevWidget",
    "PedalsWidget",
    "SceneHeaderWidget",
    "SpeedWidget",
    "TitleWidget",
    "WheelWidget",
    "build_driving_overlay",
]
