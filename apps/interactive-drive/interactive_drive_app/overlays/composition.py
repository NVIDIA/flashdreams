# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default native driving panel composition."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from flashdreams.serving.presentation import PanelOverlay
from PIL import Image

from interactive_drive_app.overlays.bev import BevWidget
from interactive_drive_app.overlays.controls import PedalsWidget, WheelWidget
from interactive_drive_app.overlays.header import SceneHeaderWidget
from interactive_drive_app.overlays.panel import (
    MIN_CAMERA_WIDTH,
    PANEL_WIDTH,
    TitleWidget,
)
from interactive_drive_app.overlays.speed import SpeedWidget
from interactive_drive_app.overlays.theme import NVIDIA_GREEN, PANEL_BG
from interactive_drive_app.state import DrivingViewState


def build_driving_overlay(
    state: DrivingViewState,
    *,
    control_assets: Any | None = None,
    marker_y_fraction: Callable[[], float] = lambda: 0.5,
    recolor_bev: Callable[[Image.Image], Image.Image] | None = None,
) -> PanelOverlay:
    """Compose the default driving chrome around app-owned view state."""
    return PanelOverlay(
        width=PANEL_WIDTH,
        min_camera_width=MIN_CAMERA_WIDTH,
        background=PANEL_BG,
        accent=NVIDIA_GREEN,
        children=(
            TitleWidget(),
            SceneHeaderWidget(
                scene_label=lambda: state.scene_label,
                variant_label=lambda: state.variant_label,
            ),
            SpeedWidget(lambda: state.speed_mps),
            WheelWidget(lambda: state, control_assets=control_assets),
            PedalsWidget(lambda: state, control_assets=control_assets),
            BevWidget(
                marker_y_fraction=marker_y_fraction,
                recolor=recolor_bev,
            ),
        ),
    )


__all__ = ["build_driving_overlay"]
