# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Application view state and generated-result metadata projection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from flashdreams.infra.video_output import VideoStepResult
from flashdreams.runtime import StepResult
from flashdreams.serving.presentation import DisplayFrame

from interactive_drive_app.overlays.bev import BEV_OVERLAY_KEY

DRIVING_METADATA_KEY = "interactive_drive"
"""``VideoStepResult.metadata`` key carrying driving-demo state."""


@dataclass(slots=True)
class DrivingViewState:
    """Latest model/session values consumed by native driving chrome."""

    scene_label: str = "Scene"
    variant_label: str = "default"
    speed_mps: float = 0.0
    steering: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    reverse: bool = False
    bev: Any = None
    status_message: str | None = None

    def project_frame(
        self,
        result: StepResult,
        video: VideoStepResult,
        frame_index: int,
        image: Any,
        timestamp_us: int,
    ) -> DisplayFrame:
        """Update app state and project one generated frame for presentation."""
        del frame_index
        metadata = video.metadata.get(DRIVING_METADATA_KEY, {})
        if isinstance(metadata, Mapping):
            self._update(metadata)
        status = result.metadata.get("status_message", self.status_message)
        return DisplayFrame(
            image=image,
            timestamp_us=timestamp_us,
            status_message=status if isinstance(status, str) else None,
            overlay_data={BEV_OVERLAY_KEY: self.bev},
        )

    def _update(self, values: Mapping[str, Any]) -> None:
        self.speed_mps = _float_value(values, "speed_mps", self.speed_mps)
        self.steering = _float_value(values, "steering", self.steering)
        self.throttle = _float_value(values, "throttle", self.throttle)
        self.brake = _float_value(values, "brake", self.brake)
        reverse = values.get("reverse")
        if isinstance(reverse, bool):
            self.reverse = reverse
        if "bev" in values:
            self.bev = values["bev"]
        status = values.get("status_message")
        if status is None or isinstance(status, str):
            self.status_message = status


def _float_value(values: Mapping[str, Any], name: str, default: float) -> float:
    value = values.get(name)
    return float(value) if isinstance(value, int | float) else default


__all__ = ["DRIVING_METADATA_KEY", "DrivingViewState"]
