# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cosmos 3 MV Auto camera metadata and recorded sensor-view adapter."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from interactive_drive.multiview import (
    CameraView,
    MultiviewUILoop,
    MultiviewUIState,
    validate_cameras,
)

from cosmos3_nano_sim_auto.impl.conditioning.captions import CAMERA_CAPTION_PREFIXES
from cosmos3_nano_sim_auto.inputs import MultiviewSample
from flashdreams.api_v2.loop import IModelLoop, invoke_async
from flashdreams.core.visualization.lidar import LidarView
from flashdreams.runtime_v2.user_input_events import UserInputEvents

# ponytail: nominal Hyperion bearings place at most eleven named cameras;
# use calibrated camera-to-rig extrinsics when supporting another sensor rig.
_BEARINGS = {
    "camera_front_wide_120fov": 0,
    "camera_cross_right_120fov": -90,
    "camera_rear_right_70fov": -135,
    "camera_rear_tele_30fov": 180,
    "camera_rear_left_70fov": 135,
    "camera_cross_left_120fov": 90,
    "camera_front_tele_30fov": 0,
    "camera_front_fisheye_200fov": 0,
    "camera_left_fisheye_200fov": 90,
    "camera_right_fisheye_200fov": -90,
    "camera_rear_fisheye_200fov": 180,
}
"""Display bearings only; these never replace the renderer's calibration."""


def camera_views(sample: MultiviewSample) -> tuple[CameraView, ...]:
    """Keep model channel order and attach Hyperion bearings and real captions."""
    return camera_views_for_names(sample.cameras, sample.view_captions)


def camera_views_for_names(
    cameras: Sequence[str], captions: Sequence[Any] = ()
) -> tuple[CameraView, ...]:
    """Attach display bearings to a camera rig without loading scene assets."""
    views = []
    for index, name in enumerate(cameras):
        if name not in _BEARINGS:
            raise ValueError(f"No UI bearing is defined for camera {name!r}.")
        caption = captions[index] if captions else CAMERA_CAPTION_PREFIXES[name]
        if not isinstance(caption, str):
            caption = json.dumps(caption, ensure_ascii=False)
        label, fov = name.removeprefix("camera_").rsplit("_", 1)
        views.append(
            CameraView(
                name,
                label.replace("_", " ").title(),
                _BEARINGS[name],
                caption,
                float(fov.removesuffix("fov")),
            )
        )
    validate_cameras(views)
    return tuple(views)


@dataclass(slots=True)
class CosmosMultiviewUIState(MultiviewUIState):
    """UI-thread display controls for recorded RGB and LiDAR."""

    model_loop: IModelLoop[Any]
    """Model-thread message destination, independent of the drive model type."""
    lidar_view: LidarView = LidarView.POINT_CLOUD
    """Selected LiDAR renderer; the angled point cloud is the initial default."""


class CosmosMultiviewUILoop(MultiviewUILoop):
    """Bind the shared grid viewer to LiDAR mode changes and runtime restarts."""

    state: CosmosMultiviewUIState
    """The main rank's UI state."""

    def _draw_view_controls(self, imgui: Any) -> None:
        if not self.state.auxiliary_views:
            return
        modes = list(LidarView)
        changed, index = imgui.combo(
            "LiDAR view",
            modes.index(self.state.lidar_view),
            ["Angled point cloud", "Range", "Intensity"],
        )
        if changed:
            selected = modes[index]
            invoke_async(
                self.state.model_loop,
                lambda state: setattr(state, "lidar_view", selected),
            )
            self.state.lidar_view = selected
        imgui.text_wrapped(
            "LiDAR mode applies to the next generated chunk. Focus LiDAR to enlarge."
        )
        imgui.text_wrapped(
            {
                LidarView.POINT_CLOUD: "Height: blue to red. Ego: magenta, white nose.",
                LidarView.RANGE: "Range: blue 0 m to red 100 m. Invalid: black.",
                LidarView.INTENSITY: "Intensity: black 0 to white 1. Invalid: black.",
            }[self.state.lidar_view]
        )

    def _apply_inputs(self, events: UserInputEvents) -> None:
        """Recorded controls have no interactive steering input."""

    def _draw_drive_hud(self, imgui: Any) -> None:
        imgui.text_wrapped("Recorded conditioning: vehicle controls are inactive.")

    def _draw_help(self, imgui: Any) -> None:
        imgui.text_wrapped(
            "Focus enlarges any camera or LiDAR without changing generation."
        )

    def _restart(self) -> None:
        # Session replacement is coordinated by the runtime across every rank.
        self.request_new_session(self.session_desc)
        self.state.status = "Restart requested."
