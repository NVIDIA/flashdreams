# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SANA-WM live-control adaptation for the shared Cam2V application."""

from __future__ import annotations

import numpy as np
from cam2v import CameraPoseIntegrator, PoseSegment

from sana_wm.impl.camera import (
    CameraPoseIntegrator as SanaCameraPoseIntegrator,
)
from sana_wm.impl.camera import (
    VelocityState,
    controls_to_target_velocity,
)

_KEY_TO_CONTROL = {
    "w": "forward",
    "s": "back",
    "a": "yaw_left",
    "j": "yaw_left",
    "d": "yaw_right",
    "l": "yaw_right",
    "i": "pitch_up",
    "k": "pitch_down",
    "q": "strafe_left",
    "e": "strafe_right",
}


class SanaWMCameraPoseIntegrator(CameraPoseIntegrator):
    """Map Cam2V keys through SANA-WM's trained motion calibration."""

    def __init__(self) -> None:
        self.reset()

    def reset(self, pose: np.ndarray | None = None) -> None:
        """Reset SANA-WM pose and smoothed velocity state."""
        self._integrator = SanaCameraPoseIntegrator()
        if pose is not None:
            if pose.shape != (4, 4):
                raise ValueError(f"Expected pose shape (4, 4), got {pose.shape}")
            self._integrator.pose = pose.astype(np.float64, copy=True)
            self._integrator.pitch = float(np.arctan2(pose[2, 1], pose[1, 1]))
        self._velocity = VelocityState()
        self._last_controls: set[str] = set()

    def current_pose(self) -> np.ndarray:
        """Return a copy of the most recently integrated camera pose."""
        return self._integrator.pose.astype(np.float32, copy=True)

    def integrate_chunk(
        self,
        *,
        segments: list[PoseSegment],
        frame_times: list[float],
    ) -> np.ndarray:
        """Return SANA-WM camera poses at the requested frame times."""
        if not segments or not frame_times:
            raise ValueError("SANA-WM Cam2V requires control segments and frame times.")
        poses: list[np.ndarray] = []
        segment_index = 0
        previous_time = float(segments[0][0])
        for frame_time in frame_times:
            while (
                segment_index + 1 < len(segments)
                and frame_time > segments[segment_index][1]
            ):
                segment_index += 1
            keys = segments[segment_index][2]
            controls = {_KEY_TO_CONTROL[key] for key in keys if key in _KEY_TO_CONTROL}
            target = controls_to_target_velocity(controls)
            dt = max(0.0, float(frame_time) - previous_time)
            if controls - self._last_controls:
                self._velocity.snap_to(target)
            else:
                self._velocity.step_toward(target, dt)
            poses.append(self._integrator.step(self._velocity))
            self._last_controls = controls
            previous_time = float(frame_time)
        return np.stack(poses).astype(np.float32)


__all__ = ["SanaWMCameraPoseIntegrator"]
