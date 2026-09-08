# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Traffic-agent recovery policy for physically simulated scene vehicles."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from ludus_renderer import BodyState, SceneObject

_RESTART_AFTER_STOPPED_S = 1.0
"""Continuous stopped time required before a struck vehicle drives again."""

_STOPPED_LINEAR_SPEED_MPS = 0.10
"""Maximum horizontal speed considered stationary by traffic AI."""

_STOPPED_ANGULAR_SPEED_RADPS = 0.10
"""Maximum angular speed considered stationary by traffic AI."""


def _yaw_from_quaternion_xyzw(quaternion: np.ndarray) -> float:
    x, y, z, w = (float(value) for value in quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


@dataclass(frozen=True)
class TrafficDriverDecision:
    """Traffic-AI outputs consumed by the native vehicle actuator."""

    drive_enabled: bool
    """Whether the actor may apply its track-following driving command."""

    detached_from_track: bool
    """Whether rendering should use the simulated recovery trajectory."""


@dataclass
class _TrafficDriverState:
    drive_enabled: bool = True
    """Whether track-following control is active."""

    detached_from_track: bool = False
    """Whether a collision has displaced the actor from its track."""

    stopped_duration_s: float = 0.0
    """Continuous stationary time accumulated after the latest strike."""


class TrafficDriverAI:
    """Own collision recovery decisions for tracked road vehicles."""

    def __init__(self) -> None:
        self._states: dict[str, _TrafficDriverState] = {}

    def synchronize(self, objects: tuple[SceneObject, ...]) -> None:
        """Synchronize AI state with the active vehicle object set."""
        vehicle_ids = {
            scene_object.object_id
            for scene_object in objects
            if scene_object.model.vehicle is not None
        }
        self._states = {
            object_id: self._states.get(object_id, _TrafficDriverState())
            for object_id in vehicle_ids
        }

    def update(
        self,
        object_id: str,
        *,
        struck: bool,
        body: BodyState,
        track_position: np.ndarray,
        track_orientation_xyzw: np.ndarray,
        track_velocity_mps: np.ndarray,
        dt_s: float,
    ) -> TrafficDriverDecision | None:
        """Advance one vehicle's recovery policy from observed physics state."""
        state = self._states.get(object_id)
        if state is None:
            return None

        if struck:
            state.drive_enabled = False
            state.detached_from_track = True
            state.stopped_duration_s = 0.0

        if not state.drive_enabled:
            linear_speed = float(np.linalg.norm(body.linear_velocity_mps[:2]))
            angular_speed = float(np.linalg.norm(body.angular_velocity_radps))
            stopped = (
                linear_speed <= _STOPPED_LINEAR_SPEED_MPS
                and angular_speed <= _STOPPED_ANGULAR_SPEED_RADPS
            )
            state.stopped_duration_s = (
                state.stopped_duration_s + dt_s if stopped else 0.0
            )
            if state.stopped_duration_s >= _RESTART_AFTER_STOPPED_S:
                state.drive_enabled = True
                state.stopped_duration_s = 0.0
        elif state.detached_from_track and self._is_recovered(
            body,
            track_position,
            track_orientation_xyzw,
            track_velocity_mps,
        ):
            state.detached_from_track = False

        return TrafficDriverDecision(
            drive_enabled=state.drive_enabled,
            detached_from_track=state.detached_from_track,
        )

    @staticmethod
    def _is_recovered(
        body: BodyState,
        track_position: np.ndarray,
        track_orientation_xyzw: np.ndarray,
        track_velocity_mps: np.ndarray,
    ) -> bool:
        displacement = np.asarray(track_position[:2] - body.position_m[:2])
        heading_error = math.atan2(
            math.sin(
                _yaw_from_quaternion_xyzw(track_orientation_xyzw)
                - _yaw_from_quaternion_xyzw(body.orientation_xyzw)
            ),
            math.cos(
                _yaw_from_quaternion_xyzw(track_orientation_xyzw)
                - _yaw_from_quaternion_xyzw(body.orientation_xyzw)
            ),
        )
        velocity_error = np.asarray(
            track_velocity_mps[:2] - body.linear_velocity_mps[:2]
        )
        return (
            float(np.linalg.norm(displacement)) <= 0.60
            and abs(heading_error) <= math.radians(8.0)
            and float(np.linalg.norm(velocity_error)) <= 0.75
        )


__all__ = ["TrafficDriverAI", "TrafficDriverDecision"]
