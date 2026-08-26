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

"""Game-owned adaptation of Ludus PhysX tracks for procedural actors."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from ludus_renderer import PhysicsObjectGraph, PhysXWorld, RigidBodyModel, SceneObject

from omnidreams_game_engine.simulation.actor_controller import ActorTrackTarget

_TRACK_SAMPLE_WINDOW_US = 100_000
"""Half-width of the local native track window around each physics step."""


def _sample_target_track(
    scene_object: SceneObject,
    timestamp_us: int,
    *,
    loop_duration_us: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if loop_duration_us is not None:
        timestamp_us %= loop_duration_us
    position, orientation, _ = scene_object.sample(timestamp_us)
    return position, orientation


class GameplayPhysXWorld(PhysXWorld):
    """Adapt gameplay-owned actor clocks to Ludus's global track clock.

    Ludus samples every native track from the rollout timestamp. This adapter
    publishes a small, rollout-aligned window for each procedural actor while
    leaving recorded scene objects on Ludus's normal global timeline.
    """

    def __init__(
        self,
        graph: PhysicsObjectGraph,
        ego_model: RigidBodyModel,
        *,
        actor_collision_enabled: bool = True,
        max_actor_drive_speed_mps: float | None = None,
        max_actor_drive_speeds_mps: Mapping[str, float] | None = None,
        capacity: int | None = None,
    ) -> None:
        self._gameplay_drive_speed_caps: dict[str, float] = {}
        self.set_actor_drive_speed_caps(max_actor_drive_speeds_mps or {})
        super().__init__(
            graph,
            ego_model,
            actor_collision_enabled=actor_collision_enabled,
            max_actor_drive_speed_mps=max_actor_drive_speed_mps,
            capacity=capacity,
        )
        if not hasattr(self._scene, "set_body_track") or not hasattr(
            self, "_object_native_ids"
        ):
            raise RuntimeError(
                "the installed ludus_renderer lacks the native track replacement "
                "bridge required by gameplay actors"
            )

    def set_actor_drive_speed_caps(self, speed_caps_mps: Mapping[str, float]) -> None:
        """Replace drive-speed caps applied when gameplay actors are inserted."""
        caps = {object_id: float(speed) for object_id, speed in speed_caps_mps.items()}
        if any(not math.isfinite(speed) or speed <= 0.0 for speed in caps.values()):
            raise ValueError("per-actor drive speeds must be finite and positive")
        self._gameplay_drive_speed_caps = caps

    def add_object(
        self, scene_object: SceneObject, *, timestamp_us: int | None = None
    ) -> None:
        """Add an object with its configured gameplay drive-speed cap."""
        default_speed = self.max_actor_drive_speed_mps
        self.max_actor_drive_speed_mps = self._gameplay_drive_speed_caps.get(
            scene_object.object_id, default_speed
        )
        try:
            super().add_object(scene_object, timestamp_us=timestamp_us)
        finally:
            self.max_actor_drive_speed_mps = default_speed

    def synchronize(
        self,
        graph: PhysicsObjectGraph,
        *,
        timestamp_us: int | None = None,
        initial_object_timestamps_us: Mapping[str, int] | None = None,
    ) -> None:
        """Synchronize topology with per-object initial logical timestamps."""
        incoming_objects = {value.object_id: value for value in graph.objects}
        for object_id in tuple(self._objects):
            if object_id not in incoming_objects:
                self.remove_object(object_id)
        for object_id, scene_object in incoming_objects.items():
            current = self._objects.get(object_id)
            if current is scene_object:
                continue
            if current is not None:
                self.remove_object(object_id)
            initial_timestamp = (
                None
                if initial_object_timestamps_us is None
                else initial_object_timestamps_us.get(object_id)
            )
            self.add_object(
                scene_object,
                timestamp_us=(
                    timestamp_us if initial_timestamp is None else initial_timestamp
                ),
            )
        super().synchronize(graph, timestamp_us=timestamp_us)

    def apply_actor_track_targets(
        self,
        targets: tuple[ActorTrackTarget, ...],
        *,
        rollout_timestamp_us: int,
    ) -> None:
        """Publish logical actor targets on Ludus's rollout clock."""
        if not targets:
            return
        native_timestamps = np.asarray(
            [
                rollout_timestamp_us - _TRACK_SAMPLE_WINDOW_US,
                rollout_timestamp_us,
                rollout_timestamp_us + _TRACK_SAMPLE_WINDOW_US,
            ],
            dtype=np.int64,
        )
        for target in targets:
            if target.object_id not in self._objects:
                raise KeyError(target.object_id)
            if not math.isfinite(target.velocity_scale) or not (
                0.0 <= target.velocity_scale <= 1.0
            ):
                raise ValueError("track velocity scale must be within [0, 1]")
            if target.loop_duration_us is not None and target.loop_duration_us <= 0:
                raise ValueError("track loop duration must be positive")

            logical_window_us = int(
                round(_TRACK_SAMPLE_WINDOW_US * target.velocity_scale)
            )
            logical_timestamps = (
                target.timestamp_us - logical_window_us,
                target.timestamp_us,
                target.timestamp_us + logical_window_us,
            )
            scene_object = self._objects[target.object_id]
            samples = tuple(
                _sample_target_track(
                    scene_object,
                    timestamp,
                    loop_duration_us=target.loop_duration_us,
                )
                for timestamp in logical_timestamps
            )
            self._scene.set_body_track(
                self._object_native_ids[target.object_id],
                native_timestamps,
                np.ascontiguousarray(
                    np.stack([sample[0] for sample in samples]), dtype=np.float32
                ),
                np.ascontiguousarray(
                    np.stack([sample[1] for sample in samples]), dtype=np.float32
                ),
                -1.0,
            )


__all__ = ["GameplayPhysXWorld"]
