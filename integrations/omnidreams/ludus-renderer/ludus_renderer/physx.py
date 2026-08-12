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

"""Standalone native PhysX implementation isolated behind Ludus data types."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass

import numpy as np

from ._physx_native import load_native_physx
from .object_graph import (
    BodyState,
    InvisibleBarrier,
    ObjectPose,
    PhysicsObjectGraph,
    PhysicsStep,
    RigidBodyModel,
    SceneObject,
)

_EGO_BODY_ID = 0


@dataclass(frozen=True)
class PhysXStepTimings:
    """Measured components of one compact native physics step."""

    total_ms: float
    actor_update_ms: float
    solver_ms: float
    readback_ms: float
    bridge_ms: float
    visible_actor_count: int
    detached_actor_count: int


@dataclass(frozen=True)
class _CompactPhysicsStep:
    ego: BodyState
    actor_samples: tuple[tuple[str, BodyState, bool], ...]
    track_samples: tuple[tuple[str, np.ndarray, np.ndarray, np.ndarray], ...]
    struck_object_ids: frozenset[str]
    impact: bool
    timings: PhysXStepTimings


def prepare_physx() -> None:
    """Build and cache the pinned standalone PhysX module if necessary."""
    load_native_physx()


def _native_id(value: str, *, barrier: bool = False) -> int:
    namespace = b"barrier:" if barrier else b"body:"
    encoded = hashlib.blake2b(namespace + value.encode(), digest_size=8).digest()
    result = int.from_bytes(encoded, "little", signed=True)
    return result if result not in {-1, _EGO_BODY_ID} else result + 2


def _state_arrays(state: BodyState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pose = np.concatenate((state.position_m, state.orientation_xyzw)).astype(
        np.float32, copy=False
    )
    return (
        np.ascontiguousarray(pose),
        np.ascontiguousarray(state.linear_velocity_mps, dtype=np.float32),
        np.ascontiguousarray(state.angular_velocity_radps, dtype=np.float32),
    )


def _body_state(row: np.ndarray, *, copy: bool = True) -> BodyState:
    """Create a body state, optionally borrowing one stable native-buffer row."""
    values = row.copy() if copy else row
    return BodyState(
        position_m=values[:3],
        orientation_xyzw=values[3:7],
        linear_velocity_mps=values[7:10],
        angular_velocity_radps=values[10:13],
    )


def _yaw_from_xyzw(quaternion: np.ndarray) -> float:
    x, y, z, w = (float(value) for value in quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _overlap(
    ego: BodyState,
    ego_half: tuple[float, float, float],
    actor_position: np.ndarray,
    actor_quaternion: np.ndarray,
    actor_half: tuple[float, float, float],
) -> bool:
    ego_yaw = _yaw_from_xyzw(ego.orientation_xyzw)
    actor_yaw = _yaw_from_xyzw(actor_quaternion)
    axes = (
        np.asarray([math.cos(ego_yaw), math.sin(ego_yaw)]),
        np.asarray([-math.sin(ego_yaw), math.cos(ego_yaw)]),
        np.asarray([math.cos(actor_yaw), math.sin(actor_yaw)]),
        np.asarray([-math.sin(actor_yaw), math.cos(actor_yaw)]),
    )
    delta = actor_position[:2] - ego.position_m[:2]
    for axis in axes:
        ego_radius = sum(
            ego_half[index] * abs(float(np.dot(axes[index], axis)))
            for index in range(2)
        )
        actor_radius = sum(
            actor_half[index] * abs(float(np.dot(axes[index + 2], axis)))
            for index in range(2)
        )
        if abs(float(np.dot(delta, axis))) > ego_radius + actor_radius:
            return False
    return True


def _barrier_rebound_velocity(
    barriers: tuple[InvisibleBarrier, ...],
    ego: BodyState,
    ego_model: RigidBodyModel,
) -> np.ndarray | None:
    position = np.asarray(ego.position_m[:2], dtype=np.float32)
    velocity = np.asarray(ego.linear_velocity_mps, dtype=np.float32)
    yaw = _yaw_from_xyzw(ego.orientation_xyzw)
    forward = np.asarray([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
    left = np.asarray([-forward[1], forward[0]], dtype=np.float32)
    for barrier in barriers:
        start = np.asarray(barrier.start_xy_m, dtype=np.float32)
        end = np.asarray(barrier.end_xy_m, dtype=np.float32)
        segment = end - start
        segment_length_sq = float(np.dot(segment, segment))
        if segment_length_sq <= 1e-8:
            continue
        alpha = float(
            np.clip(np.dot(position - start, segment) / segment_length_sq, 0.0, 1.0)
        )
        closest = start + segment * alpha
        offset = position - closest
        distance = float(np.linalg.norm(offset))
        if distance > 1e-6:
            normal = offset / distance
        else:
            speed = float(np.linalg.norm(velocity[:2]))
            normal = -velocity[:2] / speed if speed > 1e-6 else np.asarray([1.0, 0.0])
        support = (
            abs(float(np.dot(normal, forward))) * ego_model.half_extents_m[0]
            + abs(float(np.dot(normal, left))) * ego_model.half_extents_m[1]
        )
        if distance > support + barrier.thickness_m * 0.5 + 0.05:
            continue
        normal_speed = float(np.dot(velocity[:2], normal))
        if normal_speed >= 0.0:
            return velocity.copy()
        reflected = velocity.copy()
        reflected[:2] -= ((1.0 + ego_model.restitution) * normal_speed * normal).astype(
            np.float32
        )
        return reflected
    return None


class PhysXWorld:
    """Own a reusable standalone PhysX scene and stable body-state buffers."""

    def __init__(
        self,
        graph: PhysicsObjectGraph,
        ego_model: RigidBodyModel,
        *,
        actor_collision_enabled: bool = True,
        max_actor_drive_speed_mps: float | None = None,
        capacity: int | None = None,
    ) -> None:
        if max_actor_drive_speed_mps is not None and (
            not math.isfinite(max_actor_drive_speed_mps)
            or max_actor_drive_speed_mps <= 0.0
        ):
            raise ValueError("max_actor_drive_speed_mps must be finite and positive")
        native = load_native_physx()
        minimum_capacity = len(graph.objects) + 1
        self._scene = native.NativeScene(capacity or max(256, minimum_capacity * 2))
        self.graph = graph
        self.ego_model = ego_model
        self.actor_collision_enabled = actor_collision_enabled
        self.max_actor_drive_speed_mps = max_actor_drive_speed_mps
        self._closed = False
        self._objects: dict[str, SceneObject] = {}
        self._object_slots: dict[str, int] = {}
        self._slot_object_ids: dict[int, str] = {}
        self._object_native_ids: dict[str, int] = {}
        self._object_collision_active: dict[str, bool] = {}
        self._track_drive_enabled: dict[str, bool] = {}
        self._detached_object_ids: set[str] = set()
        self._barriers: dict[str, InvisibleBarrier] = {}
        self._state_buffer = self._scene.state_buffer()
        self._track_state_buffer = self._scene.track_state_buffer()
        self._id_buffer = self._scene.id_buffer()
        self._active_buffer = self._scene.active_buffer()
        self._collision_active_buffer = self._scene.collision_active_buffer()
        self._detached_buffer = self._scene.detached_buffer()
        self._struck_buffer = self._scene.struck_buffer()
        self._half_extents_buffer = np.zeros(
            (len(self._active_buffer), 3), dtype=np.float32
        )
        self.last_step_timings: PhysXStepTimings | None = None

        ego_state = BodyState(
            position_m=np.asarray(
                [0.0, 0.0, ego_model.half_extents_m[2]], dtype=np.float32
            ),
            orientation_xyzw=np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            linear_velocity_mps=np.zeros(3, dtype=np.float32),
            angular_velocity_radps=np.zeros(3, dtype=np.float32),
        )
        self._ego_slot = self._add_body(
            _EGO_BODY_ID, ego_model, ego_state, False, True, None
        )
        for scene_object in graph.objects:
            self.add_object(scene_object)
        for index, barrier in enumerate(graph.barriers):
            self.add_barrier(barrier.barrier_id or f"barrier-{index}", barrier)

    @property
    def state_buffer(self) -> np.ndarray:
        """Return the stable ``[capacity, 13]`` native body-state view."""
        return self._state_buffer

    @property
    def active_buffer(self) -> np.ndarray:
        """Return the stable active-slot mask paired with :attr:`state_buffer`."""
        return self._active_buffer

    @property
    def body_count(self) -> int:
        """Return the number of live bodies, including the ego chassis."""
        return int(self._scene.body_count)

    @property
    def barrier_count(self) -> int:
        """Return the number of live invisible walls."""
        return int(self._scene.barrier_count)

    @property
    def active_collider_ids(self) -> frozenset[str]:
        """Return tracked objects whose PhysX collision shapes are enabled."""
        return frozenset(
            object_id
            for object_id, slot in self._object_slots.items()
            if bool(self._collision_active_buffer[slot])
        )

    def body_state(self, object_id: str) -> BodyState:
        """Return the current authoritative state for one tracked body."""
        return _body_state(self._state_buffer[self._object_slots[object_id]])

    def set_track_drive_enabled(self, object_id: str, enabled: bool) -> None:
        """Enable or suppress the tracked body's driving actuator."""
        self.apply_track_controls(
            ((object_id, enabled, self._objects[object_id].detached),)
        )

    def set_detached(self, object_id: str, detached: bool) -> None:
        """Publish the integration-owned track attachment state to PhysX."""
        self.apply_track_controls(
            ((object_id, self._track_drive_enabled[object_id], detached),)
        )

    def apply_track_controls(
        self, controls: tuple[tuple[str, bool, bool], ...]
    ) -> None:
        """Publish changed traffic controls through one native bridge call."""
        changed = [
            (object_id, bool(drive_enabled), bool(detached))
            for object_id, drive_enabled, detached in controls
            if self._track_drive_enabled.get(object_id) != bool(drive_enabled)
            or self._objects[object_id].detached != bool(detached)
        ]
        if not changed:
            return
        self._scene.set_body_track_controls(
            np.fromiter(
                (self._object_native_ids[item[0]] for item in changed),
                dtype=np.int64,
                count=len(changed),
            ),
            np.fromiter(
                (item[1] for item in changed), dtype=np.uint8, count=len(changed)
            ),
            np.fromiter(
                (item[2] for item in changed), dtype=np.uint8, count=len(changed)
            ),
        )
        for object_id, drive_enabled, detached in changed:
            self._track_drive_enabled[object_id] = drive_enabled
            self._objects[object_id].detached = detached

    def _add_body(
        self,
        native_id: int,
        model: RigidBodyModel,
        state: BodyState,
        kinematic: bool,
        collision_enabled: bool,
        max_drive_speed_mps: float | None,
    ) -> int:
        pose, linear, angular = _state_arrays(state)
        vehicle = model.vehicle
        chassis_half_extents = (
            model.half_extents_m if vehicle is None else vehicle.chassis_half_extents_m
        )
        chassis_offset = (
            (0.0, 0.0, 0.0) if vehicle is None else vehicle.chassis_offset_m
        )
        suspension_mounts = (
            np.empty((0, 3), dtype=np.float32)
            if vehicle is None
            else np.asarray(vehicle.suspension_mounts_m, dtype=np.float32)
        )
        return int(
            self._scene.add_body(
                native_id,
                np.asarray(model.half_extents_m, dtype=np.float32),
                np.asarray(chassis_half_extents, dtype=np.float32),
                np.asarray(chassis_offset, dtype=np.float32),
                model.mass_kg,
                model.friction,
                model.restitution,
                pose,
                linear,
                angular,
                kinematic,
                collision_enabled,
                suspension_mounts,
                0.0 if vehicle is None else vehicle.wheel_radius_m,
                0.0 if vehicle is None else vehicle.suspension_rest_length_m,
                (0.0 if vehicle is None else vehicle.suspension_max_compression_m),
                0.0 if vehicle is None else vehicle.spring_stiffness_n_per_m,
                0.0 if vehicle is None else vehicle.damper_rate_n_s_per_m,
                0.0 if vehicle is None else vehicle.tire_friction,
                (0.0 if vehicle is None else vehicle.cornering_stiffness_n_per_rad),
                0.0 if vehicle is None else vehicle.rolling_resistance,
                0.0 if vehicle is None else vehicle.max_engine_force_n,
                0.0 if vehicle is None else vehicle.max_brake_force_n,
                0.0 if max_drive_speed_mps is None else max_drive_speed_mps,
            )
        )

    def add_object(
        self, scene_object: SceneObject, *, timestamp_us: int | None = None
    ) -> None:
        """Add one tracked object without rebuilding the PhysX scene.

        Args:
            scene_object: Object and recorded track to add.
            timestamp_us: Initial pose time; ``None`` uses the first track sample.
        """
        if scene_object.object_id in self._objects:
            raise ValueError(f"object {scene_object.object_id!r} already exists")
        position, quaternion, velocity = scene_object.sample(
            int(scene_object.timestamps_us[0])
            if timestamp_us is None
            else int(timestamp_us)
        )
        state = BodyState(
            position_m=position,
            orientation_xyzw=quaternion,
            linear_velocity_mps=velocity,
            angular_velocity_radps=np.zeros(3, dtype=np.float32),
        )
        native_id = _native_id(scene_object.object_id)
        slot = self._add_body(
            native_id,
            scene_object.model,
            state,
            False,
            self.actor_collision_enabled,
            self.max_actor_drive_speed_mps,
        )
        self._scene.set_body_track(
            native_id,
            scene_object.timestamps_us,
            scene_object.positions_m,
            scene_object.orientations_xyzw,
            (
                -1.0
                if scene_object.max_extrapolation_us is None
                else float(scene_object.max_extrapolation_us)
            ),
        )
        self._objects[scene_object.object_id] = scene_object
        self._object_slots[scene_object.object_id] = slot
        self._slot_object_ids[slot] = scene_object.object_id
        self._object_native_ids[scene_object.object_id] = native_id
        self._half_extents_buffer[slot] = np.asarray(
            scene_object.model.half_extents_m, dtype=np.float32
        )
        self._object_collision_active[scene_object.object_id] = (
            self.actor_collision_enabled
        )
        self._track_drive_enabled[scene_object.object_id] = True

    def remove_object(self, object_id: str) -> None:
        """Remove one object and recycle its stable native slot."""
        if object_id not in self._objects:
            return
        slot = self._object_slots[object_id]
        self._scene.remove_body(self._object_native_ids[object_id])
        del self._objects[object_id]
        del self._object_slots[object_id]
        del self._slot_object_ids[slot]
        del self._object_native_ids[object_id]
        del self._object_collision_active[object_id]
        del self._track_drive_enabled[object_id]
        self._detached_object_ids.discard(object_id)
        self._half_extents_buffer[slot] = 0.0

    def add_barrier(self, barrier_id: str, barrier: InvisibleBarrier) -> None:
        """Add one invisible static wall without rebuilding the scene."""
        if barrier_id in self._barriers:
            raise ValueError(f"barrier {barrier_id!r} already exists")
        self._scene.add_barrier(
            _native_id(barrier_id, barrier=True),
            np.asarray(barrier.start_xy_m, dtype=np.float32),
            np.asarray(barrier.end_xy_m, dtype=np.float32),
            barrier.thickness_m,
            barrier.height_m,
            self.ego_model.friction,
            self.ego_model.restitution,
        )
        self._barriers[barrier_id] = barrier

    def remove_barrier(self, barrier_id: str) -> None:
        """Remove one invisible wall without rebuilding the scene."""
        if barrier_id not in self._barriers:
            return
        self._scene.remove_barrier(_native_id(barrier_id, barrier=True))
        del self._barriers[barrier_id]

    def synchronize(
        self, graph: PhysicsObjectGraph, *, timestamp_us: int | None = None
    ) -> None:
        """Apply graph additions, replacements, and removals incrementally.

        Args:
            graph: Desired active topology.
            timestamp_us: Initial pose time for newly added objects.
        """
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
            self.add_object(scene_object, timestamp_us=timestamp_us)

        incoming_barriers = {
            barrier.barrier_id or f"barrier-{index}": barrier
            for index, barrier in enumerate(graph.barriers)
        }
        for barrier_id in tuple(self._barriers):
            if barrier_id not in incoming_barriers:
                self.remove_barrier(barrier_id)
        for barrier_id, barrier in incoming_barriers.items():
            if self._barriers.get(barrier_id) == barrier:
                continue
            if barrier_id in self._barriers:
                self.remove_barrier(barrier_id)
            self.add_barrier(barrier_id, barrier)
        self.graph = graph

    def step_compact(
        self, ego: BodyState, timestamp_us: int, dt_s: float
    ) -> _CompactPhysicsStep:
        """Advance all tracked actors with one native call and narrow readback."""
        if self._closed:
            raise RuntimeError("PhysX world is closed")
        total_started_at = time.perf_counter()
        ego_pose, ego_linear, ego_angular = _state_arrays(ego)
        (
            impact,
            actor_update_ms,
            solver_ms,
            readback_ms,
            visible_actor_count,
            detached_actor_count,
        ) = self._scene.step_tracked(
            ego_pose,
            ego_linear,
            ego_angular,
            int(timestamp_us),
            float(dt_s),
            self.actor_collision_enabled,
        )
        # The ego escapes through the public result, so keep its snapshot stable.
        # Actor states remain borrowed below and are materialized once by the game
        # adapter instead of being copied twice on every frame.
        simulated_ego = _body_state(self._state_buffer[self._ego_slot])
        visible_slots = tuple(
            (object_id, slot)
            for object_id, slot in self._object_slots.items()
            if self._objects[object_id].is_visible_at(timestamp_us)
        )
        actor_samples = tuple(
            (
                object_id,
                _body_state(self._state_buffer[slot], copy=False),
                bool(self._detached_buffer[slot]),
            )
            for object_id, slot in visible_slots
        )
        track_samples = tuple(
            (
                object_id,
                self._track_state_buffer[slot, :3],
                self._track_state_buffer[slot, 3:7],
                self._track_state_buffer[slot, 7:10],
            )
            for object_id, slot in visible_slots
        )
        struck_object_ids = frozenset(
            object_id
            for object_id, slot in self._object_slots.items()
            if bool(self._struck_buffer[slot])
        )
        detached_ids = {sample[0] for sample in actor_samples if sample[2]}
        for object_id in self._detached_object_ids - detached_ids:
            self._objects[object_id].detached = False
        for object_id in detached_ids - self._detached_object_ids:
            self._objects[object_id].detached = True
        self._detached_object_ids = detached_ids
        total_ms = (time.perf_counter() - total_started_at) * 1000.0
        native_ms = float(actor_update_ms) + float(solver_ms) + float(readback_ms)
        timings = PhysXStepTimings(
            total_ms=total_ms,
            actor_update_ms=float(actor_update_ms),
            solver_ms=float(solver_ms),
            readback_ms=float(readback_ms),
            bridge_ms=max(0.0, total_ms - native_ms),
            visible_actor_count=int(visible_actor_count),
            detached_actor_count=int(detached_actor_count),
        )
        self.last_step_timings = timings
        return _CompactPhysicsStep(
            ego=simulated_ego,
            actor_samples=actor_samples,
            track_samples=track_samples,
            struck_object_ids=struck_object_ids,
            impact=bool(impact),
            timings=timings,
        )

    def collider_state_arrays(
        self,
    ) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]:
        """Return compact state arrays for currently collidable tracked bodies."""
        slots = np.flatnonzero(self._collision_active_buffer)
        slots = slots[slots != self._ego_slot]
        if len(slots) == 0:
            return (
                (),
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 4), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32),
            )
        return (
            tuple(self._slot_object_ids[int(slot)] for slot in slots),
            self._state_buffer[slots, :3].copy(),
            self._state_buffer[slots, 3:7].copy(),
            (self._half_extents_buffer[slots] * 2.0).astype(np.float32, copy=False),
        )

    def step(self, ego: BodyState, timestamp_us: int, dt_s: float) -> PhysicsStep:
        """Advance PhysX and materialize the compatibility object-pose result."""
        compact = self.step_compact(ego, timestamp_us, dt_s)
        object_poses = []
        for object_id, slot in self._object_slots.items():
            state = _body_state(self._state_buffer[slot])
            collision_active = bool(self._collision_active_buffer[slot])
            self._object_collision_active[object_id] = collision_active
            object_poses.append(
                ObjectPose(
                    object_id=object_id,
                    position_m=state.position_m,
                    orientation_xyzw=state.orientation_xyzw,
                    linear_velocity_mps=state.linear_velocity_mps,
                    angular_velocity_radps=state.angular_velocity_radps,
                    detached=self._objects[object_id].detached,
                    collision_active=collision_active,
                )
            )
        objects = tuple(object_poses)
        return PhysicsStep(ego=compact.ego, objects=objects, impact=compact.impact)

    def close(self) -> None:
        """Release the native PhysX scene and its reusable buffers."""
        if self._closed:
            return
        self._closed = True
        self._scene.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = ["PhysXWorld", "prepare_physx"]
