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

"""PhysX-first object graph shared by Ludus simulation and rendering."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]
Int64Array = npt.NDArray[np.int64]


@dataclass(frozen=True)
class VehicleModel:
    """Per-instance four-wheel contact and suspension model."""

    chassis_half_extents_m: tuple[float, float, float]
    """Half extents of the sprung chassis collision shape."""

    chassis_offset_m: tuple[float, float, float]
    """Chassis-shape center relative to the visual body origin."""

    suspension_mounts_m: tuple[tuple[float, float, float], ...]
    """Four suspension mounts in body-local coordinates."""

    wheel_radius_m: float
    """Wheel radius used by suspension raycasts."""

    suspension_rest_length_m: float
    """Uncompressed suspension length from mount to wheel center."""

    suspension_max_compression_m: float
    """Maximum spring compression before the bump stop."""

    spring_stiffness_n_per_m: float
    """Linear spring rate for each wheel."""

    damper_rate_n_s_per_m: float
    """Linear compression and rebound damping rate for each wheel."""

    tire_friction: float
    """Coulomb friction coefficient limiting each tire's contact force."""

    cornering_stiffness_n_per_rad: float
    """Lateral tire force per slip angle, divided across four wheels."""

    rolling_resistance: float
    """Longitudinal rolling-resistance coefficient."""

    max_engine_force_n: float
    """Maximum propulsion force available to the shared drive actuator."""

    max_brake_force_n: float
    """Maximum braking force available to the shared drive actuator."""


@dataclass(frozen=True)
class RigidBodyModel:
    """Immutable engine-neutral body topology and material parameters."""

    mass_kg: float
    half_extents_m: tuple[float, float, float]
    restitution: float = 0.22
    friction: float = 0.65
    vehicle: VehicleModel | None = None
    """Wheel and suspension model; ``None`` uses the box collider directly."""


@dataclass
class SceneObject:
    """Typed object with contiguous tracks and a fixed PhysX body model."""

    object_id: str
    object_type: str
    model: RigidBodyModel
    timestamps_us: Int64Array
    positions_m: FloatArray
    orientations_xyzw: FloatArray
    max_extrapolation_us: float | None = None
    detached: bool = False

    def __post_init__(self) -> None:
        self.timestamps_us = np.ascontiguousarray(self.timestamps_us, dtype=np.int64)
        self.positions_m = np.ascontiguousarray(self.positions_m, dtype=np.float32)
        self.orientations_xyzw = np.ascontiguousarray(
            self.orientations_xyzw, dtype=np.float32
        )
        count = len(self.timestamps_us)
        if count == 0:
            raise ValueError("object tracks must contain at least one sample")
        if self.positions_m.shape != (count, 3):
            raise ValueError("positions_m must have shape [samples, 3]")
        if self.orientations_xyzw.shape != (count, 4):
            raise ValueError("orientations_xyzw must have shape [samples, 4]")

    def sample(self, timestamp_us: int) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Interpolate pose and finite-difference velocity at a timestamp."""
        index = int(np.searchsorted(self.timestamps_us, timestamp_us, side="left"))
        if index <= 0:
            lo = hi = 0
            alpha = 0.0
        elif index >= len(self.timestamps_us):
            lo = hi = len(self.timestamps_us) - 1
            alpha = 0.0
        else:
            lo, hi = index - 1, index
            span = int(self.timestamps_us[hi] - self.timestamps_us[lo])
            alpha = 0.0 if span == 0 else (timestamp_us - self.timestamps_us[lo]) / span
        position = (
            self.positions_m[lo] * (1.0 - alpha) + self.positions_m[hi] * alpha
        ).astype(np.float32)
        quaternion = (
            self.orientations_xyzw[lo] * (1.0 - alpha)
            + self.orientations_xyzw[hi] * alpha
        ).astype(np.float32)
        norm = float(np.linalg.norm(quaternion))
        if norm > 1e-8:
            quaternion /= norm
        if lo == hi:
            velocity = np.zeros(3, dtype=np.float32)
        else:
            dt_s = float(self.timestamps_us[hi] - self.timestamps_us[lo]) / 1_000_000.0
            velocity = ((self.positions_m[hi] - self.positions_m[lo]) / dt_s).astype(
                np.float32
            )
        return position, quaternion, velocity

    def is_visible_at(self, timestamp_us: int) -> bool:
        """Keep a track visible after it starts, holding its final pose."""
        if self.max_extrapolation_us is None:
            return True
        first_timestamp_us = int(self.timestamps_us[0])
        if timestamp_us < first_timestamp_us:
            return (
                len(self.timestamps_us) >= 2
                and first_timestamp_us - timestamp_us <= self.max_extrapolation_us
            )
        return True


@dataclass(frozen=True)
class InvisibleBarrier:
    """Non-rendered collision wall represented by one map-line segment."""

    start_xy_m: tuple[float, float]
    end_xy_m: tuple[float, float]
    thickness_m: float = 0.30
    height_m: float = 3.0
    barrier_id: str | None = None


@dataclass
class PhysicsObjectGraph:
    """Contiguous, typed scene data used to build and update one PhysX world."""

    objects: tuple[SceneObject, ...] = ()
    barriers: tuple[InvisibleBarrier, ...] = ()
    object_index: dict[str, int] = field(init=False)
    revision: int = field(init=False, default=0)
    _spatial_cell_size_m: float = field(
        init=False, default=64.0, repr=False, compare=False
    )
    _object_cells: dict[tuple[int, int], tuple[int, ...]] = field(
        init=False, default_factory=dict, repr=False, compare=False
    )
    _barrier_cells: dict[tuple[int, int], tuple[int, ...]] = field(
        init=False, default_factory=dict, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        self.object_index = {
            scene_object.object_id: index
            for index, scene_object in enumerate(self.objects)
        }
        if len(self.object_index) != len(self.objects):
            raise ValueError("object_id values must be unique")
        self._rebuild_spatial_index()

    def _cells_for_bounds(
        self, minimum_xy: np.ndarray, maximum_xy: np.ndarray
    ) -> tuple[tuple[int, int], ...]:
        minimum_cell = np.floor(minimum_xy / self._spatial_cell_size_m).astype(np.int64)
        maximum_cell = np.floor(maximum_xy / self._spatial_cell_size_m).astype(np.int64)
        return tuple(
            (x, y)
            for x in range(int(minimum_cell[0]), int(maximum_cell[0]) + 1)
            for y in range(int(minimum_cell[1]), int(maximum_cell[1]) + 1)
        )

    def _rebuild_spatial_index(self) -> None:
        object_cells: dict[tuple[int, int], list[int]] = {}
        for index, scene_object in enumerate(self.objects):
            positions_xy = scene_object.positions_m[:, :2]
            footprint_radius_m = float(
                np.linalg.norm(scene_object.model.half_extents_m[:2])
            )
            extent = np.full(2, footprint_radius_m, dtype=np.float32)
            occupied_cells: set[tuple[int, int]] = set()
            if len(positions_xy) == 1:
                occupied_cells.update(
                    self._cells_for_bounds(
                        positions_xy[0] - extent, positions_xy[0] + extent
                    )
                )
            else:
                # Index swept segment bounds, not only recorded samples. Sparse
                # tracks can cross a simulation window without a keyframe in it.
                for start, end in zip(positions_xy[:-1], positions_xy[1:], strict=True):
                    occupied_cells.update(
                        self._cells_for_bounds(
                            np.minimum(start, end) - extent,
                            np.maximum(start, end) + extent,
                        )
                    )
            for cell in occupied_cells:
                object_cells.setdefault(cell, []).append(index)
        barrier_cells: dict[tuple[int, int], list[int]] = {}
        for index, barrier in enumerate(self.barriers):
            endpoints = np.asarray(
                [barrier.start_xy_m, barrier.end_xy_m], dtype=np.float32
            )
            extent = np.full(2, barrier.thickness_m * 0.5, dtype=np.float32)
            for cell in self._cells_for_bounds(
                np.min(endpoints, axis=0) - extent,
                np.max(endpoints, axis=0) + extent,
            ):
                barrier_cells.setdefault(cell, []).append(index)
        self._object_cells = {
            cell: tuple(indices) for cell, indices in object_cells.items()
        }
        self._barrier_cells = {
            cell: tuple(indices) for cell, indices in barrier_cells.items()
        }

    def _spatial_candidates(
        self,
        index: dict[tuple[int, int], tuple[int, ...]],
        center: np.ndarray,
        radius_m: float,
    ) -> tuple[int, ...]:
        extent = np.full(2, radius_m, dtype=np.float32)
        candidates: set[int] = set()
        for cell in self._cells_for_bounds(center - extent, center + extent):
            candidates.update(index.get(cell, ()))
        return tuple(sorted(candidates))

    def upsert_object(self, scene_object: SceneObject) -> None:
        """Add or replace one object while preserving all unrelated topology."""
        existing = self.object_index.get(scene_object.object_id)
        objects = list(self.objects)
        if existing is None:
            objects.append(scene_object)
        else:
            objects[existing] = scene_object
        self.objects = tuple(objects)
        self.object_index = {
            value.object_id: index for index, value in enumerate(self.objects)
        }
        self.revision += 1
        self._rebuild_spatial_index()

    def remove_object(self, object_id: str) -> None:
        """Remove one object without rebuilding unrelated graph data."""
        if object_id not in self.object_index:
            return
        self.objects = tuple(
            scene_object
            for scene_object in self.objects
            if scene_object.object_id != object_id
        )
        self.object_index = {
            value.object_id: index for index, value in enumerate(self.objects)
        }
        self.revision += 1
        self._rebuild_spatial_index()

    def upsert_barrier(self, barrier_id: str, barrier: InvisibleBarrier) -> None:
        """Add or replace one named invisible wall."""
        normalized = InvisibleBarrier(
            start_xy_m=barrier.start_xy_m,
            end_xy_m=barrier.end_xy_m,
            thickness_m=barrier.thickness_m,
            height_m=barrier.height_m,
            barrier_id=barrier_id,
        )
        barriers = list(self.barriers)
        for index, existing in enumerate(barriers):
            if existing.barrier_id == barrier_id:
                barriers[index] = normalized
                break
        else:
            barriers.append(normalized)
        self.barriers = tuple(barriers)
        self.revision += 1
        self._rebuild_spatial_index()

    def remove_barrier(self, barrier_id: str) -> None:
        """Remove one named invisible wall."""
        barriers = tuple(
            barrier for barrier in self.barriers if barrier.barrier_id != barrier_id
        )
        if len(barriers) == len(self.barriers):
            return
        self.barriers = barriers
        self.revision += 1
        self._rebuild_spatial_index()

    def copy_for_physx(
        self,
        center_xy_m: npt.ArrayLike,
        radius_m: float,
        timestamp_us: int | None = None,
    ) -> PhysicsObjectGraph:
        """Copy nearby topology into the active PhysX simulation window.

        Args:
            center_xy_m: Center of the simulation window.
            radius_m: Simulation radius in metres.
            timestamp_us: Track-sampling time; ``None`` considers every recorded
                position.
        """
        center = np.asarray(center_xy_m, dtype=np.float32)
        if center.shape != (2,):
            raise ValueError("center_xy_m must have shape (2,)")
        if radius_m <= 0.0:
            raise ValueError("radius_m must be positive")
        object_candidates = self._spatial_candidates(
            self._object_cells, center, float(radius_m)
        )

        def _object_is_near(scene_object: SceneObject) -> bool:
            footprint_radius_m = float(
                np.linalg.norm(scene_object.model.half_extents_m[:2])
            )
            expanded_radius_sq = (float(radius_m) + footprint_radius_m) ** 2
            if timestamp_us is not None:
                if not scene_object.is_visible_at(timestamp_us):
                    return False
                position, _, _ = scene_object.sample(timestamp_us)
                delta = position[:2] - center
                return float(np.dot(delta, delta)) <= expanded_radius_sq
            points = scene_object.positions_m[:, :2]
            if len(points) == 1:
                delta = points[0] - center
                return float(np.dot(delta, delta)) <= expanded_radius_sq
            segments = points[1:] - points[:-1]
            length_sq = np.sum(segments * segments, axis=1)
            offset = center[None, :] - points[:-1]
            alpha = np.divide(
                np.sum(offset * segments, axis=1),
                length_sq,
                out=np.zeros_like(length_sq),
                where=length_sq > 1e-8,
            )
            alpha = np.clip(alpha, 0.0, 1.0)
            closest = points[:-1] + segments * alpha[:, None]
            distance_sq = np.sum((closest - center[None, :]) ** 2, axis=1)
            return float(np.min(distance_sq)) <= expanded_radius_sq

        objects = tuple(
            self.objects[index]
            for index in object_candidates
            if _object_is_near(self.objects[index])
        )

        def _barrier_is_near(barrier: InvisibleBarrier) -> bool:
            start = np.asarray(barrier.start_xy_m, dtype=np.float32)
            end = np.asarray(barrier.end_xy_m, dtype=np.float32)
            segment = end - start
            length_sq = float(np.dot(segment, segment))
            if length_sq <= 1e-8:
                closest = start
            else:
                alpha = float(
                    np.clip(np.dot(center - start, segment) / length_sq, 0.0, 1.0)
                )
                closest = start + segment * alpha
            expanded_radius = float(radius_m) + barrier.thickness_m * 0.5
            return float(np.dot(center - closest, center - closest)) <= (
                expanded_radius * expanded_radius
            )

        barrier_candidates = self._spatial_candidates(
            self._barrier_cells, center, float(radius_m)
        )
        barriers = tuple(
            self.barriers[index]
            for index in barrier_candidates
            if _barrier_is_near(self.barriers[index])
        )
        return PhysicsObjectGraph(objects=objects, barriers=barriers)


@dataclass(frozen=True)
class BodyState:
    """Engine-neutral rigid-body state crossing the Ludus PhysX boundary."""

    position_m: FloatArray
    orientation_xyzw: FloatArray
    linear_velocity_mps: FloatArray
    angular_velocity_radps: FloatArray


@dataclass(frozen=True)
class ObjectPose:
    """One simulated object pose returned by PhysX."""

    object_id: str
    position_m: FloatArray
    orientation_xyzw: FloatArray
    linear_velocity_mps: FloatArray
    angular_velocity_radps: FloatArray
    detached: bool
    collision_active: bool


@dataclass(frozen=True)
class PhysicsStep:
    """Authoritative ego and object state after a PhysX step."""

    ego: BodyState
    objects: tuple[ObjectPose, ...]
    impact: bool


__all__ = [
    "BodyState",
    "InvisibleBarrier",
    "ObjectPose",
    "PhysicsObjectGraph",
    "PhysicsStep",
    "RigidBodyModel",
    "SceneObject",
    "VehicleModel",
]
