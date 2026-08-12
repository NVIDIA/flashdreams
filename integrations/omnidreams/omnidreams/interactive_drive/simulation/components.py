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

"""Engine-neutral components for interactive driving physics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from ludus_renderer import RigidBodyModel, VehicleModel
from omnidreams.interactive_drive.config import VehicleConfig
from omnidreams.interactive_drive.types import VehicleState

FloatArray = npt.NDArray[np.float32]

# Keep struck vehicles far enough ahead of the front camera that their HD-map
# boxes remain recognizable.  A near-plane-sized actor is outside the world
# model's driving-data distribution and causes the generated object to smear or
# disappear after impact.  Lateral padding stays small so parked curbside cars
# do not create phantom side contacts.
_VEHICLE_LONGITUDINAL_COLLISION_PADDING_M = 0.2
_VEHICLE_LATERAL_COLLISION_PADDING_M = 0.1

_FIXED_OBJECT_MASS_KG = {
    "car": 1_550.0,
    "truck": 8_000.0,
    "bus": 12_000.0,
    "trailer": 10_000.0,
    "pedestrian": 80.0,
    "cyclist": 100.0,
    "motorcycle": 220.0,
    "other": 500.0,
}


def canonical_object_type(object_type: str) -> str:
    """Map scene labels to the simulation categories owned by interactive-drive."""
    normalized = object_type.strip().lower()
    # Check compound and overlapping labels before their more general forms.
    if "trailer" in normalized:
        return "trailer"
    if "motor" in normalized:
        return "motorcycle"
    if "bus" in normalized:
        return "bus"
    if "truck" in normalized:
        return "truck"
    if "pedestrian" in normalized or "person" in normalized:
        return "pedestrian"
    if "cycl" in normalized or "bicycle" in normalized:
        return "cyclist"
    if "car" in normalized or "vehicle" in normalized:
        return "car"
    return "other"


def _vector3(value: FloatArray, name: str) -> FloatArray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {result.shape}")
    return result.copy()


@dataclass
class TransformComponent:
    """World-space transform component."""

    position_m: FloatArray
    """World position in metres."""

    orientation_xyzw: FloatArray
    """Normalized world orientation quaternion in ``xyzw`` order."""

    def __post_init__(self) -> None:
        self.position_m = _vector3(self.position_m, "position_m")
        orientation = np.asarray(self.orientation_xyzw, dtype=np.float32)
        if orientation.shape != (4,):
            raise ValueError(
                f"orientation_xyzw must have shape (4,), got {orientation.shape}"
            )
        norm = float(np.linalg.norm(orientation))
        if norm <= 1e-8:
            raise ValueError("orientation_xyzw must have non-zero norm")
        self.orientation_xyzw = (orientation / norm).astype(np.float32)


@dataclass
class RigidBodyComponent:
    """Linear and angular rigid-body state in SI units."""

    mass_kg: float
    """Positive body mass; trucks use a larger value than cars."""

    linear_velocity_mps: FloatArray
    """World linear velocity."""

    angular_velocity_radps: FloatArray
    """Body angular velocity around world axes."""

    restitution: float = 0.2
    """Normal collision bounciness in the closed interval ``[0, 1]``."""

    friction: float = 0.7
    """Tangential collision friction coefficient."""

    dynamic: bool = True
    """Whether impulses and integration can move the body."""

    def __post_init__(self) -> None:
        if self.mass_kg <= 0.0:
            raise ValueError("mass_kg must be positive")
        self.linear_velocity_mps = _vector3(
            self.linear_velocity_mps, "linear_velocity_mps"
        )
        self.angular_velocity_radps = _vector3(
            self.angular_velocity_radps, "angular_velocity_radps"
        )
        self.restitution = float(np.clip(self.restitution, 0.0, 1.0))
        self.friction = max(0.0, float(self.friction))


@dataclass(frozen=True)
class BoxColliderComponent:
    """Oriented box collider component."""

    half_extents_m: tuple[float, float, float]
    """Positive half dimensions ordered as length, width, and height."""

    is_trigger: bool = False
    """Whether overlap emits events without applying impulses."""

    def __post_init__(self) -> None:
        if len(self.half_extents_m) != 3 or any(v <= 0.0 for v in self.half_extents_m):
            raise ValueError("half_extents_m must contain three positive values")


@dataclass(frozen=True)
class VehicleDynamicsComponent:
    """Four-wheel game-car layout, drivetrain, and tire parameters."""

    wheel_base_m: float
    track_width_m: float
    front_axle_to_cg_m: float
    rear_axle_to_cg_m: float
    center_of_mass_height_m: float
    wheel_radius_m: float
    wheel_width_m: float
    max_engine_force_n: float
    max_brake_force_n: float
    max_lateral_accel_mps2: float
    cornering_stiffness_n_per_rad: float
    yaw_inertia_kg_m2: float
    tire_grip: float
    rolling_resistance: float
    aero_drag_coefficient: float

    @property
    def wheel_offsets_m(self) -> tuple[tuple[float, float, float], ...]:
        """Return front-left, front-right, rear-left, and rear-right offsets."""
        half_track = self.track_width_m * 0.5
        wheel_z = -self.center_of_mass_height_m + self.wheel_radius_m
        return (
            (self.front_axle_to_cg_m, half_track, wheel_z),
            (self.front_axle_to_cg_m, -half_track, wheel_z),
            (-self.rear_axle_to_cg_m, half_track, wheel_z),
            (-self.rear_axle_to_cg_m, -half_track, wheel_z),
        )


@dataclass(frozen=True)
class SuspensionComponent:
    """Visual spring-damper suspension parameters."""

    stiffness: float
    damping: float
    travel_m: float
    visual_gain: float
    max_roll_rad: float
    max_pitch_rad: float


@dataclass
class GameEntity:
    """Entity composed from engine-neutral transform and physics components."""

    entity_id: str
    transform: TransformComponent
    rigid_body: RigidBodyComponent
    collider: BoxColliderComponent
    vehicle: VehicleDynamicsComponent | None = None
    suspension: SuspensionComponent | None = None
    object_type: str = "Car"
    detached_from_track: bool = False

    def to_game_engine_dict(self) -> dict[str, Any]:
        """Return JSON-compatible component data for an external engine."""
        components: dict[str, Any] = {
            "transform": {
                "position_m": self.transform.position_m.tolist(),
                "orientation_xyzw": self.transform.orientation_xyzw.tolist(),
            },
            "rigid_body": {
                "mass_kg": self.rigid_body.mass_kg,
                "linear_velocity_mps": self.rigid_body.linear_velocity_mps.tolist(),
                "angular_velocity_radps": self.rigid_body.angular_velocity_radps.tolist(),
                "restitution": self.rigid_body.restitution,
                "friction": self.rigid_body.friction,
                "dynamic": self.rigid_body.dynamic,
            },
            "box_collider": {
                "half_extents_m": list(self.collider.half_extents_m),
                "is_trigger": self.collider.is_trigger,
            },
        }
        if self.vehicle is not None:
            components["vehicle_dynamics"] = {
                "wheel_base_m": self.vehicle.wheel_base_m,
                "track_width_m": self.vehicle.track_width_m,
                "front_axle_to_cg_m": self.vehicle.front_axle_to_cg_m,
                "rear_axle_to_cg_m": self.vehicle.rear_axle_to_cg_m,
                "center_of_mass_height_m": self.vehicle.center_of_mass_height_m,
                "wheel_radius_m": self.vehicle.wheel_radius_m,
                "wheel_width_m": self.vehicle.wheel_width_m,
                "wheel_offsets_m": self.vehicle.wheel_offsets_m,
                "max_engine_force_n": self.vehicle.max_engine_force_n,
                "max_brake_force_n": self.vehicle.max_brake_force_n,
                "max_lateral_accel_mps2": self.vehicle.max_lateral_accel_mps2,
                "cornering_stiffness_n_per_rad": (
                    self.vehicle.cornering_stiffness_n_per_rad
                ),
                "yaw_inertia_kg_m2": self.vehicle.yaw_inertia_kg_m2,
                "tire_grip": self.vehicle.tire_grip,
                "rolling_resistance": self.vehicle.rolling_resistance,
                "aero_drag_coefficient": self.vehicle.aero_drag_coefficient,
            }
        if self.suspension is not None:
            components["suspension"] = {
                "stiffness": self.suspension.stiffness,
                "damping": self.suspension.damping,
                "travel_m": self.suspension.travel_m,
                "visual_gain": self.suspension.visual_gain,
                "max_roll_rad": self.suspension.max_roll_rad,
                "max_pitch_rad": self.suspension.max_pitch_rad,
            }
        return {
            "entity_id": self.entity_id,
            "object_type": self.object_type,
            "components": components,
        }


def rigid_body_model_for_object(
    object_type: str,
    dimensions_lwh: npt.ArrayLike,
    *,
    restitution: float = 0.22,
    friction: float = 0.65,
) -> RigidBodyModel:
    """Build a category-weighted PhysX body from an interactive scene label."""
    dimensions = np.asarray(dimensions_lwh, dtype=np.float32)
    if dimensions.shape != (3,) or bool(np.any(dimensions <= 0.0)):
        raise ValueError("dimensions_lwh must contain three positive values")
    kind = canonical_object_type(object_type)
    mass_kg = _FIXED_OBJECT_MASS_KG[kind]
    dynamics = vehicle_dynamics_for_object(object_type, dimensions, mass_kg)
    return RigidBodyModel(
        mass_kg=mass_kg,
        half_extents_m=(
            float(dimensions[0]) * 0.5,
            float(dimensions[1]) * 0.5,
            float(dimensions[2]) * 0.5,
        ),
        restitution=float(np.clip(restitution, 0.0, 1.0)),
        friction=max(0.0, float(friction)),
        vehicle=(
            None
            if dynamics is None
            else _physx_vehicle_model(
                dynamics,
                dimensions,
                mass_kg,
                suspension_travel_m=0.22 if kind == "car" else 0.32,
                longitudinal_collision_padding_m=(
                    _VEHICLE_LONGITUDINAL_COLLISION_PADDING_M
                ),
                lateral_collision_padding_m=_VEHICLE_LATERAL_COLLISION_PADDING_M,
            )
        ),
    )


def _vehicle_dynamics(
    *,
    kind: str,
    dimensions_lwh: npt.ArrayLike,
    mass_kg: float,
    max_accel_mps2: float,
    max_brake_mps2: float,
    max_lateral_accel_mps2: float,
    tire_grip: float,
    rolling_resistance: float,
    aero_drag_coefficient: float,
    wheel_base_m: float | None = None,
) -> VehicleDynamicsComponent | None:
    if kind not in {"car", "truck", "bus", "trailer"}:
        return None
    length, width, height = (
        float(value) for value in np.asarray(dimensions_lwh, dtype=np.float32)
    )
    wheel_base = (
        float(wheel_base_m)
        if wheel_base_m is not None
        else float(np.clip(length * 0.58, 1.8, max(1.8, length - 0.8)))
    )
    front_weight_fraction = 0.55 if kind == "car" else 0.52
    rear_axle_to_cg = wheel_base * front_weight_fraction
    front_axle_to_cg = wheel_base - rear_axle_to_cg
    track_width = max(0.9, width * 0.78)
    wheel_radius = float(np.clip(height * 0.2125, 0.30, 0.58))
    wheel_width = 0.24 if kind == "car" else 0.34
    yaw_inertia = mass_kg * (length * length + width * width) / 12.0
    return VehicleDynamicsComponent(
        wheel_base_m=wheel_base,
        track_width_m=track_width,
        front_axle_to_cg_m=front_axle_to_cg,
        rear_axle_to_cg_m=rear_axle_to_cg,
        center_of_mass_height_m=max(wheel_radius, height * 0.34),
        wheel_radius_m=wheel_radius,
        wheel_width_m=wheel_width,
        max_engine_force_n=mass_kg * max_accel_mps2,
        max_brake_force_n=mass_kg * max_brake_mps2,
        max_lateral_accel_mps2=max_lateral_accel_mps2,
        cornering_stiffness_n_per_rad=mass_kg * 9.81 * 5.9,
        yaw_inertia_kg_m2=yaw_inertia,
        tire_grip=tire_grip,
        rolling_resistance=rolling_resistance,
        aero_drag_coefficient=aero_drag_coefficient,
    )


def _physx_vehicle_model(
    dynamics: VehicleDynamicsComponent,
    dimensions_lwh: npt.ArrayLike,
    mass_kg: float,
    *,
    suspension_travel_m: float,
    longitudinal_collision_padding_m: float | None = None,
    lateral_collision_padding_m: float | None = None,
) -> VehicleModel:
    """Build a dimensioned four-wheel model at its static ride height."""
    length, width, height = (
        float(value) for value in np.asarray(dimensions_lwh, dtype=np.float32)
    )
    sprung_mass_kg = mass_kg * 0.90
    natural_frequency_hz = 1.5 if mass_kg < 4_000.0 else 1.2
    angular_frequency = 2.0 * math.pi * natural_frequency_hz
    spring_rate = sprung_mass_kg * angular_frequency * angular_frequency / 4.0
    corner_mass_kg = sprung_mass_kg / 4.0
    damper_rate = 2.0 * 0.70 * math.sqrt(spring_rate * corner_mass_kg)
    static_compression = mass_kg * 9.81 / (4.0 * spring_rate)
    rest_length = max(
        suspension_travel_m,
        static_compression + suspension_travel_m * 0.35,
    )

    half_track = dynamics.track_width_m * 0.5
    wheel_center_z = -height * 0.5 + dynamics.wheel_radius_m
    mount_z = wheel_center_z + rest_length - static_compression
    mounts = (
        (dynamics.front_axle_to_cg_m, half_track, mount_z),
        (dynamics.front_axle_to_cg_m, -half_track, mount_z),
        (-dynamics.rear_axle_to_cg_m, half_track, mount_z),
        (-dynamics.rear_axle_to_cg_m, -half_track, mount_z),
    )

    ground_clearance = float(np.clip(height * 0.10, 0.12, 0.35))
    chassis_bottom = -height * 0.5 + ground_clearance
    chassis_top = height * 0.45
    chassis_center_z = (chassis_bottom + chassis_top) * 0.5
    chassis_half_length = (
        length * 0.46
        if longitudinal_collision_padding_m is None
        else length * 0.5 + longitudinal_collision_padding_m
    )
    chassis_half_width = (
        width * 0.44
        if lateral_collision_padding_m is None
        else width * 0.5 + lateral_collision_padding_m
    )
    return VehicleModel(
        chassis_half_extents_m=(
            chassis_half_length,
            chassis_half_width,
            (chassis_top - chassis_bottom) * 0.5,
        ),
        chassis_offset_m=(0.0, 0.0, chassis_center_z),
        suspension_mounts_m=mounts,
        wheel_radius_m=dynamics.wheel_radius_m,
        suspension_rest_length_m=rest_length,
        suspension_max_compression_m=suspension_travel_m,
        spring_stiffness_n_per_m=spring_rate,
        damper_rate_n_s_per_m=damper_rate,
        tire_friction=dynamics.tire_grip,
        cornering_stiffness_n_per_rad=dynamics.cornering_stiffness_n_per_rad,
        rolling_resistance=dynamics.rolling_resistance,
        max_engine_force_n=dynamics.max_engine_force_n,
        max_brake_force_n=dynamics.max_brake_force_n,
    )


def vehicle_dynamics_for_object(
    object_type: str,
    dimensions_lwh: npt.ArrayLike,
    mass_kg: float,
) -> VehicleDynamicsComponent | None:
    """Build the classified four-wheel design for a recorded scene object."""
    kind = canonical_object_type(object_type)
    max_accel = 3.5 if kind == "car" else 2.0
    max_lateral = 5.5 if kind == "car" else 3.6
    return _vehicle_dynamics(
        kind=kind,
        dimensions_lwh=dimensions_lwh,
        mass_kg=mass_kg,
        max_accel_mps2=max_accel,
        max_brake_mps2=6.0,
        max_lateral_accel_mps2=max_lateral,
        tire_grip=0.92 if kind == "car" else 0.82,
        rolling_resistance=0.015,
        aero_drag_coefficient=0.42 if kind == "car" else 0.70,
    )


def vehicle_dynamics_from_config(config: VehicleConfig) -> VehicleDynamicsComponent:
    """Build the ego four-wheel design used by the interactive drive solver."""
    result = _vehicle_dynamics(
        kind="car",
        dimensions_lwh=(
            config.aabb_length_m,
            config.aabb_width_m,
            config.aabb_height_m,
        ),
        mass_kg=config.mass_kg,
        max_accel_mps2=config.max_accel_mps2,
        max_brake_mps2=config.max_brake_mps2,
        max_lateral_accel_mps2=config.max_lateral_accel_mps2,
        tire_grip=config.tire_grip,
        rolling_resistance=config.rolling_resistance,
        aero_drag_coefficient=config.aero_drag_coefficient,
        wheel_base_m=config.wheel_base_m,
    )
    assert result is not None
    return result


def rigid_body_model_from_vehicle_config(config: VehicleConfig) -> RigidBodyModel:
    """Build the ego chassis, wheels, and suspension from its runtime config."""
    dimensions = (
        config.aabb_length_m,
        config.aabb_width_m,
        config.aabb_height_m,
    )
    dynamics = vehicle_dynamics_from_config(config)
    return RigidBodyModel(
        mass_kg=config.mass_kg,
        half_extents_m=tuple(value * 0.5 for value in dimensions),
        restitution=config.collision_restitution,
        friction=config.collision_friction,
        vehicle=_physx_vehicle_model(
            dynamics,
            dimensions,
            config.mass_kg,
            suspension_travel_m=config.suspension_travel_m,
        ),
    )


def suspension_for_object(object_type: str) -> SuspensionComponent | None:
    """Return game-car suspension only for self-propelled four-wheel classes."""
    if canonical_object_type(object_type) not in {"car", "truck", "bus", "trailer"}:
        return None
    return SuspensionComponent(
        stiffness=42.0,
        damping=9.0,
        travel_m=0.22,
        visual_gain=0.65,
        max_roll_rad=0.16,
        max_pitch_rad=0.10,
    )


def game_entity_from_vehicle_state(
    state: VehicleState,
    config: VehicleConfig,
    *,
    entity_id: str = "ego",
) -> GameEntity:
    """Build an engine-neutral ego entity from authoritative simulation state."""
    half_yaw = state.yaw_rad * 0.5
    quaternion = np.asarray(
        [0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)], dtype=np.float32
    )
    velocity = np.asarray(
        [
            state.velocity_x_mps
            if state.velocity_x_mps is not None
            else math.cos(state.yaw_rad) * state.speed_mps,
            state.velocity_y_mps
            if state.velocity_y_mps is not None
            else math.sin(state.yaw_rad) * state.speed_mps,
            0.0,
        ],
        dtype=np.float32,
    )
    dynamics = vehicle_dynamics_from_config(config)
    return GameEntity(
        entity_id=entity_id,
        object_type="Car",
        transform=TransformComponent(
            np.asarray([state.x_m, state.y_m, state.z_m], dtype=np.float32),
            quaternion,
        ),
        rigid_body=RigidBodyComponent(
            mass_kg=config.mass_kg,
            linear_velocity_mps=velocity,
            angular_velocity_radps=np.asarray(
                [0.0, 0.0, state.yaw_rate_radps], dtype=np.float32
            ),
            restitution=config.collision_restitution,
            friction=config.collision_friction,
        ),
        collider=BoxColliderComponent(
            (
                config.aabb_length_m * 0.5,
                config.aabb_width_m * 0.5,
                config.aabb_height_m * 0.5,
            )
        ),
        vehicle=dynamics,
        suspension=SuspensionComponent(
            stiffness=config.suspension_stiffness,
            damping=config.suspension_damping,
            travel_m=config.suspension_travel_m,
            visual_gain=config.suspension_visual_gain,
            max_roll_rad=config.max_body_roll_rad,
            max_pitch_rad=config.max_body_pitch_rad,
        ),
        detached_from_track=state.ragdoll_active,
    )
