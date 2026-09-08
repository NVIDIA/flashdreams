# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import replace

import numpy as np
from loguru import logger

from omnidreams_game_engine.config import ChunkConfig, VehicleConfig
from omnidreams_game_engine.math3d import rig_pose_from_vehicle_state
from omnidreams_game_engine.simulation.components import (
    GameEntity,
    game_entity_from_vehicle_state,
    vehicle_dynamics_from_config,
)
from omnidreams_game_engine.simulation.game_physics import GamePhysicsWorld
from omnidreams_game_engine.simulation.ground_snap import GroundSnapper
from omnidreams_game_engine.types import (
    DriverCommand,
    PhysXChunkTimings,
    SceneDefinition,
    TrajectoryChunk,
    VehicleState,
)

PhysicsActorSamples = tuple[tuple[str, np.ndarray, np.ndarray, bool], ...]
PhysicsStepFn = Callable[
    [GamePhysicsWorld, VehicleState, DriverCommand, int, float],
    tuple[VehicleState, PhysicsActorSamples],
]


def step_physics_world(
    physics_world: GamePhysicsWorld,
    state: VehicleState,
    command: DriverCommand,
    timestamp_us: int,
    dt_s: float,
) -> tuple[VehicleState, PhysicsActorSamples]:
    del command
    return physics_world.step(state, timestamp_us, dt_s)


def _move_towards(current: float, target: float, max_delta: float) -> float:
    if current < target:
        return min(current + max_delta, target)
    return max(current - max_delta, target)


def integrate_vehicle(
    state: VehicleState,
    command: DriverCommand,
    dt_s: float,
    vehicle: VehicleConfig,
) -> VehicleState:
    steer_rad = state.steer_rad
    if command.steer_is_direct:
        max_steer = 0.4 if command.manual_control else vehicle.max_steer_rad
        steer_rad = command.steer * max_steer
    elif abs(command.steer) > 1e-5:
        steer_rad += command.steer * vehicle.steer_rate_rad_per_s * dt_s
    else:
        steer_rad = _move_towards(
            steer_rad, 0.0, vehicle.steer_return_rate_rad_per_s * dt_s
        )
    steer_rad = float(np.clip(steer_rad, -vehicle.max_steer_rad, vehicle.max_steer_rad))

    speed = state.speed_mps
    if command.stop:
        speed = 0.0
    elif command.manual_control:
        intended_direction = -1.0 if command.reverse else 1.0
        # Brake wins over throttle: holding both pedals bleeds speed toward a
        # stop, matching real cars and the demo.py target-speed integrators.
        if command.brake > 0.01:
            decel = 12.0 * command.brake * dt_s
            if speed > 0:
                speed = max(0.0, speed - decel)
            elif speed < 0:
                speed = min(0.0, speed + decel)
        elif command.throttle > 0.01:
            accel = 2.0 * command.throttle * dt_s
            if intended_direction < 0.0:
                speed -= accel
            elif vehicle.speed_limit_enabled:
                max_speed = vehicle.max_speed_mps
                current = abs(speed)
                high_speed_knee = max_speed * 0.62
                if current < high_speed_knee:
                    taper = max(0.2, 1.0 - (current / high_speed_knee) ** 2 * 0.5)
                else:
                    excess = (current - high_speed_knee) / max(
                        1e-6, max_speed - high_speed_knee
                    )
                    taper = max(0.05, 0.5 * (1.0 - excess) ** 3)
                speed += accel * taper
            else:
                speed += accel
        else:
            if speed > 0.0:
                speed = max(0.0, speed - 0.5 * dt_s)
            elif speed < 0.0:
                speed = min(0.0, speed + 0.5 * dt_s)
        if vehicle.speed_limit_enabled:
            speed = float(
                np.clip(speed, -vehicle.max_reverse_speed_mps, vehicle.max_speed_mps)
            )
    else:
        intended_direction = -1.0 if command.reverse else 1.0
        accel = command.throttle * vehicle.max_accel_mps2 * dt_s
        brake = command.brake * vehicle.max_brake_mps2 * dt_s
        if brake > 0.0:
            speed = _move_towards(speed, 0.0, brake)
        elif accel > 0.0:
            speed += intended_direction * accel
        else:
            if speed > 0.0:
                speed = max(0.0, speed - vehicle.drag_mps2 * dt_s)
            else:
                speed = min(0.0, speed + vehicle.drag_mps2 * dt_s)
        if vehicle.speed_limit_enabled:
            speed = float(
                np.clip(speed, -vehicle.max_reverse_speed_mps, vehicle.max_speed_mps)
            )

    commanded_yaw_rate = 0.0
    if abs(steer_rad) > 1e-5 and abs(speed) > 1e-5:
        commanded_yaw_rate = speed / vehicle.wheel_base_m * math.tan(steer_rad)
        # A fixed steering angle becomes unrealistically aggressive as speed
        # rises because bicycle-model lateral acceleration scales with v^2.
        # Limit yaw rate by the configured grip envelope while preserving the
        # full steering response at parking and neighbourhood speeds.
        max_yaw_rate = vehicle.max_lateral_accel_mps2 / abs(speed)
        commanded_yaw_rate = float(
            np.clip(commanded_yaw_rate, -max_yaw_rate, max_yaw_rate)
        )

    design = vehicle_dynamics_from_config(vehicle)
    forward = np.asarray(
        [math.cos(state.yaw_rad), math.sin(state.yaw_rad)], dtype=np.float32
    )
    left = np.asarray([-forward[1], forward[0]], dtype=np.float32)
    velocity = np.asarray(
        [
            state.velocity_x_mps
            if state.velocity_x_mps is not None
            else forward[0] * state.speed_mps,
            state.velocity_y_mps
            if state.velocity_y_mps is not None
            else forward[1] * state.speed_mps,
        ],
        dtype=np.float32,
    )
    if state.ragdoll_active:
        lateral_speed = float(np.dot(velocity, left))
        grip = float(np.clip(vehicle.tire_grip * dt_s * 4.0, 0.0, 1.0))
        velocity -= left * lateral_speed * grip
        longitudinal_speed = float(np.dot(velocity, forward))
        velocity += forward * (speed - longitudinal_speed)
        yaw_rate = state.yaw_rate_radps * max(
            0.0, 1.0 - 1.8 * dt_s
        ) + commanded_yaw_rate * min(1.0, 2.5 * dt_s)
    else:
        speed_abs = abs(speed)
        if speed_abs < 0.75 or speed < 0.0:
            response = 1.0 - math.exp(-8.0 * dt_s)
            yaw_rate = (
                state.yaw_rate_radps
                + (commanded_yaw_rate - state.yaw_rate_radps) * response
            )
            # The state pose is at the vehicle CG, not at the rear axle. In a
            # no-slip bicycle turn the CG therefore has lateral velocity
            # ``rear_axle_to_cg * yaw_rate``. Keeping it here also makes the
            # transition into the dynamic tire model continuous.
            lateral_speed = design.rear_axle_to_cg_m * yaw_rate
        else:
            lateral_speed = float(np.dot(velocity, left))
            front_slip = steer_rad - math.atan2(
                lateral_speed + design.front_axle_to_cg_m * state.yaw_rate_radps,
                speed_abs,
            )
            rear_slip = -math.atan2(
                lateral_speed - design.rear_axle_to_cg_m * state.yaw_rate_radps,
                speed_abs,
            )
            front_load_fraction = design.rear_axle_to_cg_m / design.wheel_base_m
            rear_load_fraction = 1.0 - front_load_fraction
            front_force = float(
                np.clip(
                    design.cornering_stiffness_n_per_rad
                    * front_load_fraction
                    * front_slip,
                    -vehicle.mass_kg
                    * front_load_fraction
                    * design.max_lateral_accel_mps2,
                    vehicle.mass_kg
                    * front_load_fraction
                    * design.max_lateral_accel_mps2,
                )
            )
            rear_force = float(
                np.clip(
                    design.cornering_stiffness_n_per_rad
                    * rear_load_fraction
                    * rear_slip,
                    -vehicle.mass_kg
                    * rear_load_fraction
                    * design.max_lateral_accel_mps2,
                    vehicle.mass_kg
                    * rear_load_fraction
                    * design.max_lateral_accel_mps2,
                )
            )
            steered_front_force = front_force * math.cos(steer_rad)
            lateral_accel = (
                steered_front_force + rear_force
            ) / vehicle.mass_kg - state.yaw_rate_radps * speed
            yaw_accel = (
                design.front_axle_to_cg_m * steered_front_force
                - design.rear_axle_to_cg_m * rear_force
            ) / design.yaw_inertia_kg_m2
            lateral_speed += lateral_accel * dt_s
            yaw_rate = state.yaw_rate_radps + yaw_accel * dt_s
            max_yaw_rate = design.max_lateral_accel_mps2 / speed_abs
            yaw_rate = float(np.clip(yaw_rate, -max_yaw_rate, max_yaw_rate))

    yaw = state.yaw_rad + yaw_rate * dt_s
    if not state.ragdoll_active:
        new_forward = np.asarray([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
        new_left = np.asarray([-new_forward[1], new_forward[0]], dtype=np.float32)
        velocity = new_forward * np.float32(speed) + new_left * np.float32(
            lateral_speed
        )
    x_m = state.x_m + float(velocity[0]) * dt_s
    y_m = state.y_m + float(velocity[1]) * dt_s

    longitudinal_accel = (speed - state.speed_mps) / max(dt_s, 1e-6)
    lateral_accel = speed * yaw_rate
    target_pitch = float(
        np.clip(
            -longitudinal_accel
            / 9.81
            * vehicle.suspension_visual_gain
            * vehicle.max_body_pitch_rad,
            -vehicle.max_body_pitch_rad,
            vehicle.max_body_pitch_rad,
        )
    )
    target_roll = float(
        np.clip(
            -lateral_accel
            / 9.81
            * vehicle.suspension_visual_gain
            * vehicle.max_body_roll_rad,
            -vehicle.max_body_roll_rad,
            vehicle.max_body_roll_rad,
        )
    )
    pitch_accel = (
        vehicle.suspension_stiffness * (target_pitch - state.suspension_pitch_rad)
        - vehicle.suspension_damping * state.suspension_pitch_rate_radps
    )
    roll_accel = (
        vehicle.suspension_stiffness * (target_roll - state.suspension_roll_rad)
        - vehicle.suspension_damping * state.suspension_roll_rate_radps
    )
    pitch_rate = state.suspension_pitch_rate_radps + pitch_accel * dt_s
    roll_rate = state.suspension_roll_rate_radps + roll_accel * dt_s
    suspension_pitch = float(
        np.clip(
            state.suspension_pitch_rad + pitch_rate * dt_s,
            -vehicle.max_body_pitch_rad,
            vehicle.max_body_pitch_rad,
        )
    )
    suspension_roll = float(
        np.clip(
            state.suspension_roll_rad + roll_rate * dt_s,
            -vehicle.max_body_roll_rad,
            vehicle.max_body_roll_rad,
        )
    )

    return VehicleState(
        x_m=x_m,
        y_m=y_m,
        z_m=state.z_m,
        yaw_rad=yaw,
        speed_mps=speed,
        steer_rad=steer_rad,
        pitch_rad=state.pitch_rad,
        roll_rad=state.roll_rad,
        velocity_x_mps=float(velocity[0]),
        velocity_y_mps=float(velocity[1]),
        yaw_rate_radps=yaw_rate,
        suspension_pitch_rad=suspension_pitch,
        suspension_roll_rad=suspension_roll,
        suspension_pitch_rate_radps=pitch_rate,
        suspension_roll_rate_radps=roll_rate,
        ragdoll_active=state.ragdoll_active,
    )


def sample_chunk_trajectory(
    start_state: VehicleState,
    start_timestamp_us: int,
    commands: Sequence[DriverCommand],
    chunk_size: int,
    chunk_config: ChunkConfig,
    vehicle_config: VehicleConfig,
    ground_snapper: GroundSnapper | None,
    physics_world: GamePhysicsWorld | None = None,
    capture_physics_debug: bool = False,
    integrate_fn: Callable[
        [VehicleState, DriverCommand, float, VehicleConfig], VehicleState
    ] = integrate_vehicle,
    physics_step_fn: PhysicsStepFn = step_physics_world,
    include_start_state: bool = False,
) -> TrajectoryChunk:
    if len(commands) != chunk_size:
        raise ValueError(
            f"commands must match chunk_size; got {len(commands)} for {chunk_size}"
        )
    timestamps = np.array(
        [
            start_timestamp_us + frame_idx * chunk_config.frame_interval_us
            for frame_idx in range(chunk_size)
        ],
        dtype=np.int64,
    )
    poses = np.zeros((chunk_size, 4, 4), dtype=np.float32)

    state = replace(start_state)
    vehicle_states: list[VehicleState] = []
    actor_samples: list[tuple[tuple[str, np.ndarray, np.ndarray, bool], ...]] = []
    physics_debug_frames = []
    physx_elapsed_s = 0.0
    physx_sync_s = 0.0
    actor_update_ms = 0.0
    solver_ms = 0.0
    readback_ms = 0.0
    bridge_ms = 0.0
    traffic_prepare_ms = 0.0
    barrier_rebound_ms = 0.0
    traffic_update_ms = 0.0
    state_materialize_ms = 0.0
    bridge_other_ms = 0.0
    max_visible_actors = 0
    max_detached_actors = 0
    actor_collision_detected = False
    actor_collision_frame_index: int | None = None
    static_collision_detected = False
    static_collision_frame_index: int | None = None
    if physics_world is not None:
        physx_started_at = time.perf_counter()
        physics_world.synchronize_window(
            np.asarray([state.x_m, state.y_m], dtype=np.float32),
            timestamp_us=start_timestamp_us,
        )
        sync_elapsed_s = time.perf_counter() - physx_started_at
        physx_sync_s += sync_elapsed_s
        physx_elapsed_s += sync_elapsed_s
    for frame_idx in range(chunk_size):
        command = commands[frame_idx]
        use_start_state = include_start_state and frame_idx == 0
        if not use_start_state:
            state = integrate_fn(
                state, command, chunk_config.frame_interval_s, vehicle_config
            )
        if physics_world is not None and use_start_state:
            frame_actor_samples = tuple(
                (
                    entity.entity_id,
                    entity.transform.position_m.copy(),
                    entity.transform.orientation_xyzw.copy(),
                    entity.detached_from_track,
                )
                for entity in physics_world.entities
            )
            actor_samples.append(frame_actor_samples)
            if capture_physics_debug:
                physics_debug_frames.append(physics_world.debug_frame(state))
        elif physics_world is not None:
            physx_started_at = time.perf_counter()
            state, frame_actor_samples = physics_step_fn(
                physics_world,
                state,
                command,
                int(timestamps[frame_idx]),
                chunk_config.frame_interval_s,
            )
            actor_collision_this_frame = bool(
                getattr(physics_world, "last_step_actor_collision", False)
            )
            actor_collision_detected |= actor_collision_this_frame
            if actor_collision_this_frame and actor_collision_frame_index is None:
                actor_collision_frame_index = frame_idx
            static_collision_this_frame = bool(
                getattr(physics_world, "last_step_static_barrier_impact", False)
            )
            static_collision_detected |= static_collision_this_frame
            if static_collision_this_frame and static_collision_frame_index is None:
                static_collision_frame_index = frame_idx
            physx_elapsed_s += time.perf_counter() - physx_started_at
            step_timings = getattr(physics_world, "last_step_timings", None)
            if step_timings is not None:
                actor_update_ms += step_timings.actor_update_ms
                solver_ms += step_timings.solver_ms
                readback_ms += step_timings.readback_ms
                bridge_ms += step_timings.bridge_ms
                max_visible_actors = max(
                    max_visible_actors, step_timings.visible_actor_count
                )
                max_detached_actors = max(
                    max_detached_actors, step_timings.detached_actor_count
                )
            bridge_timings = getattr(physics_world, "last_step_bridge_timings", None)
            if bridge_timings is not None:
                traffic_prepare_ms += bridge_timings.traffic_prepare_ms
                barrier_rebound_ms += bridge_timings.barrier_rebound_ms
                traffic_update_ms += bridge_timings.traffic_update_ms
                state_materialize_ms += bridge_timings.state_materialize_ms
                bridge_other_ms += bridge_timings.other_ms
            actor_samples.append(frame_actor_samples)
            if capture_physics_debug:
                physics_debug_frames.append(physics_world.debug_frame(state))
        if ground_snapper is not None:
            state = ground_snapper.snap(state, vehicle_config)
        vehicle_states.append(state)
        poses[frame_idx] = rig_pose_from_vehicle_state(state)

    dynamic_actors = (
        physics_world.build_trajectories(timestamps, actor_samples)
        if physics_world is not None
        else ()
    )
    return TrajectoryChunk(
        timestamps_us=timestamps,
        rig_poses_world=poses,
        vehicle_states=tuple(vehicle_states),
        boundary_state_after_chunk=state,
        applied_commands=tuple(commands),
        dynamic_actors=dynamic_actors,
        physics_debug_frames=tuple(physics_debug_frames),
        actor_collision_detected=actor_collision_detected,
        actor_collision_frame_index=actor_collision_frame_index,
        static_collision_detected=static_collision_detected,
        static_collision_frame_index=static_collision_frame_index,
        physx_elapsed_s=physx_elapsed_s if physics_world is not None else None,
        physx_timings=(
            PhysXChunkTimings(
                total_ms=physx_elapsed_s * 1000.0,
                synchronize_ms=physx_sync_s * 1000.0,
                actor_update_ms=actor_update_ms,
                solver_ms=solver_ms,
                readback_ms=readback_ms,
                bridge_ms=bridge_ms,
                traffic_prepare_ms=traffic_prepare_ms,
                barrier_rebound_ms=barrier_rebound_ms,
                traffic_update_ms=traffic_update_ms,
                state_materialize_ms=state_materialize_ms,
                bridge_other_ms=bridge_other_ms,
                step_count=chunk_size,
                max_visible_actors=max_visible_actors,
                max_detached_actors=max_detached_actors,
            )
            if physics_world is not None
            else None
        ),
    )


def state_from_initial_pose(
    initial_rig_to_world: np.ndarray,
    initial_yaw_rad: float,
    initial_speed_mps: float,
) -> VehicleState:
    return VehicleState(
        x_m=float(initial_rig_to_world[0, 3]),
        y_m=float(initial_rig_to_world[1, 3]),
        z_m=float(initial_rig_to_world[2, 3]),
        yaw_rad=initial_yaw_rad,
        speed_mps=initial_speed_mps,
        steer_rad=0.0,
        velocity_x_mps=math.cos(initial_yaw_rad) * initial_speed_mps,
        velocity_y_mps=math.sin(initial_yaw_rad) * initial_speed_mps,
    )


def build_ground_snapper(scene: SceneDefinition) -> GroundSnapper | None:
    if scene.ground_mesh_vertices is None or scene.ground_mesh_faces is None:
        logger.info(
            "[ego_vehicle_kinematics] no ground mesh in scene; z/pitch/roll will not be snapped.",
        )
        return None
    return GroundSnapper(scene.ground_mesh_vertices, scene.ground_mesh_faces)


class EgoVehicleKinematics:
    def __init__(
        self,
        initial_state: VehicleState,
        vehicle_config: VehicleConfig,
        ground_snapper: GroundSnapper | None,
        initial_timestamp_us: int,
        scene: SceneDefinition | None = None,
        integrate_fn: Callable[
            [VehicleState, DriverCommand, float, VehicleConfig], VehicleState
        ] = integrate_vehicle,
        physics_world_factory: Callable[
            [SceneDefinition, VehicleConfig], GamePhysicsWorld
        ] = GamePhysicsWorld,
        physics_step_fn: PhysicsStepFn = step_physics_world,
        include_initial_state_in_first_chunk: bool = False,
    ) -> None:
        self._state = initial_state
        self._vehicle_config = vehicle_config
        self._ground_snapper = ground_snapper
        self._next_timestamp_us = initial_timestamp_us
        self._integrate_fn = integrate_fn
        self._physics_step_fn = physics_step_fn
        self._include_initial_state_in_next_chunk = bool(
            include_initial_state_in_first_chunk
        )
        self._physics_world = (
            physics_world_factory(scene, vehicle_config) if scene is not None else None
        )
        self._capture_physics_debug = False

    def set_physx_debug_enabled(self, enabled: bool) -> None:
        """Capture debug collider snapshots only for the active PhysX view."""
        self._capture_physics_debug = bool(enabled)

    @property
    def current_state(self) -> VehicleState:
        return self._state

    @property
    def game_entities(self) -> tuple[GameEntity, ...]:
        """Return the ego and actor objects as engine-neutral components."""
        actors = self._physics_world.entities if self._physics_world is not None else ()
        return (
            game_entity_from_vehicle_state(self._state, self._vehicle_config),
            *actors,
        )

    def close(self) -> None:
        """Release the Ludus PhysX scene owned by this rollout."""
        if self._physics_world is not None:
            self._physics_world.close()
            self._physics_world = None

    def pose_chunk(
        self,
        commands: Sequence[DriverCommand],
        chunk_size: int,
        frame_interval_s: float,
        extrapolation_offset_s: float,
    ) -> TrajectoryChunk:
        if extrapolation_offset_s != 0.0:
            raise NotImplementedError(
                "Nonzero extrapolation_offset_s is not implemented in Stage 1."
            )
        chunk_config = ChunkConfig(
            fps=round(1.0 / frame_interval_s),
            initial_chunk_frames=chunk_size,
            chunk_frames=chunk_size,
        )
        trajectory = sample_chunk_trajectory(
            start_state=self._state,
            start_timestamp_us=self._next_timestamp_us,
            commands=commands,
            chunk_size=chunk_size,
            chunk_config=chunk_config,
            vehicle_config=self._vehicle_config,
            ground_snapper=self._ground_snapper,
            physics_world=self._physics_world,
            capture_physics_debug=self._capture_physics_debug,
            integrate_fn=self._integrate_fn,
            physics_step_fn=self._physics_step_fn,
            include_start_state=self._include_initial_state_in_next_chunk,
        )
        self._include_initial_state_in_next_chunk = False
        self._state = trajectory.boundary_state_after_chunk
        self._next_timestamp_us = int(
            trajectory.timestamps_us[-1] + chunk_config.frame_interval_us
        )
        return trajectory
