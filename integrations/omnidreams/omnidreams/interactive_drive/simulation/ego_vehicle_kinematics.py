# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math
import time

import numpy as np
from loguru import logger
from omnidreams.interactive_drive.config import ChunkConfig, VehicleConfig
from omnidreams.interactive_drive.math3d import rig_pose_from_state
from omnidreams.interactive_drive.simulation.components import (
    GameEntity,
    game_entity_from_vehicle_state,
    vehicle_dynamics_from_config,
)
from omnidreams.interactive_drive.simulation.game_physics import GamePhysicsWorld
from omnidreams.interactive_drive.simulation.ground_snap import GroundSnapper
from omnidreams.interactive_drive.simulation.map_bounds import MapBounds
from omnidreams.interactive_drive.types import (
    DriverCommand,
    PhysXChunkTimings,
    SceneBundle,
    TrajectoryChunk,
    VehicleState,
)


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
    command: DriverCommand,
    chunk_size: int,
    chunk_config: ChunkConfig,
    vehicle_config: VehicleConfig,
    ground_snapper: GroundSnapper | None,
    physics_world: GamePhysicsWorld | None = None,
    capture_physics_debug: bool = False,
) -> TrajectoryChunk:
    timestamps = np.array(
        [
            start_timestamp_us + frame_idx * chunk_config.frame_interval_us
            for frame_idx in range(chunk_size)
        ],
        dtype=np.int64,
    )
    poses = np.zeros((chunk_size, 4, 4), dtype=np.float32)

    state = VehicleState(**start_state.__dict__)
    actor_samples: list[tuple[tuple[str, np.ndarray, np.ndarray, bool], ...]] = []
    physics_debug_frames = []
    physx_elapsed_s = 0.0
    physx_sync_s = 0.0
    actor_update_ms = 0.0
    solver_ms = 0.0
    readback_ms = 0.0
    bridge_ms = 0.0
    max_visible_actors = 0
    max_detached_actors = 0
    actor_collision_detected = False
    actor_collision_frame_index: int | None = None
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
        state = integrate_vehicle(
            state, command, chunk_config.frame_interval_s, vehicle_config
        )
        if physics_world is not None:
            physx_started_at = time.perf_counter()
            state, frame_actor_samples = physics_world.step(
                state,
                int(timestamps[frame_idx]),
                chunk_config.frame_interval_s,
            )
            actor_collision_this_frame = bool(
                getattr(physics_world, "last_step_actor_collision", False)
            )
            actor_collision_detected |= actor_collision_this_frame
            if actor_collision_this_frame and actor_collision_frame_index is None:
                actor_collision_frame_index = frame_idx
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
            actor_samples.append(frame_actor_samples)
            if capture_physics_debug:
                physics_debug_frames.append(physics_world.debug_frame(state))
        if ground_snapper is not None:
            state = ground_snapper.snap(state, vehicle_config)
        poses[frame_idx] = rig_pose_from_state(
            x_m=state.x_m,
            y_m=state.y_m,
            z_m=state.z_m,
            yaw_rad=state.yaw_rad,
            pitch_rad=state.pitch_rad + state.suspension_pitch_rad,
            roll_rad=state.roll_rad + state.suspension_roll_rad,
        )

    dynamic_actors = (
        physics_world.build_trajectories(timestamps, actor_samples)
        if physics_world is not None
        else ()
    )
    return TrajectoryChunk(
        timestamps_us=timestamps,
        rig_poses_world=poses,
        boundary_state_after_chunk=state,
        dynamic_actors=dynamic_actors,
        physics_debug_frames=tuple(physics_debug_frames),
        actor_collision_detected=actor_collision_detected,
        actor_collision_frame_index=actor_collision_frame_index,
        physx_elapsed_s=physx_elapsed_s if physics_world is not None else None,
        physx_timings=(
            PhysXChunkTimings(
                total_ms=physx_elapsed_s * 1000.0,
                synchronize_ms=physx_sync_s * 1000.0,
                actor_update_ms=actor_update_ms,
                solver_ms=solver_ms,
                readback_ms=readback_ms,
                bridge_ms=bridge_ms,
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


def build_ground_snapper(scene: SceneBundle) -> GroundSnapper | None:
    if scene.ground_mesh_vertices is None or scene.ground_mesh_faces is None:
        logger.info(
            "[ego_vehicle_kinematics] no ground mesh in scene; z/pitch/roll will not be snapped.",
        )
        return None
    return GroundSnapper(scene.ground_mesh_vertices, scene.ground_mesh_faces)


def build_map_bounds(scene: SceneBundle) -> MapBounds | None:
    """Compute OOB bounds from every spatial layer in ``scene``.

    Decoupled from :func:`build_ground_snapper` because the OOB check
    cares about the union of all geometry (lane markers, vehicle
    tracks, polygons, ground), not just the ground mesh -- many scenes
    ship a ground mesh that's a small strip representing only the road
    surface, which would respawn the user the moment they drove onto a
    sidewalk. Logs the resulting AABB so it's easy to confirm the
    bounds match the scene's playable area.
    """
    bounds = MapBounds.from_scene(scene)
    if bounds is None:
        logger.info(
            "[ego_vehicle_kinematics] scene has no spatial geometry; "
            "OOB respawn will not fire.",
        )
        return None
    logger.info(
        f"[ego_vehicle_kinematics] map bounds: "
        f"x=[{bounds.x_min:.1f}, {bounds.x_max:.1f}] ({bounds.width_m:.1f} m), "
        f"y=[{bounds.y_min:.1f}, {bounds.y_max:.1f}] ({bounds.height_m:.1f} m). "
        "Adds 50 m margin + 100 m warning zone for OOB.",
    )
    return bounds


class EgoVehicleKinematics:
    def __init__(
        self,
        initial_state: VehicleState,
        vehicle_config: VehicleConfig,
        ground_snapper: GroundSnapper | None,
        initial_timestamp_us: int,
        map_bounds: MapBounds | None = None,
        oob_margin_m: float = 50.0,
        oob_warning_zone_m: float = 100.0,
        scene: SceneBundle | None = None,
    ) -> None:
        self._state = initial_state
        self._vehicle_config = vehicle_config
        self._ground_snapper = ground_snapper
        self._next_timestamp_us = initial_timestamp_us
        self._map_bounds = map_bounds
        self._oob_margin_m = float(oob_margin_m)
        self._oob_warning_zone_m = float(oob_warning_zone_m)
        self._physics_world = (
            GamePhysicsWorld(scene, vehicle_config) if scene is not None else None
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

    @property
    def last_proximity(self) -> float:
        """Out-of-bounds proximity of the latest simulated frame.

        Delegates to :meth:`MapBounds.proximity` (see it for the
        0.0 / (0,1] / 2.0 semantics) against the union AABB of every spatial
        layer, not just ``mesh_ground.ply`` -- a road-only ground mesh would
        respawn the ego the moment it touched a sidewalk. Returns ``0.0`` when
        the scene has no geometry (the OOB respawn path no-ops).
        """
        if self._map_bounds is None:
            return 0.0
        return self._map_bounds.proximity(
            (self._state.x_m, self._state.y_m),
            margin_m=self._oob_margin_m,
            warning_zone_m=self._oob_warning_zone_m,
        )

    def close(self) -> None:
        """Release the Ludus PhysX scene owned by this rollout."""
        if self._physics_world is not None:
            self._physics_world.close()
            self._physics_world = None

    def pose_chunk(
        self,
        command: DriverCommand,
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
            command=command,
            chunk_size=chunk_size,
            chunk_config=chunk_config,
            vehicle_config=self._vehicle_config,
            ground_snapper=self._ground_snapper,
            physics_world=self._physics_world,
            capture_physics_debug=self._capture_physics_debug,
        )
        self._state = trajectory.boundary_state_after_chunk
        self._next_timestamp_us = int(
            trajectory.timestamps_us[-1] + chunk_config.frame_interval_us
        )
        return trajectory
