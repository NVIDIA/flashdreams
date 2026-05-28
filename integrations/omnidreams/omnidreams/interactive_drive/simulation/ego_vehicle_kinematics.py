# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math

import numpy as np
from omnidreams.interactive_drive.config import ChunkConfig, VehicleConfig
from omnidreams.interactive_drive.math3d import rig_pose_from_state
from omnidreams.interactive_drive.simulation.ground_snap import GroundSnapper
from omnidreams.interactive_drive.types import (
    DriverCommand,
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
        if command.throttle > 0.01:
            max_speed = vehicle.max_speed_mps
            accel = 2.0 * command.throttle * dt_s
            if speed * intended_direction < 0:
                engine_brake_multiplier = 1.5
                if speed > 0:
                    speed = max(0.0, speed - accel * engine_brake_multiplier)
                else:
                    speed = min(0.0, speed + accel * engine_brake_multiplier)
            else:
                current = abs(speed)
                high_speed_knee = max_speed * 0.62
                if current < high_speed_knee:
                    taper = max(0.2, 1.0 - (current / high_speed_knee) ** 2 * 0.5)
                else:
                    excess = (current - high_speed_knee) / max(
                        1e-6, max_speed - high_speed_knee
                    )
                    taper = max(0.05, 0.5 * (1.0 - excess) ** 3)
                speed += intended_direction * accel * taper
        elif command.brake > 0.01:
            decel = 12.0 * command.brake * dt_s
            if speed > 0:
                speed = max(0.0, speed - decel)
            elif speed < 0:
                speed = min(0.0, speed + decel)
        else:
            creep_target = (
                4.47  # 10 mph, matching the HUD and AlpaSim manual-driver crawl.
            )
            if speed < creep_target + 0.1:
                speed += (creep_target - speed) * 0.18 * dt_s
            elif speed > 0.0:
                speed = max(0.0, speed - 0.5 * dt_s)
            else:
                speed = min(0.0, speed + 0.5 * dt_s)
        # Honour the configured forward / reverse caps instead of the
        # previous hardcoded ``-8.0, 36.0`` values, which silently
        # ignored ``VehicleConfig.max_reverse_speed_mps``.
        speed = float(
            np.clip(speed, -vehicle.max_reverse_speed_mps, vehicle.max_speed_mps)
        )
    else:
        accel = command.throttle * vehicle.max_accel_mps2
        brake = command.brake * vehicle.max_brake_mps2
        speed = speed + (accel - brake) * dt_s
        if abs(command.throttle) < 1e-3 and abs(command.brake) < 1e-3:
            if speed > 0.0:
                speed = max(0.0, speed - vehicle.drag_mps2 * dt_s)
            else:
                speed = min(0.0, speed + vehicle.drag_mps2 * dt_s)
        speed = float(np.clip(speed, 0.0, vehicle.max_speed_mps))

    yaw_rate = 0.0
    if abs(steer_rad) > 1e-5 and abs(speed) > 1e-5:
        yaw_rate = speed / vehicle.wheel_base_m * math.tan(steer_rad)

    yaw = state.yaw_rad + yaw_rate * dt_s
    x_m = state.x_m + math.cos(yaw) * speed * dt_s
    y_m = state.y_m + math.sin(yaw) * speed * dt_s

    return VehicleState(
        x_m=x_m,
        y_m=y_m,
        z_m=state.z_m,
        yaw_rad=yaw,
        speed_mps=speed,
        steer_rad=steer_rad,
        pitch_rad=state.pitch_rad,
        roll_rad=state.roll_rad,
    )


def sample_chunk_trajectory(
    start_state: VehicleState,
    start_timestamp_us: int,
    command: DriverCommand,
    chunk_size: int,
    chunk_config: ChunkConfig,
    vehicle_config: VehicleConfig,
    ground_snapper: GroundSnapper | None,
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
    for frame_idx in range(chunk_size):
        state = integrate_vehicle(
            state, command, chunk_config.frame_interval_s, vehicle_config
        )
        if ground_snapper is not None:
            state = ground_snapper.snap(state, vehicle_config)
        poses[frame_idx] = rig_pose_from_state(
            x_m=state.x_m,
            y_m=state.y_m,
            z_m=state.z_m,
            yaw_rad=state.yaw_rad,
            pitch_rad=state.pitch_rad,
            roll_rad=state.roll_rad,
        )

    return TrajectoryChunk(
        timestamps_us=timestamps,
        rig_poses_world=poses,
        boundary_state_after_chunk=state,
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
    )


def build_ground_snapper(scene: SceneBundle) -> GroundSnapper | None:
    if scene.ground_mesh_vertices is None or scene.ground_mesh_faces is None:
        print(
            "[ego_vehicle_kinematics] no ground mesh in scene; z/pitch/roll will not be snapped.",
            flush=True,
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
    ) -> None:
        self._state = initial_state
        self._vehicle_config = vehicle_config
        self._ground_snapper = ground_snapper
        self._next_timestamp_us = initial_timestamp_us

    @property
    def current_state(self) -> VehicleState:
        return self._state

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
            fps=int(round(1.0 / frame_interval_s)),
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
        )
        self._state = trajectory.boundary_state_after_chunk
        self._next_timestamp_us = int(
            trajectory.timestamps_us[-1] + chunk_config.frame_interval_us
        )
        return trajectory
