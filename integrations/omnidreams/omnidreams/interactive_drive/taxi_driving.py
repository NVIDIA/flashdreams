# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Taxi-game-only arcade vehicle integration."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from omnidreams.interactive_drive.config import VehicleConfig
from omnidreams.interactive_drive.simulation.components import (
    vehicle_dynamics_from_config,
)
from omnidreams.interactive_drive.types import DriverCommand, VehicleState


@dataclass(frozen=True)
class TaxiVehicleConfig(VehicleConfig):
    """Arcade vehicle values used only when the Taxi game is active."""

    max_accel_mps2: float = 10.0
    reverse_accel_mps2: float = 10.0
    max_brake_mps2: float = 14.0
    handbrake_decel_mps2: float = 18.0
    handbrake_yaw_gain: float = 3.25
    max_handbrake_yaw_rate_radps: float = 1.5
    max_lateral_accel_mps2: float = 8.5
    max_body_roll_rad: float = 0.16
    speed_limit_enabled: bool = True
    actor_collision_enabled: bool = True
    static_collision_enabled: bool = True


@dataclass
class TaxiKeyboardState:
    """Display and control state for Taxi keyboard driving."""

    steering: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    target_speed_mps: float = 0.0
    connected: bool = False
    reverse: bool = False


class TaxiKeyboardDriveState:
    """Taxi-only snappy steering, handbrake, and brake-to-reverse controls."""

    def __init__(self, control: Any) -> None:
        self._control = control
        self._pressed: set[str] = set()
        self._state = TaxiKeyboardState()
        self._last_update_s = time.monotonic()

    @property
    def state(self) -> TaxiKeyboardState:
        """Return a detached copy of the current display state."""
        return TaxiKeyboardState(**self._state.__dict__)

    @property
    def has_active_input(self) -> bool:
        """Return whether a keyboard drive key is held."""
        return bool(self._pressed)

    def set_key(self, keysym: str, down: bool) -> bool:
        """Apply a normalized drive key event."""
        key = keysym.lower()
        aliases = {
            "arrowup": "up",
            "arrowdown": "down",
            "arrowleft": "left",
            "arrowright": "right",
            " ": "space",
        }
        key = aliases.get(key, key)
        if key not in {"w", "a", "s", "d", "up", "down", "left", "right", "space"}:
            return False
        if down:
            self._pressed.add(key)
        else:
            self._pressed.discard(key)
        return True

    def update(self) -> TaxiKeyboardState:
        """Advance the input smoother and publish one Taxi drive command."""
        now = time.monotonic()
        dt_s = max(0.0, min(0.1, now - self._last_update_s))
        self._last_update_s = now
        target_steer = 0.0
        if {"a", "left"} & self._pressed:
            target_steer += 1.0
        if {"d", "right"} & self._pressed:
            target_steer -= 1.0
        steer_rate = 3.5 if abs(target_steer) > 0.0 else 5.0
        steer = _move_towards(self._state.steering, target_steer, steer_rate * dt_s)
        throttle = 1.0 if {"w", "up"} & self._pressed else 0.0
        brake = 1.0 if {"s", "down"} & self._pressed else 0.0
        handbrake = "space" in self._pressed
        target_speed = self._update_target_speed(
            throttle=throttle,
            brake=brake,
            handbrake=handbrake,
            dt_s=dt_s,
        )
        self._state = TaxiKeyboardState(
            steering=steer,
            throttle=throttle,
            brake=brake,
            target_speed_mps=target_speed,
            reverse=target_speed < -0.05,
        )
        self._control.set_drive(
            steer=steer,
            throttle=throttle,
            brake=brake,
            handbrake=handbrake,
        )
        return self.state

    def clear(self) -> None:
        """Release all Taxi drive keys and clear the published command."""
        self._pressed.clear()
        self._state = TaxiKeyboardState()
        self._control.set_drive(
            steer=0.0,
            throttle=0.0,
            brake=0.0,
            handbrake=False,
        )

    def release_control(self) -> None:
        """Release this input source without changing display state."""
        self._control.release_all()

    def _update_target_speed(
        self, *, throttle: float, brake: float, handbrake: bool, dt_s: float
    ) -> float:
        vehicle = TaxiVehicleConfig()
        speed = self._state.target_speed_mps
        if handbrake:
            speed = _move_towards(speed, 0.0, vehicle.handbrake_decel_mps2 * dt_s)
        elif throttle > 0.01:
            accel = vehicle.max_accel_mps2 * throttle * dt_s
            if speed < 0.0:
                speed = min(0.0, speed + accel * 1.5)
            else:
                speed += accel
        elif brake > 0.01:
            if speed > 0.0:
                speed = max(0.0, speed - vehicle.max_brake_mps2 * brake * dt_s)
            else:
                speed = max(
                    -vehicle.max_reverse_speed_mps,
                    speed - vehicle.reverse_accel_mps2 * brake * dt_s,
                )
        else:
            speed = _move_towards(speed, 0.0, vehicle.drag_mps2 * dt_s)
        return max(-vehicle.max_reverse_speed_mps, min(vehicle.max_speed_mps, speed))


def _move_towards(current: float, target: float, max_delta: float) -> float:
    if current < target:
        return min(current + max_delta, target)
    return max(current - max_delta, target)


def _apply_brake_or_reverse(
    speed_mps: float,
    command: DriverCommand,
    *,
    dt_s: float,
    brake_decel_mps2: float,
    reverse_accel_mps2: float,
    max_reverse_speed_mps: float,
) -> float:
    brake_delta = brake_decel_mps2 * command.brake * dt_s
    if command.throttle > 0.01 or command.reverse:
        return _move_towards(speed_mps, 0.0, brake_delta)
    reverse_dt_s = dt_s
    if speed_mps > 0.0:
        if brake_delta <= speed_mps:
            return max(0.0, speed_mps - brake_delta)
        reverse_dt_s -= speed_mps / (brake_decel_mps2 * command.brake)
    reverse_delta = reverse_accel_mps2 * command.brake * reverse_dt_s
    return max(-max_reverse_speed_mps, min(0.0, speed_mps) - reverse_delta)


def integrate_taxi_vehicle(
    state: VehicleState,
    command: DriverCommand,
    dt_s: float,
    vehicle: TaxiVehicleConfig,
) -> VehicleState:
    steer_rad = state.steer_rad
    if command.steer_is_direct:
        steer_rad = command.steer * vehicle.max_steer_rad
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
    elif command.handbrake:
        speed = _move_towards(speed, 0.0, vehicle.handbrake_decel_mps2 * dt_s)
    elif command.manual_control:
        intended_direction = -1.0 if command.reverse else 1.0
        if command.brake > 0.01:
            speed = _apply_brake_or_reverse(
                speed,
                command,
                dt_s=dt_s,
                brake_decel_mps2=vehicle.max_brake_mps2,
                reverse_accel_mps2=vehicle.reverse_accel_mps2,
                max_reverse_speed_mps=vehicle.max_reverse_speed_mps,
            )
        elif command.throttle > 0.01:
            accel = vehicle.max_accel_mps2 * command.throttle * dt_s
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
            speed = _move_towards(speed, 0.0, 0.5 * dt_s)
        if vehicle.speed_limit_enabled:
            speed = float(
                np.clip(speed, -vehicle.max_reverse_speed_mps, vehicle.max_speed_mps)
            )
    else:
        if command.brake > 0.01:
            speed = _apply_brake_or_reverse(
                speed,
                command,
                dt_s=dt_s,
                brake_decel_mps2=vehicle.max_brake_mps2,
                reverse_accel_mps2=vehicle.reverse_accel_mps2,
                max_reverse_speed_mps=vehicle.max_reverse_speed_mps,
            )
        elif command.throttle > 0.01:
            intended_direction = -1.0 if command.reverse else 1.0
            accel_delta = command.throttle * vehicle.max_accel_mps2 * dt_s
            if speed * intended_direction < 0.0:
                speed = _move_towards(speed, 0.0, accel_delta * 1.5)
            else:
                speed += intended_direction * accel_delta
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
        if command.handbrake:
            commanded_yaw_rate *= vehicle.handbrake_yaw_gain
            max_yaw_rate = vehicle.max_handbrake_yaw_rate_radps
        else:
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
        response = 1.0 - math.exp(-8.0 * dt_s)
        yaw_rate = (
            state.yaw_rate_radps
            + (commanded_yaw_rate - state.yaw_rate_radps) * response
        )
    elif command.handbrake:
        response = 1.0 - math.exp(-4.0 * dt_s)
        yaw_rate = (
            state.yaw_rate_radps
            + (commanded_yaw_rate - state.yaw_rate_radps) * response
        )
        lateral_speed = float(np.dot(velocity, left))
        lateral_speed *= max(0.0, 1.0 - 2.0 * dt_s)
    else:
        # Normal steering is an arcade control target, while PhysX remains
        # responsible for contact impulses and tire forces. Running a second
        # stateful tire-slip model here made the same input depend on speed,
        # residual side-slip, and collision history before PhysX saw it.
        # Publish the driver's target directly; the PhysX follower supplies
        # the one physical response curve. Smoothing here as well created two
        # serial low-pass filters and made steering unexpectedly stiff.
        yaw_rate = commanded_yaw_rate
        lateral_speed = design.rear_axle_to_cg_m * yaw_rate

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
    lateral_accel = speed * yaw_rate * (0.35 if command.handbrake else 1.0)
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
