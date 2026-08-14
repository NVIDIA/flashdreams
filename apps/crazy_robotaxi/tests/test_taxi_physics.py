# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU regression tests for the Taxi-only PhysX adapter."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from crazy_robotaxi.driving import (
    TaxiVehicleConfig,
    integrate_taxi_vehicle,
)
from crazy_robotaxi.physics import (
    TaxiPhysicsWorld,
    inset_vehicle_chassis,
    select_traffic_tracks,
    step_taxi_physics_world,
)
from omnidreams_game_engine.config import ChunkConfig
from omnidreams_game_engine.simulation.components import (
    rigid_body_model_from_vehicle_config,
)
from omnidreams_game_engine.simulation.ego_vehicle_kinematics import (
    sample_chunk_trajectory,
)
from omnidreams_game_engine.simulation.game_physics import GamePhysicsWorld
from omnidreams_game_engine.types import (
    DriverCommand,
    PhysicsDebugFrame,
    VehicleState,
    WorldLineSegments,
)

pytestmark = pytest.mark.ci_cpu


@dataclass(frozen=True)
class _Scene:
    scene_id: str = "taxi-physics-test"
    vehicle_bbox_tracks: tuple[object, ...] = ()
    line_layers: tuple[WorldLineSegments, ...] = ()
    polygon_layers: tuple[object, ...] = ()


def _scene(*, line_layers: tuple[WorldLineSegments, ...] = ()) -> _Scene:
    return _Scene(line_layers=line_layers)


def _yaw_from_quaternion_xyzw(quaternion: np.ndarray) -> float:
    x, y, z, w = [float(value) for value in quaternion]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def test_taxi_traffic_filter_is_stable_and_keeps_non_motor_actors() -> None:
    tracks = tuple(
        SimpleNamespace(track_id=f"car-{index}", object_type="Car")
        for index in range(10)
    ) + (SimpleNamespace(track_id="person-1", object_type="Pedestrian"),)

    selected = select_traffic_tracks(tracks, 0.4, "scene-a")

    assert selected == select_traffic_tracks(tracks, 0.4, "scene-a")
    assert len([track for track in selected if track.object_type == "Car"]) == 4
    assert tracks[-1] in selected


def test_taxi_chassis_inset_does_not_change_visual_extents() -> None:
    model = rigid_body_model_from_vehicle_config(TaxiVehicleConfig())
    assert model.vehicle is not None

    inset = inset_vehicle_chassis(model)

    assert inset.half_extents_m == model.half_extents_m
    assert inset.vehicle is not None
    assert inset.vehicle.chassis_half_extents_m[0] == pytest.approx(
        model.vehicle.chassis_half_extents_m[0] - 0.16
    )
    assert inset.vehicle.chassis_half_extents_m[1] == pytest.approx(
        model.vehicle.chassis_half_extents_m[1] - 0.16
    )


def test_taxi_enclosure_is_added_only_to_private_physics_scene() -> None:
    scene = _scene()
    enclosure = np.asarray([[[5.0, -3.0, 0.0], [5.0, 3.0, 0.0]]], dtype=np.float32)

    with patch.object(GamePhysicsWorld, "__init__", return_value=None) as initialize:
        TaxiPhysicsWorld(
            scene,
            TaxiVehicleConfig(),
            traffic_density=1.0,
            enclosure_segments_world=enclosure,
        )

    physics_scene = initialize.call_args.args[0]
    assert scene.line_layers == ()
    assert len(physics_scene.line_layers) == 1
    assert physics_scene.line_layers[0].layer_name == "crazy_robotaxi_enclosure_walls"
    np.testing.assert_allclose(physics_scene.line_layers[0].segments_world, enclosure)
    assert len(GamePhysicsWorld._build_barriers(physics_scene)) == 1


def test_taxi_physics_keeps_app_heading_after_contact_resolution() -> None:
    incoming = VehicleState(
        x_m=1.0,
        y_m=2.0,
        z_m=0.0,
        yaw_rad=0.75,
        speed_mps=4.0,
        steer_rad=0.2,
        velocity_x_mps=3.0,
        velocity_y_mps=1.0,
        yaw_rate_radps=0.4,
    )
    physx_state = replace(
        incoming,
        x_m=1.2,
        y_m=2.1,
        yaw_rad=-1.0,
        yaw_rate_radps=-2.0,
        velocity_x_mps=2.0,
        velocity_y_mps=-0.5,
    )
    world = object.__new__(TaxiPhysicsWorld)
    with (
        patch.object(GamePhysicsWorld, "step", return_value=(physx_state, ())),
        patch.object(TaxiPhysicsWorld, "synchronize_ego_state") as synchronize,
    ):
        resolved, _samples = world.step(incoming, timestamp_us=1, dt_s=1.0 / 30.0)

    assert resolved.x_m == pytest.approx(physx_state.x_m)
    assert resolved.y_m == pytest.approx(physx_state.y_m)
    assert resolved.velocity_x_mps == pytest.approx(physx_state.velocity_x_mps)
    assert resolved.velocity_y_mps == pytest.approx(physx_state.velocity_y_mps)
    assert resolved.yaw_rad == pytest.approx(incoming.yaw_rad)
    assert resolved.yaw_rate_radps == pytest.approx(incoming.yaw_rate_radps)
    expected_speed = float(
        np.dot(
            np.asarray([2.0, -0.5]),
            np.asarray([np.cos(incoming.yaw_rad), np.sin(incoming.yaw_rad)]),
        )
    )
    assert resolved.speed_mps == pytest.approx(expected_speed)
    synchronize.assert_called_once_with(resolved)


def test_taxi_debug_frame_preserves_pre_policy_contact_pose() -> None:
    incoming = VehicleState(
        x_m=1.0,
        y_m=2.0,
        z_m=0.0,
        yaw_rad=0.75,
        speed_mps=4.0,
        steer_rad=0.2,
    )
    contact_state = replace(incoming, x_m=1.4, y_m=2.3, yaw_rad=-0.5)
    world = object.__new__(TaxiPhysicsWorld)
    world._ego_model = SimpleNamespace(half_extents_m=(2.4, 1.0, 0.8))
    debug = PhysicsDebugFrame(
        ego_position_m=np.zeros(3, dtype=np.float32),
        ego_orientation_xyzw=np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        ego_dimensions_lwh=np.asarray([4.8, 2.0, 1.6], dtype=np.float32),
        actor_positions_m=np.empty((0, 3), dtype=np.float32),
        actor_orientations_xyzw=np.empty((0, 4), dtype=np.float32),
        actor_dimensions_lwh=np.empty((0, 3), dtype=np.float32),
        barrier_segments_xy_m=np.empty((0, 2, 2), dtype=np.float32),
        barrier_thicknesses_m=np.empty((0,), dtype=np.float32),
        barrier_heights_m=np.empty((0,), dtype=np.float32),
    )
    with (
        patch.object(GamePhysicsWorld, "step", return_value=(contact_state, ())),
        patch.object(TaxiPhysicsWorld, "synchronize_ego_state"),
        patch.object(GamePhysicsWorld, "debug_frame", return_value=debug),
    ):
        world.step(incoming, timestamp_us=1, dt_s=1.0 / 30.0)
        captured = world.debug_frame(incoming)

    assert captured.ego_position_m[:2] == pytest.approx([1.4, 2.3])
    assert _yaw_from_quaternion_xyzw(captured.ego_orientation_xyzw) == pytest.approx(
        -0.5
    )


def test_taxi_handbrake_keeps_arcade_velocity_without_contact() -> None:
    incoming = VehicleState(
        x_m=1.0,
        y_m=2.0,
        z_m=0.0,
        yaw_rad=0.5,
        speed_mps=5.0,
        steer_rad=0.4,
        velocity_x_mps=3.0,
        velocity_y_mps=4.0,
        yaw_rate_radps=0.8,
    )
    physx_state = replace(
        incoming,
        velocity_x_mps=1.0,
        velocity_y_mps=0.0,
    )
    world = object.__new__(TaxiPhysicsWorld)
    with (
        patch.object(GamePhysicsWorld, "step", return_value=(physx_state, ())),
        patch.object(TaxiPhysicsWorld, "synchronize_ego_state"),
    ):
        resolved, _samples = world.step_with_command(
            incoming,
            DriverCommand(handbrake=True),
            timestamp_us=1,
            dt_s=1.0 / 30.0,
        )

    assert resolved.velocity_x_mps == pytest.approx(incoming.velocity_x_mps)
    assert resolved.velocity_y_mps == pytest.approx(incoming.velocity_y_mps)


def test_taxi_step_adapter_rejects_generic_physics_world() -> None:
    with pytest.raises(TypeError, match="TaxiPhysicsWorld"):
        step_taxi_physics_world(
            object(),  # type: ignore[arg-type]
            VehicleState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            DriverCommand(),
            timestamp_us=0,
            dt_s=1.0 / 30.0,
        )


def test_taxi_physics_hook_receives_the_active_driver_command() -> None:
    commands: list[DriverCommand] = []

    class _PhysicsWorld:
        last_step_actor_collision = False
        last_step_timings = None

        def synchronize_window(
            self, center_xy_m: np.ndarray, timestamp_us: int | None = None
        ) -> None:
            del center_xy_m, timestamp_us

        def step(
            self, state: VehicleState, timestamp_us: int, dt_s: float
        ) -> tuple[VehicleState, tuple[object, ...]]:
            raise AssertionError("the default physics step was used")

        def build_trajectories(
            self, timestamps_us: np.ndarray, samples_by_frame: list[tuple[object, ...]]
        ) -> tuple[object, ...]:
            del timestamps_us, samples_by_frame
            return ()

    def command_aware_step(
        physics_world: GamePhysicsWorld,
        state: VehicleState,
        command: DriverCommand,
        timestamp_us: int,
        dt_s: float,
    ) -> tuple[VehicleState, tuple[tuple[str, np.ndarray, np.ndarray, bool], ...]]:
        del physics_world, timestamp_us, dt_s
        commands.append(command)
        return state, ()

    command = DriverCommand(handbrake=True, steer=0.5)
    sample_chunk_trajectory(
        start_state=VehicleState(0.0, 0.0, 0.0, 0.0, 5.0, 0.0),
        start_timestamp_us=0,
        command=command,
        chunk_size=2,
        chunk_config=ChunkConfig(fps=30),
        vehicle_config=TaxiVehicleConfig(),
        ground_snapper=None,
        physics_world=_PhysicsWorld(),  # type: ignore[arg-type]
        integrate_fn=integrate_taxi_vehicle,
        physics_step_fn=command_aware_step,
    )

    assert commands == [command, command]


def test_taxi_native_heading_matches_app_heading_after_boundary_contact() -> None:
    boundary = WorldLineSegments(
        segments_world=np.asarray(
            [[[-20.0, 0.0, 0.0], [20.0, 0.0, 0.0]]], dtype=np.float32
        ),
        color_rgba=(1.0, 1.0, 1.0, 1.0),
        width_px=2.0,
        layer_name="road_boundaries",
    )
    config = TaxiVehicleConfig(drag_mps2=0.0)
    world = TaxiPhysicsWorld(
        _scene(line_layers=(boundary,)),
        config,
        traffic_density=1.0,
    )
    initial_yaw = math.radians(15.0)
    state = VehicleState(
        x_m=-5.0,
        y_m=-3.0,
        z_m=0.0,
        yaw_rad=initial_yaw,
        speed_mps=8.0,
        steer_rad=0.0,
        velocity_x_mps=8.0 * math.cos(initial_yaw),
        velocity_y_mps=8.0 * math.sin(initial_yaw),
    )
    command = DriverCommand(throttle=1.0, steer_is_direct=True, manual_control=True)
    contact_detected = False

    try:
        for frame_index in range(90):
            state = integrate_taxi_vehicle(state, command, 1.0 / 30.0, config)
            state, _ = world.step_with_command(
                state,
                command,
                timestamp_us=frame_index * 33_333,
                dt_s=1.0 / 30.0,
            )
            native_state = world._world.state_buffer[world._world._ego_slot]
            native_yaw = _yaw_from_quaternion_xyzw(native_state[3:7])
            assert native_yaw == pytest.approx(state.yaw_rad, abs=1.0e-5)
            contact_detected |= state.ragdoll_active
            if contact_detected and frame_index > 20:
                break
    finally:
        world.close()

    assert contact_detected is True
    assert state.yaw_rad == pytest.approx(initial_yaw, abs=1.0e-5)


def test_taxi_handbrake_turn_remains_bounded_through_physx() -> None:
    config = TaxiVehicleConfig(drag_mps2=0.0)
    world = TaxiPhysicsWorld(_scene(), config, traffic_density=1.0)
    state = VehicleState(
        x_m=0.0,
        y_m=0.0,
        z_m=0.0,
        yaw_rad=0.0,
        speed_mps=15.0,
        steer_rad=0.0,
        velocity_x_mps=15.0,
        velocity_y_mps=0.0,
    )
    command = DriverCommand(
        steer=1.0,
        handbrake=True,
        steer_is_direct=True,
        manual_control=True,
    )
    yaws = [state.yaw_rad]

    try:
        for frame_index in range(72):
            state = integrate_taxi_vehicle(state, command, 1.0 / 30.0, config)
            state, _ = world.step_with_command(
                state,
                command,
                timestamp_us=frame_index * 33_333,
                dt_s=1.0 / 30.0,
            )
            yaws.append(state.yaw_rad)
    finally:
        world.close()

    assert abs(state.yaw_rad) > math.radians(25.0)
    assert abs(state.speed_mps) < 2.0
    per_frame_turns = np.abs(np.diff(np.unwrap(np.asarray(yaws))))
    assert np.max(per_frame_turns) <= (
        config.max_handbrake_yaw_rate_radps / 30.0 + 1.0e-5
    )
    assert np.max(per_frame_turns[-10:]) < math.radians(0.1)


def test_taxi_normal_steering_tracks_arcade_heading_through_physx() -> None:
    config = TaxiVehicleConfig(drag_mps2=0.0)
    world = TaxiPhysicsWorld(_scene(), config, traffic_density=1.0)
    state = VehicleState(
        x_m=0.0,
        y_m=0.0,
        z_m=0.0,
        yaw_rad=0.0,
        speed_mps=8.0,
        steer_rad=0.0,
        velocity_x_mps=8.0,
        velocity_y_mps=0.0,
    )
    command = DriverCommand(steer=1.0, steer_is_direct=True)

    try:
        for frame_index in range(30):
            state = integrate_taxi_vehicle(state, command, 1.0 / 30.0, config)
            state, _ = world.step_with_command(
                state,
                command,
                timestamp_us=frame_index * 33_333,
                dt_s=1.0 / 30.0,
            )
    finally:
        world.close()

    assert state.yaw_rad > math.radians(42.0)
    assert state.yaw_rate_radps > 0.75


def test_taxi_acceleration_and_braking_remain_arcade_responsive_through_physx() -> None:
    config = TaxiVehicleConfig()
    world = TaxiPhysicsWorld(_scene(), config, traffic_density=1.0)
    state = VehicleState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    throttle = DriverCommand(throttle=1.0, steer_is_direct=True, manual_control=True)

    try:
        for frame_index in range(45):
            state = integrate_taxi_vehicle(state, throttle, 1.0 / 30.0, config)
            state, _ = world.step_with_command(
                state,
                throttle,
                timestamp_us=frame_index * 33_333,
                dt_s=1.0 / 30.0,
            )
        assert state.speed_mps > 13.0

        brake = DriverCommand(brake=1.0, steer_is_direct=True, manual_control=True)
        stop_frame = None
        for brake_frame in range(50):
            state = integrate_taxi_vehicle(state, brake, 1.0 / 30.0, config)
            state, _ = world.step_with_command(
                state,
                brake,
                timestamp_us=(45 + brake_frame) * 33_333,
                dt_s=1.0 / 30.0,
            )
            if state.speed_mps <= 0.0:
                stop_frame = brake_frame
                break
    finally:
        world.close()

    assert stop_frame is not None
    assert stop_frame < 45


def test_taxi_pedal_brake_builds_reverse_speed_through_physx() -> None:
    config = TaxiVehicleConfig()
    world = TaxiPhysicsWorld(_scene(), config, traffic_density=1.0)
    state = VehicleState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    reverse = DriverCommand(brake=1.0, steer_is_direct=True, manual_control=True)

    try:
        for frame_index in range(30):
            state = integrate_taxi_vehicle(state, reverse, 1.0 / 30.0, config)
            state, _ = world.step_with_command(
                state,
                reverse,
                timestamp_us=frame_index * 33_333,
                dt_s=1.0 / 30.0,
            )
    finally:
        world.close()

    assert state.speed_mps < -5.5
    assert state.x_m < -2.0
