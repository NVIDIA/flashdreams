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
    step_taxi_physics_world,
)
from ludus_renderer import BodyState
from omnidreams_game_engine.config import ChunkConfig, VehicleConfig
from omnidreams_game_engine.game_map.types import GameMapTrafficVehicle
from omnidreams_game_engine.game_map.vicinity import GameMapVicinity
from omnidreams_game_engine.simulation.components import (
    rigid_body_model_from_vehicle_config,
)
from omnidreams_game_engine.simulation.ego_vehicle_kinematics import (
    sample_chunk_trajectory,
)
from omnidreams_game_engine.simulation.game_physics import GamePhysicsWorld
from omnidreams_game_engine.simulation.map_traffic import (
    MapTrafficController,
    MapTrafficPhase,
)
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
    line_layers: tuple[WorldLineSegments, ...] = ()
    polygon_layers: tuple[object, ...] = ()


def _scene(*, line_layers: tuple[WorldLineSegments, ...] = ()) -> _Scene:
    return _Scene(line_layers=line_layers)


def _yaw_from_quaternion_xyzw(quaternion: np.ndarray) -> float:
    x, y, z, w = [float(value) for value in quaternion]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


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


def test_map_traffic_controller_holds_a_car_for_same_lane_headway() -> None:
    centerline = np.asarray(
        [[0, 0, 0], [100, 0, 0], [100, 50, 0], [0, 50, 0], [0, 0, 0]],
        dtype=np.float32,
    )
    speed_limits = np.full(len(centerline), 10.0, dtype=np.float32)

    def definition(vehicle_id: str, start_distance_m: float) -> GameMapTrafficVehicle:
        return GameMapTrafficVehicle(
            vehicle_id=vehicle_id,
            node_ids=("a", "b"),
            end_behavior="wrap",
            vehicle_type="car",
            dimensions_lwh_m=(4.5, 1.8, 1.5),
            speed_mps=None,
            start_distance_m=start_distance_m,
            centerline_world=centerline,
            speed_limits_mps=speed_limits,
            route_element_ids=("test-road",) * (len(centerline) - 1),
        )

    controller = MapTrafficController(
        (definition("following", 0), definition("leading", 10)), VehicleConfig()
    )
    controller.set_vicinity(
        GameMapVicinity("test-road", frozenset({"test-road"}), frozenset({"test-road"}))
    )
    assert controller.max_drive_speeds_mps == {
        "map-traffic:following": 10.0,
        "map-traffic:leading": 10.0,
    }
    bodies = {
        "map-traffic:following": BodyState(
            position_m=np.asarray([0, 0, 0.75], dtype=np.float32),
            orientation_xyzw=np.asarray([0, 0, 0, 1], dtype=np.float32),
            linear_velocity_mps=np.zeros(3, dtype=np.float32),
            angular_velocity_radps=np.zeros(3, dtype=np.float32),
        ),
        "map-traffic:leading": BodyState(
            position_m=np.asarray([10, 0, 0.75], dtype=np.float32),
            orientation_xyzw=np.asarray([0, 0, 0, 1], dtype=np.float32),
            linear_velocity_mps=np.zeros(3, dtype=np.float32),
            angular_velocity_radps=np.zeros(3, dtype=np.float32),
        ),
    }
    published: list[tuple[tuple[str, int, float], ...]] = []
    world = SimpleNamespace(
        body_state=lambda object_id: bodies[object_id],
        ego_model=SimpleNamespace(half_extents_m=(2.4, 1.0, 0.8)),
        apply_track_progress=published.append,
    )
    ego = BodyState(
        position_m=np.asarray([-100, 0, 0.8], dtype=np.float32),
        orientation_xyzw=np.asarray([0, 0, 0, 1], dtype=np.float32),
        linear_velocity_mps=np.zeros(3, dtype=np.float32),
        angular_velocity_radps=np.zeros(3, dtype=np.float32),
    )

    controller.prepare_step(world, ego, 0.0)  # type: ignore[arg-type]

    scales = {object_id: scale for object_id, _, scale in published[-1]}
    assert scales["map-traffic:following"] == 0.0
    assert scales["map-traffic:leading"] == 1.0


def test_map_traffic_advances_inactive_car_without_publishing_it_to_physx() -> None:
    centerline = np.asarray(
        [[0, 0, 0], [100, 0, 0], [100, 50, 0], [0, 50, 0], [0, 0, 0]],
        dtype=np.float32,
    )
    speed_limits = np.full(len(centerline), 10.0, dtype=np.float32)

    def definition(vehicle_id: str, element_id: str, start_m: float):
        return GameMapTrafficVehicle(
            vehicle_id=vehicle_id,
            node_ids=("a", "b"),
            end_behavior="wrap",
            vehicle_type="car",
            dimensions_lwh_m=(4.5, 1.8, 1.5),
            speed_mps=None,
            start_distance_m=start_m,
            centerline_world=centerline,
            speed_limits_mps=speed_limits,
            route_element_ids=(element_id,) * (len(centerline) - 1),
        )

    controller = MapTrafficController(
        (definition("near", "near-road", 1.0), definition("far", "far-road", 20.0)),
        VehicleConfig(),
    )
    controller.set_vicinity(
        GameMapVicinity("near-road", frozenset({"near-road"}), frozenset({"near-road"}))
    )
    near_id = "map-traffic:near"
    far_id = "map-traffic:far"
    near_timestamp_before = controller._states_by_id[near_id].timestamp_us
    far_timestamp_before = controller._states_by_id[far_id].timestamp_us
    published: list[tuple[tuple[str, int, float], ...]] = []
    near_position, near_orientation, _ = controller.active_objects[0].sample(
        int(near_timestamp_before)
    )
    near_body = BodyState(
        position_m=near_position,
        orientation_xyzw=near_orientation,
        linear_velocity_mps=np.zeros(3, dtype=np.float32),
        angular_velocity_radps=np.zeros(3, dtype=np.float32),
    )
    world = SimpleNamespace(
        body_state=lambda object_id: {near_id: near_body}[object_id],
        ego_model=SimpleNamespace(half_extents_m=np.asarray([2.4, 1.0, 0.8])),
        apply_track_progress=published.append,
    )
    ego = BodyState(
        position_m=np.asarray([-100, -100, 0.8], dtype=np.float32),
        orientation_xyzw=np.asarray([0, 0, 0, 1], dtype=np.float32),
        linear_velocity_mps=np.zeros(3, dtype=np.float32),
        angular_velocity_radps=np.zeros(3, dtype=np.float32),
    )

    controller.prepare_step(world, ego, 1.0 / 30.0)

    assert controller.active_object_ids == {near_id}
    assert tuple(item[0] for item in published[0]) == (near_id,)
    assert controller._states_by_id[near_id].timestamp_us > near_timestamp_before
    assert controller._states_by_id[far_id].timestamp_us > far_timestamp_before


def _recovery_controller() -> tuple[MapTrafficController, str]:
    centerline = np.asarray(
        [[0, 0, 0], [20, 0, 0], [20, 20, 0], [0, 20, 0], [0, 0, 0]],
        dtype=np.float32,
    )
    definition = GameMapTrafficVehicle(
        vehicle_id="recovering",
        node_ids=("a", "b"),
        end_behavior="wrap",
        vehicle_type="car",
        dimensions_lwh_m=(4.5, 1.8, 1.5),
        speed_mps=None,
        start_distance_m=2.0,
        centerline_world=centerline,
        speed_limits_mps=np.full(len(centerline), 10.0, dtype=np.float32),
        route_element_ids=("south", "east", "north", "west"),
    )
    controller = MapTrafficController((definition,), VehicleConfig())
    controller.set_vicinity(
        GameMapVicinity(
            "south",
            frozenset({"south", "east", "north", "west"}),
            frozenset({"south", "east", "north", "west"}),
        )
    )
    return controller, "map-traffic:recovering"


def _body_at(
    position_xy: tuple[float, float],
    *,
    yaw_rad: float = 0.0,
    linear_velocity_xy: tuple[float, float] = (0.0, 0.0),
    angular_speed_radps: float = 0.0,
) -> BodyState:
    return BodyState(
        position_m=np.asarray([*position_xy, 0.75], dtype=np.float32),
        orientation_xyzw=np.asarray(
            [0.0, 0.0, math.sin(yaw_rad * 0.5), math.cos(yaw_rad * 0.5)],
            dtype=np.float32,
        ),
        linear_velocity_mps=np.asarray([*linear_velocity_xy, 0.0], dtype=np.float32),
        angular_velocity_radps=np.asarray(
            [0.0, 0.0, angular_speed_radps], dtype=np.float32
        ),
    )


def test_map_traffic_collision_freezes_route_until_nearest_route_recovery() -> None:
    controller, object_id = _recovery_controller()
    state = controller.state(object_id)
    assert state is not None
    frozen_timestamp_us = state.timestamp_us

    decision = controller.observe_physics(
        object_id,
        struck=True,
        body=_body_at((18.0, 8.0), yaw_rad=math.pi),
        dt_s=0.25,
    )
    assert decision is not None
    assert decision.drive_enabled is False
    assert decision.detached_from_track is True
    assert state.phase is MapTrafficPhase.COLLISION

    published: list[tuple[tuple[str, int, float], ...]] = []
    world = SimpleNamespace(
        body_state=lambda _: _body_at((18.0, 8.0), yaw_rad=math.pi),
        ego_model=SimpleNamespace(half_extents_m=(2.4, 1.0, 0.8)),
        apply_track_progress=published.append,
    )
    controller.prepare_step(
        world,  # type: ignore[arg-type]
        _body_at((-100.0, -100.0)),
        0.5,
    )
    assert state.timestamp_us == frozen_timestamp_us
    assert published[-1][0][2] == 0.0

    stopped = _body_at((18.0, 8.0), yaw_rad=math.pi)
    for _ in range(4):
        decision = controller.observe_physics(
            object_id, struck=False, body=stopped, dt_s=0.25
        )
    assert decision is not None
    assert decision.drive_enabled is True
    assert decision.detached_from_track is True
    assert state.phase is MapTrafficPhase.RECOVERING
    projected_position, projected_orientation, projected_velocity = (
        state.scene_object.sample(int(state.timestamp_us))
    )
    np.testing.assert_allclose(projected_position[:2], [20.0, 8.0], atol=1.0e-4)

    # A car facing backward remains physically detached; recovery does not snap it.
    controller.observe_physics(
        object_id,
        struck=False,
        body=_body_at(tuple(projected_position[:2]), yaw_rad=math.pi),
        dt_s=0.25,
    )
    assert state.phase is MapTrafficPhase.RECOVERING
    np.testing.assert_allclose(state.position_m[:2], projected_position[:2])

    recovered = BodyState(
        position_m=projected_position.copy(),
        orientation_xyzw=projected_orientation.copy(),
        linear_velocity_mps=projected_velocity.copy(),
        angular_velocity_radps=np.zeros(3, dtype=np.float32),
    )
    decision = controller.observe_physics(
        object_id, struck=False, body=recovered, dt_s=0.25
    )
    assert decision is not None
    assert decision.detached_from_track is False
    assert state.phase is MapTrafficPhase.TRAVERSING


def test_map_traffic_recollision_and_offscreen_reset_share_one_state() -> None:
    controller, object_id = _recovery_controller()
    state = controller.state(object_id)
    assert state is not None
    displaced = _body_at((19.0, 9.0))
    controller.observe_physics(object_id, struck=True, body=displaced, dt_s=0.25)
    for _ in range(4):
        controller.observe_physics(object_id, struck=False, body=displaced, dt_s=0.25)
    assert state.phase is MapTrafficPhase.RECOVERING

    decision = controller.observe_physics(
        object_id, struck=True, body=_body_at((17.0, 10.0)), dt_s=0.25
    )
    assert decision is not None
    assert decision.drive_enabled is False
    assert state.phase is MapTrafficPhase.COLLISION

    changed = controller.set_vicinity(
        GameMapVicinity("other", frozenset({"other"}), frozenset({"other"}))
    )
    assert changed is True
    assert controller.active_object_ids == frozenset()
    assert state.phase is MapTrafficPhase.TRAVERSING
    assert state.decision.detached_from_track is False
    route_position, route_orientation, route_velocity = state.scene_object.sample(
        int(state.timestamp_us)
    )
    np.testing.assert_array_equal(state.position_m, route_position)
    np.testing.assert_array_equal(state.orientation_xyzw, route_orientation)
    np.testing.assert_array_equal(state.linear_velocity_mps, route_velocity)


def test_taxi_curbs_are_physics_barriers_without_a_render_layer() -> None:
    scene = _scene()
    curbs = np.asarray([[[5.0, -3.0, 0.0], [5.0, 3.0, 0.0]]], dtype=np.float32)
    config = TaxiVehicleConfig()

    with patch.object(GamePhysicsWorld, "__init__", return_value=None) as initialize:
        TaxiPhysicsWorld(
            scene,
            config,
            curb_segments_world=curbs,
        )

        physics_scene = initialize.call_args.args[0]
        assert scene.line_layers == ()
        assert physics_scene.line_layers == ()
        np.testing.assert_array_equal(
            initialize.call_args.kwargs["static_barrier_segments_world"], curbs
        )
        assert initialize.call_args.kwargs["static_barrier_restitution"] == 0.45
        assert config.collision_restitution == VehicleConfig().collision_restitution


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
            self,
            center_xy_m: np.ndarray,
            timestamp_us: int | None = None,
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
        commands=(command,) * 2,
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
        curb_segments_world=boundary.segments_world,
    )
    initial_yaw = math.radians(30.0)
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
    contact_velocity_y_mps = 0.0
    contact_requested_speed_mps = 0.0
    contact_resolved_speed_mps = 0.0

    try:
        for frame_index in range(90):
            state = integrate_taxi_vehicle(state, command, 1.0 / 30.0, config)
            requested_speed_mps = state.speed_mps
            state, _ = world.step_with_command(
                state,
                command,
                timestamp_us=frame_index * 33_333,
                dt_s=1.0 / 30.0,
            )
            native_state = world._world.state_buffer[world._world._ego_slot]
            native_yaw = _yaw_from_quaternion_xyzw(native_state[3:7])
            assert native_yaw == pytest.approx(state.yaw_rad, abs=1.0e-5)
            if state.ragdoll_active and not contact_detected:
                contact_detected = True
                contact_velocity_y_mps = float(state.velocity_y_mps or 0.0)
                contact_requested_speed_mps = requested_speed_mps
                contact_resolved_speed_mps = state.speed_mps
            if contact_detected and frame_index > 20:
                break
    finally:
        world.close()

    assert contact_detected is True
    assert contact_velocity_y_mps < -0.75
    assert contact_resolved_speed_mps >= (
        contact_requested_speed_mps * config.curb_forward_momentum_retention - 1.0e-5
    )
    assert state.yaw_rad == pytest.approx(initial_yaw, abs=1.0e-5)


def test_taxi_handbrake_turn_remains_bounded_through_physx() -> None:
    config = TaxiVehicleConfig(drag_mps2=0.0)
    world = TaxiPhysicsWorld(_scene(), config)
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
    world = TaxiPhysicsWorld(_scene(), config)
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
    world = TaxiPhysicsWorld(_scene(), config)
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
    world = TaxiPhysicsWorld(_scene(), config)
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
