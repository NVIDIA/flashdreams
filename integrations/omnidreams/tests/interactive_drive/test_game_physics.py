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

"""CPU tests for componentized vehicle and collision physics."""

from __future__ import annotations

import gc
import math
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from ludus_renderer import (
    PRIM_OBSTACLE,
    BodyState,
    CubePool,
    InvisibleBarrier,
    LudusCudaTimestampedContext,
    PhysicsObjectGraph,
    PhysXWorld,
    RigidBodyModel,
    SceneObject,
    TimestampedScene,
)
from omnidreams.interactive_drive.config import VehicleConfig
from omnidreams.interactive_drive.input.keyboard import (
    KeyboardInputBackend,
    KeyboardState,
)
from omnidreams.interactive_drive.physx_debug import (
    build_physx_debug_cube_pool,
    select_presented_rgb,
)
from omnidreams.interactive_drive.rasterizer import _LudusConditionRasterizerImpl
from omnidreams.interactive_drive.simulation.components import (
    BoxColliderComponent,
    GameEntity,
    RigidBodyComponent,
    TransformComponent,
    canonical_object_type,
    game_entity_from_vehicle_state,
    rigid_body_model_for_object,
    rigid_body_model_from_vehicle_config,
)
from omnidreams.interactive_drive.simulation.ego_vehicle_kinematics import (
    EgoVehicleKinematics,
    integrate_vehicle,
)
from omnidreams.interactive_drive.simulation.game_physics import (
    GamePhysicsWorld,
    _is_visual_flare_impact,
    _recorded_actor_trajectory,
    _simplify_barrier_segments,
    _yaw_from_quaternion_xyzw,
)
from omnidreams.interactive_drive.simulation.traffic_ai import TrafficDriverAI
from omnidreams.interactive_drive.types import (
    DriverCommand,
    DynamicActorTrajectory,
    PhysicsDebugFrame,
    PresentedFrame,
    VehicleState,
    WorldLineSegments,
    WorldVehicleBBoxTrack,
)

pytestmark = pytest.mark.ci_cpu


def _track(
    object_type: str = "Car", *, x_m: float = 5.0, y_m: float = 0.0
) -> WorldVehicleBBoxTrack:
    timestamps = np.asarray([-1_000_000, 1_000_000], dtype=np.int64)
    return WorldVehicleBBoxTrack(
        track_id=f"{object_type.lower()}-1",
        object_type=object_type,
        timestamps_us=timestamps,
        centers_world=np.asarray([[x_m, y_m, 0.8], [x_m, y_m, 0.8]], dtype=np.float32),
        dimensions_lwh=np.asarray([[4.0, 1.9, 1.6], [4.0, 1.9, 1.6]], dtype=np.float32),
        orientations_xyzw=np.asarray(
            [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32
        ),
        max_extrapolation_us=2_000_000.0,
    )


def _moving_track() -> WorldVehicleBBoxTrack:
    timestamps = np.asarray([-1_000_000, 12_000_000], dtype=np.int64)
    return WorldVehicleBBoxTrack(
        track_id="car-moving",
        object_type="Car",
        timestamps_us=timestamps,
        centers_world=np.asarray([[5.0, 0.0, 0.8], [18.0, 0.0, 0.8]], dtype=np.float32),
        dimensions_lwh=np.asarray([[4.0, 1.9, 1.6], [4.0, 1.9, 1.6]], dtype=np.float32),
        orientations_xyzw=np.asarray(
            [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32
        ),
        max_extrapolation_us=2_000_000.0,
    )


def _fast_moving_track() -> WorldVehicleBBoxTrack:
    timestamps = np.asarray([-1_000_000, 9_000_000], dtype=np.int64)
    return WorldVehicleBBoxTrack(
        track_id="car-fast",
        object_type="Car",
        timestamps_us=timestamps,
        centers_world=np.asarray(
            [[20.0, 0.0, 0.8], [220.0, 0.0, 0.8]], dtype=np.float32
        ),
        dimensions_lwh=np.asarray([[4.0, 1.9, 1.6], [4.0, 1.9, 1.6]], dtype=np.float32),
        orientations_xyzw=np.asarray(
            [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32
        ),
        max_extrapolation_us=2_000_000.0,
    )


def _scene(
    *tracks: WorldVehicleBBoxTrack,
    line_layers: tuple[WorldLineSegments, ...] = (),
) -> SimpleNamespace:
    return SimpleNamespace(
        vehicle_bbox_tracks=tracks,
        line_layers=line_layers,
        polygon_layers=(),
    )


def _moving_ego() -> VehicleState:
    return VehicleState(
        x_m=2.0,
        y_m=0.0,
        z_m=0.0,
        yaw_rad=0.0,
        speed_mps=10.0,
        steer_rad=0.0,
        velocity_x_mps=10.0,
        velocity_y_mps=0.0,
    )


@pytest.mark.parametrize(
    ("collision_occurred", "before_mph", "after_mph", "expected"),
    [
        (False, 20.0, 10.0, False),
        (True, 20.0, 15.01, False),
        (True, 20.0, 15.0, True),
        (True, -20.0, -15.0, True),
        (True, 10.0, 15.0, True),
        (True, 3.0, -2.0, False),
    ],
)
def test_visual_flare_requires_collision_and_five_mph_speed_delta(
    collision_occurred: bool,
    before_mph: float,
    after_mph: float,
    expected: bool,
) -> None:
    mph_to_mps = 0.44704
    before_velocity = np.asarray([before_mph * mph_to_mps, 0.0, 0.0])
    after_velocity = np.asarray([after_mph * mph_to_mps, 0.0, 0.0])

    assert (
        _is_visual_flare_impact(
            collision_occurred,
            before_velocity_mps=before_velocity,
            after_velocity_mps=after_velocity,
            driving_direction_xy=np.asarray([1.0, 0.0]),
        )
        is expected
    )


def test_visual_flare_ignores_side_swipe_that_only_adds_lateral_velocity() -> None:
    mph_to_mps = 0.44704

    assert not _is_visual_flare_impact(
        True,
        before_velocity_mps=np.asarray([20.0 * mph_to_mps, 0.0, 0.0]),
        after_velocity_mps=np.asarray([20.0 * mph_to_mps, 5.0 * mph_to_mps, 0.0]),
        driving_direction_xy=np.asarray([1.0, 0.0]),
    )


def test_visual_flare_counts_side_impact_that_reduces_driving_speed() -> None:
    mph_to_mps = 0.44704
    lateral_mph = np.sqrt(20.0**2 - 15.0**2)

    assert _is_visual_flare_impact(
        True,
        before_velocity_mps=np.asarray([20.0 * mph_to_mps, 0.0, 0.0]),
        after_velocity_mps=np.asarray(
            [15.0 * mph_to_mps, lateral_mph * mph_to_mps, 0.0]
        ),
        driving_direction_xy=np.asarray([1.0, 0.0]),
    )


def test_visual_flare_counts_external_hit_across_driving_direction() -> None:
    mph_to_mps = 0.44704

    assert _is_visual_flare_impact(
        True,
        before_velocity_mps=np.asarray([20.0 * mph_to_mps, 0.0, 0.0]),
        after_velocity_mps=np.asarray([20.0 * mph_to_mps, 5.0 * mph_to_mps, 0.0]),
        driving_direction_xy=np.asarray([1.0, 0.0]),
        impact_normal_xy=np.asarray([0.0, 1.0]),
    )


def test_collision_detaches_actor_and_applies_physics_response() -> None:
    world = GamePhysicsWorld(_scene(_track()), VehicleConfig())
    state = _moving_ego()
    try:
        for frame_index in range(16):
            state, first_samples = world.step(
                state,
                timestamp_us=frame_index * 33_333,
                dt_s=1.0 / 30.0,
            )
            if world.last_step_actor_collision:
                break

        actor = world.entities[0]
        assert world.last_step_actor_collision is True
        assert actor.detached_from_track is True
        assert state.ragdoll_active is True
        assert first_samples[0][3] is True
        assert world._world._track_drive_enabled["car-1"] is False
        impact_position = actor.transform.position_m.copy()

        next_timestamp_us = (frame_index + 1) * 33_333
        _, second_samples = world.step(
            state, timestamp_us=next_timestamp_us, dt_s=1.0 / 30.0
        )

        assert second_samples[0][3] is True
        assert actor.transform.position_m[0] > impact_position[0]
        trajectory = world.build_trajectories(
            np.asarray([frame_index * 33_333, next_timestamp_us], dtype=np.int64),
            [first_samples, second_samples],
        )[0]
        np.testing.assert_array_equal(
            trajectory.timestamps_us, [frame_index * 33_333, next_timestamp_us]
        )
        assert trajectory.detached_from_track is True
        assert trajectory.is_simulated is True
    finally:
        world.close()


def test_minor_collision_does_not_latch_vehicle_drive_off() -> None:
    world = GamePhysicsWorld(_scene(_track()), VehicleConfig())
    state = VehicleState(
        x_m=2.0,
        y_m=0.0,
        z_m=0.0,
        yaw_rad=0.0,
        speed_mps=1.0,
        steer_rad=0.0,
        velocity_x_mps=1.0,
        velocity_y_mps=0.0,
    )

    try:
        for frame_index in range(17):
            state, samples = world.step(
                state,
                timestamp_us=frame_index * 33_333,
                dt_s=1.0 / 30.0,
            )
            assert world.last_step_actor_collision is False
            assert samples[0][3] is False
            if frame_index == 0:
                assert world._world._track_drive_enabled["car-1"] is False
        assert world._traffic_ai._states["car-1"].drive_enabled is True
        assert world._world._track_drive_enabled["car-1"] is True
    finally:
        world.close()


def test_recorded_renderer_trajectory_is_cached_and_holds_final_pose() -> None:
    scene_object = GamePhysicsWorld._object_from_track(
        _track(x_m=500.0), VehicleConfig()
    )
    cached = _recorded_actor_trajectory(scene_object)
    world = object.__new__(GamePhysicsWorld)
    world.graph = PhysicsObjectGraph(objects=(scene_object,))
    world._recorded_trajectories_by_id = {scene_object.object_id: cached}
    timestamps = np.asarray([0, 33_333], dtype=np.int64)

    first = world.build_trajectories(timestamps, [(), ()])[0]
    second = world.build_trajectories(timestamps, [(), ()])[0]

    assert first is second
    assert first.is_simulated is False
    assert int(first.timestamps_us[-1]) > int(timestamps[-1])
    np.testing.assert_array_equal(
        first.translations_world[-1], first.translations_world[-2]
    )
    np.testing.assert_array_equal(
        first.orientations_xyzw[-1], first.orientations_xyzw[-2]
    )


def test_side_impact_is_resolved_by_physx_contact() -> None:
    world = GamePhysicsWorld(_scene(_track(x_m=0.0, y_m=3.0)), VehicleConfig())
    state = VehicleState(
        x_m=0.0,
        y_m=0.0,
        z_m=0.0,
        yaw_rad=np.pi / 2.0,
        speed_mps=10.0,
        steer_rad=0.0,
        velocity_x_mps=0.0,
        velocity_y_mps=10.0,
    )

    resolved, samples = world.step(state, timestamp_us=0, dt_s=1.0 / 30.0)
    actor = world._world.body_state("car-1")

    assert samples[0][3] is True
    assert resolved.ragdoll_active is True
    assert actor.linear_velocity_mps[1] > 0.0
    actor_heights = [float(actor.position_m[2])]
    actor_tilts = [float(np.linalg.norm(actor.orientation_xyzw[:2]))]
    for frame_index in range(1, 121):
        resolved, _ = world.step(
            resolved,
            timestamp_us=frame_index * 33_333,
            dt_s=1.0 / 30.0,
        )
        actor = world._world.body_state("car-1")
        actor_heights.append(float(actor.position_m[2]))
        actor_tilts.append(float(np.linalg.norm(actor.orientation_xyzw[:2])))

    assert max(actor_heights) < 1.6
    assert max(actor_tilts) < 0.5
    assert actor_heights[-1] == pytest.approx(0.8, abs=0.08)
    world.close()


def test_physx_vehicle_yaw_is_free_away_from_road_boundaries() -> None:
    config = VehicleConfig()
    world = PhysXWorld(
        PhysicsObjectGraph(objects=()),
        rigid_body_model_from_vehicle_config(config),
    )
    initial_yaw = math.radians(70.0)
    ego = BodyState(
        position_m=np.asarray([0.0, 0.0, 2.0], dtype=np.float32),
        orientation_xyzw=np.asarray(
            [0.0, 0.0, math.sin(initial_yaw * 0.5), math.cos(initial_yaw * 0.5)],
            dtype=np.float32,
        ),
        linear_velocity_mps=np.zeros(3, dtype=np.float32),
        angular_velocity_radps=np.zeros(3, dtype=np.float32),
    )

    try:
        resolved = world.step(ego, timestamp_us=0, dt_s=1.0 / 120.0).ego
    finally:
        world.close()

    assert _yaw_from_quaternion_xyzw(resolved.orientation_xyzw) == pytest.approx(
        initial_yaw, abs=1.0e-5
    )


def test_physx_vehicle_yaw_is_limited_at_road_boundary() -> None:
    config = VehicleConfig()
    world = PhysXWorld(
        PhysicsObjectGraph(
            objects=(),
            barriers=(InvisibleBarrier((-10.0, 0.0), (10.0, 0.0)),),
        ),
        rigid_body_model_from_vehicle_config(config),
    )
    initial_yaw = math.radians(70.0)
    ego = BodyState(
        position_m=np.asarray([0.0, 0.0, 2.0], dtype=np.float32),
        orientation_xyzw=np.asarray(
            [0.0, 0.0, math.sin(initial_yaw * 0.5), math.cos(initial_yaw * 0.5)],
            dtype=np.float32,
        ),
        linear_velocity_mps=np.zeros(3, dtype=np.float32),
        angular_velocity_radps=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
    )

    try:
        resolved = world.step(ego, timestamp_us=0, dt_s=1.0 / 120.0).ego
    finally:
        world.close()

    assert (
        abs(_yaw_from_quaternion_xyzw(resolved.orientation_xyzw))
        <= math.radians(25.0) + 1.0e-5
    )
    assert resolved.angular_velocity_radps[2] <= 1.0e-5


def test_collision_yaw_impulse_stays_within_camera_continuity_envelope() -> None:
    config = VehicleConfig(max_collision_yaw_rate_radps=0.35)
    world = GamePhysicsWorld(_scene(_track(x_m=5.0, y_m=0.75)), config)
    before = _moving_ego()
    dt_s = 1.0 / 30.0

    resolved, _ = world.step(before, timestamp_us=0, dt_s=dt_s)

    yaw_delta = math.atan2(
        math.sin(resolved.yaw_rad - before.yaw_rad),
        math.cos(resolved.yaw_rad - before.yaw_rad),
    )
    assert resolved.ragdoll_active is True
    assert abs(yaw_delta) <= config.max_collision_yaw_rate_radps * dt_s
    assert abs(resolved.yaw_rate_radps) <= config.max_collision_yaw_rate_radps
    world.close()


def test_held_throttle_cannot_drive_ego_inside_another_vehicle() -> None:
    config = VehicleConfig()
    ego_model = rigid_body_model_from_vehicle_config(config)
    actor_model = rigid_body_model_for_object("Car", (4.0, 1.9, 1.6))
    assert ego_model.vehicle is not None
    assert actor_model.vehicle is not None
    chassis_contact_distance = (
        ego_model.vehicle.chassis_half_extents_m[0]
        + actor_model.vehicle.chassis_half_extents_m[0]
    )
    world = GamePhysicsWorld(_scene(_track(x_m=8.0)), config)
    state = VehicleState(
        x_m=0.0,
        y_m=0.0,
        z_m=0.0,
        yaw_rad=0.0,
        speed_mps=10.0,
        steer_rad=0.0,
        velocity_x_mps=10.0,
        velocity_y_mps=0.0,
    )
    command = DriverCommand(throttle=1.0, manual_control=True)
    actor_slot = world._world._object_slots["car-1"]
    minimum_center_separation = float("inf")

    try:
        for frame_index in range(180):
            state = integrate_vehicle(state, command, 1.0 / 30.0, config)
            state, _ = world.step(state, frame_index * 33_333, 1.0 / 30.0)
            actor_x = float(world._world.state_buffer[actor_slot, 0])
            center_separation = actor_x - state.x_m
            minimum_center_separation = min(
                minimum_center_separation, center_separation
            )
            assert state.x_m < actor_x
    finally:
        world.close()

    assert minimum_center_separation >= chassis_contact_distance - 0.08


def test_recorded_track_applies_force_without_prescribing_actor_pose() -> None:
    track = _moving_track()
    world = GamePhysicsWorld(_scene(track), VehicleConfig())
    parked_ego = VehicleState(
        x_m=-20.0,
        y_m=0.0,
        z_m=0.0,
        yaw_rad=0.0,
        speed_mps=0.0,
        steer_rad=0.0,
    )
    timestamp_us = 5_000_000
    target_x = track.centers_world[0, 0] + (
        (track.centers_world[1, 0] - track.centers_world[0, 0])
        * (timestamp_us - track.timestamps_us[0])
        / (track.timestamps_us[1] - track.timestamps_us[0])
    )
    _, first_samples = world.step(parked_ego, timestamp_us, 1.0 / 30.0)
    _, second_samples = world.step(parked_ego, timestamp_us + 33_333, 1.0 / 30.0)

    first_x = float(first_samples[0][1][0])
    second_x = float(second_samples[0][1][0])
    assert first_samples[0][3] is False
    assert first_x < target_x - 1.0
    assert second_x > first_x

    trajectory = world.build_trajectories(
        np.asarray([timestamp_us, timestamp_us + 33_333], dtype=np.int64),
        [first_samples, second_samples],
    )[0]
    np.testing.assert_allclose(trajectory.translations_world[:, 0], [first_x, second_x])
    world.close()


def test_non_ego_track_drive_is_limited_to_fifteen_mph() -> None:
    world = GamePhysicsWorld(_scene(_fast_moving_track()), VehicleConfig())
    parked_ego = VehicleState(
        x_m=-100.0,
        y_m=0.0,
        z_m=0.0,
        yaw_rad=0.0,
        speed_mps=0.0,
        steer_rad=0.0,
    )
    actor_speeds_mps = []

    try:
        for frame_index in range(180):
            world.step(parked_ego, frame_index * 33_333, 1.0 / 30.0)
            actor_velocity = world._world.body_state("car-fast").linear_velocity_mps
            actor_speeds_mps.append(float(np.linalg.norm(actor_velocity[:2])))
    finally:
        world.close()

    max_drive_speed_mps = 15.0 * 0.44704
    assert max(actor_speeds_mps) <= max_drive_speed_mps + 0.05
    assert actor_speeds_mps[-1] == pytest.approx(max_drive_speed_mps, abs=0.20)


def test_struck_vehicle_ai_waits_until_stopped_for_one_second() -> None:
    track = _track()
    scene_object = GamePhysicsWorld._object_from_track(track, VehicleConfig())
    traffic_ai = TrafficDriverAI()
    traffic_ai.synchronize((scene_object,))
    track_position, track_orientation, track_velocity = scene_object.sample(0)

    moving = BodyState(
        position_m=track_position.copy(),
        orientation_xyzw=track_orientation.copy(),
        linear_velocity_mps=np.asarray([2.0, 0.0, 0.0], dtype=np.float32),
        angular_velocity_radps=np.zeros(3, dtype=np.float32),
    )
    decision = traffic_ai.update(
        scene_object.object_id,
        struck=True,
        body=moving,
        track_position=track_position,
        track_orientation_xyzw=track_orientation,
        track_velocity_mps=track_velocity,
        dt_s=0.25,
    )
    assert decision is not None
    assert decision.drive_enabled is False

    stopped = BodyState(
        position_m=track_position.copy(),
        orientation_xyzw=track_orientation.copy(),
        linear_velocity_mps=np.zeros(3, dtype=np.float32),
        angular_velocity_radps=np.zeros(3, dtype=np.float32),
    )
    for _ in range(2):
        decision = traffic_ai.update(
            scene_object.object_id,
            struck=False,
            body=stopped,
            track_position=track_position,
            track_orientation_xyzw=track_orientation,
            track_velocity_mps=track_velocity,
            dt_s=0.25,
        )
        assert decision is not None
        assert decision.drive_enabled is False

    decision = traffic_ai.update(
        scene_object.object_id,
        struck=False,
        body=moving,
        track_position=track_position,
        track_orientation_xyzw=track_orientation,
        track_velocity_mps=track_velocity,
        dt_s=0.25,
    )
    assert decision is not None
    assert decision.drive_enabled is False

    for _ in range(3):
        decision = traffic_ai.update(
            scene_object.object_id,
            struck=False,
            body=stopped,
            track_position=track_position,
            track_orientation_xyzw=track_orientation,
            track_velocity_mps=track_velocity,
            dt_s=0.25,
        )
        assert decision is not None
        assert decision.drive_enabled is False

    decision = traffic_ai.update(
        scene_object.object_id,
        struck=False,
        body=stopped,
        track_position=track_position,
        track_orientation_xyzw=track_orientation,
        track_velocity_mps=track_velocity,
        dt_s=0.25,
    )
    assert decision is not None
    assert decision.drive_enabled is True
    assert decision.detached_from_track is True

    decision = traffic_ai.update(
        scene_object.object_id,
        struck=False,
        body=stopped,
        track_position=track_position,
        track_orientation_xyzw=track_orientation,
        track_velocity_mps=track_velocity,
        dt_s=0.25,
    )
    assert decision is not None
    assert decision.detached_from_track is False


def test_ego_remains_driveable_after_collision_physics_takes_authority() -> None:
    config = VehicleConfig()
    world = GamePhysicsWorld(_scene(_track()), config)
    state, _ = world.step(_moving_ego(), timestamp_us=0, dt_s=1.0 / 30.0)
    impact_position = np.asarray([state.x_m, state.y_m])
    command = DriverCommand(
        throttle=1.0,
        steer=0.5,
        steer_is_direct=True,
        manual_control=True,
    )

    for frame_index in range(1, 61):
        state = integrate_vehicle(state, command, 1.0 / 30.0, config)
        state, _ = world.step(state, frame_index * 33_333, 1.0 / 30.0)

    assert state.ragdoll_active is False
    assert np.linalg.norm(np.asarray([state.x_m, state.y_m]) - impact_position) > 3.0
    assert abs(state.y_m - impact_position[1]) > 0.5
    world.close()


def test_approaching_truck_triggers_flare_and_allows_reverse_after_impact() -> None:
    config = VehicleConfig()
    world = GamePhysicsWorld(_scene(_track("Truck", x_m=15.0)), config)
    state = VehicleState(
        x_m=0.0,
        y_m=0.0,
        z_m=0.0,
        yaw_rad=0.0,
        speed_mps=10.0,
        steer_rad=0.0,
        velocity_x_mps=10.0,
        velocity_y_mps=0.0,
    )
    forward = DriverCommand(
        throttle=1.0,
        steer_is_direct=True,
        manual_control=True,
    )
    collision_detected = False
    frame_index = 0
    for frame_index in range(90):
        state = integrate_vehicle(state, forward, 1.0 / 30.0, config)
        state, _ = world.step(state, frame_index * 33_333, 1.0 / 30.0)
        collision_detected |= world.last_step_actor_collision
        if state.ragdoll_active:
            break

    impact_x = state.x_m
    reverse = DriverCommand(
        throttle=1.0,
        reverse=True,
        steer_is_direct=True,
        manual_control=True,
    )
    for reverse_frame in range(120):
        state = integrate_vehicle(state, reverse, 1.0 / 30.0, config)
        state, _ = world.step(
            state,
            (frame_index + reverse_frame + 1) * 33_333,
            1.0 / 30.0,
        )
        collision_detected |= world.last_step_actor_collision

    assert collision_detected is True
    assert state.speed_mps < -1.0
    assert state.x_m < impact_x - 1.0
    world.close()


def test_vehicle_instances_have_dimensioned_wheels_and_suspension() -> None:
    car = rigid_body_model_for_object("Car", (4.0, 1.9, 1.6))
    truck = rigid_body_model_for_object("Truck", (8.0, 2.5, 3.2))
    pedestrian = rigid_body_model_for_object("Pedestrian", (0.6, 0.6, 1.8))

    assert car.vehicle is not None
    assert truck.vehicle is not None
    assert len(car.vehicle.suspension_mounts_m) == 4
    assert len(truck.vehicle.suspension_mounts_m) == 4
    assert truck.vehicle.wheel_radius_m > car.vehicle.wheel_radius_m
    assert truck.vehicle.spring_stiffness_n_per_m > (
        car.vehicle.spring_stiffness_n_per_m
    )
    assert pedestrian.vehicle is None


@pytest.mark.parametrize(
    ("object_type", "expected_mass_kg"),
    [
        ("Car", 1_550.0),
        ("Truck", 8_000.0),
        ("Pedestrian", 80.0),
        ("Cyclist", 100.0),
        ("Others", 500.0),
        ("Bus", 12_000.0),
        ("Trailer", 10_000.0),
        ("Motorcycle", 220.0),
    ],
)
def test_object_types_have_category_specific_mass(
    object_type: str, expected_mass_kg: float
) -> None:
    model = rigid_body_model_for_object(object_type, (4.0, 1.9, 1.6))

    assert model.mass_kg == expected_mass_kg


def test_overlapping_and_unknown_object_labels_use_the_right_mass_category() -> None:
    assert canonical_object_type("truck_trailer") == "trailer"
    assert canonical_object_type("motorcycle") == "motorcycle"
    assert canonical_object_type("unclassified_object") == "other"


def test_physx_buffers_retain_native_scene_after_world_is_released() -> None:
    world = PhysXWorld(
        PhysicsObjectGraph(),
        RigidBodyModel(1_550.0, (2.4, 1.0, 0.8)),
        capacity=4,
    )
    native_scene = world._scene
    buffers = (
        world._state_buffer,
        world._track_state_buffer,
        world._id_buffer,
        world._active_buffer,
        world._collision_active_buffer,
        world._detached_buffer,
        world._struck_buffer,
    )

    assert all(buffer.base is native_scene for buffer in buffers)

    del native_scene
    del world
    gc.collect()

    for buffer in buffers:
        buffer[...] = 1
        assert np.all(buffer == 1)


def test_physx_world_applies_incremental_graph_changes_in_stable_buffers() -> None:
    ego_model = RigidBodyModel(1_550.0, (2.4, 1.0, 0.8))
    graph = PhysicsObjectGraph()
    world = PhysXWorld(graph, ego_model, capacity=8)
    state_pointer = world.state_buffer.__array_interface__["data"][0]
    track_state_pointer = world._track_state_buffer.__array_interface__["data"][0]
    active_pointer = world.active_buffer.__array_interface__["data"][0]
    track = _track()
    actor = SceneObject(
        object_id=track.track_id,
        object_type=track.object_type,
        model=rigid_body_model_for_object(track.object_type, track.dimensions_lwh[0]),
        timestamps_us=track.timestamps_us,
        positions_m=track.centers_world,
        orientations_xyzw=track.orientations_xyzw,
    )

    graph.upsert_object(actor)
    graph.upsert_barrier("road-edge", InvisibleBarrier((2.0, -5.0), (2.0, 5.0)))
    world.synchronize(graph)

    assert world.body_count == 2
    assert world.barrier_count == 1
    assert world.state_buffer.__array_interface__["data"][0] == state_pointer
    assert world._track_state_buffer.__array_interface__["data"][0] == (
        track_state_pointer
    )
    assert world.active_buffer.__array_interface__["data"][0] == active_pointer

    graph.remove_object(actor.object_id)
    graph.remove_barrier("road-edge")
    world.synchronize(graph)

    assert world.body_count == 1
    assert world.barrier_count == 0
    assert int(world.active_buffer.sum()) == 1
    assert world.state_buffer.__array_interface__["data"][0] == state_pointer
    world.close()


def test_game_world_reuses_native_track_samples_during_step() -> None:
    world = GamePhysicsWorld(_scene(_track()), VehicleConfig())
    parked_ego = VehicleState(
        x_m=-20.0,
        y_m=0.0,
        z_m=0.0,
        yaw_rad=0.0,
        speed_mps=0.0,
        steer_rad=0.0,
    )

    try:
        with patch.object(
            SceneObject,
            "sample",
            side_effect=AssertionError("track was sampled again in Python"),
        ):
            _, samples = world.step(parked_ego, timestamp_us=0, dt_s=1.0 / 30.0)

        assert len(samples) == 1
    finally:
        world.close()


def test_game_world_recenters_topology_without_reallocating_state_buffers() -> None:
    world = GamePhysicsWorld(
        _scene(_track("Car", x_m=5.0), _track("Truck", x_m=500.0)),
        VehicleConfig(),
    )
    state_pointer = world._world.state_buffer.__array_interface__["data"][0]

    assert [entity.entity_id for entity in world.entities] == ["car-1"]
    assert world.synchronize_window(np.asarray([500.0, 0.0], dtype=np.float32))
    assert [entity.entity_id for entity in world.entities] == ["truck-1"]
    assert world._world.state_buffer.__array_interface__["data"][0] == state_pointer
    assert not world.synchronize_window(np.asarray([501.0, 0.0], dtype=np.float32))
    world.close()


def test_game_world_bounds_physics_topology_around_ego() -> None:
    world = GamePhysicsWorld(
        _scene(
            _track("Car", x_m=95.0),
            _track("Truck", x_m=100.0),
        ),
        VehicleConfig(),
    )

    assert [entity.entity_id for entity in world.entities] == ["car-1"]
    assert world.synchronize_window(np.asarray([32.0, 0.0], dtype=np.float32))
    assert [entity.entity_id for entity in world.entities] == ["car-1", "truck-1"]
    world.close()


def test_physics_topology_keeps_collider_crossing_window_boundary() -> None:
    world = GamePhysicsWorld(_scene(_track("Truck", x_m=97.0)), VehicleConfig())

    assert [entity.entity_id for entity in world.entities] == ["truck-1"]
    world.close()


def test_physics_topology_indexes_sparse_track_segments() -> None:
    track = _track()
    actor = SceneObject(
        object_id=track.track_id,
        object_type=track.object_type,
        model=rigid_body_model_for_object(track.object_type, track.dimensions_lwh[0]),
        timestamps_us=np.asarray([0, 1_000_000], dtype=np.int64),
        positions_m=np.asarray([[-200.0, 0.0, 0.8], [200.0, 0.0, 0.8]]),
        orientations_xyzw=track.orientations_xyzw,
    )

    culled = PhysicsObjectGraph(objects=(actor,)).copy_for_physx(
        np.zeros(2, dtype=np.float32), 10.0
    )

    assert culled.objects == (actor,)


def test_game_world_activates_actor_at_current_track_pose() -> None:
    timestamps_us = np.asarray([0, 2_000_000], dtype=np.int64)
    approaching_track = WorldVehicleBBoxTrack(
        track_id="car-approaching",
        object_type="Car",
        timestamps_us=timestamps_us,
        centers_world=np.asarray(
            [[120.0, 0.0, 0.8], [90.0, 0.0, 0.8]], dtype=np.float32
        ),
        dimensions_lwh=np.asarray([[4.0, 1.9, 1.6], [4.0, 1.9, 1.6]], dtype=np.float32),
        orientations_xyzw=np.asarray(
            [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32
        ),
        max_extrapolation_us=2_000_000.0,
    )
    world = GamePhysicsWorld(_scene(approaching_track), VehicleConfig())

    assert world.entities == ()
    assert world.synchronize_window(
        np.asarray([0.0, 0.0], dtype=np.float32), timestamp_us=2_000_000
    )
    np.testing.assert_array_equal(
        world._world.body_state("car-approaching").position_m,
        np.asarray([90.0, 0.0, 0.8], dtype=np.float32),
    )
    world.close()


def test_game_world_keeps_out_of_window_actor_on_recorded_trajectory() -> None:
    world = GamePhysicsWorld(
        _scene(
            _track("Car", x_m=5.0),
            _track("Truck", x_m=100.0),
        ),
        VehicleConfig(),
    )
    _, samples = world.step(_moving_ego(), timestamp_us=0, dt_s=1.0 / 30.0)

    trajectories = {
        trajectory.entity_id: trajectory
        for trajectory in world.build_trajectories(
            np.asarray([0], dtype=np.int64), [samples]
        )
    }

    assert trajectories["car-1"].is_simulated is True
    assert trajectories["truck-1"].is_simulated is False
    np.testing.assert_array_equal(
        trajectories["truck-1"].translations_world[0],
        np.asarray([100.0, 0.0, 0.8], dtype=np.float32),
    )
    world.close()


def test_actor_collision_can_be_disabled() -> None:
    world = GamePhysicsWorld(
        _scene(_track()), VehicleConfig(actor_collision_enabled=False)
    )

    state, samples = world.step(_moving_ego(), timestamp_us=0, dt_s=1.0 / 30.0)

    assert state.ragdoll_active is False
    assert len(samples) == 1
    assert samples[0][3] is False
    world.close()


def test_actor_collider_stays_enabled_while_hdmap_track_is_visible() -> None:
    track = _track()
    track = WorldVehicleBBoxTrack(
        track_id=track.track_id,
        object_type=track.object_type,
        timestamps_us=track.timestamps_us,
        centers_world=track.centers_world,
        dimensions_lwh=track.dimensions_lwh,
        orientations_xyzw=track.orientations_xyzw,
        max_extrapolation_us=500_000.0,
    )
    world = GamePhysicsWorld(_scene(track), VehicleConfig())

    state, _ = world.step(
        _moving_ego(),
        timestamp_us=1_500_000,
        dt_s=1.0 / 30.0,
    )

    assert state.ragdoll_active is True
    assert world._active_collider_ids == {track.track_id}
    assert world.debug_frame(state).actor_positions_m.shape == (1, 3)
    world.close()


def test_actor_and_hdmap_hold_final_pose_after_track_ends() -> None:
    track = _track()
    track = WorldVehicleBBoxTrack(
        track_id=track.track_id,
        object_type=track.object_type,
        timestamps_us=track.timestamps_us,
        centers_world=track.centers_world,
        dimensions_lwh=track.dimensions_lwh,
        orientations_xyzw=track.orientations_xyzw,
        max_extrapolation_us=500_000.0,
    )
    world = GamePhysicsWorld(_scene(track), VehicleConfig())

    parked_ego = VehicleState(
        x_m=-20.0,
        y_m=0.0,
        z_m=0.0,
        yaw_rad=0.0,
        speed_mps=0.0,
        steer_rad=0.0,
    )
    state, first_samples = world.step(
        parked_ego,
        timestamp_us=1_500_001,
        dt_s=1.0 / 30.0,
    )
    _, second_samples = world.step(
        state,
        timestamp_us=1_533_334,
        dt_s=1.0 / 30.0,
    )

    assert state.ragdoll_active is False
    assert len(first_samples) == 1
    assert len(second_samples) == 1
    assert world._active_collider_ids == {track.track_id}
    debug_frame = world.debug_frame(state)
    assert debug_frame.actor_positions_m.shape == (1, 3)
    np.testing.assert_allclose(
        debug_frame.actor_positions_m[0], track.centers_world[-1], atol=0.05
    )
    trajectory = world.build_trajectories(
        np.asarray([1_500_001, 1_533_334], dtype=np.int64),
        [first_samples, second_samples],
    )[0]
    np.testing.assert_array_equal(trajectory.timestamps_us, [1_500_001, 1_533_334])
    np.testing.assert_allclose(
        trajectory.translations_world,
        np.stack((first_samples[0][1], second_samples[0][1])),
    )
    assert trajectory.detached_from_track is False
    world.close()

    collision_world = GamePhysicsWorld(_scene(track), VehicleConfig())
    collision_state, _ = collision_world.step(
        _moving_ego(),
        timestamp_us=3_000_000,
        dt_s=1.0 / 30.0,
    )
    assert collision_state.ragdoll_active is True
    assert collision_world._active_collider_ids == {track.track_id}
    assert collision_world.debug_frame(collision_state).actor_positions_m.shape == (
        1,
        3,
    )
    collision_world.close()


def test_held_throttle_advances_ego_through_physx_world() -> None:
    config = VehicleConfig()
    world = GamePhysicsWorld(_scene(), config)
    state = VehicleState(
        x_m=0.0,
        y_m=0.0,
        z_m=0.0,
        yaw_rad=0.0,
        speed_mps=0.0,
        steer_rad=0.0,
    )
    command = DriverCommand(throttle=1.0, steer_is_direct=True, manual_control=True)

    for frame_index in range(60):
        state = integrate_vehicle(state, command, 1.0 / 30.0, config)
        state, _ = world.step(state, frame_index * 33_333, 1.0 / 30.0)

    assert state.x_m > 3.5
    assert state.speed_mps > 3.5
    assert state.ragdoll_active is False
    world.close()


def test_held_s_reverses_runtime_ego_through_physx_world() -> None:
    """Exercise the same keyboard, chunking, kinematics, and PhysX path as the app."""
    config = VehicleConfig()
    simulation = EgoVehicleKinematics(
        initial_state=VehicleState(
            x_m=0.0,
            y_m=0.0,
            z_m=0.0,
            yaw_rad=0.0,
            speed_mps=10.0,
            steer_rad=0.0,
        ),
        vehicle_config=config,
        ground_snapper=None,
        initial_timestamp_us=0,
        scene=_scene(),
    )
    keyboard = KeyboardState()
    keyboard.set_key("s", True)
    input_backend = KeyboardInputBackend(keyboard)

    boundary_x = []
    for _ in range(16):
        chunk = simulation.pose_chunk(
            command=input_backend.sample().command,
            chunk_size=8,
            frame_interval_s=1.0 / 30.0,
            extrapolation_offset_s=0.0,
        )
        assert chunk.boundary_state_after_chunk.ragdoll_active is False
        boundary_x.append(chunk.boundary_state_after_chunk.x_m)

    # S applies ordinary throttle in the negative direction through the native
    # PhysX step without a reset, separate braking mode, or forward creep.
    final_speed_mps = simulation.current_state.speed_mps
    final_x_m = boundary_x[-1]
    simulation.close()

    assert final_speed_mps < -4.0
    assert final_x_m < max(boundary_x)


def test_runtime_pose_chunks_keep_ground_anchored_ego_driving_forward() -> None:
    """The app's z=0 rig anchor must not embed the PhysX chassis in ground."""
    config = VehicleConfig()
    simulation = EgoVehicleKinematics(
        initial_state=VehicleState(
            x_m=0.0,
            y_m=0.0,
            z_m=0.0,
            yaw_rad=0.0,
            speed_mps=10.0,
            steer_rad=0.0,
        ),
        vehicle_config=config,
        ground_snapper=None,
        initial_timestamp_us=0,
        scene=_scene(),
    )
    simulation.set_physx_debug_enabled(True)
    keyboard = KeyboardState()
    keyboard.set_key("w", True)
    input_backend = KeyboardInputBackend(keyboard)

    boundary_x = []
    first_chunk = None
    # Eight chunks crosses the speed at which ground-plane friction used to
    # exceed the generic 0.5 m/s impact heuristic and latch ragdoll on.
    for _ in range(8):
        sampled = input_backend.sample()
        chunk = simulation.pose_chunk(
            command=sampled.command,
            chunk_size=8,
            frame_interval_s=1.0 / 30.0,
            extrapolation_offset_s=0.0,
        )
        first_chunk = first_chunk or chunk
        boundary_x.append(simulation.current_state.x_m)
        assert simulation.current_state.ragdoll_active is False

    assert first_chunk is not None
    assert first_chunk.physics_debug_frames[0].ego_position_m[2] == pytest.approx(
        config.aabb_height_m * 0.5,
        abs=1e-3,
    )
    assert boundary_x == sorted(boundary_x)
    assert boundary_x[-1] > 25.0
    assert simulation.current_state.speed_mps > 17.0
    simulation.close()


def test_dense_boundary_samples_are_coalesced_for_physx() -> None:
    points = np.linspace(0.0, 20.0, 401, dtype=np.float32)
    segments = np.stack(
        [
            np.stack([points[:-1], np.zeros(400), np.zeros(400)], axis=1),
            np.stack([points[1:], np.zeros(400), np.zeros(400)], axis=1),
        ],
        axis=1,
    )

    simplified = _simplify_barrier_segments(segments)

    assert len(simplified) <= 11
    np.testing.assert_allclose(simplified[0][0], [0.0, 0.0])
    np.testing.assert_allclose(simplified[-1][1], [20.0, 0.0])


def test_road_boundary_behaves_as_solid_wall() -> None:
    boundary = WorldLineSegments(
        segments_world=np.asarray(
            [[[2.0, -5.0, 0.0], [2.0, 5.0, 0.0]]], dtype=np.float32
        ),
        color_rgba=(1.0, 1.0, 1.0, 1.0),
        width_px=2.0,
        layer_name="road_boundaries",
    )
    world = GamePhysicsWorld(_scene(line_layers=(boundary,)), VehicleConfig())
    state = VehicleState(
        x_m=1.5,
        y_m=0.0,
        z_m=0.0,
        yaw_rad=0.0,
        speed_mps=8.0,
        steer_rad=0.0,
        velocity_x_mps=8.0,
        velocity_y_mps=0.0,
    )

    resolved = state
    visual_flare_triggered = False
    for frame_index in range(6):
        resolved, _ = world.step(
            resolved,
            timestamp_us=frame_index * 33_333,
            dt_s=1.0 / 30.0,
        )
        visual_flare_triggered |= world.last_step_actor_collision

    assert resolved.x_m < 2.0
    assert resolved.velocity_x_mps is not None
    assert resolved.velocity_x_mps < 0.0
    assert resolved.ragdoll_active is True
    assert visual_flare_triggered is True
    world.close()


def test_physx_debug_view_packs_active_colliders_and_invisible_walls_for_ludus() -> (
    None
):
    boundary = WorldLineSegments(
        segments_world=np.asarray(
            [[[-10.0, -4.0, 0.0], [20.0, -4.0, 0.0]]], dtype=np.float32
        ),
        color_rgba=(1.0, 1.0, 1.0, 1.0),
        width_px=2.0,
        layer_name="road_boundaries",
    )
    world = GamePhysicsWorld(
        _scene(
            _track(),
            _track("Truck", x_m=250.0),
            line_layers=(boundary,),
        ),
        VehicleConfig(),
    )
    state, _ = world.step(_moving_ego(), timestamp_us=0, dt_s=1.0 / 30.0)

    snapshot = world.debug_frame(state)
    pool = build_physx_debug_cube_pool(
        (snapshot,), np.asarray([0], dtype=np.int64), device=torch.device("cpu")
    )

    assert snapshot.actor_positions_m.shape == (1, 3)
    assert snapshot.barrier_segments_xy_m.shape == (1, 2, 2)
    assert snapshot.barrier_thicknesses_m.tolist() == pytest.approx([0.3])
    assert snapshot.barrier_heights_m.tolist() == pytest.approx([3.0])
    assert pool.translations.shape == (2, 3)
    assert pool.quaternions.shape == (2, 4)
    assert pool.scales.shape == (2, 3)
    np.testing.assert_allclose(
        pool.translations[0].numpy(), snapshot.actor_positions_m[0]
    )
    np.testing.assert_allclose(pool.scales[0].numpy(), snapshot.actor_dimensions_lwh[0])
    assert pool.render_flags == 0

    lazy_debug = object()
    frame = PresentedFrame(
        timestamp_us=0,
        rgb_host_uint8=np.zeros((180, 320, 3), dtype=np.uint8),
        depth_host_f32=None,
        physx_debug=snapshot,
        physx_rgb_host_uint8=lazy_debug,
    )
    first = select_presented_rgb(frame, "physx", width=320, height=180)
    second = select_presented_rgb(frame, "physx", width=320, height=180)
    assert first is lazy_debug
    assert second is lazy_debug
    world.close()


def _debug_snapshot_at(
    positions_m: list[tuple[float, float, float]],
) -> PhysicsDebugFrame:
    count = len(positions_m)
    return PhysicsDebugFrame(
        ego_position_m=np.asarray([0.0, 0.0, 0.8], dtype=np.float32),
        ego_orientation_xyzw=np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        ego_dimensions_lwh=np.asarray([4.8, 2.0, 1.6], dtype=np.float32),
        actor_positions_m=np.asarray(positions_m, dtype=np.float32).reshape(count, 3),
        actor_orientations_xyzw=np.tile(
            np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (count, 1)
        ),
        actor_dimensions_lwh=np.tile(
            np.asarray([4.0, 3.0, 3.0], dtype=np.float32), (count, 1)
        ),
        barrier_segments_xy_m=np.empty((0, 2, 2), dtype=np.float32),
        barrier_thicknesses_m=np.empty((0,), dtype=np.float32),
        barrier_heights_m=np.empty((0,), dtype=np.float32),
    )


def test_physx_debug_pool_reuses_one_track_per_collider_across_frames() -> None:
    first = _debug_snapshot_at([(12.0, 0.0, 1.5)])
    second = _debug_snapshot_at([(24.0, 2.0, 1.5), (30.0, -2.0, 1.5)])
    pool = build_physx_debug_cube_pool(
        (first, second),
        np.asarray([10, 20], dtype=np.int64),
        device=torch.device("cpu"),
    )

    tracks = pool.translations.reshape(-1, 2, 3).numpy()
    assert pool.scales.shape[0] == 2
    np.testing.assert_allclose(tracks[0, 0], first.actor_positions_m[0])
    np.testing.assert_allclose(tracks[0, 1], second.actor_positions_m[0])
    assert np.linalg.norm(tracks[1, 0]) > 100_000.0
    np.testing.assert_allclose(tracks[1, 1], second.actor_positions_m[1])


def test_physx_debug_culling_keeps_colliders_crossing_view_boundary() -> None:
    world = GamePhysicsWorld.__new__(GamePhysicsWorld)
    world._ego_model = RigidBodyModel(mass_kg=1.0, half_extents_m=(2.0, 1.0, 1.0))
    world._world = SimpleNamespace(
        collider_state_arrays=lambda: (
            ("forward-edge", "outside", "lateral-edge"),
            np.asarray(
                [[127.0, 0.0, 1.0], [129.0, 0.0, 1.0], [0.0, 101.0, 1.0]],
                dtype=np.float32,
            ),
            np.tile(np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (3, 1)),
            np.asarray(
                [[6.0, 2.0, 2.0], [6.0, 2.0, 2.0], [4.0, 4.0, 2.0]],
                dtype=np.float32,
            ),
        )
    )
    world._debug_barrier_ids = ()
    world._debug_barrier_segments = np.empty((0, 2, 2), dtype=np.float32)
    world._debug_barrier_thicknesses = np.empty((0,), dtype=np.float32)
    world._debug_barrier_heights = np.empty((0,), dtype=np.float32)

    snapshot = world.debug_frame(
        VehicleState(
            x_m=0.0,
            y_m=0.0,
            z_m=0.0,
            yaw_rad=0.0,
            speed_mps=0.0,
            steer_rad=0.0,
        )
    )

    assert snapshot.actor_ids == ("forward-edge", "lateral-edge")


def test_track_control_bridge_batches_only_changed_actor_state() -> None:
    calls: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    world = PhysXWorld.__new__(PhysXWorld)
    world._scene = SimpleNamespace(
        set_body_track_controls=lambda ids, drive, detached: calls.append(
            (ids.copy(), drive.copy(), detached.copy())
        )
    )
    world._object_native_ids = {"car-1": 11, "car-2": 22}
    world._track_drive_enabled = {"car-1": True, "car-2": True}
    world._objects = {
        "car-1": SimpleNamespace(detached=False),
        "car-2": SimpleNamespace(detached=False),
    }

    controls = (("car-1", False, True), ("car-2", True, False))
    world.apply_track_controls(controls)
    world.apply_track_controls(controls)

    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], np.asarray([11], dtype=np.int64))
    np.testing.assert_array_equal(calls[0][1], np.asarray([0], dtype=np.uint8))
    np.testing.assert_array_equal(calls[0][2], np.asarray([1], dtype=np.uint8))


def test_physx_debug_view_rejects_missing_ludus_frame() -> None:
    frame = PresentedFrame(
        timestamp_us=0,
        rgb_host_uint8=np.zeros((4, 4, 3), dtype=np.uint8),
        depth_host_f32=None,
        physx_debug=_debug_snapshot_at([]),
    )

    with pytest.raises(RuntimeError, match="Ludus-rendered lazy debug frame"):
        select_presented_rgb(frame, "physx", width=4, height=4)


def test_acceleration_and_cornering_drive_suspension() -> None:
    config = VehicleConfig()
    state = VehicleState(
        x_m=0.0,
        y_m=0.0,
        z_m=0.0,
        yaw_rad=0.0,
        speed_mps=4.0,
        steer_rad=0.0,
    )

    accelerated = integrate_vehicle(
        state,
        DriverCommand(throttle=1.0, steer=1.0),
        dt_s=0.1,
        vehicle=config,
    )

    assert accelerated.suspension_pitch_rad < 0.0
    assert accelerated.suspension_roll_rad < 0.0
    assert abs(accelerated.suspension_pitch_rad) <= config.max_body_pitch_rad
    assert abs(accelerated.suspension_roll_rad) <= config.max_body_roll_rad


def test_high_speed_steering_stays_inside_lateral_grip_envelope() -> None:
    config = VehicleConfig()
    state = VehicleState(
        x_m=0.0,
        y_m=0.0,
        z_m=0.0,
        yaw_rad=0.0,
        speed_mps=config.max_speed_mps,
        steer_rad=config.max_steer_rad,
    )

    advanced = integrate_vehicle(
        state, DriverCommand(), dt_s=1.0 / 30.0, vehicle=config
    )

    lateral_accel = abs(advanced.speed_mps * advanced.yaw_rate_radps)
    assert lateral_accel <= config.max_lateral_accel_mps2 + 1e-6


def test_suspension_cornering_response_is_smooth_and_settles() -> None:
    config = VehicleConfig()
    state = VehicleState(
        x_m=0.0,
        y_m=0.0,
        z_m=0.0,
        yaw_rad=0.0,
        speed_mps=config.max_speed_mps,
        steer_rad=config.max_steer_rad,
    )
    dt_s = 1.0 / 30.0
    cornering_roll: list[float] = []
    for _ in range(90):
        state = integrate_vehicle(state, DriverCommand(), dt_s=dt_s, vehicle=config)
        cornering_roll.append(state.suspension_roll_rad)

    release_roll: list[float] = []
    for _ in range(120):
        state = integrate_vehicle(
            state,
            DriverCommand(steer_is_direct=True),
            dt_s=dt_s,
            vehicle=config,
        )
        release_roll.append(state.suspension_roll_rad)

    # The grip envelope keeps the visual lean modest instead of pinning the
    # suspension at its hard stop, and the spring-damper returns without a snap.
    assert max(abs(value) for value in cornering_roll) < config.max_body_roll_rad * 0.75
    assert abs(release_roll[0]) < abs(cornering_roll[-1])
    assert abs(release_roll[-1]) < 1e-4


def test_entity_export_is_json_compatible_component_data() -> None:
    entity = GameEntity(
        entity_id="car-7",
        object_type="Car",
        transform=TransformComponent(
            np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
            np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        ),
        rigid_body=RigidBodyComponent(
            mass_kg=1_500.0,
            linear_velocity_mps=np.asarray([4.0, 0.0, 0.0], dtype=np.float32),
            angular_velocity_radps=np.zeros(3, dtype=np.float32),
        ),
        collider=BoxColliderComponent((2.2, 0.9, 0.7)),
    )

    exported = entity.to_game_engine_dict()

    assert exported["entity_id"] == "car-7"
    assert exported["components"]["rigid_body"]["mass_kg"] == 1_500.0
    assert exported["components"]["transform"]["position_m"] == [1.0, 2.0, 3.0]


def test_ego_export_includes_drivetrain_and_suspension_components() -> None:
    config = VehicleConfig(mass_kg=1_700.0)
    exported = game_entity_from_vehicle_state(
        _moving_ego(), config
    ).to_game_engine_dict()

    assert exported["components"]["rigid_body"]["mass_kg"] == 1_700.0
    assert exported["components"]["vehicle_dynamics"][
        "max_engine_force_n"
    ] == pytest.approx(1_700.0 * config.max_accel_mps2)
    assert (
        exported["components"]["vehicle_dynamics"]["max_lateral_accel_mps2"]
        == config.max_lateral_accel_mps2
    )
    assert (
        exported["components"]["suspension"]["travel_m"] == config.suspension_travel_m
    )


class _FakeLudusContext:
    def __init__(self) -> None:
        self.clear_count = 0
        self.replace_count = 0
        self.update_count = 0
        self.uploaded_scene: TimestampedScene | None = None

    def clear_scenes(self) -> None:
        self.clear_count += 1

    def upload_scene(self, scene: TimestampedScene) -> int:
        self.uploaded_scene = scene
        return 17

    def replace_scene(self, scene_id: int, scene: TimestampedScene) -> int:
        self.replace_count += 1
        self.uploaded_scene = scene
        return scene_id

    def update_cube_pool(
        self, scene_id: int, prim_type_id: int, pool: CubePool
    ) -> bool:
        self.update_count += 1
        assert scene_id == 3
        assert prim_type_id == PRIM_OBSTACLE
        return True

    def update_cube_pool_at_index(
        self, scene_id: int, pool_index: int, pool: CubePool
    ) -> bool:
        self.update_count += 1
        assert scene_id == 3
        assert pool_index >= 0
        assert pool.prim_type_id == PRIM_OBSTACLE
        return True


def _cube_pool(prim_type_id: int) -> CubePool:
    return CubePool(
        timestamps_us=torch.tensor([0], dtype=torch.int64),
        cube_ts_prefix_sum=torch.tensor([1], dtype=torch.int32),
        track_timestamps_us=torch.tensor([0], dtype=torch.int64),
        translations=torch.zeros((1, 3), dtype=torch.float32),
        quaternions=torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
        scales=torch.ones((1, 3), dtype=torch.float32),
        colors=torch.ones((1, 6), dtype=torch.float32),
        prim_type_id=prim_type_id,
    )


def test_ludus_replacement_removes_recorded_actor_without_dropping_static_pools() -> (
    None
):
    context = _FakeLudusContext()
    renderer = object.__new__(_LudusConditionRasterizerImpl)
    renderer._device = torch.device("cpu")
    renderer.ctx = context
    static_pool = _cube_pool(999)
    renderer._base_timestamped_scene = TimestampedScene(
        polyline_pools=[],
        polygon_pools=[],
        cube_pools=[_cube_pool(PRIM_OBSTACLE), static_pool],
    )
    renderer._scene_id = 3
    renderer._dynamic_scene_initialized = False
    actor = DynamicActorTrajectory(
        entity_id="car-1",
        object_type="Car",
        timestamps_us=np.asarray([10, 20], dtype=np.int64),
        translations_world=np.asarray(
            [[1.0, 2.0, 0.8], [1.5, 2.0, 0.8]], dtype=np.float32
        ),
        orientations_xyzw=np.asarray(
            [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32
        ),
        dimensions_lwh=np.asarray([4.0, 2.0, 1.6], dtype=np.float32),
        detached_from_track=True,
        is_simulated=True,
    )
    recorded_actor = DynamicActorTrajectory(
        entity_id="car-recorded",
        object_type="Car",
        timestamps_us=actor.timestamps_us,
        translations_world=actor.translations_world
        + np.asarray([20.0, 0.0, 0.0], dtype=np.float32),
        orientations_xyzw=actor.orientations_xyzw,
        dimensions_lwh=actor.dimensions_lwh,
    )

    renderer._replace_dynamic_actor_scene((recorded_actor, actor))

    assert context.clear_count == 0
    assert context.replace_count == 1
    assert renderer._scene_id == 3
    assert context.uploaded_scene is not None
    assert static_pool in context.uploaded_scene.cube_pools
    obstacle_pools = [
        pool
        for pool in context.uploaded_scene.cube_pools
        if pool.prim_type_id == PRIM_OBSTACLE
    ]
    assert len(obstacle_pools) == 2
    np.testing.assert_allclose(
        obstacle_pools[0].translations.numpy(), recorded_actor.translations_world
    )
    np.testing.assert_allclose(
        obstacle_pools[1].translations.numpy(), actor.translations_world
    )

    renderer._replace_dynamic_actor_scene((recorded_actor, actor))

    assert context.replace_count == 1
    assert context.update_count == 1

    renderer._replace_dynamic_actor_scene(())

    assert context.replace_count == 2
    assert context.uploaded_scene is not None
    assert all(
        pool.prim_type_id != PRIM_OBSTACLE for pool in context.uploaded_scene.cube_pools
    )


def test_ludus_cube_pool_update_reuses_packed_storage() -> None:
    context = object.__new__(LudusCudaTimestampedContext)
    packed_floats = torch.zeros(23, dtype=torch.float32)
    context._scenes = [
        {
            "timestamps": torch.zeros(4, dtype=torch.int64),
            "int32": torch.zeros(1, dtype=torch.int32),
            "floats": packed_floats,
            "cube_pools": torch.zeros((1, 16), dtype=torch.int32),
            "cube_pool_metadata": [
                {
                    "prim_type_id": PRIM_OBSTACLE,
                    "n_cubes": 1,
                    "n_global_ts": 2,
                    "n_track_poses": 2,
                    "timestamp_offset": 0,
                    "track_timestamp_offset": 2,
                    "int32_offset": 0,
                    "translation_offset": 0,
                    "quaternion_offset": 6,
                    "scale_offset": 14,
                    "color_offset": 17,
                }
            ],
        }
    ]
    pool = CubePool(
        timestamps_us=torch.tensor([10, 20], dtype=torch.int64),
        cube_ts_prefix_sum=torch.tensor([2], dtype=torch.int32),
        track_timestamps_us=torch.tensor([10, 20], dtype=torch.int64),
        translations=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        quaternions=torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]),
        scales=torch.tensor([[4.0, 2.0, 1.6]]),
        colors=torch.ones((1, 6), dtype=torch.float32),
        prim_type_id=PRIM_OBSTACLE,
        render_flags=7,
    )
    storage_pointer = packed_floats.data_ptr()

    assert context.update_cube_pool(0, PRIM_OBSTACLE, pool) is True
    assert context._scenes[0]["floats"].data_ptr() == storage_pointer
    np.testing.assert_allclose(
        packed_floats[:6].numpy(), pool.translations.numpy().reshape(-1)
    )
    assert int(context._scenes[0]["cube_pools"][0, 11]) == 7
