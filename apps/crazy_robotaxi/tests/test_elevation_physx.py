# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native PhysX driving checks for the compiled elevated Raceway.

These manual tests build the pinned CPU PhysX SDK on first use. Driving checks
measure the physical pose without snapping; curb checks use the full simulation
and inspect the physical pose before render alignment.
"""

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from crazy_robotaxi.driving_script import load_driving_script
from crazy_robotaxi.dynamics import TaxiVehicleConfig, integrate_taxi_vehicle
from crazy_robotaxi.factory import _build_ground_snapper, _integrate_taxi_vehicle
from crazy_robotaxi.physics import TaxiPhysicsWorld, step_taxi_physics_world
from crazy_robotaxi.rules import TaxiGameConfig
from crazy_robotaxi.scene import load_scene_data
from ludus_renderer import prepare_physx
from omnidreams_game_engine.config import RasterConfig
from omnidreams_game_engine.game_map import compile_game_map
from omnidreams_game_engine.math3d import rig_pose_from_vehicle_state
from omnidreams_game_engine.scene import load_scene_bundle
from omnidreams_game_engine.simulation.ego_vehicle_kinematics import (
    EgoVehicleKinematics,
)
from omnidreams_game_engine.simulation.ground_snap import GroundSnapper
from omnidreams_game_engine.types import DriverCommand, SceneDefinition, VehicleState

pytestmark = pytest.mark.manual


@pytest.fixture(scope="module", autouse=True)
def native_physx() -> None:
    prepare_physx()


@pytest.fixture(scope="module")
def raceway(tmp_path_factory: pytest.TempPathFactory) -> SceneDefinition:
    source = (
        Path(__file__).parents[1]
        / "crazy_robotaxi/maps/flashdreams_raceway.robotaxi.yaml"
    )
    compiled = compile_game_map(
        source, cache_root=tmp_path_factory.mktemp("elevated-raceway")
    )
    return load_scene_bundle(
        compiled.archive_path,
        "camera_front_wide_120fov",
        RasterConfig(width=64, height=32),
    )


@pytest.mark.parametrize(
    ("lane_id", "expected_change"),
    [
        ("back_straight:lane:2", 1),
        ("west_straight:lane:2", -1),
        ("infield_straight:lane:2", 0),
    ],
)
def test_taxi_drives_authored_grade_without_ground_snap(
    raceway: SceneDefinition, lane_id: str, expected_change: int
) -> None:
    assert raceway.game_map is not None
    lane = next(lane for lane in raceway.game_map.lanes if lane.lane_id == lane_id)
    points = lane.centerline_world
    direction = points[6] - points[5]
    yaw = math.atan2(float(direction[1]), float(direction[0]))
    pitch = -math.atan2(float(direction[2]), float(np.linalg.norm(direction[:2])))
    state = VehicleState(
        float(points[5, 0]),
        float(points[5, 1]),
        float(points[5, 2]),
        yaw,
        0,
        0,
        pitch_rad=pitch,
    )
    start = np.asarray([state.x_m, state.y_m])
    forward = direction[:2] / np.linalg.norm(direction[:2])
    distance = float((points[-6, :2] - start) @ forward)
    assert raceway.ground_mesh_vertices is not None
    assert raceway.ground_mesh_faces is not None
    sampler = GroundSnapper(raceway.ground_mesh_vertices, raceway.ground_mesh_faces)
    vehicle = TaxiVehicleConfig()
    world = TaxiPhysicsWorld(
        raceway,
        vehicle,
        curb_segments_world=load_scene_data(raceway).curb_segments_world,
        ground_snapper=sampler,
    )
    start_height = state.z_m
    maximum_height_error = 0.0
    try:
        for tick in range(2000):
            world.synchronize_window(
                np.asarray([state.x_m, state.y_m], dtype=np.float32)
            )
            command = DriverCommand(throttle=1 if state.speed_mps < 10 else 0)
            intent = integrate_taxi_vehicle(state, command, 1 / 30, vehicle)
            state, _ = world.step_with_command(intent, command, tick * 33333, 1 / 30)
            surface = sampler.ground_surface_at(state.x_m, state.y_m, state.z_m)
            assert surface is not None
            error = abs(state.z_m - surface[0])
            maximum_height_error = max(maximum_height_error, error)
            assert error < 0.10, (tick, state, surface[0])
            assert not state.ragdoll_active, (tick, state)
            assert not world.last_step_static_barrier_collision
            progress = float((np.asarray([state.x_m, state.y_m]) - start) @ forward)
            if progress >= distance:
                break
        else:
            pytest.fail(f"Taxi did not complete {lane_id}: {state}")
        change = state.z_m - start_height
        if expected_change:
            assert expected_change * change > 3
        else:
            assert abs(change) < 0.10
        print(
            f"{lane_id}: drove {progress:.1f} m, height change {change:.2f} m, "
            f"maximum ground error {maximum_height_error:.3f} m"
        )
    finally:
        world.close()


@pytest.mark.parametrize(
    "lane_id",
    [
        "back_straight:lane:2",
        "west_straight:lane:2",
        "infield_straight:lane:2",
    ],
)
@pytest.mark.parametrize("speed_mps", [10.0, 25.0])
def test_curb_impacts_keep_the_physical_taxi_above_the_road(
    raceway: SceneDefinition, lane_id: str, speed_mps: float
) -> None:
    assert raceway.game_map is not None
    lane = next(lane for lane in raceway.game_map.lanes if lane.lane_id == lane_id)
    points = lane.centerline_world
    midpoint = len(points) // 2
    direction = points[midpoint + 1] - points[midpoint]
    yaw = math.atan2(float(direction[1]), float(direction[0])) + 0.55
    pitch = -math.atan2(float(direction[2]), float(np.linalg.norm(direction[:2])))
    state = VehicleState(
        float(points[midpoint, 0]),
        float(points[midpoint, 1]),
        float(points[midpoint, 2]),
        yaw,
        speed_mps,
        0,
        pitch_rad=pitch,
    )
    vehicle = TaxiVehicleConfig()
    sampler = _build_ground_snapper(raceway, TaxiGameConfig())
    assert sampler is not None
    curbs = load_scene_data(raceway).curb_segments_world
    simulation = EgoVehicleKinematics(
        state,
        vehicle,
        sampler,
        0,
        scene=raceway,
        integrate_fn=_integrate_taxi_vehicle,
        physics_world_factory=lambda scene, config: TaxiPhysicsWorld(
            scene, vehicle, curb_segments_world=curbs, ground_snapper=sampler
        ),
        physics_step_fn=step_taxi_physics_world,
    )
    world = simulation._physics_world
    assert isinstance(world, TaxiPhysicsWorld)
    world.synchronize_window(np.asarray([state.x_m, state.y_m], dtype=np.float32))
    hit_curb = False
    try:
        for tick in range(300):
            simulation.pose_chunk([DriverCommand(throttle=0.25)], 1, 1 / 30, 0)
            hit_curb |= world.last_step_static_barrier_collision
            contact = world._last_contact_resolved_state
            assert contact is not None
            surface = sampler.ground_surface_at(contact.x_m, contact.y_m, contact.z_m)
            assert surface is not None
            # Allow suspension compression while detecting chassis fall-through.
            assert contact.z_m >= surface[0] - 0.25, (tick, contact, surface[0])
            camera = (
                rig_pose_from_vehicle_state(simulation.current_state)
                @ raceway.selected_camera.sensor_to_rig_flu
            )[:3, 3]
            camera_ground = sampler.ground_surface_at(*map(float, camera))
            assert camera_ground is not None
            assert camera[2] >= camera_ground[0] + 0.35, (tick, camera, camera_ground)
        assert hit_curb
    finally:
        simulation.close()


@pytest.mark.parametrize("speed_mps", [0.0, 25.0])
def test_flat_starting_straight_has_no_false_suspension_motion(
    raceway: SceneDefinition, speed_mps: float
) -> None:
    state = VehicleState(
        float(raceway.initial_rig_to_world[0, 3]),
        float(raceway.initial_rig_to_world[1, 3]),
        float(raceway.initial_rig_to_world[2, 3]),
        raceway.initial_yaw_rad,
        speed_mps,
        0,
    )
    vehicle = TaxiVehicleConfig()
    sampler = _build_ground_snapper(raceway, TaxiGameConfig())
    assert sampler is not None
    curbs = load_scene_data(raceway).curb_segments_world
    simulation = EgoVehicleKinematics(
        state,
        vehicle,
        sampler,
        0,
        scene=raceway,
        integrate_fn=_integrate_taxi_vehicle,
        physics_world_factory=lambda scene, config: TaxiPhysicsWorld(
            scene, vehicle, curb_segments_world=curbs, ground_snapper=sampler
        ),
        physics_step_fn=step_taxi_physics_world,
        include_initial_state_in_first_chunk=True,
    )
    assert raceway.game_map is not None
    world = simulation._physics_world
    assert isinstance(world, TaxiPhysicsWorld)
    lane = next(
        lane
        for lane in raceway.game_map.lanes
        if lane.lane_id == "main_straight:lane:2"
    )
    finish_x = float(lane.centerline_world[-1, 0]) - 5
    command = DriverCommand(throttle=1, manual_control=True)
    try:
        simulation.pose_chunk([command], 1, 1 / 30, 0)
        for tick in range(240):
            before = simulation.current_state
            simulation.pose_chunk([command], 1, 1 / 30, 0)
            contact = world._last_contact_resolved_state
            assert contact is not None
            assert not world.last_step_static_barrier_collision
            assert not contact.ragdoll_active
            assert abs(contact.z_m) < 0.01, (tick, contact)
            assert contact.velocity_z_mps is not None
            assert abs(contact.velocity_z_mps) < 0.015, (tick, contact)
            assert contact.speed_mps > before.speed_mps, (tick, before, contact)
            if contact.x_m >= finish_x:
                break
        else:
            pytest.fail("Taxi did not reach the end of the starting straight")
    finally:
        simulation.close()


def test_fast_lap_keeps_the_taxi_supported_without_road_jolts(
    raceway: SceneDefinition,
) -> None:
    assert raceway.game_map is not None
    lanes = {lane.lane_id: lane for lane in raceway.game_map.lanes}
    lane = lanes["main_straight:lane:2"]
    sequence = [lane]
    while lane.successor_ids[0] != sequence[0].lane_id:
        lane = lanes[lane.successor_ids[0]]
        sequence.append(lane)
    points = np.concatenate([lane.centerline_world[:-1] for lane in sequence])
    lengths = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    distances = np.concatenate(([0], np.cumsum(lengths)))
    state = VehicleState(
        float(points[0, 0]), float(points[0, 1]), float(points[0, 2]), 0, 10, 0
    )
    vehicle = TaxiVehicleConfig()
    sampler = _build_ground_snapper(raceway, TaxiGameConfig())
    assert sampler is not None
    curbs = load_scene_data(raceway).curb_segments_world
    simulation = EgoVehicleKinematics(
        state,
        vehicle,
        sampler,
        0,
        scene=raceway,
        integrate_fn=_integrate_taxi_vehicle,
        physics_world_factory=lambda scene, config: TaxiPhysicsWorld(
            scene, vehicle, curb_segments_world=curbs, ground_snapper=sampler
        ),
        physics_step_fn=step_taxi_physics_world,
    )
    world = simulation._physics_world
    assert isinstance(world, TaxiPhysicsWorld)
    index = 0
    try:
        for tick in range(6000):
            state = simulation.current_state
            candidates = np.arange(index, min(index + 30, len(points)))
            xy = np.asarray([state.x_m, state.y_m])
            index = int(
                candidates[
                    np.argmin(np.linalg.norm(points[candidates, :2] - xy, axis=1))
                ]
            )
            ahead = min(
                len(points) - 1, int(np.searchsorted(distances, distances[index] + 16))
            )
            delta = points[ahead, :2] - xy
            angle = math.atan2(float(delta[1]), float(delta[0])) - state.yaw_rad
            steer = (
                math.atan2(
                    2 * vehicle.wheel_base_m * math.sin(angle),
                    max(1, np.linalg.norm(delta)),
                )
                / vehicle.max_steer_rad
            )
            command = DriverCommand(
                throttle=1 if state.speed_mps < 25 else 0,
                brake=0.3 if state.speed_mps > 26 else 0,
                steer=steer,
                steer_is_direct=True,
                manual_control=True,
            )
            simulation.pose_chunk([command], 1, 1 / 30, 0)
            contact = world._last_contact_resolved_state
            assert contact is not None
            surface = sampler.ground_surface_at(contact.x_m, contact.y_m, contact.z_m)
            assert surface is not None
            assert abs(contact.z_m - surface[0]) < 0.08, (tick, contact, surface[0])
            normal_velocity = (
                np.asarray(
                    [
                        contact.velocity_x_mps,
                        contact.velocity_y_mps,
                        contact.velocity_z_mps,
                    ]
                )
                @ surface[1]
            )
            assert abs(normal_velocity) < 0.25, (tick, contact, normal_velocity)
            assert not contact.ragdoll_active, (tick, contact)
            assert not world.last_step_static_barrier_collision
            if index >= len(points) - 3:
                break
        else:
            pytest.fail(f"Taxi did not complete the circuit: {index}/{len(points)}")
    finally:
        simulation.close()


def test_taxi_heading_is_unaffected_by_a_curb_above_it(
    raceway: SceneDefinition,
) -> None:
    vertices = np.asarray(
        [[-20, -20, 0], [20, -20, 0], [20, 20, 0], [-20, 20, 0]], dtype=np.float32
    )
    scene = replace(
        raceway,
        game_map=None,
        line_layers=(),
        polygon_layers=(),
        initial_rig_to_world=np.eye(4),
        ground_mesh_vertices=vertices,
        ground_mesh_faces=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
    )
    curbs = np.asarray([[[-10, 0, 8], [10, 0, 8]]], dtype=np.float32)
    trajectories = []
    for segments in (np.empty((0, 2, 3), dtype=np.float32), curbs):
        world = TaxiPhysicsWorld(
            scene, TaxiVehicleConfig(), curb_segments_world=segments
        )
        state = VehicleState(0, 0.2, 0, 0.7, 0, 0)
        samples = []
        try:
            for tick in range(30):
                state, _ = world.step_with_command(
                    state, DriverCommand(), tick * 33333, 1 / 30
                )
                assert not world.last_step_static_barrier_collision
                contact = world._last_contact_resolved_state
                assert contact is not None
                samples.append(
                    [
                        state.x_m,
                        state.y_m,
                        state.z_m,
                        contact.yaw_rad,
                        state.ragdoll_active,
                    ]
                )
            if len(segments):
                heights = world.debug_frame(state).barrier_base_heights_m
                assert heights is not None
                np.testing.assert_allclose(heights, 8)
            trajectories.append(samples)
        finally:
            world.close()
    np.testing.assert_allclose(trajectories[1], trajectories[0], atol=1e-5)


def test_headless_curb_script_does_not_roll_the_taxi_over(tmp_path: Path) -> None:
    tests = Path(__file__).parent
    compiled = compile_game_map(
        tests / "maps/debug_flat.robotaxi.yaml", cache_root=tmp_path
    )
    scene = load_scene_bundle(
        compiled.archive_path,
        "camera_front_wide_120fov",
        RasterConfig(width=64, height=32),
    )
    vehicle = TaxiVehicleConfig()
    sampler = _build_ground_snapper(scene, TaxiGameConfig())
    assert sampler is not None
    curbs = load_scene_data(scene).curb_segments_world
    simulation = EgoVehicleKinematics(
        VehicleState(
            float(scene.initial_rig_to_world[0, 3]),
            float(scene.initial_rig_to_world[1, 3]),
            float(scene.initial_rig_to_world[2, 3]),
            scene.initial_yaw_rad,
            0,
            0,
        ),
        vehicle,
        sampler,
        scene.initial_timestamp_us,
        scene=scene,
        integrate_fn=_integrate_taxi_vehicle,
        physics_world_factory=lambda scene, config: TaxiPhysicsWorld(
            scene, vehicle, curb_segments_world=curbs, ground_snapper=sampler
        ),
        physics_step_fn=step_taxi_physics_world,
        include_initial_state_in_first_chunk=True,
    )
    script = load_driving_script(tests / "scenarios/curb_impact.yaml")
    commands, _ = script.batch(0, script.frame_count)
    hit_curb = recovered = False
    try:
        for tick, command in enumerate(commands):
            simulation.pose_chunk([command], 1, 1 / 30, 0)
            world = simulation._physics_world
            assert isinstance(world, TaxiPhysicsWorld)
            hit_curb |= world.last_step_static_barrier_collision
            contact = world._last_contact_resolved_state or simulation.current_state
            surface = sampler.ground_surface_at(contact.x_m, contact.y_m, contact.z_m)
            assert surface is not None
            assert abs(contact.z_m - surface[0]) < 0.15, (tick, contact)
            assert abs(contact.roll_rad) < 0.5, (tick, contact)
            recovered |= hit_curb and not contact.ragdoll_active
        assert hit_curb and recovered
    finally:
        simulation.close()
