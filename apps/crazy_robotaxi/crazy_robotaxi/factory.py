# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Crazy Robotaxi composition inside the generic model-thread engine."""

from __future__ import annotations

from dataclasses import replace
from functools import partial

from omnidreams_game_engine.conditioning import LudusConditionRenderer
from omnidreams_game_engine.config import BevConfig, RasterConfig, VehicleConfig
from omnidreams_game_engine.engine import GameEngine
from omnidreams_game_engine.game_map.vicinity import GameMapVicinityResolver
from omnidreams_game_engine.simulation.ego_vehicle_kinematics import (
    EgoVehicleKinematics,
    state_from_initial_pose,
)
from omnidreams_game_engine.simulation.ground_snap import GroundSnapper
from omnidreams_game_engine.types import DriverCommand, SceneDefinition, VehicleState

from crazy_robotaxi.dynamics import TaxiVehicleConfig, integrate_taxi_vehicle
from crazy_robotaxi.physics import TaxiPhysicsWorld, step_taxi_physics_world
from crazy_robotaxi.rules import TaxiGameConfig, TaxiGameController, TaxiGameRules
from crazy_robotaxi.scene import load_scene_data


def build_taxi_engine(
    *,
    scene: SceneDefinition,
    game_config: TaxiGameConfig,
    raster: RasterConfig,
    bev: BevConfig,
    frame_interval_s: float,
    device: str,
) -> GameEngine:
    """Construct every mutable Taxi subsystem on the calling model thread."""
    if scene.game_map is None:
        raise ValueError("Crazy Robotaxi requires a compiled semantic game map")
    scene_data = load_scene_data(scene)
    ground_snapper = _build_ground_snapper(scene, game_config)
    simulation = EgoVehicleKinematics(
        initial_state=state_from_initial_pose(
            initial_rig_to_world=scene.initial_rig_to_world,
            initial_yaw_rad=scene.initial_yaw_rad,
            initial_speed_mps=0.0,
        ),
        vehicle_config=game_config.vehicle,
        ground_snapper=ground_snapper,
        initial_timestamp_us=scene.initial_timestamp_us,
        scene=scene,
        integrate_fn=_integrate_taxi_vehicle,
        physics_world_factory=lambda active_scene, vehicle: TaxiPhysicsWorld(
            active_scene,
            game_config.vehicle,
            curb_segments_world=scene_data.curb_segments_world,
        ),
        physics_step_fn=step_taxi_physics_world,
        include_initial_state_in_first_chunk=True,
    )
    controller = TaxiGameController(
        scene_id=scene.scene_id,
        reference_route_world=scene_data.reference_route_world,
        navigation_lanes=scene_data.navigation_lanes,
        fare_regions=scene_data.fare_regions,
        initial_state=simulation.current_state,
        config=game_config,
        initial_camera=scene.selected_camera,
        vicinity_resolver=GameMapVicinityResolver(scene.game_map),
    )
    renderer = LudusConditionRenderer(raster, bev, device=device)
    renderer.load_scene(scene)
    return GameEngine(
        simulation=simulation,
        rules=TaxiGameRules(controller),
        condition_renderer=renderer,
        frame_interval_s=frame_interval_s,
    )


def _integrate_taxi_vehicle(
    state: VehicleState,
    command: DriverCommand,
    dt_s: float,
    vehicle: VehicleConfig,
) -> VehicleState:
    if not isinstance(vehicle, TaxiVehicleConfig):
        raise TypeError("Crazy Robotaxi requires TaxiVehicleConfig")
    return integrate_taxi_vehicle(state, command, dt_s, vehicle)


def _build_ground_snapper(
    scene: SceneDefinition,
    config: TaxiGameConfig,
) -> GroundSnapper | None:
    if scene.ground_mesh_vertices is None or scene.ground_mesh_faces is None:
        return None
    return GroundSnapper(
        scene.ground_mesh_vertices,
        scene.ground_mesh_faces,
        max_absolute_rotation_deg=config.ground_snap_max_absolute_rotation_deg,
        invalid_sample_handler=partial(
            settle_invalid_ground_attitude,
            settle_fraction=config.ground_snap_settle_fraction,
        ),
    )


def settle_invalid_ground_attitude(
    state: VehicleState,
    *,
    settle_fraction: float = 0.25,
) -> VehicleState:
    """Ease stale ground attitude toward level after an invalid sample."""
    pitch = state.pitch_rad * (1.0 - settle_fraction)
    roll = state.roll_rad * (1.0 - settle_fraction)
    return replace(
        state,
        pitch_rad=0.0 if abs(pitch) < 1.0e-4 else pitch,
        roll_rad=0.0 if abs(roll) < 1.0e-4 else roll,
    )
