# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU check: an external engine over TCP gets the in-process Ludus conditioning.

Manual tier (needs CUDA and the Ludus renderer; no model weights). A Crazy
Robotaxi authored map is compiled and drawn by two ``LudusConditionRenderer``
instances. The in-process ``GameEngine`` drives a taxi with a car ahead; the
stub engine then sends the same states in ``flu/0`` over TCP to the
``ExternalSessionDriver``. Every chunk's conditioning must be bit-identical. Run with::

    uv run pytest -p no:manual_marker -m manual \
        apps/omnidreams_game_engine/tests/test_api_gpu_external_loop.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
import torch
from omnidreams_game_engine.api import v0
from omnidreams_game_engine.api.v0.stub_engine import StubEngine, StubScript
from omnidreams_game_engine.api.v0.testing import FakePipeline
from omnidreams_game_engine.config import BevConfig, RasterConfig, VehicleConfig
from omnidreams_game_engine.simulation.ego_vehicle_kinematics import (
    EgoVehicleKinematics,
    state_from_initial_pose,
)

pytestmark = pytest.mark.manual

MAP = (
    Path(__file__).resolve().parents[2]
    / "crazy_robotaxi"
    / "crazy_robotaxi"
    / "maps"
    / "boulevard_district.robotaxi.yaml"
)
CHUNKS = 6


class _CarAheadRules:
    """Free driving with one simulated car 15 m ahead of the taxi."""

    is_running = True

    def snapshot(self, vehicle_state: v0.VehicleState) -> object:
        return vehicle_state.x_m

    def advance_frames(
        self, trajectory: v0.TrajectoryChunk, frame_interval_s: float
    ) -> v0.GameUpdate:
        del frame_interval_s
        poses = trajectory.rig_poses_world
        ahead = poses[:, :3, 3] + 15.0 * poses[:, :3, 0]
        yaw = np.arctan2(poses[:, 1, 0], poses[:, 0, 0])
        quats = np.stack(
            [np.zeros_like(yaw), np.zeros_like(yaw), np.sin(yaw / 2), np.cos(yaw / 2)],
            axis=1,
        )
        car = v0.DynamicActorTrajectory(
            entity_id="car-1",
            object_type="Car",
            timestamps_us=np.asarray(trajectory.timestamps_us, dtype=np.int64),
            translations_world=ahead.astype(np.float32),
            orientations_xyzw=quats.astype(np.float32),
            dimensions_lwh=np.asarray((4.5, 1.9, 1.6), dtype=np.float32),
            is_simulated=True,
        )
        return v0.GameUpdate(
            frames=tuple(s.x_m for s in trajectory.vehicle_states),
            dynamic_actors=(car,),
        )

    def submit_text(self, value: str, vehicle_state: v0.VehicleState) -> object:
        del value, vehicle_state
        return None


class _SameScene:
    def __init__(self, scene: v0.SceneDefinition) -> None:
        self.scene = scene

    def scene_for_seed(
        self, hello: v0.messages.Hello, seed: v0.SeedFrame
    ) -> v0.SceneDefinition:
        del hello, seed
        return self.scene


def test_external_engine_matches_in_process_conditioning_on_gpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if not MAP.is_file():
        pytest.skip("needs the Crazy Robotaxi boulevard map")
    from omnidreams_game_engine.conditioning import LudusConditionRenderer
    from omnidreams_game_engine.scene import SceneRequest, load_scene

    raster = RasterConfig(width=1024, height=560)
    bev = BevConfig(enabled=False)
    scene = load_scene(SceneRequest(map_path=MAP), raster)
    commands = [
        v0.DriverCommand(throttle=0.7, steer=0.2 * np.sin(g / 9.0)) for g in range(200)
    ]

    # In-process: GameEngine with kinematics, rules and Ludus.
    in_renderer = LudusConditionRenderer(raster, bev, device="cuda")
    in_renderer.load_scene(scene)
    in_pipeline = FakePipeline(device="cuda")

    def in_process_engine() -> v0.GameEngine:
        simulation = EgoVehicleKinematics(
            initial_state=state_from_initial_pose(
                initial_rig_to_world=scene.initial_rig_to_world,
                initial_yaw_rad=scene.initial_yaw_rad,
                initial_speed_mps=0.0,
            ),
            vehicle_config=VehicleConfig(),
            ground_snapper=None,
            initial_timestamp_us=scene.initial_timestamp_us,
            include_initial_state_in_first_chunk=True,
        )
        return v0.GameEngine(
            simulation=simulation,
            rules=_CarAheadRules(),
            condition_renderer=in_renderer,
            frame_interval_s=1.0 / 30.0,
        )

    rollout = v0.WorldModelRollout(
        pipeline=in_pipeline, scene=scene, engine_factory=in_process_engine
    )
    states: dict[int, v0.FrameState] = {}
    g = 0
    for k in range(CHUNKS + 4):
        count = rollout.frame_count(k)
        result = rollout.step(
            autoregressive_index=k, commands=tuple(commands[g : g + count])
        )
        trajectory = result.engine.trajectory
        car = trajectory.dynamic_actors[0]
        for index in range(count):
            states[2 * g] = v0.FrameState(
                g=g,
                tick=2 * g,
                rig_to_world=trajectory.rig_poses_world[index],
                vehicle=trajectory.vehicle_states[index],
                command=trajectory.applied_commands[index],
                actors=(
                    v0.ActorState(
                        entity_id=car.entity_id,
                        object_type=car.object_type,
                        position_world=car.translations_world[index],
                        orientation_xyzw=car.orientations_xyzw[index],
                        dimensions_lwh=car.dimensions_lwh,
                    ),
                ),
            )
            g += 1
    rollout.close()

    # External: the stub engine sends those states over TCP.
    codec = v0.FluStateCodec()
    ex_renderer = LudusConditionRenderer(raster, bev, device="cuda")
    ex_pipeline = FakePipeline(device="cuda")
    link = v0.TcpEngineLink(codec, host="127.0.0.1", port=0)
    driver = v0.ExternalSessionDriver(
        source=link,
        pipeline=ex_pipeline,
        scene_provider=_SameScene(scene),
        condition_renderer=ex_renderer,
    )
    stub = StubEngine(
        *link.address,
        state_fn=lambda tick: codec.encode(states[tick]),
        script=StubScript(),
    ).start()
    started = time.monotonic()
    try:
        while driver.stats.chunks < CHUNKS:
            assert time.monotonic() - started < 120.0, "external loop timed out"
            driver.poll(0.05)
    finally:
        stub.stop()
        stub.join(5.0)
        driver.close()
        link.close()
        ex_renderer.close()
    assert stub.failure is None

    assert len(ex_pipeline.inputs) >= CHUNKS
    for k in range(CHUNKS):
        assert torch.equal(ex_pipeline.inputs[k], in_pipeline.inputs[k]), f"chunk {k}"
    frames = [f for f in stub.summary.gen_frames if f[0] == 1]
    assert [f[2] for f in frames] == list(range(len(frames)))
    assert all(size == 1024 * 560 * 3 for *_, size in frames)
