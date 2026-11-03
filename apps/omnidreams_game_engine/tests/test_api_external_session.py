# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for ExternalEngineSession inside WorldModelRollout."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import pytest
import torch
from omnidreams_game_engine.api import v0
from omnidreams_game_engine.api.v0.assembly import ChunkAssembler, ExternalChunk
from omnidreams_game_engine.api.v0.stub_engine import ScriptedDrive
from omnidreams_game_engine.api.v0.testing import (
    FakePipeline,
    PoseConditionRenderer,
    minimal_scene,
)
from omnidreams_game_engine.config import VehicleConfig
from omnidreams_game_engine.contracts import GameUpdate
from omnidreams_game_engine.simulation.ego_vehicle_kinematics import (
    EgoVehicleKinematics,
    state_from_initial_pose,
)

pytestmark = pytest.mark.ci_cpu


class _AssemblerSource:
    """Minimal EngineSource: chunks come from a pre-filled assembler."""

    def __init__(self, assembler: ChunkAssembler) -> None:
        self.assembler = assembler
        self.waits: list[tuple[int, int]] = []

    def poll_events(self, timeout_s: float = 0.0) -> list[v0.SourceEvent]:
        del timeout_s
        return []

    def begin_epoch(self, epoch: int, timeline: v0.ChunkTimeline) -> None:
        self.assembler.begin_epoch(epoch, timeline)

    def wait_chunk(
        self, epoch: int, ar_index: int, timeout_s: float
    ) -> ExternalChunk | None:
        del timeout_s
        self.waits.append((epoch, ar_index))
        if self.assembler.epoch != epoch:
            return None
        return self.assembler.take(ar_index)

    def send(self, message: v0.messages.Message, payload: bytes = b"") -> None:
        del message, payload

    def send_frames(self, frames: Sequence[tuple[v0.messages.Message, bytes]]) -> None:
        del frames

    def close(self) -> None:
        pass


def _drive_frames(count: int, *, t0: int = 0) -> list[v0.FrameState]:
    drive = ScriptedDrive()
    frames = []
    for g in range(count):
        state = drive.frame_state(t0 + 2 * g)
        frames.append(replace(state, g=g))
    return frames


def _session_rollout(
    frames: list[v0.FrameState], *, epoch: int = 1
) -> tuple[v0.WorldModelRollout, FakePipeline, PoseConditionRenderer, _AssemblerSource]:
    timeline = v0.ChunkTimeline()
    assembler = ChunkAssembler(timeline)
    assembler.begin_epoch(epoch)
    for frame in frames:
        assembler.add(epoch, frame)
    source = _AssemblerSource(assembler)
    pipeline = FakePipeline()
    renderer = PoseConditionRenderer()
    scene = minimal_scene()

    def factory() -> v0.ExternalEngineSession:
        return v0.ExternalEngineSession(
            source=source,
            epoch=epoch,
            condition_renderer=renderer,
            timeline=timeline,
            base_timestamp_us=scene.initial_timestamp_us,
            chunk_timeout_s=0.01,
        )

    rollout = v0.WorldModelRollout(
        pipeline=pipeline, scene=scene, engine_factory=factory
    )
    return rollout, pipeline, renderer, source


def _g(frame: object) -> int:
    assert isinstance(frame, v0.FrameState)
    return frame.g


def _commands(count: int) -> tuple[v0.DriverCommand, ...]:
    return tuple(v0.DriverCommand() for _ in range(count))


def test_external_session_drives_the_rollout_and_video_aligns_with_frames() -> None:
    frames = _drive_frames(21)
    rollout, pipeline, renderer, source = _session_rollout(frames)
    results = []
    for k in range(3):
        count = rollout.frame_count(k)
        results.append(rollout.step(autoregressive_index=k, commands=_commands(count)))
    assert [r.video_bvtchw.shape[2] for r in results] == [5, 8, 8]
    assert [tuple(_g(f) for f in r.engine.game_frames) for r in results] == [
        (0, 1, 2, 3, 4),
        tuple(range(5, 13)),
        tuple(range(13, 21)),
    ]
    assert source.waits == [(1, 0), (1, 1), (1, 2)]
    assert [a for a, _ in pipeline.generated] == [0, 1, 2]
    # The "generated" video is the condition, so frame i of a chunk shows frame i's pose.
    for result in results:
        np.testing.assert_array_equal(
            result.video_bvtchw.float().numpy(),
            result.engine.condition.hdmap_bvtchw.float().numpy(),
        )
    trajectory = renderer.trajectories[1]
    np.testing.assert_array_equal(
        trajectory.rig_poses_world, np.stack([f.rig_to_world for f in frames[5:13]])
    )
    np.testing.assert_array_equal(
        trajectory.timestamps_us, 1_000_000_000 + np.arange(5, 13) * 33333
    )
    assert {a.entity_id for a in trajectory.dynamic_actors} >= {"car-1"}
    assert results[2].engine.metrics["engine_step_wall_ms"] >= 0.0
    assert rollout.engine.current_game_frame is results[2].engine.game_frames[-1]
    rollout.close()


def test_missing_chunk_times_out() -> None:
    rollout, *_ = _session_rollout(_drive_frames(4))
    with pytest.raises(v0.ChunkTimeoutError):
        rollout.step(autoregressive_index=0, commands=_commands(5))


def test_step_rejects_a_frame_count_mismatch_and_a_foreign_chunk() -> None:
    frames = _drive_frames(5)
    rollout, *_ = _session_rollout(frames)
    engine = rollout.engine
    assert isinstance(engine, v0.ExternalEngineSession)
    with pytest.raises(ValueError):
        engine.step(_commands(4))
    with pytest.raises(ValueError):
        engine.supply(
            ExternalChunk(epoch=9, ar_index=0, t0_tick=0, frames=tuple(frames))
        )
    with pytest.raises(ValueError):
        engine.step(())


def test_reset_with_a_new_scene_uses_the_new_seed() -> None:
    rollout, pipeline, *_ = _session_rollout(_drive_frames(5))
    first_cache = rollout.cache
    assert first_cache is not None
    bright = replace(
        minimal_scene(), initial_rgb=np.full((4, 8, 3), 255, dtype=np.uint8)
    )
    old_engine = rollout.engine
    rollout.reset(scene=bright)
    assert rollout.scene is bright
    assert rollout.engine is not old_engine and not old_engine.is_running
    second_cache = rollout.cache
    assert second_cache is not None
    assert first_cache["seed_mean"] == pytest.approx(-1.0)
    assert second_cache["seed_mean"] == pytest.approx(1.0)
    rollout.reset()
    assert rollout.scene is bright
    assert pipeline.caches == 3


class _FreeDriveRules:
    """Minimal GameRules: one snapshot per frame, no actors."""

    is_running = True

    def snapshot(self, vehicle_state: v0.VehicleState) -> object:
        return vehicle_state.x_m

    def advance_frames(
        self, trajectory: v0.TrajectoryChunk, frame_interval_s: float
    ) -> GameUpdate:
        del frame_interval_s
        return GameUpdate(frames=tuple(s.x_m for s in trajectory.vehicle_states))

    def submit_text(self, value: str, vehicle_state: v0.VehicleState) -> object:
        del value, vehicle_state
        return None


def test_external_engine_reproduces_the_in_process_engine_bit_for_bit() -> None:
    """An engine that sends the in-process engine's states gets the same model input and video."""
    scene = minimal_scene()
    plan = (5, 8)
    chunks = 4
    throttle = [
        v0.DriverCommand(throttle=0.8, steer=0.15 * ((g // 7) % 3 - 1))
        for g in range(60)
    ]

    in_pipeline = FakePipeline(plan)
    in_renderer = PoseConditionRenderer()

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
            rules=_FreeDriveRules(),
            condition_renderer=in_renderer,
            frame_interval_s=1.0 / 30.0,
        )

    in_rollout = v0.WorldModelRollout(
        pipeline=in_pipeline, scene=scene, engine_factory=in_process_engine
    )
    in_results = []
    g = 0
    for k in range(chunks):
        count = in_rollout.frame_count(k)
        in_results.append(
            in_rollout.step(
                autoregressive_index=k, commands=tuple(throttle[g : g + count])
            )
        )
        g += count

    # The "external engine": every in-process state sent over the wire in flu/0.
    codec = v0.FluStateCodec()
    frames: list[v0.FrameState] = []
    g = 0
    for result in in_results:
        trajectory = result.engine.trajectory
        for index in range(len(trajectory.timestamps_us)):
            state = v0.FrameState(
                g=g,
                tick=2 * g,
                rig_to_world=trajectory.rig_poses_world[index],
                vehicle=trajectory.vehicle_states[index],
                command=trajectory.applied_commands[index],
            )
            wire = json.loads(json.dumps(codec.encode(state)))
            frames.append(codec.decode(wire, g=g, tick=2 * g))
            g += 1

    ex_rollout, ex_pipeline, ex_renderer, _ = _session_rollout(frames)
    for k in range(chunks):
        ex_rollout.step(
            autoregressive_index=k, commands=_commands(ex_rollout.frame_count(k))
        )

    assert len(ex_renderer.trajectories) == len(in_renderer.trajectories) == chunks
    for got, want in zip(
        ex_renderer.trajectories, in_renderer.trajectories, strict=True
    ):
        np.testing.assert_array_equal(got.timestamps_us, want.timestamps_us)
        np.testing.assert_array_equal(got.rig_poses_world, want.rig_poses_world)
        assert got.vehicle_states == want.vehicle_states
        assert got.applied_commands == want.applied_commands
    for got, want in zip(ex_pipeline.inputs, in_pipeline.inputs, strict=True):
        assert torch.equal(got, want)
    assert ex_pipeline.generated == in_pipeline.generated
