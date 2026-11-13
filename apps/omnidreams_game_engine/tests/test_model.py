# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the direct OmniDreams rollout contract."""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
import torch
from omnidreams_game_engine.engine import (
    EngineStep,
    SimulationStep,
    render_recorded_step,
)
from omnidreams_game_engine.model import WorldModelRollout, _initial_image_tensor
from omnidreams_game_engine.timeline import WorldSample, WorldTimeline
from omnidreams_game_engine.types import (
    CameraCalibration,
    ConditionBatch,
    DriverCommand,
    DynamicActorTrajectory,
    SceneDefinition,
    TrajectoryChunk,
    VehicleState,
)

pytestmark = pytest.mark.ci_cpu


def _scene() -> SceneDefinition:
    calibration = CameraCalibration(
        clipgt_name="front",
        logical_name="camera_front_wide_120fov",
        width=8,
        height=4,
        cx=4.0,
        cy=2.0,
        polynomial=np.zeros(6, dtype=np.float32),
        is_backward_polynomial=False,
        linear_cde=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        sensor_to_rig_flu=np.eye(4, dtype=np.float32),
    )
    return SceneDefinition(
        scene_path=Path("scene.arrow"),
        scene_id="cpu-scene",
        metadata={},
        selected_camera=calibration,
        initial_rig_to_world=np.eye(4, dtype=np.float32),
        initial_timestamp_us=0,
        initial_yaw_rad=0.0,
        initial_speed_mps=0.0,
        initial_rgb=np.zeros((4, 8, 3), dtype=np.uint8),
        prompt="a yellow taxi",
        line_layers=(),
        triangle_layers=(),
    )


class _Pipeline:
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def initialize_cache(self, **kwargs):
        self.calls.append(("initialize_cache", kwargs))
        return {"rollout": len(self.calls)}

    def get_num_output_frames(self, autoregressive_index):
        self.calls.append(("get_num_output_frames", autoregressive_index))
        return 2

    def generate(self, *, autoregressive_index, cache, input):
        self.calls.append(("generate", (autoregressive_index, cache, input.shape)))
        return torch.zeros(1, 1, 2, 3, 4, 8)

    def finalize(self, *, autoregressive_index, cache):
        self.calls.append(("finalize", (autoregressive_index, cache)))
        return {"model_ms": 1.25}


@dataclass
class _Engine:
    closed: bool = False
    is_running: bool = True
    live_edit: object | None = None

    @property
    def current_game_frame(self):
        return None

    def submit_text(self, value):
        return value

    def step(self, commands):
        count = len(commands)
        state = VehicleState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        trajectory = TrajectoryChunk(
            timestamps_us=np.arange(count, dtype=np.int64),
            rig_poses_world=np.repeat(np.eye(4, dtype=np.float32)[None], count, axis=0),
            vehicle_states=(state,) * count,
            boundary_state_after_chunk=state,
            applied_commands=commands,
        )
        return self.render(SimulationStep(trajectory, tuple(range(count))))

    def render(self, recorded):
        return EngineStep(
            recorded.trajectory,
            recorded.game_frames,
            ConditionBatch(
                torch.zeros(1, 1, len(recorded.trajectory.timestamps_us), 3, 4, 8)
            ),
        )

    def close(self):
        self.closed = True


def test_rollout_calls_pipeline_directly_and_owns_its_cache() -> None:
    pipeline = _Pipeline()
    engines: list[_Engine] = []

    def engine_factory() -> _Engine:
        engine = _Engine()
        engines.append(engine)
        return engine

    rollout = WorldModelRollout(
        pipeline=pipeline,
        scene=_scene(),
        engine_factory=engine_factory,
        trace_chunk_lifecycle=True,
    )
    result = rollout.step(
        autoregressive_index=0,
        commands=(DriverCommand(), DriverCommand()),
    )

    assert result.video_bvtchw.shape == (1, 1, 2, 3, 4, 8)
    assert result.metrics["model_ms"] == 1.25
    assert result.metrics["engine_wall_ms"] >= 0.0
    assert result.metrics["engine_cpu_ms"] >= 0.0
    assert result.metrics["pipeline_wall_ms"] >= 0.0
    assert result.metrics["pipeline_cpu_ms"] >= 0.0
    assert result.metrics["rollout_wall_ms"] >= 0.0
    assert result.metrics["rollout_cpu_ms"] >= 0.0
    assert result._trace is not None
    assert (
        result._trace.engine_step_started_ns
        <= result._trace.engine_step_returned_ns
        <= result._trace.generate_started_ns
        <= result._trace.generate_returned_ns
        <= result._trace.cache_finalize_returned_ns
        <= result._trace.rollout_step_returned_ns
    )
    assert [call[0] for call in pipeline.calls] == [
        "initialize_cache",
        "get_num_output_frames",
        "generate",
        "finalize",
    ]

    rollout.reset()
    assert engines[0].closed
    assert len(engines) == 2
    assert [call[0] for call in pipeline.calls].count("initialize_cache") == 2

    rollout.close()
    assert engines[1].closed


def test_rollout_attaches_live_edit_style_without_removed_timing_argument() -> None:
    pipeline = _Pipeline()
    engine = _Engine()
    attached: list[tuple[object, object, str]] = []

    class Style:
        def attach_v2(self, pipeline, cache, prompt):
            attached.append((pipeline, cache, prompt))

    engine.live_edit = type("LiveEdit", (), {"style": Style()})()

    rollout = WorldModelRollout(
        pipeline=pipeline,
        scene=_scene(),
        engine_factory=lambda: engine,
    )

    assert attached == [(pipeline, rollout.cache, "a yellow taxi")]
    rollout.close()


def test_initial_image_tensor_owns_writable_numpy_storage(monkeypatch) -> None:
    source = np.zeros((4, 8, 4), dtype=np.uint8)
    source.setflags(write=False)
    writable_flags: list[bool] = []
    torch_from_numpy = torch.from_numpy

    def record_writable(array):
        writable_flags.append(array.flags.writeable)
        return torch_from_numpy(array)

    monkeypatch.setattr(torch, "from_numpy", record_writable)

    tensor = _initial_image_tensor(source, device="cpu")

    assert writable_flags == [True]
    assert tensor.shape == (1, 1, 1, 3, 4, 8)


@dataclass(frozen=True)
class _ExternalFrame:
    epoch: int
    tick: int


def _external_sample(
    tick: int, *, epoch: int = 7, sim_hz: int = 60, actor_present: bool = False
) -> WorldSample:
    state = VehicleState(tick / 1000, 0, 0, 0, 1, 0)
    pose = np.eye(4, dtype=np.float32)[None]
    pose[0, 0, 3] = state.x_m
    timestamps = np.array([round(tick * 1_000_000 / sim_hz)], dtype=np.int64)
    actors = (
        (
            DynamicActorTrajectory(
                "external-car",
                "Car",
                timestamps,
                pose[:, :3, 3].copy(),
                np.array([[0, 0, 0, 1]], dtype=np.float32),
                np.ones(3, dtype=np.float32),
                is_simulated=True,
            ),
        )
        if actor_present
        else ()
    )
    trajectory = TrajectoryChunk(
        timestamps, pose, (state,), state, dynamic_actors=actors
    )
    return WorldSample(
        SimulationStep(
            trajectory, (_ExternalFrame(epoch, tick),), {"external_build_ms": 1.0}
        )
    )


class _ObservationRenderer:
    def load_scene(self, scene: SceneDefinition) -> None:
        pass

    def render(self, trajectory: TrajectoryChunk) -> ConditionBatch:
        positions = torch.from_numpy(trajectory.rig_poses_world[:, 0, 3].copy())
        return ConditionBatch(
            positions.reshape(1, 1, -1, 1, 1, 1).expand(-1, -1, -1, 3, 4, 8)
        )

    def close(self) -> None:
        pass


class _ExternalEngine(_Engine):
    def step(self, commands: tuple[DriverCommand, ...]) -> EngineStep:
        raise AssertionError("An external recording must not advance local simulation")

    def render(self, recorded: SimulationStep) -> EngineStep:
        return render_recorded_step(recorded, _ObservationRenderer())


@pytest.mark.parametrize("sim_hz", [60, 120])
def test_external_observations_share_timeline_and_recorded_rollout(
    monkeypatch: pytest.MonkeyPatch, sim_hz: int
) -> None:
    stride = sim_hz // 30
    timeline = WorldTimeline(ticks_per_frame=stride)
    origin = 600
    for tick in range(stride + 1):
        timeline.append(_external_sample(origin + tick, sim_hz=sim_hz))
    pipeline = _Pipeline()
    original_generate = pipeline.generate
    elapsed_ticks = sim_hz // 2

    def generate(*, autoregressive_index, cache, input):
        original_generate(
            autoregressive_index=autoregressive_index, cache=cache, input=input
        )
        if autoregressive_index == 0:
            # The external producer progresses while generation holds its snapshot.
            for tick in range(stride + 1, stride + elapsed_ticks + 1):
                timeline.append(
                    _external_sample(
                        origin + tick,
                        sim_hz=sim_hz,
                        actor_present=tick <= stride + 4,
                    )
                )
        return input.clone()

    monkeypatch.setattr(pipeline, "generate", generate)
    rollout = WorldModelRollout(
        pipeline=pipeline, scene=_scene(), engine_factory=_ExternalEngine
    )
    first = timeline.prepare(2, threading.Event())
    assert first is not None
    result = rollout.step(autoregressive_index=0, recorded_step=first.recorded_step())
    assert result.engine.game_frames == (
        _ExternalFrame(7, origin),
        _ExternalFrame(7, origin + stride),
    )
    assert timeline.gameplay_tick == stride + elapsed_ticks
    assert timeline.generation_tick == -1
    assert pipeline.calls[-1][0] == "finalize"
    timeline.commit(first)

    second = timeline.prepare(2, threading.Event())
    assert second is not None
    result = rollout.step(autoregressive_index=1, recorded_step=second.recorded_step())
    ticks = (origin + stride + elapsed_ticks // 2, origin + stride + elapsed_ticks)
    assert result.engine.game_frames == tuple(_ExternalFrame(7, tick) for tick in ticks)
    np.testing.assert_array_equal(
        result.engine.trajectory.timestamps_us,
        [round(tick * 1_000_000 / sim_hz) for tick in ticks],
    )
    np.testing.assert_allclose(
        result.video_bvtchw[0, 0, :, 0, 0, 0], np.array(ticks) / 1000
    )
    assert result.metrics["external_build_ms"] == elapsed_ticks
    actor = result.engine.trajectory.dynamic_actors[0]
    assert actor.timestamps_us[0] == round((origin + stride + 1) * 1_000_000 / sim_hz)
    assert actor.timestamps_us[-1] == round((origin + stride + 4) * 1_000_000 / sim_hz)
    timeline.commit(second)
    assert timeline.generation_tick == timeline.gameplay_tick
    rollout.close()


@pytest.mark.parametrize("failure_stage", ["generate", "finalize"])
def test_external_recording_failure_and_new_epoch(
    monkeypatch: pytest.MonkeyPatch, failure_stage: str
) -> None:
    pipeline = _Pipeline()
    rollout = WorldModelRollout(
        pipeline=pipeline, scene=_scene(), engine_factory=_ExternalEngine
    )
    timeline = WorldTimeline(ticks_per_frame=2)
    for tick in range(600, 603):
        timeline.append(_external_sample(tick))
    window = timeline.prepare(2, threading.Event())
    assert window is not None

    def fail(**kwargs):
        raise RuntimeError("model failed")

    with monkeypatch.context() as patch:
        patch.setattr(pipeline, failure_stage, fail)
        with pytest.raises(RuntimeError, match="model failed"):
            rollout.step(autoregressive_index=0, recorded_step=window.recorded_step())
            timeline.commit(window)
    assert timeline.generation_tick == -1
    with pytest.raises(RuntimeError, match="Commit"):
        timeline.prepare(2, threading.Event())

    # A failed model/cache step restarts the epoch instead of replaying stale history.
    timeline.close()
    previous_engine = rollout.engine
    previous_cache = rollout.cache
    rollout.reset()
    assert isinstance(previous_engine, _ExternalEngine) and previous_engine.closed
    assert rollout.cache is not previous_cache
    timeline = WorldTimeline(ticks_per_frame=2)
    for tick in range(30, 33):
        timeline.append(_external_sample(tick, epoch=8))
    window = timeline.prepare(2, threading.Event())
    assert window is not None
    result = rollout.step(autoregressive_index=0, recorded_step=window.recorded_step())
    assert result.engine.game_frames == (_ExternalFrame(8, 30), _ExternalFrame(8, 32))
    timeline.commit(window)
    rollout.close()


@pytest.mark.parametrize(
    "input_kind", ["missing", "both", "commands", "recorded", "unaligned"]
)
def test_rollout_rejects_invalid_inputs_before_rendering(
    monkeypatch: pytest.MonkeyPatch, input_kind: str
) -> None:
    pipeline = _Pipeline()
    engine = _ExternalEngine()
    rollout = WorldModelRollout(
        pipeline=pipeline, scene=_scene(), engine_factory=lambda: engine
    )
    recorded = _external_sample(600).step
    commands = (DriverCommand(),) if input_kind in {"both", "commands"} else None
    if input_kind == "unaligned":
        recorded = replace(recorded, game_frames=recorded.game_frames * 2)
    recorded_step = (
        recorded if input_kind in {"both", "recorded", "unaligned"} else None
    )

    def unexpected_render(recorded):
        pytest.fail("Invalid recordings must be rejected before rendering")

    monkeypatch.setattr(engine, "render", unexpected_render)
    with pytest.raises(ValueError):
        rollout.step(
            autoregressive_index=0, commands=commands, recorded_step=recorded_step
        )
    assert not any(name in {"generate", "finalize"} for name, _ in pipeline.calls)
    rollout.close()


def test_recorded_rendering_rejects_unaligned_game_frames() -> None:
    recorded = _external_sample(600).step
    with pytest.raises(ValueError, match="Game frames must align"):
        render_recorded_step(replace(recorded, game_frames=()), _ObservationRenderer())


def test_observer_receives_simulation_before_failed_generation(monkeypatch):
    pipeline = _Pipeline()
    observed = []

    def fail(**kwargs):
        assert len(observed) == 1
        raise RuntimeError("test generation failure")

    monkeypatch.setattr(pipeline, "generate", fail)
    rollout = WorldModelRollout(
        pipeline=pipeline, scene=_scene(), engine_factory=_Engine
    )
    try:
        with pytest.raises(RuntimeError, match="test generation failure"):
            rollout.step(
                autoregressive_index=0,
                commands=(DriverCommand(),) * 2,
                observe_engine_step=observed.append,
            )
        assert len(observed[0].trajectory.vehicle_states) == 2
    finally:
        rollout.close()
