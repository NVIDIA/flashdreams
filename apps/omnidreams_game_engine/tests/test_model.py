# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the direct OmniDreams rollout contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from omnidreams_game_engine.engine import EngineStep
from omnidreams_game_engine.model import WorldModelRollout, _initial_image_tensor
from omnidreams_game_engine.types import (
    CameraCalibration,
    ConditionBatch,
    DriverCommand,
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
        return EngineStep(
            trajectory=trajectory,
            game_frames=tuple(range(count)),
            condition=ConditionBatch(torch.zeros(1, 1, count, 3, 4, 8)),
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
