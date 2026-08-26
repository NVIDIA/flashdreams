# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for Crazy Robotaxi's application boundary against FlashDreams V2."""

from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest
import torch
from crazy_robotaxi.application import _MODEL_PRESETS, CrazyRobotaxiApplication
from crazy_robotaxi.physics import TaxiPhysicsWorld
from crazy_robotaxi.session import CrazyRobotaxiModelLoop
from crazy_robotaxi.ui import CrazyRobotaxiSlangPyUILoop
from omnidreams_game_engine.simulation.game_physics import GamePhysicsWorld
from omnidreams_game_engine.types import (
    CameraCalibration,
    DriverCommand,
    SceneDefinition,
)

from flashdreams.runtime_v2.native_window_client_window import (
    NativeWindowClientWindow,
)
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu


def _scene() -> SceneDefinition:
    calibration = CameraCalibration(
        clipgt_name="front",
        logical_name="camera_front_wide_120fov",
        width=1280,
        height=704,
        cx=640.0,
        cy=352.0,
        polynomial=np.zeros(6, dtype=np.float32),
        is_backward_polynomial=False,
        linear_cde=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        sensor_to_rig_flu=np.eye(4, dtype=np.float32),
    )
    return SceneDefinition(
        scene_path=Path("scene.arrow"),
        scene_id="test",
        metadata={},
        selected_camera=calibration,
        initial_rig_to_world=np.eye(4, dtype=np.float32),
        initial_timestamp_us=0,
        initial_yaw_rad=0.0,
        initial_speed_mps=0.0,
        initial_rgb=np.zeros((704, 1280, 3), dtype=np.uint8),
        prompt="taxi",
        line_layers=(),
        triangle_layers=(),
    )


def test_application_registers_model_and_slangpy_ui_loops() -> None:
    pipeline = object()
    app = CrazyRobotaxiApplication(
        pipeline_factory=lambda config, device: pipeline,
        scene_factory=lambda request, raster: _scene(),
    )
    desc = app.session_desc()
    app.init(
        [
            "--device",
            "cpu",
            "--total-blocks",
            "2",
            "--profile-input-latency",
        ]
    )

    session = app.create_session(desc)
    session.init()
    ui_loop, model_loop = session._take_loops()

    assert desc.output_layout is VideoTensorLayout.tchw
    assert desc.frames_per_second_for_ui == 30
    assert desc.frames_per_second_for_step == 30
    assert isinstance(model_loop, CrazyRobotaxiModelLoop)
    assert isinstance(ui_loop, CrazyRobotaxiSlangPyUILoop)
    assert model_loop.state.pipeline is pipeline
    assert model_loop.state.rollout is None
    assert model_loop.state.ui_loop is ui_loop
    assert ui_loop.state.model_loop is model_loop
    assert ui_loop.state.profile_input_latency


def test_native_window_accepts_crazy_robotaxi_output_contract() -> None:
    """Keep the app's fixed output contract compatible with V2 native output."""

    class Presenter:
        should_close = False

        def __init__(self) -> None:
            self.frames: list[torch.Tensor] = []
            self.closed = False

        def set_input_callbacks(self, **callbacks: object) -> None:
            assert set(callbacks) == {"on_keyboard_event", "on_mouse_event"}

        def present_frame(self, frame: torch.Tensor) -> bool:
            self.frames.append(frame)
            return True

        def close(self) -> None:
            self.closed = True

    desc = CrazyRobotaxiApplication().session_desc()
    presenter = Presenter()
    presenter_arguments: dict[str, object] = {}

    def create_presenter(**arguments: object) -> Presenter:
        presenter_arguments.update(arguments)
        return presenter

    window = NativeWindowClientWindow(
        title="Crazy Robotaxi",
        presenter_factory=cast(Any, create_presenter),
    )
    source = torch.zeros(
        (1, 3, desc.video_height, desc.video_width),
        dtype=torch.float32,
    )

    window.open(desc)
    window.write(
        StepResult(
            step_index=0,
            output=source,
            frame_count=1,
            output_layout=desc.output_layout,
        )
    )
    window.close()

    assert presenter_arguments == {
        "width": desc.video_width,
        "height": desc.video_height,
        "title": "Crazy Robotaxi",
    }
    assert len(presenter.frames) == 1
    assert presenter.frames[0].shape == (
        desc.video_height,
        desc.video_width,
        3,
    )
    assert presenter.frames[0].device == source.device
    assert presenter.frames[0].dtype is torch.uint8
    assert torch.all(presenter.frames[0] == 128)
    assert presenter.closed


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ([], False),
        (["--profile-pipeline"], True),
    ],
)
def test_pipeline_profiling_is_an_app_local_opt_in(
    arguments: list[str],
    expected: bool,
) -> None:
    configured = []
    app = CrazyRobotaxiApplication(
        pipeline_factory=lambda config, device: configured.append(config) or object(),
        scene_factory=lambda request, raster: _scene(),
    )
    app.init(arguments)

    app.create_session(app.session_desc())

    assert configured[0].enable_sync_and_profile is expected
    assert app._config is not None
    assert app._config.pipeline_profiling is expected
    assert _MODEL_PRESETS["standard"].enable_sync_and_profile


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ([], False),
        (["--profile-input-latency"], True),
    ],
)
def test_input_latency_profiling_is_an_app_local_opt_in(
    arguments: list[str],
    expected: bool,
) -> None:
    app = CrazyRobotaxiApplication()

    app.init(arguments)

    assert app._config is not None
    assert app._config.profile_input_latency is expected


@pytest.mark.parametrize("prewarm_blocks", [0, 4, 7])
def test_application_configures_prepresentation_warmup(prewarm_blocks: int) -> None:
    app = CrazyRobotaxiApplication(
        pipeline_factory=lambda config, device: object(),
        scene_factory=lambda request, raster: _scene(),
    )
    app.init(["--prewarm-blocks", str(prewarm_blocks)])

    assert app._config is not None
    assert app._config.prewarm_blocks == prewarm_blocks


def test_application_rejects_negative_prewarm_blocks() -> None:
    app = CrazyRobotaxiApplication()

    with pytest.raises(ValueError, match="must be non-negative"):
        app.init(["--prewarm-blocks", "-1"])


def test_model_state_prewarms_neutral_blocks_once_then_resets(monkeypatch) -> None:
    class FakeRollout:
        def __init__(self, **kwargs) -> None:
            del kwargs
            self.steps: list[tuple[int, tuple[DriverCommand, ...]]] = []
            self.reset_count = 0

        def frame_count(self, autoregressive_index: int) -> int:
            return autoregressive_index + 1

        def step(self, *, autoregressive_index: int, commands):
            self.steps.append((autoregressive_index, commands))
            return object()

        def reset(self) -> None:
            self.reset_count += 1

        def close(self) -> None:
            return

    monkeypatch.setattr("crazy_robotaxi.session.WorldModelRollout", FakeRollout)
    app = CrazyRobotaxiApplication(
        pipeline_factory=lambda config, device: object(),
        scene_factory=lambda request, raster: _scene(),
    )
    app.init(["--device", "cpu", "--prewarm-blocks", "4"])
    session = app.create_session(app.session_desc())
    session.init()
    ui_loop, model_loop = session._take_loops()

    rollout = model_loop.state.ensure_rollout()
    ui_loop._run_message_batch()

    assert [index for index, _ in rollout.steps] == [0, 1, 2, 3]
    assert [len(commands) for _, commands in rollout.steps] == [1, 2, 3, 4]
    assert all(
        command == DriverCommand()
        for _, commands in rollout.steps
        for command in commands
    )
    assert rollout.reset_count == 1
    assert model_loop.state.blocks_generated == 0
    assert model_loop.state.prewarm_complete
    assert ui_loop.state._loading_status == "STARTING GAME"

    assert model_loop.state.ensure_rollout() is rollout
    assert len(rollout.steps) == 4
    model_loop.state.reset()
    assert rollout.reset_count == 2
    assert len(rollout.steps) == 4


def test_taxi_physics_uses_spatial_and_traffic_topology_refreshes_only() -> None:
    world = object.__new__(TaxiPhysicsWorld)
    world.graph = type("Graph", (), {"objects": ()})()
    world._physics_center_xy = np.zeros(2, dtype=np.float32)
    center = np.asarray([40.0, -4.0], dtype=np.float32)
    with patch.object(
        GamePhysicsWorld,
        "synchronize_window",
        return_value=True,
    ) as synchronize:
        changed = world.synchronize_window(center, timestamp_us=2_000_000)

    assert changed
    synchronize.assert_called_once_with(center, timestamp_us=None)


def test_application_rejects_geometry_the_model_does_not_produce() -> None:
    app = CrazyRobotaxiApplication(
        pipeline_factory=lambda config, device: object(),
        scene_factory=lambda request, raster: _scene(),
    )
    app.init([])
    desc = app.session_desc()
    desc = type(desc)(
        output_layout=desc.output_layout,
        presentation_mode=desc.presentation_mode,
        frames_per_second_for_ui=desc.frames_per_second_for_ui,
        frames_per_second_for_step=desc.frames_per_second_for_step,
        video_width=640,
        video_height=desc.video_height,
    )

    with pytest.raises(ValueError, match="do not match renderer"):
        app.create_session(desc)


@pytest.mark.parametrize(
    "override",
    [
        {"frames_per_second_for_step": 60},
        {"frames_per_second_for_ui": 60},
    ],
)
def test_application_rejects_mismatched_generation_or_ui_rate(
    override: dict[str, int],
) -> None:
    app = CrazyRobotaxiApplication(
        pipeline_factory=lambda config, device: object(),
        scene_factory=lambda request, raster: _scene(),
    )
    app.init([])

    with pytest.raises(ValueError, match="30 frames per second"):
        app.create_session(replace(app.session_desc(), **override))
