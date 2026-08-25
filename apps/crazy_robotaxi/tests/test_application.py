# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for Crazy Robotaxi's application boundary against FlashDreams V2."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from crazy_robotaxi.application import CrazyRobotaxiApplication
from crazy_robotaxi.session import CrazyRobotaxiModelLoop
from crazy_robotaxi.ui import CrazyRobotaxiSlangPyUILoop
from omnidreams_game_engine.types import CameraCalibration, SceneDefinition

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
    app.init(["--device", "cpu", "--total-blocks", "2"])

    session = app.create_session(desc)
    session.init()
    ui_loop, model_loop = session._take_loops()

    assert desc.output_layout is VideoTensorLayout.tchw
    assert isinstance(model_loop, CrazyRobotaxiModelLoop)
    assert isinstance(ui_loop, CrazyRobotaxiSlangPyUILoop)
    assert model_loop.state.pipeline is pipeline
    assert model_loop.state.rollout is None
    assert model_loop.state.ui_loop is ui_loop
    assert ui_loop.state.model_loop is model_loop


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


def test_application_rejects_a_frame_rate_the_model_was_not_trained_for() -> None:
    app = CrazyRobotaxiApplication(
        pipeline_factory=lambda config, device: object(),
        scene_factory=lambda request, raster: _scene(),
    )
    app.init([])

    with pytest.raises(ValueError, match="30 frames per second"):
        app.create_session(replace(app.session_desc(), frames_per_second_for_step=60))
