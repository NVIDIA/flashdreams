# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU tests for Crazy Robotaxi alignment diagnostics."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from crazy_robotaxi.alignment_diagnostics import (
    AlignmentDiagnosticPresenter,
)
from omnidreams_game_engine.math3d import rig_pose_from_vehicle_state
from omnidreams_game_engine.types import (
    CameraCalibration,
    PhysicsDebugFrame,
    PresentedFrame,
    VehicleState,
)
from PIL import Image

pytestmark = pytest.mark.ci_cpu


class _Presenter:
    def __init__(self) -> None:
        self.presented: list[PresentedFrame] = []
        self.closed = False
        self.camera: CameraCalibration | None = None
        self.scene: tuple[object, str] | None = None

    def configure_taxi_camera(self, calibration: CameraCalibration) -> None:
        self.camera = calibration

    def acknowledge_scene_change(self, scene_path: object, variant: str) -> None:
        self.scene = (scene_path, variant)

    def present_frame(self, frame: PresentedFrame, view_mode: str) -> None:
        del view_mode
        self.presented.append(frame)

    def close(self) -> None:
        self.closed = True


def _calibration() -> CameraCalibration:
    return CameraCalibration(
        clipgt_name="camera:test",
        logical_name="camera_test",
        width=8,
        height=6,
        cx=4.0,
        cy=3.0,
        polynomial=np.asarray([0.0, 0.01], dtype=np.float32),
        is_backward_polynomial=True,
        linear_cde=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        sensor_to_rig_flu=np.eye(4, dtype=np.float32),
    )


def _frame() -> PresentedFrame:
    state = VehicleState(
        x_m=12.0,
        y_m=-3.0,
        z_m=1.0,
        yaw_rad=0.25,
        speed_mps=8.0,
        steer_rad=0.1,
        velocity_x_mps=7.0,
        velocity_y_mps=1.0,
    )
    debug = PhysicsDebugFrame(
        ego_position_m=np.asarray([12.0, -3.0, 1.0], dtype=np.float32),
        ego_orientation_xyzw=np.asarray(
            [0.0, 0.0, np.sin(0.125), np.cos(0.125)], dtype=np.float32
        ),
        ego_dimensions_lwh=np.asarray([4.8, 2.0, 1.6], dtype=np.float32),
        actor_positions_m=np.asarray([[18.0, -2.0, 1.0]], dtype=np.float32),
        actor_orientations_xyzw=np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        actor_dimensions_lwh=np.asarray([[4.0, 1.8, 1.5]], dtype=np.float32),
        barrier_segments_xy_m=np.asarray(
            [[[10.0, -5.0], [20.0, -5.0]]], dtype=np.float32
        ),
        barrier_thicknesses_m=np.asarray([0.2], dtype=np.float32),
        barrier_heights_m=np.asarray([1.0], dtype=np.float32),
    )
    return PresentedFrame(
        timestamp_us=123_456,
        rgb_host_uint8=np.full((6, 8, 3), 20, dtype=np.uint8),
        depth_host_f32=None,
        model_rgb_host_uint8=np.full((6, 8, 3), 80, dtype=np.uint8),
        bev_host_uint8=np.full((8, 8, 3), 140, dtype=np.uint8),
        physx_debug=debug,
        rig_to_world=rig_pose_from_vehicle_state(state),
        vehicle_state=state,
    )


def test_diagnostic_presenter_writes_synchronized_artifact(tmp_path: Path) -> None:
    wrapped = _Presenter()
    presenter = AlignmentDiagnosticPresenter(wrapped, tmp_path)
    calibration = _calibration()
    frame = _frame()

    presenter.configure_taxi_camera(calibration)
    presenter.acknowledge_scene_change(Path("scene.usdz"), "rain")
    presenter.present_frame(frame, view_mode="model_rgb")
    presenter.present_frame(frame, view_mode="model_rgb")
    presenter.close()

    metadata = json.loads((presenter.output_dir / "metadata.json").read_text())
    with (presenter.output_dir / "telemetry.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    diagnostic_frame = presenter.output_dir / "frames" / "frame_000000.png"

    assert wrapped.camera is calibration
    assert wrapped.scene == (Path("scene.usdz"), "rain")
    assert wrapped.presented == [frame, frame]
    assert wrapped.closed is True
    assert metadata["frame_count"] == 1
    assert metadata["variant"] == "rain"
    assert metadata["camera"]["logical_name"] == "camera_test"
    assert len(rows) == 1
    assert rows[0]["sequence"] == "0"
    assert float(rows[0]["state_rig_yaw_error_rad"]) == pytest.approx(0.0, abs=1e-6)
    assert float(rows[0]["state_physx_yaw_error_rad"]) == pytest.approx(0.0, abs=1e-6)
    assert float(rows[0]["state_physx_xy_error_m"]) == pytest.approx(0.0)
    assert diagnostic_frame.exists()
    with Image.open(diagnostic_frame) as image:
        assert image.width > image.height
