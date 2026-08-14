# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU tests for Taxi causal frame presentation alignment."""

from __future__ import annotations

import numpy as np
import pytest
from crazy_robotaxi.frame_alignment import (
    CausalFrameAlignmentPresenter,
)
from omnidreams_game_engine.math3d import rig_pose_from_vehicle_state
from omnidreams_game_engine.types import PresentedFrame, VehicleState

pytestmark = pytest.mark.ci_cpu


class _Presenter:
    def __init__(self) -> None:
        self.frames: list[PresentedFrame] = []
        self.scene_changes: list[tuple[object, str]] = []

    def present_frame(self, frame: PresentedFrame, view_mode: str) -> None:
        del view_mode
        self.frames.append(frame)

    def acknowledge_scene_change(self, scene_path: object, variant: str) -> None:
        self.scene_changes.append((scene_path, variant))

    def close(self) -> None:
        pass


def _frame(index: int) -> PresentedFrame:
    state = VehicleState(
        x_m=float(index),
        y_m=0.0,
        z_m=0.0,
        yaw_rad=index * 0.1,
        speed_mps=float(index),
        steer_rad=0.0,
    )
    return PresentedFrame(
        timestamp_us=100 + index,
        rgb_host_uint8=f"condition-{index}",
        depth_host_f32=None,
        model_rgb_host_uint8=f"model-{index}",
        bev_host_uint8=f"bev-{index}",
        rig_to_world=rig_pose_from_vehicle_state(state),
        vehicle_state=state,
        application_state=f"game-{index}",
    )


def test_generated_frame_uses_preceding_synchronized_state() -> None:
    wrapped = _Presenter()
    presenter = CausalFrameAlignmentPresenter(wrapped)
    first = _frame(0)
    second = _frame(1)

    presenter.present_frame(first, view_mode="model_rgb")
    presenter.present_frame(second, view_mode="model_rgb")

    assert wrapped.frames[0] is first
    aligned = wrapped.frames[1]
    assert aligned.model_rgb_host_uint8 == "model-1"
    assert aligned.rgb_host_uint8 == "condition-0"
    assert aligned.bev_host_uint8 == "bev-0"
    assert aligned.vehicle_state is first.vehicle_state
    assert aligned.rig_to_world is first.rig_to_world
    assert aligned.application_state == "game-0"
    assert aligned.timestamp_us == first.timestamp_us


def test_representing_frame_does_not_advance_alignment() -> None:
    wrapped = _Presenter()
    presenter = CausalFrameAlignmentPresenter(wrapped)
    first = _frame(0)
    second = _frame(1)

    presenter.present_frame(first, view_mode="model_rgb")
    presenter.present_frame(first, view_mode="model_rgb")
    presenter.present_frame(second, view_mode="model_rgb")

    assert wrapped.frames[0] is first
    assert wrapped.frames[1] is first
    assert wrapped.frames[2].vehicle_state is first.vehicle_state


def test_scene_change_clears_previous_rollout_frame() -> None:
    wrapped = _Presenter()
    presenter = CausalFrameAlignmentPresenter(wrapped)
    previous = _frame(5)
    new_first = _frame(0)

    presenter.present_frame(previous, view_mode="model_rgb")
    presenter.acknowledge_scene_change("scene", "rain")
    presenter.present_frame(new_first, view_mode="model_rgb")

    assert wrapped.scene_changes == [("scene", "rain")]
    assert wrapped.frames[-1] is new_first


def test_loading_frame_clears_previous_rollout_frame() -> None:
    wrapped = _Presenter()
    presenter = CausalFrameAlignmentPresenter(wrapped)
    presenter.present_frame(_frame(5), view_mode="model_rgb")
    loading = PresentedFrame(
        timestamp_us=0,
        rgb_host_uint8=np.zeros((2, 2, 3), dtype=np.uint8),
        depth_host_f32=None,
    )
    new_first = _frame(0)

    presenter.present_frame(loading, view_mode="model_rgb")
    presenter.present_frame(new_first, view_mode="model_rgb")

    assert wrapped.frames[-1] is new_first
