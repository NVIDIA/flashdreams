# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Causal live-domain labeling, without modifying controller targets."""

import copy

import numpy as np
import pytest
import torch
from cosmos3_nano_sim_bimanual.apps.handdreams.actions import (
    ControllerActionState,
    controller_targets_to_raw_action,
    stationary_camera_mask,
)

from .test_controller_actions import _targets

pytestmark = pytest.mark.ci_cpu


def _stationary_targets(count):
    frames = _targets(count)
    for frame in frames:
        for side in ("left", "right"):
            frame[side] = copy.deepcopy(frames[0][side])
            frame[side]["gripper_open"] = 1.0
    return frames


def _classify(frames, state=None):
    _, proposed = controller_targets_to_raw_action(frames, state=state)
    return stationary_camera_mask(frames, proposed)


def test_stationary_mask_waits_for_eleven_rows_and_does_not_mutate_targets():
    frames = _stationary_targets(25)
    original = copy.deepcopy(frames)
    state = ControllerActionState()
    mask, proposed = _classify(frames, state)
    assert mask.tolist() == [False] * 10 + [True] * 14
    assert proposed.stationary_rows == 11
    assert proposed.gripper_guard_rows == 0
    assert state.stationary_rows == 0
    assert frames == original


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("motion", ["translation", "rotation"])
def test_motion_of_either_wrist_immediately_restores_manipulation(side, motion):
    frames = _stationary_targets(40)
    for frame in frames[20:]:
        if motion == "translation":
            frame[side]["position"][0] += 0.002
        else:
            frame[side]["orientation"] = [0, np.sin(0.004 / 2), 0, np.cos(0.004 / 2)]
    mask, _ = _classify(frames)
    assert mask[:19].tolist() == [False] * 10 + [True] * 9
    assert not mask[19:30].any()
    assert mask[30:].all()


@pytest.mark.parametrize("side", ["left", "right"])
def test_trigger_only_motion_uses_fraction_not_metric_width_and_guards_following_rows(
    side,
):
    frames = _stationary_targets(60)
    for frame in frames[20:]:
        # 0.005 native openness is >0.002; its metric width delta is only 0.0006m.
        frame[side]["gripper_open"] = 0.995
    mask, proposed = _classify(frames)
    assert mask[10:19].all()
    assert not mask[19:45].any()
    assert mask[45:].all()
    assert proposed.gripper_guard_rows == 0


@pytest.mark.parametrize("motion", ["translation", "rotation"])
def test_motion_score_is_framewise_not_chunk_anchored(motion):
    frames = _stationary_targets(35)
    for index, frame in enumerate(frames):
        for side in ("left", "right"):
            if motion == "translation":
                frame[side]["position"][0] += 0.0009 * index
            else:
                angle = 0.003 * index
                frame[side]["orientation"] = [
                    0,
                    np.sin(angle / 2),
                    0,
                    np.cos(angle / 2),
                ]
    mask, _ = _classify(frames)
    assert mask.tolist() == [False] * 10 + [True] * 24


@pytest.mark.parametrize("chunk_rows", [1, 5, 8, 17])
def test_guard_and_stationary_counter_are_independent_of_chunk_boundaries(chunk_rows):
    frames = _stationary_targets(65)
    for frame in frames[21:]:
        frame["left"]["gripper_open"] = 0.4
    expected, expected_state = _classify(frames)
    state = ControllerActionState()
    masks = []
    for start in range(0, 64, chunk_rows):
        stop = min(64, start + chunk_rows)
        mask, state = _classify(frames[start : stop + 1], state)
        masks.append(mask)
    torch.testing.assert_close(torch.cat(masks), expected)
    assert state.rows == expected_state.rows
    assert state.stationary_rows == expected_state.stationary_rows
    assert state.gripper_guard_rows == expected_state.gripper_guard_rows
    assert state.wrist_anchors is not None
    assert expected_state.wrist_anchors is not None
    for actual, reference in zip(
        state.wrist_anchors, expected_state.wrist_anchors, strict=True
    ):
        np.testing.assert_array_equal(actual, reference)


def test_repeated_trigger_changes_extend_guard_across_action_windows():
    frames = _stationary_targets(40)
    for index, frame in enumerate(frames[12:20]):
        frame["right"]["gripper_open"] = 1 - (index + 1) * 0.01
    for frame in frames[20:]:
        frame["right"]["gripper_open"] = 0.92
    mask, _ = _classify(frames)
    assert mask[10]
    assert not mask[11:].any()
