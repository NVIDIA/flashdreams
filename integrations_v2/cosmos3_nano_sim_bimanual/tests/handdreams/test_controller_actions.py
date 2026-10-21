# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GA action layout and 16-frame anchoring of calibrated controller targets."""

import copy
from typing import Any

import numpy as np
import pytest
from cosmos3_nano_sim_bimanual import action_format
from cosmos3_nano_sim_bimanual.apps.handdreams.actions import (
    ControllerActionState,
    controller_targets_to_raw_action,
)
from handdreams.rotations import matrix_to_rot6d, quat_xyzw_to_matrix

from apps.handdreams.tests.test_teleop_pipeline import (  # ty: ignore[unresolved-import]  # pytest exposes these shared helpers via the repository root.
    _frame,
    _pipeline,
    _through_atomic_event,
)

pytestmark = pytest.mark.ci_cpu


def _targets(count: int) -> list[dict[str, Any]]:
    frames = [_frame(index + 1) for index in range(count)]
    for index, frame in enumerate(frames):
        for side in ("left", "right"):
            frame[side]["position"][0] += 0.01 * index
            frame[side]["orientation"] = [
                0.0,
                np.sin(0.03 * index),
                0.0,
                np.cos(0.03 * index),
            ]
            frame[side]["gripper_open"] = index / max(1, count - 1)
    return frames


@pytest.mark.parametrize("atomic_input", [False, True])
def test_calibrated_targets_preserve_source_timestamps_and_metric_grippers(
    atomic_input: bool,
) -> None:
    pipeline, wrist = _pipeline(slew=0.25)
    frames = [copy.deepcopy(wrist) for _ in range(3)]
    for index, frame in enumerate(frames):
        frame["client_capture_ns"] = 1 + index
        frame["left"]["position"][0] += 0.01 * index
        frame["right"]["position"][1] -= 0.02 * index
    if atomic_input:
        frames = [
            _through_atomic_event(frame, frame_id=i) for i, frame in enumerate(frames)
        ]

    prepared = pipeline.prepare_window(frames, command_sequence=7)
    before = copy.deepcopy(prepared.frames)
    actual, state = controller_targets_to_raw_action(prepared.frames)

    assert prepared.frames == before
    assert [frame["client_capture_ns"] for frame in prepared.frames] == [1, 2, 3]
    assert state.rows == 2
    assert actual.shape == (2, 59)
    for side, index in (("right", 33), ("left", 58)):
        np.testing.assert_allclose(
            actual[:, index],
            [frame[side]["gripper_open"] * 0.12 for frame in prepared.frames[1:]],
        )
    normalized = action_format.normalize_and_pad(actual)
    for start, stop in ((18, 33), (43, 58), (59, 64)):
        np.testing.assert_array_equal(normalized[:, start:stop], 0)


@pytest.mark.parametrize("custom_wrist_rotations", [False, True])
def test_actions_match_training_chunk_anchor_formula(
    custom_wrist_rotations: bool,
) -> None:
    frames = _targets(34)
    rotations = (
        {side: np.eye(3, dtype=np.float32) for side in ("left", "right")}
        if custom_wrist_rotations
        else action_format.WRIST_TO_OPENCV
    )
    actual, state = controller_targets_to_raw_action(frames, wrist_rotations=rotations)
    assert state.rows == 33

    for row in range(33):
        anchor_index = (row // 16) * 16
        for side, start in (("right", 9), ("left", 34)):
            anchor = frames[anchor_index][side]
            current = frames[row + 1][side]
            anchor_rotation = (
                quat_xyzw_to_matrix(anchor["orientation"]) @ rotations[side]
            )
            current_rotation = (
                quat_xyzw_to_matrix(current["orientation"]) @ rotations[side]
            )
            expected_translation = anchor_rotation.T @ (
                np.asarray(current["position"]) - np.asarray(anchor["position"])
            )
            np.testing.assert_allclose(
                actual[row, start : start + 3], expected_translation, atol=1e-6
            )
            np.testing.assert_allclose(
                actual[row, start + 3 : start + 9],
                matrix_to_rot6d(anchor_rotation.T @ current_rotation),
                atol=1e-6,
            )


@pytest.mark.parametrize("chunk_rows", [5, 8, 17])
def test_anchor_state_spans_inference_chunks_with_n_plus_one_overlap(
    chunk_rows,
) -> None:
    frames = _targets(49)
    expected, _ = controller_targets_to_raw_action(frames)
    state = ControllerActionState()
    chunks = []
    for start in range(0, 48, chunk_rows):
        stop = min(48, start + chunk_rows)
        previous = state
        raw, state = controller_targets_to_raw_action(
            frames[start : stop + 1], state=state
        )
        assert previous.rows == start
        assert state.rows == stop
        chunks.append(raw)
    np.testing.assert_array_equal(np.concatenate(chunks), expected)


def test_rejected_action_proposal_does_not_advance_or_alias_state() -> None:
    frames = _targets(25)
    _, committed = controller_targets_to_raw_action(frames[:9])
    assert committed.wrist_anchors is not None
    anchors = tuple(value.copy() for value in committed.wrist_anchors)
    first, proposed = controller_targets_to_raw_action(frames[8:17], state=committed)
    repeated, _ = controller_targets_to_raw_action(frames[8:17], state=committed)
    assert committed.rows == 8
    assert proposed.rows == 16
    np.testing.assert_array_equal(first, repeated)
    for actual, expected in zip(committed.wrist_anchors, anchors, strict=True):
        np.testing.assert_array_equal(actual, expected)
    frames[0]["right"]["position"][0] = 100
    np.testing.assert_array_equal(committed.wrist_anchors[0], anchors[0])


@pytest.mark.parametrize("invalid_gripper", [None, True, float("nan"), -0.1, 1.1])
def test_action_codec_rejects_invalid_command_gripper(invalid_gripper: Any) -> None:
    frames = _targets(2)
    frames[1]["right"]["gripper_open"] = invalid_gripper
    with pytest.raises((ValueError, TypeError), match="gripper_open"):
        controller_targets_to_raw_action(frames)


def test_nonzero_row_offset_requires_the_existing_anchor() -> None:
    with pytest.raises(ValueError, match="requires wrist anchors"):
        controller_targets_to_raw_action(
            _targets(2), state=ControllerActionState(rows=8)
        )


def test_source_clock_gap_does_not_restart_the_model_anchor_block() -> None:
    frames = _targets(25)
    expected, _ = controller_targets_to_raw_action(frames)
    _, state = controller_targets_to_raw_action(frames[:9])
    resumed = copy.deepcopy(frames[16:25])
    for frame in resumed:
        frame["client_capture_ns"] += 10_000_000_000
    actual, state = controller_targets_to_raw_action(resumed, state=state)
    # Eight generated rows, not the wall clock or number of source observations,
    # have elapsed. These rows must still be relative to the original pose.
    reference, _ = controller_targets_to_raw_action([frames[0], *frames[17:25]])
    np.testing.assert_array_equal(actual, reference)
    assert not np.allclose(actual[:, 9:12], expected[16:24, 9:12])
    assert state.rows == 16
