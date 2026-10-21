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

"""Synthetic public-state conversion checks without downloaded dataset content."""

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
from cosmos3_nano_sim_bimanual.apps.handdreams.agibot_poses import (
    load_initial_pose,
    load_pose_sequence,
)

pytestmark = pytest.mark.ci_cpu


def test_replay_sequence_shares_initial_calibration_and_includes_base_motion(tmp_path):
    state, camera = _source(tmp_path)
    with h5py.File(state, "r+") as source:
        source["state/robot/position"] = np.column_stack(
            (np.arange(40) * 0.01, np.zeros((40, 2)))
        )
        source["state/robot/orientation"] = np.tile([0, 0, 0, 1], (40, 1))
    kwargs: dict[str, Any] = dict(
        frame_index=7, frame_count=40, gripper_hardware="robotiq"
    )
    head, wrists, grippers = load_pose_sequence(state, camera, **kwargs)
    initial, opened = load_initial_pose(state, camera, **kwargs)
    assert head.shape == (33, 4, 4)
    for side in ("left", "right"):
        np.testing.assert_allclose(
            np.linalg.inv(head[0]) @ wrists[side][0], initial[side], atol=1e-6
        )
        assert grippers[side][0] == opened[side]
        np.testing.assert_allclose(
            wrists[side][-1, :3, 3] - wrists[side][0, :3, 3], [0.32, 0, 0], atol=1e-6
        )
    np.testing.assert_allclose(
        head[-1, :3, 3] - head[0, :3, 3], [0.32, 0, 0], atol=1e-6
    )


@pytest.mark.parametrize(
    "bad",
    ["missing_base", "external_base", "zero_quaternion", "tail_nan", "tail_camera"],
)
def test_replay_validates_entire_trajectory(tmp_path, bad):
    state, camera = _source(tmp_path)
    with h5py.File(state, "r+") as source:
        source["state/robot/position"] = np.zeros((40, 3))
        if bad != "missing_base":
            source["state/robot/orientation"] = np.tile([0, 0, 0, 1], (40, 1))
        if bad == "external_base":
            del source["state/robot/position"]
            source["state/robot/position"] = h5py.ExternalLink("other.h5", "/pose")
        if bad == "zero_quaternion":
            source["state/robot/orientation"][-1] = 0
        if bad == "tail_nan":
            source["state/end/position"][-1] = np.nan
    if bad == "tail_camera":
        data = json.loads(camera.read_text())
        data[-1]["extrinsic"]["rotation_matrix"][2][2] = -1
        camera.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_pose_sequence(
            state, camera, frame_index=0, frame_count=40, gripper_hardware="robotiq"
        )


def _source(tmp_path: Path, frames: int = 40) -> tuple[Path, Path]:
    state_path = tmp_path / "state.h5"
    with h5py.File(state_path, "w") as source:
        source["state/end/position"] = np.tile(
            [[2.0, 3.0, 4.0], [-1.0, 2.0, 3.0]], (frames, 1, 1)
        )
        # The left wrist points along +X; its local-Z offset is not world-Z.
        source["state/end/orientation"] = np.tile(
            [[0.0, 2**-0.5, 0.0, 2**-0.5], [0.0, 0.0, 0.0, 1.0]],
            (frames, 1, 1),
        )
        source["state/effector/position"] = np.tile([35.2, 120.0], (frames, 1))
    camera_path = tmp_path / "camera.json"
    camera_path.write_text(
        json.dumps(
            [
                {
                    "extrinsic": {
                        "rotation_matrix": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
                        "translation_vector": [1.0, 2.0, 3.0],
                    }
                }
                for _ in range(frames)
            ]
        ),
        encoding="utf-8",
    )
    return state_path, camera_path


def test_recorded_basis_and_local_offset_are_applied_once(tmp_path: Path) -> None:
    state_path, camera_path = _source(tmp_path)
    poses, grippers = load_initial_pose(
        state_path,
        camera_path,
        frame_index=0,
        frame_count=40,
        gripper_hardware="robotiq",
    )
    np.testing.assert_allclose(poses["left"][:3, 3], [1.0, -1.0061, 1.0], atol=1e-6)
    np.testing.assert_allclose(poses["right"][:3, 3], [0, 2, 0.0061], atol=1e-6)
    np.testing.assert_allclose(
        poses["left"][:3, :3], [[0, 1, 0], [0, 0, -1], [-1, 0, 0]], atol=1e-6
    )
    assert grippers == pytest.approx({"left": (120.0 - 35.2) / 85, "right": 0.0})
    assert poses["left"].dtype == np.float32
    np.testing.assert_array_equal(poses["left"][3], [0, 0, 0, 1])


def test_nonzero_clip_frame_uses_matching_state_and_camera(tmp_path: Path) -> None:
    state_path, camera_path = _source(tmp_path)
    with h5py.File(state_path, "r+") as source:
        source["state/end/position"][39, 0] = [9, 8, 7]
        source["state/effector/position"][39] = [0.217, 124]
    cameras = json.loads(camera_path.read_text())
    cameras[39]["extrinsic"]["translation_vector"] = [5, 6, 7]
    camera_path.write_text(json.dumps(cameras))
    poses, grippers = load_initial_pose(
        state_path,
        camera_path,
        frame_index=39,
        frame_count=40,
        gripper_hardware="omnipicker",
    )
    np.testing.assert_allclose(poses["left"][:3, 3], [2, -4.0061, 0], atol=1e-6)
    assert grippers == pytest.approx({"left": 1 - 0.217 / 120, "right": 0})


@pytest.mark.parametrize("frame", [-1, 40, 0.5, True])
def test_rejects_invalid_frame_index(tmp_path: Path, frame: object) -> None:
    with pytest.raises(ValueError, match="frame index"):
        load_initial_pose(
            *_source(tmp_path),
            frame_index=frame,  # ty: ignore[invalid-argument-type]  # Exercise invalid-input validation.
            frame_count=40,
            gripper_hardware="robotiq",
        )


@pytest.mark.parametrize(
    "key", ["state/end/position", "state/end/orientation", "state/effector/position"]
)
def test_rejects_unaligned_state_counts(tmp_path: Path, key: str) -> None:
    state_path, camera_path = _source(tmp_path)
    with h5py.File(state_path, "r+") as source:
        values = source[key][:-1]
        del source[key]
        source[key] = values
    with pytest.raises(ValueError, match="shapes do not match"):
        load_initial_pose(
            state_path,
            camera_path,
            frame_index=0,
            frame_count=40,
            gripper_hardware="robotiq",
        )


def test_rejects_unaligned_camera_count(tmp_path: Path) -> None:
    state_path, camera_path = _source(tmp_path)
    camera_path.write_text(json.dumps(json.loads(camera_path.read_text())[:-1]))
    with pytest.raises(ValueError, match="camera and video frame counts"):
        load_initial_pose(
            state_path,
            camera_path,
            frame_index=0,
            frame_count=40,
            gripper_hardware="robotiq",
        )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("state/end/position", np.nan),
        ("state/end/orientation", 0),
        ("state/effector/position", -1),
        ("state/effector/position", 126),
    ],
)
def test_rejects_invalid_state_values(tmp_path: Path, key: str, value: float) -> None:
    state_path, camera_path = _source(tmp_path)
    with h5py.File(state_path, "r+") as source:
        source[key][0] = value
    with pytest.raises(ValueError):
        load_initial_pose(
            state_path,
            camera_path,
            frame_index=0,
            frame_count=40,
            gripper_hardware="robotiq",
        )


def test_rejects_improper_camera_rotation(tmp_path: Path) -> None:
    state_path, camera_path = _source(tmp_path)
    cameras = json.loads(camera_path.read_text())
    cameras[0]["extrinsic"]["rotation_matrix"][2][2] = -1
    camera_path.write_text(json.dumps(cameras))
    with pytest.raises(ValueError):
        load_initial_pose(
            state_path,
            camera_path,
            frame_index=0,
            frame_count=40,
            gripper_hardware="robotiq",
        )


@pytest.mark.parametrize("kind", ["missing", "group", "external", "soft"])
def test_rejects_nonlocal_or_missing_datasets(tmp_path: Path, kind: str) -> None:
    state_path, camera_path = _source(tmp_path)
    with h5py.File(state_path, "r+") as source:
        del source["state/end/position"]
        if kind == "group":
            source.create_group("state/end/position")
        elif kind == "external":
            source["state/end/position"] = h5py.ExternalLink("other.h5", "/pose")
        elif kind == "soft":
            source["state/end/position"] = h5py.SoftLink("/state/end/orientation")
    with pytest.raises(ValueError):
        load_initial_pose(
            state_path,
            camera_path,
            frame_index=0,
            frame_count=40,
            gripper_hardware="robotiq",
        )


@pytest.mark.parametrize(
    ("hardware", "values", "expected"),
    [
        ("robotiq", [35, 120], [1, 0]),
        ("robotiq", [77.5, 34.8], [0.5, 1]),
        ("robotiq", [120.73, 124], [0, 0]),
        ("omnipicker", [0, 120], [1, 0]),
        ("omnipicker", [60, 35], [0.5, 1 - 35 / 120]),
    ],
)
def test_hardware_calibration_preserves_wrist_poses(
    tmp_path: Path, hardware: str, values: list[float], expected: list[float]
) -> None:
    state, camera = _source(tmp_path)
    with h5py.File(state, "r+") as source:
        source["state/effector/position"][0] = values
    poses, grippers = load_initial_pose(
        state, camera, frame_index=0, frame_count=40, gripper_hardware=hardware
    )
    baseline, _ = load_initial_pose(
        state, camera, frame_index=0, frame_count=40, gripper_hardware="omnipicker"
    )
    for index, side in enumerate(("left", "right")):
        assert grippers[side] == pytest.approx(expected[index], abs=1e-7)
        np.testing.assert_array_equal(poses[side], baseline[side])


@pytest.mark.parametrize("hardware", ["unknown", "", None])
def test_rejects_unknown_gripper_hardware(tmp_path: Path, hardware: object) -> None:
    with pytest.raises(ValueError, match="gripper hardware"):
        load_initial_pose(
            *_source(tmp_path),
            frame_index=0,
            frame_count=40,
            gripper_hardware=hardware,  # ty: ignore[invalid-argument-type]  # Exercise invalid-input validation.
        )
