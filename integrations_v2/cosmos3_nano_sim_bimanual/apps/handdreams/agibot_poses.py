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

"""Public AgiBot observations converted to initial camera-relative wrist poses."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
from handdreams.rotations import quat_xyzw_to_matrix, validate_rotation

_FLANGE_TO_WRIST_M = 0.0061
"""Local-Z displacement from recorded end pose to the model's gripper base."""


def load_initial_pose(
    state_path: Path,
    camera_path: Path,
    *,
    frame_index: int,
    frame_count: int,
    gripper_hardware: str,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Convert one synchronized source frame to scene calibration targets.

    The public ``state/end`` orientations already have the model wrist basis;
    applying the FK wrist post-rotations again would rotate the claws twice.
    Aligned head-camera extrinsics map camera coordinates into the same robot
    base as the recorded end poses. This uses each episode's camera calibration,
    not the training loader's fixed robot-camera calibration.

    Args:
        state_path: Downloaded ``proprio_stats.h5`` for the selected episode.
        camera_path: Its ``head_extrinsic_params_aligned.json`` sequence.
        frame_index: Zero-based frame shared by video, state and calibration.
        frame_count: Source video frame count.
        gripper_hardware: Audited ``robotiq`` or ``omnipicker`` episode label;
            never inferred from observed actuator values.

    Returns:
        Left/right wrist-to-camera transforms and initial gripper openness.

    Raises:
        ValueError: If source arrays are malformed or are not frame-aligned.
    """
    if (
        type(frame_count) is not int
        or frame_count <= 0
        or type(frame_index) is not int
        or not 0 <= frame_index < frame_count
    ):
        raise ValueError("AgiBot frame index must be within the source video")
    camera, wrists, openness = _load_pose_frames(
        state_path, camera_path, frame_index, frame_count, gripper_hardware
    )
    relative = wrists.copy()
    relative[:, :3, :3] = camera[:3, :3].T @ wrists[:, :3, :3]
    relative[:, :3, 3] = (camera[:3, :3].T @ (wrists[:, :3, 3] - camera[:3, 3]).T).T
    return (
        {side: relative[index] for index, side in enumerate(("left", "right"))},
        {side: float(openness[index]) for index, side in enumerate(("left", "right"))},
    )


def _read_state(
    state_path: Path,
    shapes: dict[str, tuple[int, ...]],
    selection: int | slice,
    frame_count: int,
) -> dict[str, np.ndarray]:
    rows = {}
    with h5py.File(state_path, "r") as source:
        for name, shape in shapes.items():
            key = f"state/{name}"
            # Do not follow external HDF5 links or datasets with external files.
            parts = key.split("/")
            for depth in range(1, len(parts) + 1):
                link = source.get("/".join(parts[:depth]), getlink=True)
                if not isinstance(link, h5py.HardLink):
                    raise ValueError(f"AgiBot {key} requires an internal dataset")
            dataset = source[key]
            if (
                not isinstance(dataset, h5py.Dataset)
                or dataset.shape != (frame_count, *shape)
                or dataset.external
                or dataset.is_virtual
            ):
                raise ValueError(f"AgiBot {key} and video shapes do not match")
            value = np.asarray(dataset[selection], dtype=np.float32)
            if not np.isfinite(value).all():
                raise ValueError(f"AgiBot {key} must be finite")
            rows[name] = value
    return rows


def _load_pose_frames(
    state_path: Path,
    camera_path: Path,
    selection: int | slice,
    frame_count: int,
    gripper_hardware: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if gripper_hardware not in ("robotiq", "omnipicker"):
        raise ValueError(f"Unsupported AgiBot gripper hardware: {gripper_hardware!r}")
    with camera_path.open(encoding="utf-8") as stream:
        camera_frames = json.load(stream)
    if not isinstance(camera_frames, list) or len(camera_frames) != frame_count:
        raise ValueError("AgiBot camera and video frame counts do not match")
    try:
        frames = camera_frames[selection]
        if isinstance(selection, int):
            frames = [frames]
        rotation = validate_rotation(
            [frame["extrinsic"]["rotation_matrix"] for frame in frames], "AgiBot camera"
        )
        translation = np.asarray(
            [frame["extrinsic"]["translation_vector"] for frame in frames],
            dtype=np.float32,
        )
    except (KeyError, TypeError) as error:
        raise ValueError("Malformed AgiBot head-camera extrinsics") from error
    if rotation.shape != (len(frames), 3, 3) or translation.shape != (len(frames), 3):
        raise ValueError("AgiBot camera requires one rotation and translation")
    if not np.isfinite(translation).all():
        raise ValueError("AgiBot camera translation must be finite")

    rows = _read_state(
        state_path,
        {"end/position": (2, 3), "end/orientation": (2, 4), "effector/position": (2,)},
        selection,
        frame_count,
    )

    wrist_rotations = quat_xyzw_to_matrix(rows["end/orientation"])
    grippers = rows["effector/position"]
    if np.any(grippers < -1e-4) or np.any(grippers > 125.0):
        raise ValueError("AgiBot grippers require actuator-close degrees [0,125]")
    # Both actuators close at 120; Robotiq reaches full opening at 35, not zero.
    open_position = 35.0 if gripper_hardware == "robotiq" else 0.0
    openness = 1.0 - np.clip(
        (grippers - open_position) / (120.0 - open_position), 0.0, 1.0
    )

    wrists = np.broadcast_to(
        np.eye(4, dtype=np.float32), (*openness.shape, 4, 4)
    ).copy()
    wrists[..., :3, :3] = wrist_rotations
    wrists[..., :3, 3] = (
        rows["end/position"] + wrist_rotations[..., :, 2] * _FLANGE_TO_WRIST_M
    )
    camera = np.tile(np.eye(4, dtype=np.float32), (len(frames), 1, 1))
    camera[:, :3, :3] = rotation
    camera[:, :3, 3] = translation
    return camera[0] if isinstance(selection, int) else camera, wrists, openness


def load_pose_sequence(
    state_path: Path,
    camera_path: Path,
    *,
    frame_index: int,
    frame_count: int,
    gripper_hardware: str,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load synchronized head/wrist world poses and openness for recorded replay.

    Use the episode's recorded camera calibration and end poses, including robot
    base motion. At least two observations are required for one action.
    """
    if (
        type(frame_count) is not int
        or type(frame_index) is not int
        or not 0 <= frame_index < frame_count - 1
    ):
        raise ValueError("AgiBot replay frame index requires at least two observations")
    selection = slice(frame_index, None)
    camera, wrists, openness = _load_pose_frames(
        state_path, camera_path, selection, frame_count, gripper_hardware
    )
    robot = _read_state(
        state_path,
        {"robot/position": (3,), "robot/orientation": (4,)},
        selection,
        frame_count,
    )
    base = np.tile(np.eye(4, dtype=np.float32), (len(camera), 1, 1))
    base[:, :3, :3] = quat_xyzw_to_matrix(robot["robot/orientation"])
    base[:, :3, 3] = robot["robot/position"]
    motion = np.linalg.inv(base[0]) @ base
    return (
        motion @ camera,
        {side: motion @ wrists[:, i] for i, side in enumerate(("left", "right"))},
        {side: openness[:, i] for i, side in enumerate(("left", "right"))},
    )
