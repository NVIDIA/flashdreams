# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chunk-anchored GA model actions from V2 Teleop command targets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from itertools import pairwise

import numpy as np
import torch
from handdreams.controllers.quest_controller import ControllerFrame, as_mapping
from handdreams.rotations import (
    matrix_to_rot6d,
    quat_xyzw_to_matrix,
    validate_pose_track,
    validate_rotation,
)

from cosmos3_nano_sim_bimanual import action_format

from .scenes import Scene

_IDENTITY_POSE = np.eye(4, dtype=np.float32)
# AgiBot camera-mask thresholds, evaluated per physical action row.
_STATIONARY_THRESHOLD = 1e-3
_STATIONARY_MIN_ROWS = 11
_GRIPPER_CHANGE_THRESHOLD = 2e-3
_GRIPPER_GUARD_ROWS = 15


@dataclass(frozen=True, slots=True)
class ControllerActionState:
    """Accepted anchors and causal stationary-mask history for one rollout."""

    rows: int = 0
    """Number of accepted action rows, independent of inference chunk size."""
    wrist_anchors: tuple[np.ndarray, np.ndarray] | None = None
    """Right/left absolute poses at the start of the current 16-row block."""
    stationary_rows: int = 0
    """Consecutive motion-free rows outside the gripper guard."""
    gripper_guard_rows: int = 0
    """Following rows still excluded from camera-only conditioning."""


def controller_targets_to_raw_action(
    frames: list[ControllerFrame],
    *,
    wrist_rotations: Mapping[str, np.ndarray] | None = None,
    state: ControllerActionState | None = None,
) -> tuple[np.ndarray, ControllerActionState]:
    """Encode N+1 calibrated wrist targets into N model-format action rows.

    Calibration, gain, clutch, offset and gripper shaping belong to Teleop.
    This codec rotates into the scene wrist convention and measures each target
    against the start of its 16-row action block; camera motion stays identity.
    The returned state must only replace ``state`` after diffusion accepts it.

    Args:
        frames: Chronological absolute wrist targets with explicit gripper_open.
        wrist_rotations: Scene wrist rotations, or the recorded model defaults.
        state: Last accepted row position and anchors; ``None`` for a new rollout.

    Returns:
        Raw 59-D actions and proposed state after consuming the target window.

    Raises:
        ValueError: Fewer than two targets or an invalid pose/gripper target.
        TypeError: Non-numeric gripper target.
    """
    if len(frames) < 2:
        raise ValueError("need at least two controller target frames")
    if state is None:
        state = ControllerActionState()
    rotation_source = wrist_rotations or action_format.WRIST_TO_OPENCV
    model_wrist_rotations = {
        side: validate_rotation(
            np.asarray(rotation_source[side], dtype=np.float32),
            f"{side} scene wrist rotation",
        )
        for side in ("right", "left")
    }

    raw = action_format.identity_raw_action(len(frames) - 1)
    anchors: list[np.ndarray] = []

    for side_index, (side, pose_start, gripper_index) in enumerate(
        (("right", 9, 33), ("left", 34, 58))
    ):
        poses: list[np.ndarray] = []
        grippers: list[float] = []
        for frame in frames:
            controller = as_mapping(frame.get(side))
            if controller is None:
                raise ValueError(f"command target requires the {side} controller")
            position = np.asarray(controller.get("position"), dtype=np.float32)
            orientation = np.asarray(controller.get("orientation"), dtype=np.float32)
            if position.shape != (3,) or orientation.shape != (4,):
                raise ValueError(
                    f"{side} command target requires position[3] and orientation[4]"
                )
            if not np.isfinite(position).all() or not np.isfinite(orientation).all():
                raise ValueError(f"{side} command target pose must be finite")
            pose = _IDENTITY_POSE.copy()
            pose[:3, :3] = (
                quat_xyzw_to_matrix(orientation) @ model_wrist_rotations[side]
            )
            pose[:3, 3] = position
            validate_pose_track(pose[None], f"{side} command target", minimum_frames=1)
            poses.append(pose)

            gripper_open = controller.get("gripper_open")
            if isinstance(gripper_open, (bool, np.bool_)) or not isinstance(
                gripper_open, (int, float, np.integer, np.floating)
            ):
                raise TypeError(f"{side} command target requires numeric gripper_open")
            gripper = float(gripper_open)
            if not np.isfinite(gripper) or not 0.0 <= gripper <= 1.0:
                raise ValueError(f"{side} command gripper_open must be in [0, 1]")
            grippers.append(gripper)

        anchor = (
            None if state.wrist_anchors is None else state.wrist_anchors[side_index]
        )
        for index, current in enumerate(poses[1:]):
            if (state.rows + index) % action_format.POSE_ANCHOR_FRAMES == 0:
                anchor = poses[index]
            if anchor is None:
                raise ValueError("nonzero action-row offset requires wrist anchors")
            delta = np.linalg.inv(anchor) @ current
            raw[index, pose_start : pose_start + 3] = delta[:3, 3]
            raw[index, pose_start + 3 : pose_start + 9] = matrix_to_rot6d(delta[:3, :3])
            raw[index, gripper_index] = (
                grippers[index + 1] * action_format.GRIPPER_OPEN_WIDTH_METERS
            )
        assert anchor is not None
        anchors.append(anchor.copy())
    return raw, replace(
        state,
        rows=state.rows + len(frames) - 1,
        wrist_anchors=(anchors[0], anchors[1]),
    )


def encode_controller_actions(
    frames: list[ControllerFrame], scene: Scene, state: ControllerActionState
) -> tuple[torch.Tensor, ControllerActionState]:
    """Encode calibrated wrist targets as normalized AgiBot action rows.

    Args:
        frames: Chronological absolute wrist targets with explicit gripper openness.
        scene: Conditioning scene providing the initial wrist orientations.
        state: Last accepted action block state.

    Returns:
        CPU float32 padded actions and the proposed action block state.
    """
    raw, next_state = controller_targets_to_raw_action(
        frames, wrist_rotations=scene.wrist_rotations, state=state
    )
    canonical = np.array(
        action_format.normalize_and_pad(raw), dtype=np.float32, copy=True
    )
    return torch.from_numpy(canonical), next_state


def stationary_camera_mask(
    frames: list[ControllerFrame], state: ControllerActionState
) -> tuple[torch.Tensor, ControllerActionState]:
    """Classify validated target rows using a causal stationary-camera mask.

    Use absolute successive targets, not accumulated chunk-anchor deltas. The
    demo's camera is fixed, so these targets are relative to its fixed virtual
    head. Both wrists must stay below the translation/rotation threshold for
    eleven rows. Gripper changes exclude the current and next fifteen rows.

    Unlike offline training labels, this does not relabel earlier rows or look
    ahead to future trigger changes. It adds no controller buffering latency.

    Args:
        frames: N+1 targets already validated by ``encode_controller_actions``.
        state: Proposed anchor state carrying the last accepted mask history.

    Returns:
        N boolean camera-row flags and proposed state, committed on acceptance.
    """
    stationary_rows = state.stationary_rows
    guard_rows = state.gripper_guard_rows
    camera_rows = []
    for previous, current in pairwise(frames):
        movement = 0.0
        gripper_changed = False
        for side in ("left", "right"):
            before = as_mapping(previous[side])
            after = as_mapping(current[side])
            assert before is not None and after is not None
            displacement = np.asarray(after["position"], dtype=np.float64) - np.asarray(
                before["position"], dtype=np.float64
            )
            rotation = quat_xyzw_to_matrix(
                before["orientation"]
            ).T @ quat_xyzw_to_matrix(after["orientation"])
            angle = np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1.0, 1.0))
            movement = max(movement, float(np.linalg.norm(displacement)), 0.3 * angle)
            gripper_changed |= (
                abs(float(after["gripper_open"]) - float(before["gripper_open"]))
                > _GRIPPER_CHANGE_THRESHOLD
            )

        guarded = gripper_changed or guard_rows > 0
        guard_rows = _GRIPPER_GUARD_ROWS if gripper_changed else max(0, guard_rows - 1)
        if guarded or movement > _STATIONARY_THRESHOLD:
            stationary_rows = 0
        else:
            stationary_rows = min(stationary_rows + 1, _STATIONARY_MIN_ROWS)
        camera_rows.append(stationary_rows >= _STATIONARY_MIN_ROWS)

    return torch.tensor(camera_rows, dtype=torch.bool), replace(
        state, stationary_rows=stationary_rows, gripper_guard_rows=guard_rows
    )
