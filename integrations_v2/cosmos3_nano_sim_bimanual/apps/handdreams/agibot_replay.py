# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recorded public AgiBot trajectories converted to bimanual model conditioning."""

from pathlib import Path

import numpy as np
from handdreams.rotations import matrix_to_rot6d, validate_pose_track

from ...action_format import (
    AGIBOT_DOMAIN_ID,
    CAMERA_DOMAIN_ID,
    GRIPPER_OPEN_WIDTH_METERS,
    POSE_ANCHOR_FRAMES,
    identity_raw_action,
    normalize_and_pad,
)
from .agibot_download import download_archive_member
from .agibot_poses import load_pose_sequence
from .agibot_scenes import _frame, _manifest, prepare_default_scenes


def encode_recorded_actions(
    head: np.ndarray,
    wrists: dict[str, np.ndarray],
    grippers: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Encode synchronized observations into normalized actions and domain IDs.

    All poses share a world frame. Row zero targets observation one; each group
    of 16 targets is anchored at the preceding observation. Stationary hands
    use camera-only conditioning, with the offline training rule: an entire
    run of at least 11 small transitions, excluding 15 rows either side of a
    gripper change. No live smoothing or controller calibration is applied.
    """
    head = validate_pose_track(head, "AgiBot head")
    count = len(head) - 1
    anchors = np.arange(count) // POSE_ANCHOR_FRAMES * POSE_ANCHOR_FRAMES
    raw = identity_raw_action(count)
    scores = np.zeros(count, dtype=np.float32)
    changes = np.zeros(count, dtype=bool)
    inverse_head = np.linalg.inv(head)
    for side, start, grip in (("right", 9, 33), ("left", 34, 58)):
        track = validate_pose_track(wrists[side], f"AgiBot {side} wrist")
        opened = np.asarray(grippers[side], dtype=np.float32)
        if len(track) != len(head) or opened.shape != (len(head),):
            raise ValueError("AgiBot replay tracks must have matching frame counts")
        if not np.isfinite(opened).all() or np.any((opened < 0) | (opened > 1)):
            raise ValueError("AgiBot replay gripper openness must be finite [0,1]")
        delta = np.linalg.inv(track[anchors]) @ track[1:]
        raw[:, start : start + 3] = delta[:, :3, 3]
        raw[:, start + 3 : start + 9] = matrix_to_rot6d(delta[:, :3, :3])
        raw[:, grip] = opened[1:] * GRIPPER_OPEN_WIDTH_METERS

        # Score adjacent wrist motion in the moving head frame, not chunk anchors.
        relative = inverse_head @ track
        step = np.linalg.inv(relative[:-1]) @ relative[1:]
        angle = np.arccos(
            np.clip((np.trace(step[:, :3, :3], axis1=1, axis2=2) - 1) / 2, -1, 1)
        )
        scores = np.maximum(
            scores, np.maximum(np.linalg.norm(step[:, :3, 3], axis=1), 0.3 * angle)
        )
        changes |= np.abs(np.diff(opened)) > 2e-3

    delta = np.linalg.inv(head[anchors]) @ head[1:]
    raw[:, :3] = delta[:, :3, 3]
    raw[:, 3:9] = matrix_to_rot6d(delta[:, :3, :3])
    guarded = (
        np.convolve(changes.astype(np.int32), np.ones(31, dtype=np.int32), mode="full")[
            15 : 15 + count
        ]
        > 0
    )
    still = (scores <= 1e-3) & ~guarded
    boundaries = np.flatnonzero(np.diff(np.r_[False, still, False]))
    camera_rows = np.zeros(count, dtype=bool)
    for start, stop in boundaries.reshape(-1, 2):
        if stop - start >= 11:
            camera_rows[start:stop] = True
    actions = normalize_and_pad(raw)
    # Missing hands in the camera domain are normalized zeros, not raw zeros.
    actions[camera_rows, 9:] = 0
    domains = np.where(camera_rows, CAMERA_DOMAIN_ID, AGIBOT_DOMAIN_ID).astype(np.int64)
    return actions, domains


def prepare_dataset_replays() -> dict[str, tuple[Path, np.ndarray, np.ndarray]]:
    """Prepare Fold the T-shirt and its recorded actions during init.

    Reuse authenticated, checksum-verified source downloads and scene caching.
    Convert the small state sequence in memory so stale normalized actions are
    never reused after a change to the model contract. The source camera
    calibration is episode-specific, not training's fixed URDF calibration.
    """
    manifest = _manifest()
    root = prepare_default_scenes()
    recordings = {}
    for scene in manifest["scenes"]:
        name = scene["name"]
        if name != "fold-tshirt":
            continue
        members = {
            kind: download_archive_member(
                repo_id=manifest["repo_id"],
                revision=manifest["revision"],
                **scene[kind],
            )
            for kind in ("video", "state", "camera")
        }
        _, frame_count = _frame(members["video"], scene["frame"])
        actions, domains = encode_recorded_actions(
            *load_pose_sequence(
                members["state"],
                members["camera"],
                frame_index=scene["frame"],
                frame_count=frame_count,
                gripper_hardware=scene["gripper_hardware"],
            )
        )
        print(
            f"[handdreams-replay] {name}: {len(actions)} recorded actions at 30 FPS; "
            f"{np.count_nonzero(domains == CAMERA_DOMAIN_ID)} stationary camera-domain rows",
            flush=True,
        )
        recordings[name] = root / name, actions, domains
    return recordings
