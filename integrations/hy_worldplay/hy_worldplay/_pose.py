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

"""Pose-string parser for the HY-WorldPlay action / camera conditioner.

Produces both the discrete action labels consumed by the action
conditioner and the per-latent W2C / intrinsic matrices consumed by
the camera conditioner. Discrete labels follow the 81-class encoding
``trans * 9 + rotate``, with ``trans`` and ``rotate`` each drawn from
the 9-entry one-hot mapping in ``_TRANS_ROTATE_TABLE`` below.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from scipy.spatial.transform import Rotation as _ScipyRotation
from torch import Tensor

# Maps 4-bit one-hot ``(forward, backward, right, left)`` (translation) or
# ``(right, left, up, down)`` (rotation) to a single 0..8 class label so the
# combined action lives in ``[0, 81)``.
_TRANS_ROTATE_TABLE: Mapping[tuple[int, int, int, int], int] = {
    (0, 0, 0, 0): 0,
    (1, 0, 0, 0): 1,
    (0, 1, 0, 0): 2,
    (0, 0, 1, 0): 3,
    (0, 0, 0, 1): 4,
    (1, 0, 1, 0): 5,
    (1, 0, 0, 1): 6,
    (0, 1, 1, 0): 7,
    (0, 1, 0, 1): 8,
}

_FORWARD_SPEED = 0.08
"""Per-step forward / strafe translation magnitude in world units."""

_YAW_SPEED = float(np.deg2rad(3))
"""Per-step yaw step in radians (3°)."""

_PITCH_SPEED = float(np.deg2rad(3))
"""Per-step pitch step in radians (3°)."""

_DEFAULT_INTRINSIC: list[list[float]] = [
    [969.6969696969696, 0.0, 960.0],
    [0.0, 969.6969696969696, 540.0],
    [0.0, 0.0, 1.0],
]
"""Default camera intrinsic matrix for a 1920x1080 image (pre-normalisation)."""

_MOVE_NORM_THRESHOLD = 1e-4
"""Translation magnitude below which a per-step move is treated as no translation."""


def _rot_x(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _parse_pose_string(pose_string: str) -> list[dict[str, float]]:
    """Parse a comma-separated motion script into per-frame motion dicts.

    Each command is ``action-N`` where ``N`` is the number of latent
    frames to apply the motion across. Accepted actions:

    * ``w`` / ``s``: forward / backward translation.
    * ``a`` / ``d``: left / right strafe.
    * ``up`` / ``down``: pitch up / down.
    * ``left`` / ``right``: yaw left / right.
    """
    motions: list[dict[str, float]] = []
    for raw in pose_string.split(","):
        cmd = raw.strip()
        if not cmd:
            continue
        parts = cmd.split("-")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid pose command {cmd!r}: expected 'action-duration'."
            )
        action = parts[0].strip()
        try:
            duration = float(parts[1].strip())
        except ValueError as exc:
            raise ValueError(f"Invalid duration in command {cmd!r}.") from exc
        n = int(duration)
        if action == "w":
            motions.extend({"forward": _FORWARD_SPEED} for _ in range(n))
        elif action == "s":
            motions.extend({"forward": -_FORWARD_SPEED} for _ in range(n))
        elif action == "a":
            motions.extend({"right": -_FORWARD_SPEED} for _ in range(n))
        elif action == "d":
            motions.extend({"right": _FORWARD_SPEED} for _ in range(n))
        elif action == "up":
            motions.extend({"pitch": _PITCH_SPEED} for _ in range(n))
        elif action == "down":
            motions.extend({"pitch": -_PITCH_SPEED} for _ in range(n))
        elif action == "left":
            motions.extend({"yaw": -_YAW_SPEED} for _ in range(n))
        elif action == "right":
            motions.extend({"yaw": _YAW_SPEED} for _ in range(n))
        else:
            raise ValueError(
                f"Unknown action {action!r}. "
                "Supported: w, s, a, d, up, down, left, right."
            )
    return motions


def _generate_trajectory_c2w(motions: list[dict[str, float]]) -> np.ndarray:
    """Integrate per-step motions into a camera-to-world trajectory.

    Returns:
        ``[n_motions + 1, 4, 4]`` array of C2W matrices; the first entry is
        the identity (the rollout's initial pose).
    """
    poses: list[np.ndarray] = []
    T = np.eye(4)
    poses.append(T.copy())
    for move in motions:
        if "yaw" in move:
            T[:3, :3] = T[:3, :3] @ _rot_y(move["yaw"])
        if "pitch" in move:
            T[:3, :3] = T[:3, :3] @ _rot_x(move["pitch"])
        forward = move.get("forward", 0.0)
        if forward != 0.0:
            T[:3, 3] += T[:3, :3] @ np.array([0.0, 0.0, forward])
        right = move.get("right", 0.0)
        if right != 0.0:
            T[:3, 3] += T[:3, :3] @ np.array([right, 0.0, 0.0])
        poses.append(T.copy())
    return np.stack(poses, axis=0)


def _pose_string_to_json(pose_string: str) -> dict[str, dict[str, list]]:
    """Convert a pose string into the JSON-shaped dict the parity script consumes.

    Keys are stringified indices so that pose files dumped from upstream
    round-trip identically through ``json.load`` / ``json.dump``.
    """
    motions = _parse_pose_string(pose_string)
    c2w = _generate_trajectory_c2w(motions)
    out: dict[str, dict[str, list]] = {}
    for i, p in enumerate(c2w):
        out[str(i)] = {"extrinsic": p.tolist(), "K": _DEFAULT_INTRINSIC}
    return out


def _one_hot_to_label(one_hot: np.ndarray) -> Tensor:
    """Map per-frame 4-bit one-hot rows to single 0..8 class labels."""
    return torch.tensor(
        [_TRANS_ROTATE_TABLE[tuple(row.tolist())] for row in one_hot],
        dtype=torch.long,
    )


def parse_pose_data(
    pose_data: str | Path | Mapping[str, Mapping[str, list]],
    n_latents: int,
    *,
    third_person: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Parse a pose script (or file/dict) into per-latent W2C / K / action labels.

    Args:
        pose_data: One of:

            * A path (``str`` ending in ``.json`` or ``Path``) to an
              upstream-format pose JSON.
            * A pose-script string like ``"w-3, right-0.5, d-4"``.
            * A pre-parsed mapping with the same shape as the JSON file.
        n_latents: Number of latent frames the rollout will produce
            (``len_t`` times the AR-step count). The parsed pose script
            must produce exactly this many entries.
        third_person: When ``True``, only emit translation classes for
            frames with small yaw / pitch (matches upstream's ``tps`` flag).

    Returns:
        Tuple ``(w2c, K, action_labels)``:

            * ``w2c`` has shape ``[n_latents, 4, 4]`` (world-to-camera).
            * ``K`` has shape ``[n_latents, 3, 3]`` with cx/cy renormalised
              to 0.5.
            * ``action_labels`` has shape ``[n_latents]`` and contains
              0..80 class indices (``trans * 9 + rotate``).
    """
    if isinstance(pose_data, (str, Path)):
        s = str(pose_data)
        if s.endswith(".json"):
            pose_json: Mapping[str, Mapping[str, list]] = json.loads(
                Path(s).read_text()
            )
        else:
            pose_json = _pose_string_to_json(s)
    elif isinstance(pose_data, Mapping):
        pose_json = pose_data
    else:
        raise TypeError(
            f"Invalid pose_data type: {type(pose_data).__name__}. "
            "Expected str, Path, or Mapping."
        )

    keys = list(pose_json.keys())
    if len(keys) != n_latents:
        raise ValueError(
            f"pose corresponds to {len(keys)} latents, but n_latents={n_latents}. "
            "Pass a pose script of matching length or adjust n_latents."
        )

    c2w_list: list[np.ndarray] = []
    K_list: list[np.ndarray] = []
    for i in range(n_latents):
        entry = pose_json[keys[i]]
        c2w_i = np.asarray(entry["extrinsic"], dtype=np.float64)
        K_i = np.asarray(entry["K"], dtype=np.float64)
        c2w_list.append(c2w_i)

        # Normalise principal point to (0.5, 0.5) and rescale focals so the
        # resulting intrinsic is resolution-independent.
        K_i = K_i.copy()
        K_i[0, 0] /= K_i[0, 2] * 2.0
        K_i[1, 1] /= K_i[1, 2] * 2.0
        K_i[0, 2] = 0.5
        K_i[1, 2] = 0.5
        K_list.append(K_i)

    c2w = np.stack(c2w_list, axis=0)
    K = np.stack(K_list, axis=0)
    w2c = np.linalg.inv(c2w)

    # Per-step relative C2W; frame 0 keeps its absolute pose.
    relative_c2w = np.zeros_like(c2w)
    relative_c2w[0] = c2w[0]
    relative_c2w[1:] = np.linalg.inv(c2w[:-1]) @ c2w[1:]

    trans_one_hot = np.zeros((n_latents, 4), dtype=np.int32)
    rotate_one_hot = np.zeros((n_latents, 4), dtype=np.int32)

    deg_per_rad = 180.0 / np.pi
    for i in range(1, n_latents):
        move_dirs = relative_c2w[i, :3, 3]
        move_norm = float(np.linalg.norm(move_dirs))
        if move_norm > _MOVE_NORM_THRESHOLD:
            unit = move_dirs / move_norm
            trans_angles_deg = np.arccos(np.clip(unit, -1.0, 1.0)) * deg_per_rad
        else:
            trans_angles_deg = np.zeros(3)

        rot_angles_deg = _ScipyRotation.from_matrix(relative_c2w[i, :3, :3]).as_euler(
            "xyz", degrees=True
        )

        if move_norm > _MOVE_NORM_THRESHOLD:
            yaw_calm = abs(rot_angles_deg[1]) < 5e-2 and abs(rot_angles_deg[0]) < 5e-2
            if (not third_person) or yaw_calm:
                if trans_angles_deg[2] < 60:
                    trans_one_hot[i, 0] = 1  # forward
                elif trans_angles_deg[2] > 120:
                    trans_one_hot[i, 1] = 1  # backward
                if trans_angles_deg[0] < 60:
                    trans_one_hot[i, 2] = 1  # right
                elif trans_angles_deg[0] > 120:
                    trans_one_hot[i, 3] = 1  # left

        if rot_angles_deg[1] > 5e-2:
            rotate_one_hot[i, 0] = 1  # yaw right
        elif rot_angles_deg[1] < -5e-2:
            rotate_one_hot[i, 1] = 1  # yaw left
        if rot_angles_deg[0] > 5e-2:
            rotate_one_hot[i, 2] = 1  # pitch up
        elif rot_angles_deg[0] < -5e-2:
            rotate_one_hot[i, 3] = 1  # pitch down

    trans_label = _one_hot_to_label(trans_one_hot)
    rotate_label = _one_hot_to_label(rotate_one_hot)
    action_label = trans_label * 9 + rotate_label

    return (
        torch.as_tensor(w2c, dtype=torch.float64),
        torch.as_tensor(K, dtype=torch.float64),
        action_label.to(torch.long),
    )


def parse_pose_action_labels(
    pose_data: str | Path | Mapping[str, Mapping[str, list]],
    n_latents: int,
    *,
    third_person: bool = False,
) -> Tensor:
    """Return only the per-latent action labels from :func:`parse_pose_data`."""
    _, _, action = parse_pose_data(pose_data, n_latents, third_person=third_person)
    return action
