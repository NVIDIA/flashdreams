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

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SUPPORTED_KEYS = frozenset({"w", "a", "s", "d", "q", "e", "i", "k", "j", "l"})
KEY_ALIASES = {
    "arrowup": "w",
    "arrowleft": "a",
    "arrowdown": "s",
    "arrowright": "d",
}


def normalize_key(key: str) -> str:
    normalized = key.strip().lower()
    return KEY_ALIASES.get(normalized, normalized)


@dataclass(slots=True)
class KeyboardState:
    pressed_keys: set[str] = field(default_factory=set)
    _press_order: dict[str, int] = field(default_factory=dict)
    _press_counter: int = 0

    def apply_event(self, *, event: str, key: str) -> bool:
        normalized_key = normalize_key(key)
        if normalized_key not in SUPPORTED_KEYS:
            return False

        normalized_event = event.strip().lower()
        if normalized_event == "keydown":
            self.pressed_keys.add(normalized_key)
            self._press_counter += 1
            self._press_order[normalized_key] = self._press_counter
            return True
        if normalized_event == "keyup":
            self.pressed_keys.discard(normalized_key)
            self._press_order.pop(normalized_key, None)
            return True
        return False

    def snapshot(self) -> frozenset[str]:
        return frozenset(self.pressed_keys)

    def _latest_pressed(self, keys: tuple[str, ...]) -> str | None:
        latest_key: str | None = None
        latest_idx = -1
        for key in keys:
            if key not in self.pressed_keys:
                continue
            idx = self._press_order.get(key, -1)
            if idx >= latest_idx:
                latest_idx = idx
                latest_key = key
        return latest_key

    def resolved_effective_keys(self) -> frozenset[str]:
        """Resolve per-component intent with latest-pressed precedence.

        Components:
        - forward/backward: ``w`` vs ``s``
        - turn: ``a``/``j`` vs ``d``/``l``
        - strafe: ``q`` vs ``e``
        - pitch: ``i`` vs ``k``
        """
        effective: set[str] = set()
        for key in (
            self._latest_pressed(("w", "s")),
            self._latest_pressed(("a", "d", "j", "l")),
            self._latest_pressed(("q", "e")),
            self._latest_pressed(("i", "k")),
        ):
            if key is not None:
                effective.add(key)
        return frozenset(effective)


def _rotation_matrix(axis: str, angle_rad: float) -> np.ndarray:
    cos_t = np.float32(np.cos(angle_rad))
    sin_t = np.float32(np.sin(angle_rad))
    if axis == "x":
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, cos_t, -sin_t],
                [0.0, sin_t, cos_t],
            ],
            dtype=np.float32,
        )
    if axis == "y":
        return np.array(
            [
                [cos_t, 0.0, sin_t],
                [0.0, 1.0, 0.0],
                [-sin_t, 0.0, cos_t],
            ],
            dtype=np.float32,
        )
    return np.eye(3, dtype=np.float32)


@dataclass(slots=True)
class CameraPoseIntegrator:
    # Mirrors the author trajectory update style while preserving our key bindings.
    move_speed: float = 0.05
    rotate_speed_rad: float = float(np.deg2rad(2.0))
    pitch_limit_rad: float = float(np.deg2rad(85.0))
    _current_pose: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=np.float32),
    )
    _current_pitch: float = 0.0

    def reset(self, pose: np.ndarray | None = None) -> None:
        if pose is None:
            self._current_pose = np.eye(4, dtype=np.float32)
            self._current_pitch = 0.0
            return
        if pose.shape != (4, 4):
            raise ValueError(f"Expected pose shape (4, 4), got {pose.shape}")
        self._current_pose = pose.astype(np.float32, copy=True)
        # Keep cached pitch coherent with the provided pose.
        self._current_pitch = float(np.arctan2(pose[2, 1], pose[1, 1]))

    def current_pose(self) -> np.ndarray:
        return self._current_pose.copy()

    def next_pose_chunk(
        self, *, num_frames: int, pressed_keys: frozenset[str]
    ) -> np.ndarray:
        if num_frames < 1:
            raise ValueError("num_frames must be >= 1")

        poses: list[np.ndarray] = []
        for _ in range(num_frames):
            rot = self._current_pose[:3, :3]
            trans = self._current_pose[:3, 3]

            pitch_delta = 0.0
            if "i" in pressed_keys:
                pitch_delta += self.rotate_speed_rad
            if "k" in pressed_keys:
                pitch_delta -= self.rotate_speed_rad
            new_pitch = self._current_pitch + pitch_delta
            if -self.pitch_limit_rad <= new_pitch <= self.pitch_limit_rad:
                self._current_pitch = new_pitch
            else:
                pitch_delta = 0.0

            # Keep A/D as turn keys (and support J/L as aliases from author scripts).
            yaw_delta = 0.0
            if "a" in pressed_keys or "j" in pressed_keys:
                yaw_delta -= self.rotate_speed_rad
            if "d" in pressed_keys or "l" in pressed_keys:
                yaw_delta += self.rotate_speed_rad

            rot_pitch = _rotation_matrix("x", pitch_delta)
            rot_yaw = _rotation_matrix("y", yaw_delta)
            rot_new = rot_yaw @ rot @ rot_pitch

            vec_right = rot_new[:, 0]
            vec_forward = rot_new[:, 2]
            forward_flat = np.array(
                [vec_forward[0], 0.0, vec_forward[2]], dtype=np.float32
            )
            right_flat = np.array([vec_right[0], 0.0, vec_right[2]], dtype=np.float32)
            forward_norm = np.linalg.norm(forward_flat)
            right_norm = np.linalg.norm(right_flat)
            if forward_norm > 0:
                forward_flat /= forward_norm
            if right_norm > 0:
                right_flat /= right_norm

            move_vec = np.zeros(3, dtype=np.float32)
            if "w" in pressed_keys:
                move_vec += forward_flat * self.move_speed
            if "s" in pressed_keys:
                move_vec -= forward_flat * self.move_speed
            if "e" in pressed_keys:
                move_vec += right_flat * self.move_speed
            if "q" in pressed_keys:
                move_vec -= right_flat * self.move_speed

            trans_new = trans + move_vec
            self._current_pose = np.eye(4, dtype=np.float32)
            self._current_pose[:3, :3] = rot_new
            self._current_pose[:3, 3] = trans_new
            poses.append(self._current_pose.copy())

        return np.stack(poses, axis=0).astype(np.float32)
