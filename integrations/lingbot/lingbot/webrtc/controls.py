from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SUPPORTED_KEYS = frozenset({"w", "a", "s", "d", "q", "e"})
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

    def apply_event(self, *, event: str, key: str) -> bool:
        normalized_key = normalize_key(key)
        if normalized_key not in SUPPORTED_KEYS:
            return False

        normalized_event = event.strip().lower()
        if normalized_event == "keydown":
            self.pressed_keys.add(normalized_key)
            return True
        if normalized_event == "keyup":
            self.pressed_keys.discard(normalized_key)
            return True
        return False

    def snapshot(self) -> frozenset[str]:
        return frozenset(self.pressed_keys)


def _rotation_z(theta_rad: float) -> np.ndarray:
    cos_t = np.float32(np.cos(theta_rad))
    sin_t = np.float32(np.sin(theta_rad))
    rotation = np.eye(4, dtype=np.float32)
    rotation[0, 0] = cos_t
    rotation[0, 1] = -sin_t
    rotation[1, 0] = sin_t
    rotation[1, 1] = cos_t
    return rotation


@dataclass(slots=True)
class CameraPoseIntegrator:
    forward_step: float = 0.08
    strafe_step: float = 0.04
    yaw_step_rad: float = 0.04
    _current_pose: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=np.float32),
    )

    def reset(self, pose: np.ndarray | None = None) -> None:
        if pose is None:
            self._current_pose = np.eye(4, dtype=np.float32)
            return
        if pose.shape != (4, 4):
            raise ValueError(f"Expected pose shape (4, 4), got {pose.shape}")
        self._current_pose = pose.astype(np.float32, copy=True)

    def current_pose(self) -> np.ndarray:
        return self._current_pose.copy()

    def next_pose_chunk(
        self, *, num_frames: int, pressed_keys: frozenset[str]
    ) -> np.ndarray:
        if num_frames < 1:
            raise ValueError("num_frames must be >= 1")

        forward = 0.0
        if "w" in pressed_keys:
            forward += self.forward_step
        if "s" in pressed_keys:
            forward -= self.forward_step

        strafe = 0.0
        if "e" in pressed_keys:
            strafe += self.strafe_step
        if "q" in pressed_keys:
            strafe -= self.strafe_step

        yaw = 0.0
        if "a" in pressed_keys:
            yaw += self.yaw_step_rad
        if "d" in pressed_keys:
            yaw -= self.yaw_step_rad

        delta = np.eye(4, dtype=np.float32)
        delta[:3, :3] = _rotation_z(yaw)[:3, :3]
        delta[0, 3] = forward
        delta[1, 3] = strafe

        poses: list[np.ndarray] = []
        for _ in range(num_frames):
            self._current_pose = self._current_pose @ delta
            poses.append(self._current_pose.copy())

        return np.stack(poses, axis=0).astype(np.float32)
