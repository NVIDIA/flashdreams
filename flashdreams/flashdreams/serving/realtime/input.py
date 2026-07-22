# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Input state and sparse-control helpers for realtime serving."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

DEFAULT_SUPPORTED_KEYS = frozenset({"w", "a", "s", "d", "q", "e", "i", "k", "j", "l"})
DRIVING_SUPPORTED_KEYS = frozenset(
    {"w", "a", "s", "d", "up", "down", "left", "right", "space"}
)
WSAD_SUPPORTED_KEYS = frozenset({"w", "a", "s", "d"})
KEY_ALIASES = {
    "arrowup": "w",
    "arrowleft": "a",
    "arrowdown": "s",
    "arrowright": "d",
}


@dataclass(frozen=True, slots=True)
class ResetRequest:
    """Transport-neutral request to reset the realtime rollout."""

    reason: str | None = None
    request_id: str | None = None


@dataclass(frozen=True, slots=True)
class PromptRequest:
    """Transport-neutral prompt update request."""

    prompt: str
    negative_prompt: str | None = None
    request_id: str | None = None


@dataclass(frozen=True, slots=True)
class ImageRequest:
    """Transport-neutral image update request."""

    data: bytes
    content_type: str
    request_id: str | None = None


@dataclass(frozen=True, slots=True)
class SparseInputSnapshot:
    """Sparse input state sampled at a realtime loop boundary."""

    timestamp_s: float
    pressed_keys: frozenset[str] = field(default_factory=frozenset)
    effective_keys: frozenset[str] = field(default_factory=frozenset)
    reset: ResetRequest | None = None
    prompt: PromptRequest | None = None
    image: ImageRequest | None = None


def normalize_key(key: str) -> str:
    normalized = key.strip().lower()
    return KEY_ALIASES.get(normalized, normalized)


@dataclass(slots=True)
class KeyboardState:
    pressed_keys: set[str] = field(default_factory=set)
    supported_keys: frozenset[str] = DEFAULT_SUPPORTED_KEYS
    _press_order: dict[str, int] = field(default_factory=dict)
    _press_counter: int = 0

    def apply_event(self, *, event: str, key: str) -> bool:
        normalized_key = normalize_key(key)
        if normalized_key not in self.supported_keys:
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

    def sparse_snapshot(self, *, timestamp_s: float) -> SparseInputSnapshot:
        return SparseInputSnapshot(
            timestamp_s=timestamp_s,
            pressed_keys=self.snapshot(),
            effective_keys=self.resolved_effective_keys(),
        )

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
        effective: set[str] = set()
        for key in (
            self._latest_pressed(("w", "s")),
            self._latest_pressed(("a", "d", "j", "l")),
            self._latest_pressed(("q", "e")),
            self._latest_pressed(("i", "k")),
        ):
            if key is not None:
                effective.add(key)
        return frozenset(key for key in effective if key in self.supported_keys)


PoseSegment = tuple[float, float, frozenset[str]]


class KeyboardResampler:
    """Resample sparse keydown/keyup edges into a chunk timeline."""

    def __init__(
        self,
        *,
        fps: float,
        start_v: float = 0.0,
        supported_keys: frozenset[str] = DEFAULT_SUPPORTED_KEYS,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be > 0")
        self._fps = float(fps)
        self._dt = 1.0 / self._fps
        self._supported_keys = supported_keys
        self.next_chunk_start_v = start_v
        self._event_log: deque[tuple[float, dict[str, str]]] = deque()
        self._carried_state = KeyboardState(supported_keys=supported_keys)

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def dt(self) -> float:
        return self._dt

    def on_edge(self, *, arrival_t: float, event: str, key: str) -> None:
        self._event_log.append((arrival_t, {"event": event, "key": key}))

    def sample_chunk(self, num_frames: int) -> tuple[list[PoseSegment], list[float]]:
        if num_frames < 1:
            raise ValueError("num_frames must be >= 1")

        chunk_start_v = self.next_chunk_start_v
        chunk_end_v = chunk_start_v + num_frames * self._dt

        while self._event_log and self._event_log[0][0] < chunk_start_v:
            _, payload = self._event_log.popleft()
            self._carried_state.apply_event(**payload)

        segments: list[PoseSegment] = []
        prev_t = chunk_start_v
        prev_state = self._carried_state.resolved_effective_keys()
        while self._event_log and self._event_log[0][0] <= chunk_end_v:
            event_t, payload = self._event_log.popleft()
            if event_t > prev_t:
                segments.append((prev_t, event_t, prev_state))
            self._carried_state.apply_event(**payload)
            prev_state = self._carried_state.resolved_effective_keys()
            prev_t = event_t
        if prev_t < chunk_end_v:
            segments.append((prev_t, chunk_end_v, prev_state))
        elif not segments:
            segments.append((chunk_start_v, chunk_end_v, prev_state))

        frame_times = [chunk_start_v + (i + 1) * self._dt for i in range(num_frames)]
        self.next_chunk_start_v = chunk_end_v
        return segments, frame_times

    def reset(self, *, start_v: float) -> None:
        self._event_log.clear()
        self._carried_state = KeyboardState(supported_keys=self._supported_keys)
        self.next_chunk_start_v = start_v

    def event_log_size(self) -> int:
        return len(self._event_log)


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
    if axis == "z":
        return np.array(
            [
                [cos_t, -sin_t, 0.0],
                [sin_t, cos_t, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    return np.eye(3, dtype=np.float32)


@dataclass(slots=True)
class CameraPoseIntegrator:
    """Integrate a piecewise-constant keyboard timeline into a camera trajectory."""

    move_speed_per_s: float = 0.8
    rotate_speed_rad_per_s: float = float(np.deg2rad(32.0))
    pitch_limit_rad: float = float(np.deg2rad(85.0))
    coordinate_system: Literal["RDF", "FLU"] = "RDF"
    _current_pose: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=np.float32),
    )
    _current_pitch: float = 0.0

    def __post_init__(self) -> None:
        if self.coordinate_system not in {"RDF", "FLU"}:
            raise ValueError(
                "coordinate_system must be 'RDF' (right-down-forward) "
                "or 'FLU' (forward-left-up)"
            )

    def reset(self, pose: np.ndarray | None = None) -> None:
        if pose is None:
            self._current_pose = np.eye(4, dtype=np.float32)
            self._current_pitch = 0.0
            return
        if pose.shape != (4, 4):
            raise ValueError(f"Expected pose shape (4, 4), got {pose.shape}")
        self._current_pose = pose.astype(np.float32, copy=True)
        if self.coordinate_system == "FLU":
            self._current_pitch = float(np.arcsin(np.clip(pose[2, 0], -1.0, 1.0)))
        else:
            self._current_pitch = float(np.arctan2(pose[2, 1], pose[1, 1]))

    def current_pose(self) -> np.ndarray:
        return self._current_pose.copy()

    def _advance(self, *, state: frozenset[str], duration: float) -> None:
        if duration <= 0:
            return

        yaw_rate = 0.0
        if self.coordinate_system == "FLU":
            if "a" in state or "j" in state:
                yaw_rate += self.rotate_speed_rad_per_s
            if "d" in state or "l" in state:
                yaw_rate -= self.rotate_speed_rad_per_s
        else:
            if "a" in state or "j" in state:
                yaw_rate -= self.rotate_speed_rad_per_s
            if "d" in state or "l" in state:
                yaw_rate += self.rotate_speed_rad_per_s
        pitch_rate = 0.0
        if "i" in state:
            pitch_rate += self.rotate_speed_rad_per_s
        if "k" in state:
            pitch_rate -= self.rotate_speed_rad_per_s

        yaw_delta = yaw_rate * duration
        pitch_delta = pitch_rate * duration

        new_pitch = self._current_pitch + pitch_delta
        if -self.pitch_limit_rad <= new_pitch <= self.pitch_limit_rad:
            self._current_pitch = new_pitch
        else:
            pitch_delta = 0.0

        rot = self._current_pose[:3, :3]
        trans = self._current_pose[:3, 3]
        if self.coordinate_system == "FLU":
            rot_pitch = _rotation_matrix("y", -pitch_delta)
            rot_yaw = _rotation_matrix("z", yaw_delta)
        else:
            rot_pitch = _rotation_matrix("x", pitch_delta)
            rot_yaw = _rotation_matrix("y", yaw_delta)
        rot_new = rot_yaw @ rot @ rot_pitch

        forward_rate = 0.0
        if "w" in state:
            forward_rate += self.move_speed_per_s
        if "s" in state:
            forward_rate -= self.move_speed_per_s
        right_rate = 0.0
        if "e" in state:
            right_rate += self.move_speed_per_s
        if "q" in state:
            right_rate -= self.move_speed_per_s

        if self.coordinate_system == "FLU":
            vec_forward = rot_new[:, 0]
            vec_right = -rot_new[:, 1]
            forward_flat = np.array(
                [vec_forward[0], vec_forward[1], 0.0], dtype=np.float32
            )
            right_flat = np.array([vec_right[0], vec_right[1], 0.0], dtype=np.float32)
        else:
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

        move_vec = forward_flat * (forward_rate * duration) + right_flat * (
            right_rate * duration
        )
        self._current_pose = np.eye(4, dtype=np.float32)
        self._current_pose[:3, :3] = rot_new
        self._current_pose[:3, 3] = trans + move_vec

    def integrate_chunk(
        self,
        *,
        segments: list[PoseSegment],
        frame_times: list[float],
    ) -> np.ndarray:
        if not segments:
            raise ValueError("segments must be non-empty")
        if not frame_times:
            raise ValueError("frame_times must be non-empty")
        chunk_start = segments[0][0]
        chunk_end = segments[-1][1]
        if any(
            frame_times[i] >= frame_times[i + 1] for i in range(len(frame_times) - 1)
        ):
            raise ValueError("frame_times must be strictly increasing")
        if frame_times[0] < chunk_start - 1e-9 or frame_times[-1] > chunk_end + 1e-9:
            raise ValueError(
                "frame_times must lie within the chunk window "
                f"[{chunk_start}, {chunk_end}]"
            )

        poses: list[np.ndarray] = []
        cur_t = chunk_start
        ft_idx = 0
        for _, seg_end, seg_state in segments:
            while ft_idx < len(frame_times) and frame_times[ft_idx] <= seg_end:
                target_t = frame_times[ft_idx]
                self._advance(state=seg_state, duration=target_t - cur_t)
                cur_t = target_t
                poses.append(self._current_pose.copy())
                ft_idx += 1
            if seg_end > cur_t:
                self._advance(state=seg_state, duration=seg_end - cur_t)
                cur_t = seg_end

        return np.stack(poses, axis=0).astype(np.float32)
