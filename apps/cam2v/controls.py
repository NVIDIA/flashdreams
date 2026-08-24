# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keyboard sampling and camera-pose integration shared by Cam2V apps."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from flashdreams.runtime.keyboard import DEFAULT_SUPPORTED_KEYS, KeyboardState

PoseSegment = tuple[float, float, frozenset[str]]
"""One time interval and the camera-control keys held throughout it."""


class KeyboardResampler:
    """Resample sparse key-down/key-up edges into a camera-control timeline."""

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
        """Return the target camera sampling rate."""
        return self._fps

    @property
    def dt(self) -> float:
        """Return the interval between adjacent camera samples."""
        return self._dt

    def on_edge(self, *, arrival_t: float, event: str, key: str) -> None:
        """Record one keyboard edge in timestamp order."""
        entry = (arrival_t, {"event": event, "key": key})
        if not self._event_log or arrival_t >= self._event_log[-1][0]:
            self._event_log.append(entry)
            return
        for index, (event_t, _) in enumerate(self._event_log):
            if arrival_t < event_t:
                self._event_log.insert(index, entry)
                return
        self._event_log.append(entry)

    def sample_chunk(self, num_frames: int) -> tuple[list[PoseSegment], list[float]]:
        """Return key-state segments and sample times for one model chunk."""
        if num_frames < 1:
            raise ValueError("num_frames must be >= 1")

        chunk_start_v = self.next_chunk_start_v
        chunk_end_v = chunk_start_v + num_frames * self._dt
        while self._event_log and self._event_log[0][0] < chunk_start_v:
            _, payload = self._event_log.popleft()
            self._carried_state.apply_event(**payload)

        segments: list[PoseSegment] = []
        previous_t = chunk_start_v
        previous_state = self._carried_state.resolved_effective_keys()
        while self._event_log and self._event_log[0][0] <= chunk_end_v:
            event_t, payload = self._event_log.popleft()
            if event_t > previous_t:
                segments.append((previous_t, event_t, previous_state))
            self._carried_state.apply_event(**payload)
            previous_state = self._carried_state.resolved_effective_keys()
            previous_t = event_t
        if previous_t < chunk_end_v:
            segments.append((previous_t, chunk_end_v, previous_state))
        elif not segments:
            segments.append((chunk_start_v, chunk_end_v, previous_state))

        frame_times = [
            chunk_start_v + (index + 1) * self._dt for index in range(num_frames)
        ]
        self.next_chunk_start_v = chunk_end_v
        return segments, frame_times

    def reset(self, *, start_v: float) -> None:
        """Discard queued edges and restart the virtual camera clock."""
        self._event_log.clear()
        self._carried_state = KeyboardState(supported_keys=self._supported_keys)
        self.next_chunk_start_v = start_v

    def event_log_size(self) -> int:
        """Return the number of keyboard edges awaiting consumption."""
        return len(self._event_log)


def _rotation_matrix(axis: str, angle_rad: float) -> np.ndarray:
    """Return a float32 three-dimensional rotation matrix."""
    cos_t = np.float32(np.cos(angle_rad))
    sin_t = np.float32(np.sin(angle_rad))
    if axis == "x":
        return np.array(
            [[1.0, 0.0, 0.0], [0.0, cos_t, -sin_t], [0.0, sin_t, cos_t]],
            dtype=np.float32,
        )
    if axis == "y":
        return np.array(
            [[cos_t, 0.0, sin_t], [0.0, 1.0, 0.0], [-sin_t, 0.0, cos_t]],
            dtype=np.float32,
        )
    if axis == "z":
        return np.array(
            [[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
    return np.eye(3, dtype=np.float32)


@dataclass(slots=True)
class CameraPoseIntegrator:
    """Integrate piecewise-constant keyboard intent into camera poses."""

    move_speed_per_s: float = 0.8
    """Camera translation speed in world units per second."""

    rotate_speed_rad_per_s: float = float(np.deg2rad(32.0))
    """Camera yaw and pitch speed in radians per second."""

    pitch_limit_rad: float = float(np.deg2rad(85.0))
    """Maximum absolute camera pitch."""

    coordinate_system: Literal["RDF", "FLU"] = "RDF"
    """Camera basis: right-down-forward or forward-left-up."""

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
        """Reset integration to identity or to ``pose``."""
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
        """Return a copy of the most recently integrated camera pose."""
        return self._current_pose.copy()

    def _advance(self, *, state: frozenset[str], duration: float) -> None:
        """Advance the current pose through one constant-key interval."""
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

        rotation = self._current_pose[:3, :3]
        translation = self._current_pose[:3, 3]
        if self.coordinate_system == "FLU":
            pitch_rotation = _rotation_matrix("y", -pitch_delta)
            yaw_rotation = _rotation_matrix("z", yaw_delta)
        else:
            pitch_rotation = _rotation_matrix("x", pitch_delta)
            yaw_rotation = _rotation_matrix("y", yaw_delta)
        new_rotation = yaw_rotation @ rotation @ pitch_rotation

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
            forward = new_rotation[:, 0]
            right = -new_rotation[:, 1]
            flat_forward = np.array([forward[0], forward[1], 0.0], dtype=np.float32)
            flat_right = np.array([right[0], right[1], 0.0], dtype=np.float32)
        else:
            right = new_rotation[:, 0]
            forward = new_rotation[:, 2]
            flat_forward = np.array([forward[0], 0.0, forward[2]], dtype=np.float32)
            flat_right = np.array([right[0], 0.0, right[2]], dtype=np.float32)
        forward_norm = np.linalg.norm(flat_forward)
        right_norm = np.linalg.norm(flat_right)
        if forward_norm > 0:
            flat_forward /= forward_norm
        if right_norm > 0:
            flat_right /= right_norm

        movement = flat_forward * (forward_rate * duration) + flat_right * (
            right_rate * duration
        )
        self._current_pose = np.eye(4, dtype=np.float32)
        self._current_pose[:3, :3] = new_rotation
        self._current_pose[:3, 3] = translation + movement

    def integrate_chunk(
        self,
        *,
        segments: list[PoseSegment],
        frame_times: list[float],
    ) -> np.ndarray:
        """Return one camera-to-world matrix at each requested frame time."""
        if not segments:
            raise ValueError("segments must be non-empty")
        if not frame_times:
            raise ValueError("frame_times must be non-empty")
        chunk_start = segments[0][0]
        chunk_end = segments[-1][1]
        if any(
            frame_times[index] >= frame_times[index + 1]
            for index in range(len(frame_times) - 1)
        ):
            raise ValueError("frame_times must be strictly increasing")
        if frame_times[0] < chunk_start - 1e-9 or frame_times[-1] > chunk_end + 1e-9:
            raise ValueError(
                "frame_times must lie within the chunk window "
                f"[{chunk_start}, {chunk_end}]"
            )

        poses: list[np.ndarray] = []
        current_t = chunk_start
        frame_index = 0
        for _, segment_end, segment_state in segments:
            while (
                frame_index < len(frame_times)
                and frame_times[frame_index] <= segment_end
            ):
                target_t = frame_times[frame_index]
                self._advance(state=segment_state, duration=target_t - current_t)
                current_t = target_t
                poses.append(self._current_pose.copy())
                frame_index += 1
            if segment_end > current_t:
                self._advance(state=segment_state, duration=segment_end - current_t)
                current_t = segment_end

        return np.stack(poses, axis=0).astype(np.float32)


__all__ = ["CameraPoseIntegrator", "KeyboardResampler", "PoseSegment"]
