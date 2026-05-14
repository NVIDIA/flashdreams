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

"""Keyboard state, sparse-edge event resampling, and camera pose integration."""

from __future__ import annotations

from collections import deque
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


PoseSegment = tuple[float, float, frozenset[str]]
"""One piecewise-constant interval of effective keyboard state.

The triple is ``(start_v, end_v, state)``: the resolved effective key
set ``state`` is held continuously over the half-open virtual-time
interval ``[start_v, end_v)``. A chunk's full timeline is a sequence of
such segments whose end-times match the next segment's start-time and
whose union covers the chunk window exactly.
"""


class KeyboardResampler:
    """Resample sparse keydown/keyup edges into a chunk timeline.

    Holds an arrival-timestamped FIFO of raw keyboard edges and a
    ``carried_state`` checkpoint as of the next chunk's virtual start
    time. :meth:`sample_chunk` partitions the chunk window
    ``[next_chunk_start_v, next_chunk_start_v + num_frames * dt)`` at
    every logged event in that window, producing piecewise-constant
    :data:`PoseSegment` records plus the per-frame sample times at which
    the camera pose should be evaluated.

    The downstream camera-pose integrator treats each segment as
    constant-velocity motion of duration ``end_v - start_v``: a 30 ms
    tap therefore advances the pose by 30 ms of motion regardless of
    whether it straddles a frame boundary, instead of being quantised
    to one fixed-magnitude step per sampled frame.
    """

    def __init__(self, *, fps: int, start_v: float = 0.0) -> None:
        if fps <= 0:
            raise ValueError("fps must be > 0")
        self._fps = fps
        self._dt = 1.0 / fps
        self.next_chunk_start_v = start_v
        """Virtual time at which the next chunk window starts.

        Advanced by ``num_frames * dt`` on every :meth:`sample_chunk`."""

        self._event_log: deque[tuple[float, dict[str, str]]] = deque()
        """FIFO of ``(arrival_t, payload)`` pairs sorted by ``arrival_t``.
        Holds events with ``arrival_t`` strictly greater than the last
        chunk window's end time."""

        self._carried_state = KeyboardState()
        """Effective :class:`KeyboardState` as of ``next_chunk_start_v``;
        rolled forward at each :meth:`sample_chunk` so the log can drop
        events whose timestamps are now behind the active window without
        losing held-key information."""

    @property
    def fps(self) -> int:
        """Configured resampling rate in frames per second."""
        return self._fps

    @property
    def dt(self) -> float:
        """Per-frame interval in seconds (``1 / fps``)."""
        return self._dt

    def on_edge(self, *, arrival_t: float, event: str, key: str) -> None:
        """Append one sparse keyboard edge to the event log.

        ``arrival_t`` must be on the same monotonic clock that seeds
        ``next_chunk_start_v`` (typically ``asyncio.AbstractEventLoop.time``);
        :meth:`sample_chunk` compares the two without rescaling.

        Args:
            arrival_t: Wallclock at which the edge was received.
            event: ``"keydown"`` or ``"keyup"``. Other values are stored
                and silently dropped by :class:`KeyboardState.apply_event`
                at sample time.
            key: Key name (validated downstream via
                :func:`normalize_key`).
        """
        self._event_log.append((arrival_t, {"event": event, "key": key}))

    def sample_chunk(self, num_frames: int) -> tuple[list[PoseSegment], list[float]]:
        """Build the next chunk's piecewise-constant timeline.

        Iterates the event log, folding every event with
        ``arrival_t <= chunk_end_v`` (where
        ``chunk_end_v = next_chunk_start_v + num_frames * dt``) into the
        carried state and emitting one :data:`PoseSegment` per state
        change. The chunk window is therefore covered by a sequence of
        segments whose lengths add up to ``num_frames * dt`` exactly.

        Returns the segments alongside the per-frame sample times the
        camera integrator should record poses at:
        ``next_chunk_start_v + (i + 1) * dt`` for ``i`` in
        ``[0, num_frames)``. Sampling at the *end* of each frame's dt
        interval keeps the chunk's last sample aligned with
        ``chunk_end_v`` so the integrator's terminal pose carries
        cleanly into the next chunk.

        Args:
            num_frames: Number of frames in the chunk. Must be ``>= 1``.

        Returns:
            A pair ``(segments, frame_times)``. ``segments`` is a
            non-empty list of :data:`PoseSegment` tuples covering
            ``[next_chunk_start_v, chunk_end_v)``; ``frame_times`` is a
            list of length ``num_frames`` of strictly-increasing virtual
            times.

        Raises:
            ValueError: ``num_frames < 1``.
        """
        if num_frames < 1:
            raise ValueError("num_frames must be >= 1")

        chunk_start_v = self.next_chunk_start_v
        chunk_end_v = chunk_start_v + num_frames * self._dt

        # Drain any events that arrived before the chunk window even
        # opened — they belong to the past, but still need to fold into
        # carried_state so held-key continuity is preserved.
        while self._event_log and self._event_log[0][0] < chunk_start_v:
            _, payload = self._event_log.popleft()
            self._carried_state.apply_event(**payload)

        segments: list[PoseSegment] = []
        prev_t = chunk_start_v
        prev_state = self._carried_state.resolved_effective_keys()
        # Each event in [chunk_start_v, chunk_end_v] splits the timeline
        # at its arrival time; the post-event state takes effect on the
        # next segment, the pre-event state covers up to it.
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
            # Pathological: every breakpoint exactly equals chunk_end_v,
            # which collapses the entire window to zero-length segments.
            # Emit one zero-length segment so downstream sees a
            # non-empty timeline; the integrator treats it as a no-op.
            segments.append((chunk_start_v, chunk_end_v, prev_state))

        frame_times = [chunk_start_v + (i + 1) * self._dt for i in range(num_frames)]
        self.next_chunk_start_v = chunk_end_v
        return segments, frame_times

    def reset(self, *, start_v: float) -> None:
        """Discard buffered events and reset to a fresh ``carried_state``.

        Args:
            start_v: New virtual start time for the next chunk.
        """
        self._event_log.clear()
        self._carried_state = KeyboardState()
        self.next_chunk_start_v = start_v

    def event_log_size(self) -> int:
        """Number of pending unconsumed events, for diagnostics."""
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
    return np.eye(3, dtype=np.float32)


@dataclass(slots=True)
class CameraPoseIntegrator:
    """Integrate a piecewise-constant keyboard timeline into a camera trajectory.

    All speeds are per-second so that motion magnitude scales linearly
    with key-press duration rather than being quantised to one fixed
    step per sampled frame. The integrator carries a single 4×4 pose
    and a cached pitch across calls; :meth:`integrate_chunk` integrates
    one chunk's worth of segments and emits a pose at each requested
    sample time, leaving the internal state at the chunk's end time so
    the next chunk continues without a seam.
    """

    # 0.05 world-units/frame * 16 fps from the upstream Lingbot author
    # scripts; expressed per-second so the trajectory stays consistent
    # if the playback rate is ever retuned.
    move_speed_per_s: float = 0.8
    rotate_speed_rad_per_s: float = float(np.deg2rad(32.0))
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

    def _advance(self, *, state: frozenset[str], duration: float) -> None:
        """Integrate ``state`` forward in time by ``duration`` seconds.

        Yaw and pitch are accumulated as ``rate * duration``; translation
        uses the post-yaw heading for the whole segment (a first-order
        Euler step). Zero or negative durations are no-ops.
        """
        if duration <= 0:
            return

        # Yaw and pitch rates. A/J turn left, D/L turn right; I/K pitch.
        yaw_rate = 0.0
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

        # Clamp pitch against the dataclass limit; if the proposed pitch
        # is out of range we keep the cached value and zero out the
        # rotation contribution for this segment.
        new_pitch = self._current_pitch + pitch_delta
        if -self.pitch_limit_rad <= new_pitch <= self.pitch_limit_rad:
            self._current_pitch = new_pitch
        else:
            pitch_delta = 0.0

        rot = self._current_pose[:3, :3]
        trans = self._current_pose[:3, 3]
        rot_pitch = _rotation_matrix("x", pitch_delta)
        rot_yaw = _rotation_matrix("y", yaw_delta)
        # Yaw is applied in world frame (rotate the body), pitch in body
        # frame; same convention as the author trajectory scripts so an
        # 'i' tap looks like the camera tipping up rather than spinning.
        rot_new = rot_yaw @ rot @ rot_pitch

        # Translation rates (body frame). W/S forward/backward, Q/E strafe.
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

        vec_right = rot_new[:, 0]
        vec_forward = rot_new[:, 2]
        forward_flat = np.array([vec_forward[0], 0.0, vec_forward[2]], dtype=np.float32)
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
        trans_new = trans + move_vec
        self._current_pose = np.eye(4, dtype=np.float32)
        self._current_pose[:3, :3] = rot_new
        self._current_pose[:3, 3] = trans_new

    def integrate_chunk(
        self,
        *,
        segments: list[PoseSegment],
        frame_times: list[float],
    ) -> np.ndarray:
        """Integrate ``segments`` and record a pose at each frame time.

        Walks ``segments`` left-to-right, subdividing each segment at any
        ``frame_times`` that fall inside it so the integrator records a
        pose snapshot at the exact requested virtual time. After the
        loop the integrator's internal pose corresponds to the chunk
        end (``segments[-1][1]``), so the next call resumes cleanly.

        Args:
            segments: Non-empty list of :data:`PoseSegment` triples
                covering ``[chunk_start_v, chunk_end_v)`` with no gaps
                and no overlaps; ``segments[i][1] == segments[i+1][0]``.
            frame_times: Strictly-increasing virtual times in
                ``[chunk_start_v, chunk_end_v]`` at which to record
                poses; the typical caller passes the resampler's
                ``frame_times`` output.

        Returns:
            ``np.ndarray`` of shape ``(len(frame_times), 4, 4)`` and
            dtype ``float32`` containing the recorded poses in order.

        Raises:
            ValueError: ``segments`` or ``frame_times`` is empty, a
                frame time falls outside the chunk window, or
                ``frame_times`` is not strictly increasing.
        """
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
        for seg_start, seg_end, seg_state in segments:
            # Integrate every frame time that lies strictly inside this
            # segment, recording a pose snapshot at each.
            while ft_idx < len(frame_times) and frame_times[ft_idx] <= seg_end:
                target_t = frame_times[ft_idx]
                self._advance(state=seg_state, duration=target_t - cur_t)
                cur_t = target_t
                poses.append(self._current_pose.copy())
                ft_idx += 1
            # Drain any remainder of the segment so the integrator's
            # internal state ends at ``seg_end`` regardless of whether a
            # frame time landed there exactly.
            if seg_end > cur_t:
                self._advance(state=seg_state, duration=seg_end - cur_t)
                cur_t = seg_end

        return np.stack(poses, axis=0).astype(np.float32)
