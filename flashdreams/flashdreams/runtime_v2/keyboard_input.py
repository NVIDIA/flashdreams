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

"""Timestamped keyboard state independent of an application's frame clock."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypeAlias

from flashdreams.api_v2.user_input_event import UserInputEvent
from flashdreams.runtime.keyboard import (
    DEFAULT_SUPPORTED_KEYS,
    KeyboardState,
    normalize_key,
)
from flashdreams.runtime_v2.input_timeline import InputWindow
from flashdreams.runtime_v2.user_input_event import (
    FocusUserInputEvent,
    KeyboardInputState,
    KeyboardUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

KeyboardStateProjection: TypeAlias = Callable[[KeyboardState], frozenset[str]]
"""Convert mutable keyboard state into an immutable segment value."""

KeyboardStateSegment: TypeAlias = tuple[float, float, frozenset[str]]
"""One time interval and the projected keys held throughout it."""


@dataclass(frozen=True, slots=True)
class KeyboardEventDisposition:
    """Timeline disposition for one relevant keyboard or focus event."""

    event: KeyboardUserInputEvent | FocusUserInputEvent
    """Original event retained for application-owned acknowledgement policy."""

    timestamp_s: float
    """Event timestamp converted once to session-relative seconds."""

    tracked: bool
    """Whether the keyboard track queued this event on its state timeline."""


@dataclass(frozen=True, slots=True)
class _KeyboardEdge:
    """One ordered keyboard-state transition."""

    timestamp_s: float
    """Session-relative transition time."""

    action: Literal["keydown", "keyup", "release_all"]
    """State transition applied when a sampling window reaches this edge."""

    key: str | None = None
    """Original key identifier, or ``None`` for ``release_all``."""


class KeyboardStateTrack:
    """Buffer keyboard edges and project held state over input windows.

    The track owns keyboard state but no clock. A model loop may combine it
    with
    :class:`~flashdreams.runtime_v2.input_timeline.RealtimeInputTimeline`, while
    a different input modality can reuse that clock with its own state or
    impulse semantics.

    Instances are mutable and single-loop-owned. A UI that needs immediate key
    highlights must maintain a separate state instance rather than sharing this
    model-thread track.
    """

    def __init__(
        self,
        *,
        supported_keys: frozenset[str] = DEFAULT_SUPPORTED_KEYS,
        state_projection: KeyboardStateProjection = KeyboardState.snapshot,
    ) -> None:
        if not callable(state_projection):
            raise TypeError("state_projection must be callable.")
        normalized_keys = frozenset(normalize_key(key) for key in supported_keys)
        if "" in normalized_keys:
            raise ValueError("supported_keys must not contain empty keys.")
        self._supported_keys = normalized_keys
        self._state_projection = state_projection
        self._event_log: deque[_KeyboardEdge] = deque()
        self._carried_state = KeyboardState(supported_keys=normalized_keys)
        self._sampled_through_s: float | None = None

    @property
    def supported_keys(self) -> frozenset[str]:
        """Return normalized key identifiers accepted by this track."""
        return self._supported_keys

    @property
    def pending_event_count(self) -> int:
        """Return the number of keyboard edges awaiting consumption."""
        return len(self._event_log)

    def supports_key(self, key: str) -> bool:
        """Return whether ``key`` belongs to this track."""
        return normalize_key(key) in self._supported_keys

    def ingest(
        self,
        events: UserInputEvents,
    ) -> tuple[KeyboardEventDisposition, ...]:
        """Queue supported key edges and browser focus-loss barriers.

        Keyboard events rejected by :meth:`supports_key` are returned with
        ``tracked=False`` so the application can acknowledge them without
        letting them move the input clock. Focus gain and non-keyboard events
        are unrelated to held keyboard state and are omitted.
        """
        results: list[KeyboardEventDisposition] = []
        for event in events.get_events():
            if isinstance(event, FocusUserInputEvent):
                if event.focused:
                    continue
                timestamp_s = _event_timestamp_seconds(event)
                self.release_all(timestamp_s=timestamp_s)
                results.append(
                    KeyboardEventDisposition(
                        event=event,
                        timestamp_s=timestamp_s,
                        tracked=True,
                    )
                )
                continue
            if not isinstance(event, KeyboardUserInputEvent):
                continue
            timestamp_s = _event_timestamp_seconds(event)
            tracked = self.on_edge(
                timestamp_s=timestamp_s,
                action=(
                    "keydown" if event.state is KeyboardInputState.PRESSED else "keyup"
                ),
                key=event.key,
            )
            results.append(
                KeyboardEventDisposition(
                    event=event,
                    timestamp_s=timestamp_s,
                    tracked=tracked,
                )
            )
        return tuple(results)

    def on_edge(self, *, timestamp_s: float, action: str, key: str) -> bool:
        """Queue one supported key edge and return whether it is tracked."""
        normalized_action = action.strip().lower()
        edge_action: Literal["keydown", "keyup"]
        if normalized_action == "keydown":
            edge_action = "keydown"
        elif normalized_action == "keyup":
            edge_action = "keyup"
        else:
            return False
        if not self.supports_key(key):
            return False
        self._record_edge(
            _KeyboardEdge(
                timestamp_s=_finite_timestamp(timestamp_s),
                action=edge_action,
                key=key,
            )
        )
        return True

    def release_all(self, *, timestamp_s: float) -> None:
        """Queue a barrier that releases every held key."""
        self._record_edge(
            _KeyboardEdge(
                timestamp_s=_finite_timestamp(timestamp_s),
                action="release_all",
            )
        )

    def segments(self, window: InputWindow) -> list[KeyboardStateSegment]:
        """Consume edges through ``window`` and return contiguous state segments."""
        if (
            self._sampled_through_s is not None
            and window.start_s < self._sampled_through_s
        ):
            raise ValueError(
                "window.start_s must not precede the last sampled window end."
            )

        while self._event_log and self._event_log[0].timestamp_s < window.start_s:
            self._apply_edge(self._event_log.popleft())

        segments: list[KeyboardStateSegment] = []
        previous_s = window.start_s
        previous_state = self._project_state()
        while self._event_log and self._event_log[0].timestamp_s <= window.end_s:
            edge = self._event_log.popleft()
            if edge.timestamp_s > previous_s:
                segments.append((previous_s, edge.timestamp_s, previous_state))
            self._apply_edge(edge)
            previous_state = self._project_state()
            previous_s = edge.timestamp_s

        if previous_s < window.end_s:
            segments.append((previous_s, window.end_s, previous_state))
        elif not segments:
            segments.append((window.start_s, window.end_s, previous_state))
        self._sampled_through_s = window.end_s
        return segments

    def reset(self) -> None:
        """Discard queued edges and reset held keyboard state."""
        self._event_log.clear()
        self._carried_state = KeyboardState(supported_keys=self._supported_keys)
        self._sampled_through_s = None

    def _record_edge(self, edge: _KeyboardEdge) -> None:
        """Insert ``edge`` while preserving stable timestamp order."""
        if not self._event_log or edge.timestamp_s >= self._event_log[-1].timestamp_s:
            self._event_log.append(edge)
            return
        for index, queued_edge in enumerate(self._event_log):
            if edge.timestamp_s < queued_edge.timestamp_s:
                self._event_log.insert(index, edge)
                return
        self._event_log.append(edge)

    def _apply_edge(self, edge: _KeyboardEdge) -> None:
        """Apply one queued edge to carried keyboard state."""
        if edge.action == "release_all":
            self._carried_state = KeyboardState(
                supported_keys=self._supported_keys,
            )
            return
        assert edge.key is not None
        self._carried_state.apply_event(event=edge.action, key=edge.key)

    def _project_state(self) -> frozenset[str]:
        """Return the configured immutable view of carried state."""
        return frozenset(self._state_projection(self._carried_state))


def _event_timestamp_seconds(event: UserInputEvent) -> float:
    """Convert a v2 event's microsecond timestamp to seconds."""
    return float(event.get_timestamp()) / 1_000_000.0


def _finite_timestamp(value: float) -> float:
    """Return ``value`` as a finite nonnegative session timestamp."""
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError("timestamp_s must be finite and nonnegative.")
    return parsed


__all__ = [
    "KeyboardEventDisposition",
    "KeyboardStateProjection",
    "KeyboardStateSegment",
    "KeyboardStateTrack",
]
