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

"""CPU tests for reusable realtime input timing and keyboard state."""

from __future__ import annotations

import pytest
from numpy import uint64

from flashdreams.runtime.keyboard import KeyboardState
from flashdreams.runtime_v2.input_timeline import InputWindow, RealtimeInputTimeline
from flashdreams.runtime_v2.keyboard_input import KeyboardStateTrack
from flashdreams.runtime_v2.user_input_event import (
    FocusUserInputEvent,
    KeyboardInputState,
    KeyboardUserInputEvent,
    MouseUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

pytestmark = pytest.mark.ci_cpu


def _keyboard_event(
    timestamp_s: float,
    *,
    key: str,
    state: KeyboardInputState,
) -> KeyboardUserInputEvent:
    """Build one keyboard event from a timestamp expressed in seconds."""
    return KeyboardUserInputEvent(
        timestamp=uint64(round(timestamp_s * 1_000_000.0)),
        key=key,
        state=state,
    )


def test_realtime_input_timeline_catches_up_to_only_new_accepted_input() -> None:
    """Preserve one window of a fresh input batch without moving backward."""
    timeline = RealtimeInputTimeline(samples_per_second=10)

    quick_tap = timeline.next_window(2, input_times_s=(10.0, 10.08))

    assert quick_tap.start_s == pytest.approx(10.0)
    assert quick_tap.end_s == pytest.approx(10.2)
    assert quick_tap.sample_times_s == pytest.approx((10.1, 10.2))

    old_input = timeline.next_window(2, input_times_s=(1.0,))

    assert old_input.start_s == pytest.approx(10.2)
    assert timeline.next_window_start_s == pytest.approx(10.4)


def test_realtime_input_timeline_retains_latest_part_of_a_long_batch() -> None:
    """Fold stale edges while retaining the latest full sampling window."""
    timeline = RealtimeInputTimeline(samples_per_second=10)

    window = timeline.next_window(2, input_times_s=(10.0, 10.5))

    assert window.start_s == pytest.approx(10.3)
    assert window.end_s == pytest.approx(10.5)


def test_input_at_nominal_window_end_does_not_trigger_catch_up() -> None:
    """Treat a boundary event as part of the nominal next window."""
    timeline = RealtimeInputTimeline(samples_per_second=10)

    window = timeline.next_window(2, input_times_s=(0.2,))

    assert window.start_s == 0.0
    assert window.end_s == pytest.approx(0.2)


def test_input_window_rejects_invalid_bounds_and_sample_times() -> None:
    """Fail at the public window boundary instead of corrupting a state track."""
    with pytest.raises(ValueError, match="greater than"):
        InputWindow(start_s=1.0, end_s=1.0, sample_times_s=(1.0,))
    with pytest.raises(ValueError, match="must not be empty"):
        InputWindow(start_s=0.0, end_s=1.0, sample_times_s=())
    with pytest.raises(ValueError, match="strictly increasing"):
        InputWindow(start_s=0.0, end_s=1.0, sample_times_s=(0.5, 0.5))
    with pytest.raises(ValueError, match="beyond"):
        InputWindow(start_s=0.0, end_s=1.0, sample_times_s=(1.1,))


def test_keyboard_track_uses_configured_keys_for_ingest_and_clock_updates() -> None:
    """Ignore unconfigured keys even when another application supports them."""
    track = KeyboardStateTrack(supported_keys=frozenset({"x"}))
    events = UserInputEvents(
        [
            _keyboard_event(
                0.05,
                key="x",
                state=KeyboardInputState.PRESSED,
            ),
            _keyboard_event(
                10.0,
                key="q",
                state=KeyboardInputState.PRESSED,
            ),
            FocusUserInputEvent(timestamp=uint64(11_000_000), focused=True),
            MouseUserInputEvent(timestamp=uint64(12_000_000), x=0.5, y=0.5),
        ]
    )

    results = track.ingest(events)
    timeline = RealtimeInputTimeline(samples_per_second=10)
    window = timeline.next_window(
        2,
        input_times_s=(result.timestamp_s for result in results if result.tracked),
    )
    segments = track.segments(window)

    observed_results: list[tuple[str, bool]] = []
    for result in results:
        assert isinstance(result.event, KeyboardUserInputEvent)
        observed_results.append((result.event.key, result.tracked))
    assert observed_results == [
        ("x", True),
        ("q", False),
    ]
    assert window.start_s == 0.0
    assert segments == [
        (0.0, 0.05, frozenset()),
        (0.05, 0.2, frozenset({"x"})),
    ]


def test_keyboard_track_focus_loss_releases_all_alias_sources() -> None:
    """Catch up to a delayed browser blur and release every alias source."""
    track = KeyboardStateTrack(
        state_projection=KeyboardState.resolved_effective_keys,
    )
    events = UserInputEvents(
        [
            _keyboard_event(
                0.02,
                key="w",
                state=KeyboardInputState.PRESSED,
            ),
            _keyboard_event(
                0.03,
                key="ArrowUp",
                state=KeyboardInputState.PRESSED,
            ),
            FocusUserInputEvent(timestamp=uint64(10_000_000), focused=False),
        ]
    )

    results = track.ingest(events)
    timeline = RealtimeInputTimeline(samples_per_second=10)
    window = timeline.next_window(
        1,
        input_times_s=(result.timestamp_s for result in results if result.tracked),
    )
    held_segments = track.segments(window)
    released_segments = track.segments(timeline.next_window(1))

    assert results[-1].tracked
    assert window.start_s == pytest.approx(9.9)
    assert held_segments[-1][2] == frozenset({"w"})
    assert released_segments == [(10.0, 10.1, frozenset())]
    assert track.pending_event_count == 0


def test_keyboard_track_preserves_equal_timestamp_event_order() -> None:
    """Use stable batch order when multiple edges share one timestamp."""
    key_down = _keyboard_event(
        0.05,
        key="w",
        state=KeyboardInputState.PRESSED,
    )
    focus_lost = FocusUserInputEvent(timestamp=uint64(50_000), focused=False)

    down_then_blur = KeyboardStateTrack()
    down_then_blur.ingest(UserInputEvents([key_down, focus_lost]))
    released = down_then_blur.segments(
        RealtimeInputTimeline(samples_per_second=10).next_window(1)
    )

    blur_then_down = KeyboardStateTrack()
    blur_then_down.ingest(UserInputEvents([focus_lost, key_down]))
    held = blur_then_down.segments(
        RealtimeInputTimeline(samples_per_second=10).next_window(1)
    )

    assert released[-1][2] == frozenset()
    assert held[-1][2] == frozenset({"w"})


def test_keyboard_track_orders_edges_received_across_batches() -> None:
    """Insert late-arriving older edges before already buffered transitions."""
    track = KeyboardStateTrack()
    track.ingest(
        UserInputEvents(
            [
                _keyboard_event(
                    0.08,
                    key="w",
                    state=KeyboardInputState.RELEASED,
                )
            ]
        )
    )
    track.ingest(
        UserInputEvents(
            [
                _keyboard_event(
                    0.02,
                    key="w",
                    state=KeyboardInputState.PRESSED,
                )
            ]
        )
    )

    window = RealtimeInputTimeline(samples_per_second=10).next_window(1)
    segments = track.segments(window)

    assert segments == [
        (0.0, 0.02, frozenset()),
        (0.02, 0.08, frozenset({"w"})),
        (0.08, 0.1, frozenset()),
    ]


def test_keyboard_track_reset_clears_held_state_and_future_edges() -> None:
    """Reset both carried state and unconsumed transitions."""
    track = KeyboardStateTrack()
    track.on_edge(timestamp_s=0.0, action="keydown", key="w")
    track.on_edge(timestamp_s=10.0, action="keyup", key="w")
    first_window = RealtimeInputTimeline(samples_per_second=10).next_window(1)
    assert track.segments(first_window)[-1][2] == frozenset({"w"})

    track.reset()
    reset_timeline = RealtimeInputTimeline(samples_per_second=10, start_s=5.0)
    reset_segments = track.segments(reset_timeline.next_window(1))

    assert reset_segments == [(5.0, 5.1, frozenset())]
    assert track.pending_event_count == 0


def test_keyboard_track_rejects_overlapping_windows() -> None:
    """Do not silently reuse carried state for already-consumed history."""
    track = KeyboardStateTrack()
    track.on_edge(timestamp_s=0.02, action="keydown", key="w")
    track.on_edge(timestamp_s=0.08, action="keyup", key="w")
    track.segments(RealtimeInputTimeline(samples_per_second=10).next_window(1))

    with pytest.raises(ValueError, match="must not precede"):
        track.segments(
            InputWindow(
                start_s=0.05,
                end_s=0.15,
                sample_times_s=(0.15,),
            )
        )
