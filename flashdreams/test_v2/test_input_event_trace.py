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

"""CPU tests for correlating user input with output frames."""

import pytest
from numpy import uint64

from flashdreams.runtime_v2.input_event_trace import InputEventTraceTracker
from flashdreams.runtime_v2.user_input_event import MouseUserInputEvent

pytestmark = pytest.mark.ci_cpu


def _event(event_id: str | None) -> MouseUserInputEvent:
    return MouseUserInputEvent(timestamp=uint64(0), event_id=event_id)


def test_input_event_trace_tracker_uses_the_first_strictly_later_frame() -> None:
    tracker = InputEventTraceTracker()
    tracker.track(_event("page:pointer"), effective_at_s=0.1)

    traces = tracker.resolve((0.0, 0.1, 0.2))

    assert [(trace.event_id, trace.frame_index) for trace in traces] == [
        ("page:pointer", 2)
    ]
    assert tracker.pending_count == 0


def test_input_event_trace_tracker_carries_events_across_chunks() -> None:
    tracker = InputEventTraceTracker()
    tracker.track(_event("page:future"), effective_at_s=0.3)

    assert tracker.resolve((0.0, 0.1, 0.2)) == ()
    assert tracker.pending_count == 1
    traces = tracker.resolve((0.3, 0.4))

    assert [(trace.event_id, trace.frame_index) for trace in traces] == [
        ("page:future", 1)
    ]
    assert tracker.pending_count == 0


def test_input_event_trace_tracker_does_not_block_later_untimed_events() -> None:
    tracker = InputEventTraceTracker()
    tracker.track(_event("page:future"), effective_at_s=10.0)
    tracker.track(_event("page:next"), effective_at_s=None)

    traces = tracker.resolve((1.0, 2.0))

    assert [(trace.event_id, trace.frame_index) for trace in traces] == [
        ("page:next", 0)
    ]
    assert tracker.pending_count == 1
    future_traces = tracker.resolve((10.0, 11.0))
    assert [(trace.event_id, trace.frame_index) for trace in future_traces] == [
        ("page:future", 1)
    ]


def test_input_event_trace_tracker_resolves_untimed_events_on_frame_zero() -> None:
    tracker = InputEventTraceTracker()
    tracker.track(_event("page:next"), effective_at_s=None)

    traces = tracker.resolve((1.0, 2.0))

    assert [(trace.event_id, trace.frame_index) for trace in traces] == [
        ("page:next", 0)
    ]


def test_input_event_trace_tracker_ignores_events_without_ids() -> None:
    tracker = InputEventTraceTracker()

    tracker.track(_event(None), effective_at_s=0.0)

    assert tracker.pending_count == 0
    assert tracker.resolve((1.0,)) == ()


def test_input_event_trace_tracker_clear_discards_pending_events() -> None:
    tracker = InputEventTraceTracker()
    tracker.track(_event("page:future"), effective_at_s=10.0)

    tracker.clear()

    assert tracker.pending_count == 0
    assert tracker.resolve((1.0,)) == ()


@pytest.mark.parametrize("frame_times", [(-1.0,), (1.0, 0.0), (1.0, 1.0)])
def test_input_event_trace_tracker_rejects_invalid_frame_times(
    frame_times: tuple[float, ...],
) -> None:
    tracker = InputEventTraceTracker()

    with pytest.raises(ValueError, match="frame_times_s"):
        tracker.resolve(frame_times)


def test_input_event_trace_tracker_rejects_negative_event_times() -> None:
    tracker = InputEventTraceTracker()

    with pytest.raises(ValueError, match="effective_at_s"):
        tracker.track(_event("page:pointer"), effective_at_s=-1.0)
