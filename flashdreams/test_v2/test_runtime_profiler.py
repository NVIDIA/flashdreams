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

"""CPU checks for V2 perceived input-latency profiles."""

import json

import pytest
from numpy import uint64

from flashdreams.runtime_v2.runtime_profiler import RuntimeProfiler
from flashdreams.runtime_v2.user_input_event import (
    KeyboardInputState,
    KeyboardUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

pytestmark = pytest.mark.ci_cpu


def _input(timestamp_us: int = 1_000) -> UserInputEvents:
    return UserInputEvents(
        [
            KeyboardUserInputEvent(
                timestamp=uint64(timestamp_us),
                key="w",
                state=KeyboardInputState.PRESSED,
            )
        ]
    )


def test_profile_correlates_claimed_input_with_first_following_write(tmp_path) -> None:
    path = tmp_path / "runtime.jsonl"
    profiler = RuntimeProfiler(path)
    profiler.session_started(input_timestamp_origin_ns=1_000_000, time_ns=1_000_000)

    profiler.ui_step_started(
        _input(),
        generation=2,
        step=9,
        time_ns=5_000_000,
    )
    profiler.ui_step_started(
        UserInputEvents([]),
        generation=2,
        step=10,
        time_ns=6_000_000,
    )
    profiler.window_write_completed(
        generation=2,
        ui_step=10,
        time_ns=9_000_000,
    )

    assert profiler.summary() == {
        "input_to_ui_step_s": {
            "count": 1,
            "median_s": pytest.approx(0.003),
            "p90_s": pytest.approx(0.003),
            "max_s": pytest.approx(0.003),
        },
        "input_to_window_write_s": {
            "count": 1,
            "median_s": pytest.approx(0.007),
            "p90_s": pytest.approx(0.007),
            "max_s": pytest.approx(0.007),
        },
    }

    profiler.close()
    profiler.close()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    output_record = next(
        record for record in records if record["phase"] == "input_to_window_write"
    )
    assert output_record == {
        "schema_version": 1,
        "phase": "input_to_window_write",
        "time_ns": 9_000_000,
        "generation": 2,
        "claimed_ui_step": 9,
        "presented_ui_step": 10,
        "input_type": "KeyboardUserInputEvent",
        "input_timestamp_us": 1_000,
        "duration_s": 0.007,
    }
    assert sum(record["phase"] == "profile_summary" for record in records) == 2


def test_profile_skips_sources_without_a_shared_clock_origin(tmp_path) -> None:
    profiler = RuntimeProfiler(tmp_path / "runtime.jsonl")
    profiler.session_started(input_timestamp_origin_ns=None, time_ns=1_000_000)

    profiler.ui_step_started(_input(), generation=0, step=0, time_ns=5_000_000)
    profiler.window_write_completed(generation=0, ui_step=0, time_ns=9_000_000)

    assert profiler.summary() == {
        "input_to_ui_step_s": {"count": 0},
        "input_to_window_write_s": {"count": 0},
    }
    profiler.close()


def test_generation_change_discards_unpresented_input(tmp_path) -> None:
    profiler = RuntimeProfiler(tmp_path / "runtime.jsonl")
    profiler.session_started(input_timestamp_origin_ns=0, time_ns=0)
    profiler.ui_step_started(_input(), generation=0, step=0, time_ns=2_000_000)

    profiler.ui_step_started(
        UserInputEvents([]),
        generation=1,
        step=0,
        time_ns=3_000_000,
    )
    profiler.window_write_completed(generation=1, ui_step=0, time_ns=4_000_000)

    assert profiler.summary()["input_to_window_write_s"] == {"count": 0}
    profiler.close()
