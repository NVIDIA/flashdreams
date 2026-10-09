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

"""CPU checks for V2 host-side input-latency profiles."""

import json

import pytest
from numpy import uint64

from flashdreams.runtime_v2.runtime_profiler import RuntimeProfiler
from flashdreams.runtime_v2.session_desc import (
    BackpressureMode,
    PresentationMode,
    SessionDesc,
)
from flashdreams.runtime_v2.user_input_event import (
    KeyboardInputState,
    KeyboardUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

pytestmark = pytest.mark.ci_cpu

_SESSION_DESC = SessionDesc(
    backpressure_mode=BackpressureMode.DROP_OLDEST,
    presentation_mode=PresentationMode.ON_DEMAND,
    frames_per_second_for_ui=60,
    frames_per_second_for_step=24,
    video_width=640,
    video_height=360,
)


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
    profiler.session_started(
        input_timestamp_origin_ns=1_000_000,
        session_desc=_SESSION_DESC,
        client_window_type="NativeWindowClientWindow",
        time_ns=1_000_000,
    )

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
            "quantile_sample_count": 1,
            "quantiles_approximate": False,
        },
        "input_to_window_write_s": {
            "count": 1,
            "median_s": pytest.approx(0.007),
            "p90_s": pytest.approx(0.007),
            "max_s": pytest.approx(0.007),
            "quantile_sample_count": 1,
            "quantiles_approximate": False,
        },
    }

    profiler.close()
    profiler.close()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    output_record = next(
        record for record in records if record["phase"] == "input_to_window_write"
    )
    assert output_record == {
        "artifact_type": "flashdreams.runtime_v2.input_latency_profile",
        "schema_version": 1,
        "phase": "input_to_window_write",
        "time_ns": 9_000_000,
        "generation": 2,
        "claimed_ui_step": 9,
        "presented_ui_step": 10,
        "input_type": "keyboard",
        "input_timestamp_us": 1_000,
        "duration_s": 0.007,
    }
    assert sum(record["phase"] == "profile_summary" for record in records) == 2
    session_record = records[0]
    assert session_record["measurement_endpoints"] == {
        "input_to_ui_step_s": "ui_loop_begin_run",
        "input_to_window_write_s": "client_window_write_return",
    }
    assert session_record["runtime_settings"] == {
        "client_window_type": "NativeWindowClientWindow",
        "output_layout": "tchw",
        "frames_per_second_for_ui": 60,
        "frames_per_second_for_step": 24,
        "video_width": 640,
        "video_height": 360,
        "presentation_mode": "on_demand",
        "backpressure_mode": "drop_oldest",
    }


def test_profile_skips_sources_without_a_shared_clock_origin(tmp_path) -> None:
    profiler = RuntimeProfiler(tmp_path / "runtime.jsonl")
    profiler.session_started(
        input_timestamp_origin_ns=None,
        session_desc=_SESSION_DESC,
        client_window_type="UnknownWindow",
        time_ns=1_000_000,
    )

    profiler.ui_step_started(_input(), generation=0, step=0, time_ns=5_000_000)
    profiler.window_write_completed(generation=0, ui_step=0, time_ns=9_000_000)

    assert profiler.summary() == {
        "input_to_ui_step_s": {"count": 0},
        "input_to_window_write_s": {"count": 0},
    }
    profiler.close()


def test_generation_change_discards_unpresented_input(tmp_path) -> None:
    profiler = RuntimeProfiler(tmp_path / "runtime.jsonl")
    profiler.session_started(
        input_timestamp_origin_ns=0,
        session_desc=_SESSION_DESC,
        client_window_type="UnknownWindow",
        time_ns=0,
    )
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


def test_replacement_sessions_append_independent_profile_segments(tmp_path) -> None:
    path = tmp_path / "runtime.jsonl"
    profiler = RuntimeProfiler(path)

    for origin_ns in (1_000_000, 10_000_000):
        profiler.session_started(
            input_timestamp_origin_ns=origin_ns,
            session_desc=_SESSION_DESC,
            client_window_type="NativeWindowClientWindow",
            time_ns=origin_ns,
        )
        profiler.ui_step_started(
            _input(),
            generation=0,
            step=0,
            time_ns=origin_ns + 2_000_000,
        )
        profiler.close()

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert sum(record["phase"] == "session_started" for record in records) == 2
    summaries = [
        record
        for record in records
        if record["phase"] == "profile_summary"
        and record["metric"] == "input_to_ui_step_s"
    ]
    assert [record["count"] for record in summaries] == [1, 1]


def test_profile_summary_bounds_quantile_storage(tmp_path) -> None:
    profiler = RuntimeProfiler(tmp_path / "runtime.jsonl")
    profiler.session_started(
        input_timestamp_origin_ns=0,
        session_desc=_SESSION_DESC,
        client_window_type="WebRTCClientWindow",
        time_ns=0,
    )

    for index in range(1_100):
        profiler.ui_step_started(
            _input(index),
            generation=0,
            step=index,
            time_ns=index * 1_000 + 1,
        )
        profiler.window_write_completed(
            generation=0,
            ui_step=index,
            time_ns=index * 1_000 + 2,
        )

    for summary, maximum in zip(
        profiler.summary().values(),
        (1e-9, 2e-9),
        strict=True,
    ):
        assert summary["count"] == 1_100
        assert summary["quantile_sample_count"] == 1_024
        assert summary["quantiles_approximate"] is True
        assert summary["max_s"] == pytest.approx(maximum)
    profiler.close()
