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

"""CPU checks for correlated V2 runtime profiles."""

import json

import pytest
from flashdreams.runtime_v2.runtime_profiler import RuntimeProfiler
from flashdreams.runtime_v2.user_input_event import (
    KeyboardInputState,
    KeyboardUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from numpy import uint64

pytestmark = pytest.mark.ci_cpu


class _Clock:
    def __init__(self, now_ns: int = 0) -> None:
        self.now_ns = now_ns

    def __call__(self) -> int:
        return self.now_ns


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


def test_profile_correlates_ui_claim_through_first_window_write(tmp_path) -> None:
    clock = _Clock()
    path = tmp_path / "runtime.jsonl"
    profiler = RuntimeProfiler(path, clock_ns=clock)
    profiler.session_started(input_timestamp_origin_ns=0, time_ns=0)

    profiler.model_step_started(generation=2, step=3, time_ns=11_000_000)
    profiler.model_step_completed(
        generation=2,
        step=3,
        duration_s=0.02,
        time_ns=31_000_000,
    )
    profiler.chunk_published(
        generation=2,
        step=3,
        frame_count=4,
        wait_s=0.003,
        time_ns=34_000_000,
    )
    profiler.frame_selected(
        generation=2,
        step=3,
        frame=0,
        frame_count=4,
        time_ns=41_000_000,
    )
    profiler.ui_step_started(
        _input(),
        generation=2,
        step=9,
        time_ns=11_000_000,
    )
    profiler.ui_step_completed(
        generation=2,
        step=9,
        duration_s=0.004,
        presented=True,
        time_ns=15_000_000,
    )
    profiler.window_write_completed(
        endpoint="native_presenter_return",
        generation=2,
        ui_step=9,
        time_ns=51_000_000,
    )

    summary = profiler.summary()
    assert summary["input_to_ui_step_s"]["median_s"] == pytest.approx(0.01)
    assert summary["input_to_window_write_s"]["median_s"] == pytest.approx(0.05)
    assert summary["model_step_s"]["median_s"] == pytest.approx(0.02)
    assert summary["ui_step_s"]["median_s"] == pytest.approx(0.004)
    assert summary["publish_wait_s"]["median_s"] == pytest.approx(0.003)
    assert summary["frame_to_window_write_s"]["median_s"] == pytest.approx(0.01)

    profiler.close()
    profiler.close()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    phases = [record["phase"] for record in records]
    assert {
        "session_started",
        "input_to_ui_step",
        "input_to_window_write",
        "frame_to_window_write",
    } <= set(phases)
    output_record = next(
        record for record in records if record["phase"] == "input_to_window_write"
    )
    assert output_record == {
        "schema_version": 1,
        "phase": "input_to_window_write",
        "time_ns": 51_000_000,
        "generation": 2,
        "claimed_ui_step": 9,
        "presented_ui_step": 9,
        "endpoint": "native_presenter_return",
        "input_type": "KeyboardUserInputEvent",
        "input_timestamp_us": 1_000,
        "duration_s": 0.05,
    }
    assert phases.count("profile_summary") == 6


def test_disconnected_write_preserves_correlations_until_sender_admission(
    tmp_path,
) -> None:
    profiler = RuntimeProfiler(tmp_path / "runtime.jsonl")
    profiler.session_started(input_timestamp_origin_ns=0, time_ns=0)
    profiler.ui_step_started(_input(), generation=0, step=0, time_ns=2_000_000)
    profiler.frame_selected(
        generation=0,
        step=0,
        frame=0,
        frame_count=1,
        time_ns=3_000_000,
    )

    profiler.window_write_completed(
        endpoint=None,
        generation=0,
        ui_step=0,
        time_ns=4_000_000,
    )
    assert profiler.summary()["input_to_window_write_s"] == {"count": 0}
    assert profiler.summary()["frame_to_window_write_s"] == {"count": 0}

    profiler.window_write_completed(
        endpoint="webrtc_sender_admission",
        generation=0,
        ui_step=1,
        time_ns=5_000_000,
    )
    assert profiler.summary()["input_to_window_write_s"]["median_s"] == pytest.approx(
        0.004
    )
    assert profiler.summary()["frame_to_window_write_s"]["median_s"] == pytest.approx(
        0.002
    )
    profiler.close()


def test_close_is_final_after_a_summary_write_failure(tmp_path) -> None:
    class _FailingOutput:
        def write(self, value: str) -> int:
            del value
            raise OSError("profile write failed")

        def close(self) -> None:
            return

    profiler = RuntimeProfiler(tmp_path / "runtime.jsonl")
    profiler._output.close()
    profiler._output = _FailingOutput()  # type: ignore[assignment]

    with pytest.raises(OSError, match="profile write failed"):
        profiler.close()

    profiler.close()
