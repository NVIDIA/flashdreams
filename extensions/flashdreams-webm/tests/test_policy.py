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

"""CPU tests for the machine-local VP9 latency policy."""

import json
from contextlib import nullcontext
from itertools import count
from pathlib import Path
from typing import Any

import pytest
from flashdreams_webm import policy

pytestmark = pytest.mark.ci_cpu


def _benchmark(*, p90_ms: float, frame_budget_ms: float = 1000 / 24) -> dict[str, Any]:
    """Return the benchmark fields the automatic decision consumes."""
    return {
        "codec": "vp9",
        "width": 768,
        "height": 768,
        "frames_per_second": 24,
        "warmup_frames": 8,
        "measured_frames": 24,
        "latencies_ms": [p90_ms] * 24,
        "median_ms": p90_ms,
        "p90_ms": p90_ms,
        "frame_budget_ms": frame_budget_ms,
    }


def test_default_benchmark_uses_the_required_acceptance_workload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, Any] = {}

    class FakeWriter:
        def __init__(
            self,
            path: Path,
            width: int,
            height: int,
            frames_per_second: int,
            codec: str,
        ) -> None:
            observed["arguments"] = (
                path.parent.name,
                width,
                height,
                frames_per_second,
                codec,
            )
            observed["writes"] = 0

        def write_video(self, frame: bytes) -> None:
            assert frame == b"benchmark frame"
            observed["writes"] += 1

        def close(self) -> None:
            observed["closed"] = True

        def abort(self) -> None:
            raise AssertionError("successful benchmark should not abort")

    ticks = count(start=0, step=1_000_000)
    monkeypatch.setattr(policy, "WebmWriter", FakeWriter)
    monkeypatch.setattr(
        policy,
        "_benchmark_frames",
        lambda width, height: (b"benchmark frame",),
    )
    monkeypatch.setattr(policy.time, "perf_counter_ns", lambda: next(ticks))
    monkeypatch.setattr(
        policy.tempfile,
        "TemporaryDirectory",
        lambda **kwargs: nullcontext(tmp_path),
    )

    result = policy.benchmark_video_codec("vp9")

    assert observed["arguments"] == (tmp_path.name, 768, 768, 24, "vp9")
    assert observed["writes"] == 8 + 24
    assert observed["closed"] is True
    assert result["warmup_frames"] == 8
    assert result["measured_frames"] == 24
    assert result["latencies_ms"] == [1.0] * 24
    assert result["frame_budget_ms"] == pytest.approx(41.667, rel=1e-4)


def test_vp9_is_selected_when_p90_fits_one_frame_interval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_path = tmp_path / "selection.json"
    monkeypatch.delenv("FLASHDREAMS_WEBM_CODEC", raising=False)
    monkeypatch.setattr(policy, "_fingerprint", lambda: {"machine": "test"})
    monkeypatch.setattr(
        policy,
        "benchmark_video_codec",
        lambda codec: _benchmark(p90_ms=41.0),
    )

    record = policy.codec_selection(cache_path=cache_path)

    assert record["codec"] == "vp9"
    assert record["excessive_latency"] is False
    assert record["source"] == "benchmark"
    assert json.loads(cache_path.read_text(encoding="utf-8"))["codec"] == "vp9"


def test_vp8_is_selected_only_when_vp9_p90_exceeds_the_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FLASHDREAMS_WEBM_CODEC", raising=False)
    monkeypatch.setattr(policy, "_fingerprint", lambda: {"machine": "test"})
    monkeypatch.setattr(
        policy,
        "benchmark_video_codec",
        lambda codec: _benchmark(p90_ms=41.667),
    )

    record = policy.codec_selection(cache_path=tmp_path / "selection.json")

    assert record["codec"] == "vp8"
    assert record["excessive_latency"] is True
    assert "exceeded" in record["reason"]


def test_matching_machine_record_avoids_repeating_the_benchmark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_path = tmp_path / "selection.json"
    fingerprint = {"machine": "test"}
    cache_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "codec": "vp8",
                "source": "benchmark",
                "fingerprint": fingerprint,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("FLASHDREAMS_WEBM_CODEC", raising=False)
    monkeypatch.setattr(policy, "_fingerprint", lambda: fingerprint)

    def unexpected_benchmark(codec: str) -> dict[str, Any]:
        raise AssertionError(f"unexpected benchmark for {codec}")

    monkeypatch.setattr(policy, "benchmark_video_codec", unexpected_benchmark)

    record = policy.codec_selection(cache_path=cache_path)

    assert record["codec"] == "vp8"
    assert record["source"] == "cache"


def test_environment_override_is_explicit_and_does_not_benchmark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FLASHDREAMS_WEBM_CODEC", "vp9")
    monkeypatch.setattr(policy, "_fingerprint", lambda: {"machine": "test"})

    def unexpected_benchmark(codec: str) -> dict[str, Any]:
        raise AssertionError(f"unexpected benchmark for {codec}")

    monkeypatch.setattr(policy, "benchmark_video_codec", unexpected_benchmark)

    record = policy.codec_selection(cache_path=tmp_path / "selection.json")

    assert record["codec"] == "vp9"
    assert record["source"] == "environment"
    assert not (tmp_path / "selection.json").exists()


def test_invalid_environment_override_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FLASHDREAMS_WEBM_CODEC", "av1")

    with pytest.raises(ValueError, match="must be 'vp8' or 'vp9'"):
        policy.codec_selection(cache_path=tmp_path / "selection.json")
