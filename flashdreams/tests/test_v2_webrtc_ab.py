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

"""CPU tests for the v2 WebRTC A/B benchmark artifact and analyzer."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tools.benchmarks import v2_webrtc_ab as benchmark_module
from tools.benchmarks.scenarios import load_scenario_file
from tools.benchmarks.v2_webrtc_ab import (
    build_run_matrix,
    evaluate_acceptance,
    load_benchmark_definition,
    percentile,
    render_markdown,
    run_benchmark,
    summarize_run,
)

pytestmark = pytest.mark.ci_cpu


def _config_payload() -> dict[str, Any]:
    return {
        "benchmarks": [
            {
                "id": "demo-ab",
                "application": "demo-app",
                "common_application_args": ["--total-blocks", "3"],
                "variants": {
                    "baseline": {"application_args": ["--no-ui"]},
                    "candidate": {"application_args": ["--ui"]},
                },
                "run_order": ["baseline", "candidate", "candidate", "baseline"],
                "baseline": "baseline",
                "candidate": "candidate",
                "warmup_steps": 1,
                "measured_steps": 2,
                "receiver_ready_timeout_s": 2.0,
                "receiver_drain_timeout_s": 3.0,
                "worker_timeout_s": 10.0,
                "metadata": {"seed": 42},
                "acceptance": {
                    "decoded_fps_ratio_min": 0.95,
                    "model_fps_ratio_min": 0.95,
                    "model_step_wall_median_ratio_max": 1.05,
                    "model_step_wall_p90_ratio_max": 1.05,
                    "publish_wait_p90_s_max": 0.001,
                    "decoded_interarrival_p90_s_max": 0.15,
                    "decoded_interarrival_max_s_max": 0.25,
                },
            }
        ]
    }


def _write_config(tmp_path: Path, payload: dict[str, Any] | None = None) -> Path:
    path = tmp_path / "benchmarks.json"
    path.write_text(json.dumps(payload or _config_payload()), encoding="utf-8")
    return path


def _raw_run(run_id: str, label: str, *, time_scale: float = 1.0) -> dict[str, Any]:
    model_results = [
        {
            "step_index": 0,
            "frame_count": 12,
            "recorded_at_s": 0.0,
            "metrics": {
                "model_step_wall_s": 99.0,
                "runtime_presentation_publish_wait_s": 99.0,
            },
        },
        {
            "step_index": 1,
            "frame_count": 12,
            "recorded_at_s": 1.0 * time_scale,
            "metrics": {
                "model_step_wall_s": 0.8 * time_scale,
                "runtime_presentation_publish_wait_s": 0.0004,
            },
        },
        {
            "step_index": 2,
            "frame_count": 12,
            "recorded_at_s": 2.0 * time_scale,
            "metrics": {
                "model_step_wall_s": 0.9 * time_scale,
                "runtime_presentation_publish_wait_s": 0.0006,
            },
        },
    ]
    window_writes = [
        {
            "step_index": index,
            "frame_count": 1,
            "submitted_at_s": index * 0.05 * time_scale,
            "write_wall_s": 0.001,
            "metrics": {},
        }
        for index in range(36)
    ]
    receiver_frames = [
        {
            "frame_index": index,
            "pts": index,
            "time_base": 1.0 / 60.0,
            # Transport delay must not affect ordinal warmup exclusion.
            "received_at_s": index * 0.05 * time_scale + 0.2,
            "decode_s": 0.0002,
        }
        for index in range(36)
    ]
    return {
        "run_id": run_id,
        "label": label,
        "status": "pass",
        "wall_time_s": 2.0,
        "model_results": model_results,
        "window_writes": window_writes,
        "receiver_frames": receiver_frames,
        "receiver_drain_complete": True,
        "webrtc_sender_metrics": {
            "webrtc_sender_queue_depth_count": 0,
            "webrtc_sender_queue_capacity_count": 2,
            "webrtc_sender_enqueued_count": 36,
            "webrtc_sender_handed_off_count": 36,
            "webrtc_sender_dropped_for_lag_count": 0,
            "webrtc_sender_discarded_on_close_count": 0,
            "webrtc_sender_oldest_queue_age_s": 0.0,
            "webrtc_sender_materialized_count": 36,
        },
    }


def test_shipped_lingbot_definition_builds_abba_fresh_process_matrix() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    definition = load_benchmark_definition(
        repo_root / "configs" / "v2_webrtc_benchmarks.json",
        "cam2v-lingbot-hud-ab",
    )

    requests = build_run_matrix(definition, repo_root=repo_root)

    assert [request.run_id for request in requests] == [
        "no_hud_1",
        "hud_1",
        "hud_2",
        "no_hud_2",
    ]
    assert requests[0].application == "cam2v-lingbot"
    assert requests[0].application_args[-1] == "--no-ui"
    assert requests[1].application_args[-1] == "--ui"
    assert definition.warmup_steps == 5
    assert definition.measured_steps == 20


def test_shipped_lingbot_definition_is_a_command_backed_manual_scenario() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    scenarios = load_scenario_file(repo_root / "configs" / "v2_webrtc_benchmarks.json")

    scenario = scenarios["cam2v-lingbot-hud-ab"]
    assert scenario.output_dir_arg is None
    assert {"manual", "gpu", "webrtc", "api-v2"}.issubset(scenario.tags)
    assert "tools.benchmarks.v2_webrtc_ab" in scenario.command


def test_definition_rejects_unbalanced_ab_order(tmp_path: Path) -> None:
    payload = _config_payload()
    payload["benchmarks"][0]["run_order"] = [
        "baseline",
        "candidate",
        "candidate",
    ]
    path = _write_config(tmp_path, payload)

    with pytest.raises(ValueError, match="equally many"):
        load_benchmark_definition(path, "demo-ab")


def test_definition_rejects_unknown_acceptance_threshold(tmp_path: Path) -> None:
    payload = _config_payload()
    payload["benchmarks"][0]["acceptance"]["require_receiver_drain"] = True

    with pytest.raises(ValueError, match="unknown thresholds"):
        load_benchmark_definition(_write_config(tmp_path, payload), "demo-ab")


@pytest.mark.parametrize(
    ("name", "value", "error_type", "message"),
    [
        ("decoded_fps_ratio_min", True, TypeError, "must be numeric"),
        ("decoded_fps_ratio_min", 0.0, ValueError, "must be positive"),
        ("decoded_fps_ratio_min", float("inf"), ValueError, "must be finite"),
        ("publish_wait_p90_s_max", -0.1, ValueError, "must be non-negative"),
    ],
)
def test_definition_validates_acceptance_threshold_values(
    tmp_path: Path,
    name: str,
    value: Any,
    error_type: type[Exception],
    message: str,
) -> None:
    payload = _config_payload()
    payload["benchmarks"][0]["acceptance"][name] = value

    with pytest.raises(error_type, match=message):
        load_benchmark_definition(_write_config(tmp_path, payload), "demo-ab")


def test_percentile_is_inclusive_and_interpolated() -> None:
    assert percentile([], 0.9) is None
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == pytest.approx(2.5)
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.9) == pytest.approx(3.7)


def test_summary_excludes_model_and_receiver_warmup() -> None:
    summary = summarize_run(_raw_run("baseline_1", "baseline"), warmup_steps=1)

    assert summary["steady_model_step_count"] == 2
    assert summary["warmup_frame_count"] == 12
    model_step = summary["model_metrics"]["model_step_wall_s"]
    assert model_step["count"] == 2
    assert model_step["median"] == pytest.approx(0.85)
    assert model_step["max"] == pytest.approx(0.9)
    assert summary["decoded_interarrival_s"]["count"] == 23
    assert summary["decoded_steady_fps"] == pytest.approx(20.0)
    assert summary["steady_model_fps"] == pytest.approx(24 / 1.7)
    assert summary["missing_receiver_frame_count"] == 0
    assert summary["sender_dropped_for_lag_count"] == 0
    assert summary["expected_sender_enqueued_frame_count"] == 36
    assert summary["sender_enqueued_frame_count"] == 36
    assert summary["sender_handed_off_frame_count"] == 36
    assert summary["sender_discarded_on_close_count"] == 0
    assert summary["sender_materialized_frame_count"] == 36
    assert summary["receiver_pts_strictly_increasing"] is True


def test_summary_model_fps_includes_every_measured_step() -> None:
    raw = _raw_run("baseline_1", "baseline")
    raw["model_results"][1]["metrics"]["model_step_wall_s"] = 25.0

    summary = summarize_run(raw, warmup_steps=1)

    assert summary["steady_model_fps"] == pytest.approx(24 / 25.9)


def test_summary_counts_an_unexpected_extra_receiver_frame() -> None:
    raw = _raw_run("baseline_1", "baseline")
    raw["receiver_frames"].append(
        {
            "frame_index": 36,
            "pts": 36,
            "time_base": 1.0 / 60.0,
            "received_at_s": 2.0,
            "decode_s": 0.0002,
        }
    )

    summary = summarize_run(raw, warmup_steps=1)

    assert summary["received_frame_count"] == 37
    assert summary["extra_receiver_frame_count"] == 1
    assert summary["decoded_steady_fps"] == pytest.approx(20.0)


def test_summary_rejects_missing_receiver_pts() -> None:
    raw = _raw_run("baseline_1", "baseline")
    raw["receiver_frames"][10].pop("pts")

    summary = summarize_run(raw, warmup_steps=1)

    assert summary["receiver_pts_strictly_increasing"] is False


def test_summary_separates_fifo_queue_evictions_from_decoder_loss() -> None:
    raw = _raw_run("baseline_1", "baseline")
    raw["receiver_frames"] = raw["receiver_frames"][:30]
    raw["webrtc_sender_metrics"].update(
        {
            "webrtc_sender_handed_off_count": 30,
            "webrtc_sender_dropped_for_lag_count": 6,
        }
    )

    summary = summarize_run(raw, warmup_steps=1)

    assert summary["submitted_frame_count"] == 36
    assert summary["sender_handed_off_frame_count"] == 30
    assert summary["sender_dropped_for_lag_count"] == 6
    assert summary["expected_receiver_frame_count"] == 30
    assert summary["missing_receiver_frame_count"] == 0


def test_acceptance_compares_each_abba_pair(tmp_path: Path) -> None:
    definition = load_benchmark_definition(_write_config(tmp_path), "demo-ab")
    raw_runs = [
        _raw_run("baseline_1", "baseline"),
        _raw_run("candidate_1", "candidate", time_scale=1.01),
        _raw_run("candidate_2", "candidate", time_scale=1.02),
        _raw_run("baseline_2", "baseline"),
    ]
    runs = [summarize_run(item, warmup_steps=1) for item in raw_runs]

    acceptance = evaluate_acceptance(runs, definition)

    assert acceptance["pair_count"] == 2
    assert acceptance["passed"] is True
    assert all(run["passed"] for run in acceptance["run_integrity"])
    assert all(pair["passed"] for pair in acceptance["pairs"])
    first = acceptance["pairs"][0]["criteria"]
    assert first["decoded_fps_ratio"]["observed"] == pytest.approx(1 / 1.01)
    assert first["model_fps_ratio"]["observed"] == pytest.approx(1 / 1.01)
    assert first["publish_wait_p90_s"]["passed"] is True


def test_acceptance_enforces_integrity_for_baseline_and_candidate(
    tmp_path: Path,
) -> None:
    definition = load_benchmark_definition(_write_config(tmp_path), "demo-ab")
    complete_runs = [
        summarize_run(_raw_run("baseline_1", "baseline"), warmup_steps=1),
        summarize_run(_raw_run("candidate_1", "candidate"), warmup_steps=1),
        summarize_run(_raw_run("candidate_2", "candidate"), warmup_steps=1),
        summarize_run(_raw_run("baseline_2", "baseline"), warmup_steps=1),
    ]
    invalid_values = {
        "model_step_count": 2,
        "model_step_indices_contiguous": False,
        "steady_model_step_count": 1,
        "submitted_frame_count": 35,
        "receiver_drain_complete": False,
        "receiver_pts_strictly_increasing": False,
        "sender_dropped_for_lag_count": 1,
        "sender_enqueued_frame_count": 35,
        "sender_handed_off_frame_count": 35,
        "sender_materialized_frame_count": 35,
        "sender_discarded_on_close_count": 1,
        "missing_receiver_frame_count": 1,
        "extra_receiver_frame_count": 1,
    }

    for run_index in (0, 1):
        for field_name, invalid_value in invalid_values.items():
            runs = [dict(run) for run in complete_runs]
            runs[run_index][field_name] = invalid_value

            acceptance = evaluate_acceptance(runs, definition)
            integrity = acceptance["run_integrity"][run_index]

            assert integrity["criteria"][field_name]["passed"] is False
            assert integrity["passed"] is False
            assert acceptance["passed"] is False


def test_acceptance_requires_exact_model_step_indices(tmp_path: Path) -> None:
    definition = load_benchmark_definition(_write_config(tmp_path), "demo-ab")
    baseline = summarize_run(_raw_run("baseline_1", "baseline"), warmup_steps=1)
    candidate_raw = _raw_run("candidate_1", "candidate")
    candidate_raw["model_results"][2]["step_index"] = 1
    candidate = summarize_run(candidate_raw, warmup_steps=1)

    acceptance = evaluate_acceptance([baseline, candidate], definition)

    integrity = acceptance["run_integrity"][1]
    assert integrity["criteria"]["model_step_indices_contiguous"]["passed"] is False
    assert acceptance["passed"] is False


def test_acceptance_rejects_generated_frames_not_submitted_to_webrtc(
    tmp_path: Path,
) -> None:
    definition = load_benchmark_definition(_write_config(tmp_path), "demo-ab")
    baseline = summarize_run(_raw_run("baseline_1", "baseline"), warmup_steps=1)
    candidate_raw = _raw_run("candidate_1", "candidate")
    candidate_raw["window_writes"].pop()
    candidate_raw["receiver_frames"].pop()
    candidate_raw["webrtc_sender_metrics"].update(
        {
            "webrtc_sender_enqueued_count": 36,
            "webrtc_sender_handed_off_count": 36,
            "webrtc_sender_materialized_count": 36,
        }
    )
    candidate = summarize_run(candidate_raw, warmup_steps=1)

    acceptance = evaluate_acceptance([baseline, candidate], definition)

    integrity = acceptance["run_integrity"][1]
    assert integrity["criteria"]["submitted_frame_count"]["passed"] is False
    assert acceptance["passed"] is False


def test_acceptance_rejects_bursty_decode_cadence(tmp_path: Path) -> None:
    definition = load_benchmark_definition(_write_config(tmp_path), "demo-ab")
    baseline = summarize_run(_raw_run("baseline_1", "baseline"), warmup_steps=1)
    candidate_raw = _raw_run("candidate_1", "candidate")
    for index, frame in enumerate(candidate_raw["receiver_frames"]):
        frame["received_at_s"] = (
            index * 0.05 if index <= 20 else 1.5 + (index - 21) * 0.005
        )
    candidate = summarize_run(candidate_raw, warmup_steps=1)

    acceptance = evaluate_acceptance([baseline, candidate], definition)

    cadence = acceptance["pairs"][0]["criteria"]
    assert cadence["decoded_interarrival_max_s"]["passed"] is False
    assert acceptance["passed"] is False


def test_acceptance_requires_every_measured_publish_wait_sample(
    tmp_path: Path,
) -> None:
    definition = load_benchmark_definition(_write_config(tmp_path), "demo-ab")
    baseline = summarize_run(_raw_run("baseline_1", "baseline"), warmup_steps=1)
    candidate_raw = _raw_run("candidate_1", "candidate")
    candidate_raw["model_results"][1]["metrics"].pop(
        "runtime_presentation_publish_wait_s"
    )
    candidate = summarize_run(candidate_raw, warmup_steps=1)

    acceptance = evaluate_acceptance([baseline, candidate], definition)

    integrity = acceptance["run_integrity"][1]
    criterion = integrity["criteria"]["presentation_publish_wait_sample_count"]
    assert criterion["passed"] is False
    assert acceptance["passed"] is False


def test_acceptance_fails_when_required_runtime_metric_is_missing(
    tmp_path: Path,
) -> None:
    definition = load_benchmark_definition(_write_config(tmp_path), "demo-ab")
    baseline = summarize_run(_raw_run("baseline_1", "baseline"), warmup_steps=1)
    candidate_raw = _raw_run("candidate_1", "candidate")
    for result in candidate_raw["model_results"]:
        result["metrics"].pop("runtime_presentation_publish_wait_s")
    candidate = summarize_run(candidate_raw, warmup_steps=1)

    acceptance = evaluate_acceptance([baseline, candidate], definition)

    criterion = acceptance["pairs"][0]["criteria"]["publish_wait_p90_s"]
    assert criterion["missing"] is True
    assert criterion["passed"] is False
    assert acceptance["passed"] is False


def test_markdown_calls_out_loopback_scope_and_acceptance(tmp_path: Path) -> None:
    definition = load_benchmark_definition(_write_config(tmp_path), "demo-ab")
    runs = [
        summarize_run(_raw_run("baseline_1", "baseline"), warmup_steps=1),
        summarize_run(_raw_run("candidate_1", "candidate"), warmup_steps=1),
        summarize_run(_raw_run("candidate_2", "candidate"), warmup_steps=1),
        summarize_run(_raw_run("baseline_2", "baseline"), warmup_steps=1),
    ]
    summary = {
        "benchmark_id": definition.id,
        "protocol": {
            "application": definition.application,
            "warmup_steps": definition.warmup_steps,
            "measured_steps": definition.measured_steps,
            "run_order": list(definition.run_order),
        },
        "runs": runs,
        "acceptance": evaluate_acceptance(runs, definition),
    }

    report = render_markdown(summary)

    assert "loopback server-to-aiortc-decoder" in report
    assert "| baseline_1 | pass |" in report
    assert "### Run baseline_1: PASS" in report
    assert "`sender_dropped_for_lag_count`" in report
    assert "`publish_wait_p90_s`" in report
    assert "Overall acceptance: **PASS**" in report


def test_dry_run_writes_definition_json_and_report_without_loading_model(
    tmp_path: Path,
) -> None:
    definition = load_benchmark_definition(_write_config(tmp_path), "demo-ab")
    output_dir = tmp_path / "out"

    summary, returncode = run_benchmark(
        definition,
        output_dir=output_dir,
        repo_root=tmp_path,
        dry_run=True,
        enforce_acceptance=True,
    )

    assert returncode == 0
    assert summary["dry_run"] is True
    assert [run["run_id"] for run in summary["runs"]] == [
        "baseline_1",
        "candidate_1",
        "candidate_2",
        "baseline_2",
    ]
    assert (output_dir / "definition.json").exists()
    assert (output_dir / "webrtc_ab.json").exists()
    assert "Overall acceptance: **FAIL**" in (output_dir / "webrtc_ab.md").read_text(
        encoding="utf-8"
    )


def test_worker_invalid_artifact_is_rewritten_as_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition = load_benchmark_definition(_write_config(tmp_path), "demo-ab")
    request = build_run_matrix(definition, repo_root=tmp_path)[0]
    output_dir = tmp_path / "out"

    def run_with_malformed_artifact(
        command: tuple[str, ...], **_: Any
    ) -> subprocess.CompletedProcess[str]:
        output_index = command.index("--worker-output") + 1
        Path(command[output_index]).write_text("{", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(benchmark_module.subprocess, "run", run_with_malformed_artifact)

    artifact = benchmark_module._launch_worker(
        request,
        output_dir,
        timeout_s=definition.worker_timeout_s,
    )

    assert artifact["status"] == "fail"
    assert "invalid artifact" in artifact["exception"]
    persisted = json.loads(
        (output_dir / "runs" / request.run_id / "raw.json").read_text(encoding="utf-8")
    )
    assert persisted == artifact
