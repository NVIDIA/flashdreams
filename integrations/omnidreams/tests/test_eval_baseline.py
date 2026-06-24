# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from omnidreams.eval.baseline import (
    check_summary_against_baseline,
    format_baseline_check_report,
    resolve_summary_path,
)
from omnidreams.eval.cli import _build_parser

pytestmark = pytest.mark.ci_cpu


def _summary() -> dict[str, object]:
    return {
        "generated": {
            "case_directories": 40,
            "generated_mp4_count": 40,
        },
        "validation": {
            "case_count": 40,
            "expected_frames_from_steps": {
                "157": 40,
            },
            "failure_count": 0,
            "runner_written_frames": {
                "157": 40,
            },
            "total_blocks": {
                "20": 40,
            },
        },
        "drivinggen": {
            "fvd_lite": [
                {
                    "split": "od_26_01_batch_00001",
                    "value": 388.13717153675134,
                },
                {
                    "split": "od_26_01_smoke20",
                    "value": 477.4306044078767,
                },
            ],
        },
        "worldlens": {
            "runs": [
                {
                    "split": "od_26_01_worldlens_40",
                    "artifact_metrics": [
                        {
                            "artifact": "subject_consistency/repeat_0.json",
                            "temporal_consistency_per_frame": 0.9443080609091199,
                            "ts_per_frame": 0.8726091176271439,
                        },
                        {
                            "artifact": "temporal_consistency/repeat_0.json",
                            "temporal_consistency_per_frame": 0.9520937014848758,
                            "ts_per_frame": 0.8295900329947472,
                        },
                    ],
                },
            ],
            "stage_manifest": {
                "case_count": 40,
                "frame_count_mismatch_count": 0,
                "generated_frame_counts": [
                    157,
                ],
                "reference_frame_counts": [
                    157,
                ],
            },
        },
    }


def _canary_summary() -> dict[str, object]:
    return {
        "generated": {
            "case_directories": 3,
            "generated_mp4_count": 3,
        },
        "validation": {
            "case_count": 3,
            "expected_frames_from_steps": {
                "29": 3,
            },
            "failure_count": 0,
            "runner_written_frames": {
                "29": 3,
            },
            "total_blocks": {
                "4": 3,
            },
        },
        "worldlens": {
            "runs": [
                {
                    "split": "od_26_01_canary3_tb4",
                    "artifact_metrics": [
                        {
                            "artifact": "subject_consistency/repeat_0.json",
                            "temporal_consistency_per_frame": 0.9799064011091277,
                            "ts_per_frame": 0.9036903977394104,
                            "video_count": 3,
                        },
                        {
                            "artifact": "temporal_consistency/repeat_0.json",
                            "temporal_consistency_per_frame": 0.9731416248139881,
                            "ts_per_frame": 0.8190207282702128,
                            "video_count": 3,
                        },
                    ],
                },
            ],
            "stage_manifest": {
                "case_count": 3,
                "frame_count_mismatch_count": 0,
                "generated_frame_counts": [
                    29,
                ],
                "reference_frame_counts": [
                    29,
                ],
            },
        },
    }


def _passing_baseline() -> dict[str, object]:
    return {
        "kind": "omnidreams_eval_baseline",
        "schema_version": 1,
        "baseline_id": "od-26.01-worldlens-40-v1",
        "checks": [
            {
                "name": "generated_clip_count",
                "source": "generated.generated_mp4_count",
                "op": "==",
                "expected": 40,
            },
            {
                "name": "validation_failures",
                "source": "validation.failure_count",
                "op": "==",
                "expected": 0,
            },
            {
                "name": "worldlens_temporal_consistency",
                "source": (
                    "worldlens.runs[split=od_26_01_worldlens_40]"
                    ".artifact_metrics[artifact=temporal_consistency/repeat_0.json]"
                    ".temporal_consistency_per_frame"
                ),
                "expected": 0.9521,
                "tolerance": 0.001,
            },
            {
                "name": "drivinggen_fvd_lite",
                "source": "drivinggen.fvd_lite[split=od_26_01_batch_00001].value",
                "max_allowed": 430.0,
            },
        ],
    }


def test_resolve_summary_path_supports_indices_and_list_filters() -> None:
    summary = _summary()

    assert resolve_summary_path(summary, "worldlens.runs[0].split") == (
        "od_26_01_worldlens_40"
    )
    assert (
        resolve_summary_path(
            summary,
            "validation.expected_frames_from_steps[157]",
        )
        == 40
    )
    assert resolve_summary_path(
        summary,
        "worldlens.runs[od_26_01_worldlens_40]"
        ".artifact_metrics[temporal_consistency/repeat_0.json].ts_per_frame",
    ) == pytest.approx(0.8295900329947472)


def test_check_summary_against_baseline_passes_ranges_and_tolerances() -> None:
    report = check_summary_against_baseline(_summary(), _passing_baseline())

    assert report["passed"] is True
    assert report["critical_failures"] == 0
    assert report["check_count"] == 4
    assert {check["name"] for check in report["checks"]} == {
        "generated_clip_count",
        "validation_failures",
        "worldlens_temporal_consistency",
        "drivinggen_fvd_lite",
    }


def test_check_summary_against_baseline_fails_critical_checks() -> None:
    baseline = _passing_baseline()
    checks = cast(list[dict[str, object]], baseline["checks"])
    assert isinstance(checks, list)
    checks[-1] = {
        "name": "drivinggen_fvd_lite",
        "source": "drivinggen.fvd_lite[split=od_26_01_batch_00001].value",
        "max_allowed": 300.0,
    }

    report = check_summary_against_baseline(_summary(), baseline)

    assert report["passed"] is False
    assert report["critical_failures"] == 1
    text = format_baseline_check_report(report)
    assert "Baseline check: FAIL" in text
    assert "`drivinggen_fvd_lite`" in text


def test_warning_failures_do_not_fail_the_report() -> None:
    baseline = {
        "checks": [
            {
                "name": "non_blocking_worldlens_target",
                "source": (
                    "worldlens.runs[od_26_01_worldlens_40]"
                    ".artifact_metrics[temporal_consistency/repeat_0.json]"
                    ".temporal_consistency_per_frame"
                ),
                "min_allowed": 0.99,
                "severity": "warning",
            }
        ]
    }

    report = check_summary_against_baseline(_summary(), baseline)

    assert report["passed"] is True
    assert report["critical_failures"] == 0
    assert report["warning_failures"] == 1


def test_check_baseline_cli_writes_report_and_returns_nonzero_on_failure(
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "summary.json"
    baseline_path = tmp_path / "baseline.json"
    output_path = tmp_path / "baseline-check.json"
    summary_path.write_text(json.dumps(_summary()) + "\n", encoding="utf-8")
    baseline = _passing_baseline()
    checks = cast(list[dict[str, object]], baseline["checks"])
    assert isinstance(checks, list)
    checks[0] = {
        "name": "generated_clip_count",
        "source": "generated.generated_mp4_count",
        "op": "==",
        "expected": 41,
    }
    baseline_path.write_text(json.dumps(baseline) + "\n", encoding="utf-8")

    parser = _build_parser()
    args = parser.parse_args(
        [
            "check-baseline",
            "--summary",
            str(summary_path),
            "--baseline",
            str(baseline_path),
            "--output-json",
            str(output_path),
            "--quiet",
        ]
    )

    assert args.func(args) == 1
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert report["summary_path"] == str(summary_path)
    assert report["baseline_path"] == str(baseline_path)


def test_shipped_od_26_01_baseline_passes_summary_fixture() -> None:
    baseline_path = (
        Path(__file__).resolve().parents[1]
        / "eval_baselines"
        / "od-26.01-worldlens-40-v1.json"
    )
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    report = check_summary_against_baseline(_summary(), baseline)

    assert report["passed"] is True
    assert report["check_count"] == 16


def test_shipped_od_26_01_canary_baseline_passes_summary_fixture() -> None:
    baseline_path = (
        Path(__file__).resolve().parents[1]
        / "eval_baselines"
        / "od-26.01-canary3-tb4-v1.json"
    )
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    report = check_summary_against_baseline(_canary_summary(), baseline)

    assert report["passed"] is True
    assert report["check_count"] == 17
