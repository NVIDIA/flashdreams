# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from flashdreams.quality.video_quality.evaluators import (
    get_evaluator,
    register_default_evaluators,
)
from flashdreams.quality.video_quality.evaluators.base import CheckResult, EvalSample
from flashdreams.quality.video_quality.manifest import Check, load_manifest
from flashdreams.quality.video_quality.run_benchmark import main
from flashdreams.quality.video_quality.scoring import aggregate_kpi, score_scenario

pytestmark = pytest.mark.ci_cpu

BENCHMARK_MANIFEST = Path("configs/video_quality_benchmark.yml")


def _result(check_id, category, channel, correct, *, tier="presence") -> CheckResult:
    return CheckResult(
        check_id=check_id,
        category=category,
        question="q?",
        expected="yes",
        answer="yes" if correct else "no",
        correct=correct,
        control_channel=channel,
        tier=tier,
        evaluator="stub_presence",
    )


def test_benchmark_manifest_loads_checks() -> None:
    manifest = load_manifest(BENCHMARK_MANIFEST)
    cases = manifest.select_cases(suite="nightly")
    assert len(cases) == 3

    for case in cases:
        assert case.category  # every case is categorized
        assert case.thresholds == ()  # checks-only cases, thresholds optional
        assert case.checks  # at least one check per case
        for check in case.checks:
            assert check.expected in {"yes", "no"}
            assert check.evaluator  # names an evaluator to dispatch to

    # Example scenes carry an example_data_uuid for the omnidreams generator.
    assert any(case.generation.get("example_data_uuid") for case in cases)


def test_score_scenario_counts_correct() -> None:
    results = [
        _result("a", "ped", "structured", True),
        _result("b", "ped", "structured", False),
    ]
    decision = score_scenario(results, pass_threshold=1.0)
    assert decision["check_count"] == 2
    assert decision["checks_correct"] == 1
    assert decision["score"] == 0.5
    assert decision["status"] == "fail"
    assert decision["failed_check_ids"] == ["b"]


def test_aggregate_kpi_rolls_up_by_category_and_channel() -> None:
    s1 = [
        _result("a", "ped", "structured", True),
        _result("b", "ped", "structured", True),
    ]
    s2 = [_result("c", "animal", "prompt", False)]
    scenarios = [
        {"id": "s1", "decision": score_scenario(s1, pass_threshold=1.0), "check_results": s1},
        {"id": "s2", "decision": score_scenario(s2, pass_threshold=1.0), "check_results": s2},
    ]
    agg = aggregate_kpi(scenarios, pass_threshold=1.0)
    kpi = agg["kpi"]
    assert kpi["scenario_count"] == 2
    assert kpi["check_count"] == 3
    assert kpi["checks_correct"] == 2
    assert kpi["overall_accuracy"] == pytest.approx(2 / 3)
    assert kpi["per_category"]["ped"]["accuracy"] == 1.0
    assert kpi["per_category"]["animal"]["accuracy"] == 0.0
    assert kpi["scenario_pass_rate"] == 0.5
    assert agg["coverage"]["ped"]["structured"]["checks"] == 2
    assert agg["coverage"]["animal"]["prompt"]["accuracy"] == 0.0


def test_stub_evaluator_is_contract_faithful() -> None:
    register_default_evaluators()
    evaluator = get_evaluator("stub_presence")
    sample = EvalSample(
        scenario_id="s", frames=np.zeros((4, 8, 8, 3), dtype=np.uint8), fps=8.0
    )
    check = Check(
        id="c", question="q?", expected="yes", evaluator="stub_presence", category="ped"
    )
    results = evaluator.evaluate(sample, [check])
    assert len(results) == 1
    assert results[0].answer in {"yes", "no"}
    assert isinstance(results[0].correct, bool)


def test_run_benchmark_end_to_end(tmp_path: Path) -> None:
    exit_code = main(
        [
            "--manifest",
            str(BENCHMARK_MANIFEST),
            "--suite",
            "nightly",
            "--output-dir",
            str(tmp_path),
            "--no-fail-on-regression",
        ]
    )
    assert exit_code == 0

    report = json.loads((tmp_path / "benchmark.json").read_text())
    assert report["kind"] == "video_quality_benchmark"
    assert report["kpi"]["scenario_count"] == 3
    assert report["kpi"]["overall_accuracy"] is not None
    assert report["coverage"]  # non-empty (category x control_channel) map
    assert len(report["scenarios"]) == 3
    for scenario in report["scenarios"]:
        assert scenario["checks"]
        assert "quality_metrics" in scenario
