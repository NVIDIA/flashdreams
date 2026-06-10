# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggregate per-scenario check results into the benchmark KPI + coverage map.

The KPI is a *faithfulness/canon accuracy*: the fraction of yes/no checks the
generated video gets right, reported overall, per category, and as a scenario
pass-rate. The coverage map rolls accuracy up by (category x control_channel) so
a low cell reads as a model gap vs. an interface gap.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from flashdreams.quality.video_quality.evaluators.base import CheckResult


def score_scenario(
    results: Sequence[CheckResult], *, pass_threshold: float
) -> dict[str, Any]:
    """Grade one scenario from its check results."""
    total = len(results)
    correct = sum(1 for r in results if r.correct)
    score = correct / total if total else None
    passed = score is not None and score >= pass_threshold
    return {
        "check_count": total,
        "checks_correct": correct,
        "score": score,
        "status": "pass" if passed else "fail",
        "failed_check_ids": sorted(r.check_id for r in results if not r.correct),
    }


def aggregate_kpi(
    scenarios: Sequence[dict[str, Any]], *, pass_threshold: float
) -> dict[str, Any]:
    """Roll per-scenario results into the headline KPI + coverage map.

    Each item in ``scenarios`` is a dict with keys ``id``, ``decision`` (the
    output of :func:`score_scenario`), and ``check_results`` (a list of
    :class:`CheckResult`).
    """
    all_results: list[CheckResult] = []
    per_category: dict[str, dict[str, int]] = {}
    coverage: dict[str, dict[str, dict[str, int]]] = {}
    passed_scenarios = 0

    for scenario in scenarios:
        results = list(scenario["check_results"])
        all_results.extend(results)
        if scenario["decision"]["status"] == "pass":
            passed_scenarios += 1
        for r in results:
            cat = per_category.setdefault(r.category, {"checks": 0, "correct": 0})
            cat["checks"] += 1
            cat["correct"] += int(r.correct)
            cell = coverage.setdefault(r.category, {}).setdefault(
                r.control_channel, {"checks": 0, "correct": 0}
            )
            cell["checks"] += 1
            cell["correct"] += int(r.correct)

    total_checks = len(all_results)
    total_correct = sum(1 for r in all_results if r.correct)
    overall_accuracy = total_correct / total_checks if total_checks else None

    category_accuracy = {
        cat: (v["correct"] / v["checks"] if v["checks"] else None)
        for cat, v in per_category.items()
    }
    present = [a for a in category_accuracy.values() if a is not None]
    macro = sum(present) / len(present) if present else None

    scenario_count = len(scenarios)
    scenario_pass_rate = (
        passed_scenarios / scenario_count if scenario_count else None
    )
    critical_failures = sum(
        1 for r in all_results if r.tier == "presence" and not r.correct
    )

    kpi = {
        "scenario_count": scenario_count,
        "passed": passed_scenarios,
        "failed": scenario_count - passed_scenarios,
        "check_count": total_checks,
        "checks_correct": total_correct,
        "overall_accuracy": overall_accuracy,
        "macro_category_accuracy": macro,
        "scenario_pass_rate": scenario_pass_rate,
        # Single attempt today; pass@k (k samples) lands with the real generator.
        "pass_at_1": scenario_pass_rate,
        "critical_failures": critical_failures,
        "scenario_pass_threshold": pass_threshold,
        "per_category": {
            cat: {
                "checks": per_category[cat]["checks"],
                "correct": per_category[cat]["correct"],
                "accuracy": category_accuracy[cat],
            }
            for cat in sorted(per_category)
        },
    }
    coverage_map = {
        cat: {
            channel: {
                "checks": cell["checks"],
                "correct": cell["correct"],
                "accuracy": (
                    cell["correct"] / cell["checks"] if cell["checks"] else None
                ),
            }
            for channel, cell in sorted(channels.items())
        }
        for cat, channels in sorted(coverage.items())
    }
    return {"kpi": kpi, "coverage": coverage_map}
