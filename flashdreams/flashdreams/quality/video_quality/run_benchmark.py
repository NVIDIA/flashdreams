# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the manifest-driven video-quality *capability* benchmark.

Skeleton/plumbing built on top of #317's harness: for each scenario it generates
a clip (via a pluggable :class:`~...generation.Generator`), runs the scenario's
yes/no checks through pluggable evaluators, and rolls the results into a canon
KPI + capability coverage map. The generator and evaluators are placeholder stubs
for now; the contracts and aggregation are the real deliverable.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from flashdreams.quality.video_quality.evaluators import (
    get_evaluator,
    register_default_evaluators,
)
from flashdreams.quality.video_quality.evaluators.base import CheckResult, EvalSample
from flashdreams.quality.video_quality.generation import (
    Generator,
    get_generator,
    request_from_case,
)
from flashdreams.quality.video_quality.manifest import (
    Check,
    VideoQualityCase,
    load_manifest,
)
from flashdreams.quality.video_quality.metrics import compute_video_metrics
from flashdreams.quality.video_quality.scoring import aggregate_kpi, score_scenario


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    register_default_evaluators()
    started_at = time.time()
    manifest = load_manifest(args.manifest)
    cases = manifest.select_cases(suite=args.suite, case_id=args.case_id)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = get_generator(args.generator)

    scenarios: list[dict[str, Any]] = []
    scenario_results: list[dict[str, Any]] = []
    for case in cases:
        scenario, results = _run_scenario(
            case,
            generator=generator,
            output_dir=output_dir / case.id,
            pass_threshold=args.scenario_pass_threshold,
        )
        scenarios.append(scenario)
        scenario_results.append(
            {"id": case.id, "decision": scenario["decision"], "check_results": results}
        )

    aggregate = aggregate_kpi(
        scenario_results, pass_threshold=args.scenario_pass_threshold
    )
    kpi = aggregate["kpi"]
    decision = "fail" if kpi["failed"] else "pass"
    run = {
        "schema_version": 2,
        "kind": "video_quality_benchmark",
        "decision": decision,
        "manifest_path": str(args.manifest),
        "suite": args.suite,
        "case_id": args.case_id,
        "generator": generator.name,
        "github": _github_metadata(),
        "started_at_unix": started_at,
        "duration_s": time.time() - started_at,
        "output_dir": str(output_dir),
        "kpi": kpi,
        "coverage": aggregate["coverage"],
        "scenarios": scenarios,
    }
    _write_json(output_dir / "benchmark.json", run)

    print(
        f"video-quality benchmark {decision}: "
        f"{kpi['scenario_count']} scenario(s), "
        f"pass_rate={_fmt(kpi['scenario_pass_rate'])}, "
        f"overall_accuracy={_fmt(kpi['overall_accuracy'])}; "
        f"report={output_dir / 'benchmark.json'}"
    )
    return 1 if decision == "fail" and args.fail_on_regression else 0


def _run_scenario(
    case: VideoQualityCase,
    *,
    generator: Generator,
    output_dir: Path,
    pass_threshold: float,
) -> tuple[dict[str, Any], list[CheckResult]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    request = request_from_case(case)
    clip = generator.generate(request)
    sample = EvalSample(
        scenario_id=case.id,
        frames=clip.frames,
        fps=clip.fps,
        conditioning=request.conditioning,
        metadata={"generator": generator.name},
    )

    # Quality/coherence axis (reuse #317 metrics) alongside the canon/checks axis.
    quality_metrics = compute_video_metrics(
        clip.frames,
        fps=clip.fps,
        windows=case.windows,
        metric_groups=case.metrics or None,
    )

    by_evaluator: dict[str, list[Check]] = defaultdict(list)
    for check in case.checks:
        by_evaluator[check.evaluator].append(check)
    results: list[CheckResult] = []
    for evaluator_name, checks in by_evaluator.items():
        results.extend(get_evaluator(evaluator_name).evaluate(sample, checks))

    decision = score_scenario(results, pass_threshold=pass_threshold)
    scenario = {
        "id": case.id,
        "category": case.category,
        "description": case.description,
        "suites": list(case.suites),
        "control_channels": sorted({c.control_channel for c in case.checks}),
        "generator": generator.name,
        "decision": decision,
        "quality_metrics": quality_metrics,
        "checks": [asdict(r) for r in results],
    }
    _write_json(output_dir / "scenario.json", scenario)
    return scenario, results


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/video_quality_benchmark.yml"),
        help="Path to the benchmark case manifest.",
    )
    parser.add_argument(
        "--suite",
        default="nightly",
        help="Suite label to run (e.g. nightly, per_commit, vlm_experimental).",
    )
    parser.add_argument("--case-id", help="Optional single case id to run.")
    parser.add_argument(
        "--generator",
        default="synthetic_passthrough",
        help="Generator backend name (placeholder default until real backends land).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/video_quality_benchmark"),
        help="Directory for the run report and per-scenario artifacts.",
    )
    parser.add_argument(
        "--scenario-pass-threshold",
        type=float,
        default=1.0,
        help="Fraction of a scenario's checks that must be correct to pass.",
    )
    parser.add_argument(
        "--fail-on-regression",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Exit non-zero when scenarios fail (off by default while evaluators are stubs).",
    )
    return parser.parse_args(argv)


def _github_metadata() -> dict[str, str | None]:
    return {
        "commit_sha": os.environ.get("GITHUB_SHA"),
        "ref": os.environ.get("GITHUB_REF"),
        "event_name": os.environ.get("GITHUB_EVENT_NAME"),
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "job": os.environ.get("GITHUB_JOB"),
        "actor": os.environ.get("GITHUB_ACTOR"),
    }


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
