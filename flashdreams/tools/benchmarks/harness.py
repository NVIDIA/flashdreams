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

"""Command-backed local benchmark runner."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

from tools.benchmarks.environment import collect_environment
from tools.benchmarks.metrics import (
    MetricRecord,
    lifecycle_record,
    record_from_quality_metrics,
    records_from_log,
    records_from_stats_file,
    summarize_records,
    write_metrics_csv,
    write_metrics_ndjson,
)
from tools.benchmarks.quality import (
    QualityBaselineConfig,
    find_baseline_video,
    run_baseline_clip_compare,
)
from tools.benchmarks.report import write_html_report
from tools.benchmarks.scenarios import (
    BenchmarkScenario,
    QualityCommandConfig,
    render_template,
)

_SCHEMA_VERSION = "0.1.0"
ProgressCallback = Callable[[str], None]
_PROCESS_KILL_TIMEOUT_S = 10.0
_PROCESS_WAIT_POLL_S = 0.1


@dataclass(frozen=True, kw_only=True)
class ScenarioRunResult:
    """Result for one benchmark scenario command."""

    scenario: BenchmarkScenario
    output_dir: Path
    log_path: Path
    command: tuple[str, ...]
    status: str
    returncode: int | None
    timed_out: bool
    started_at: str
    finished_at: str
    wall_time_s: float
    artifacts: Mapping[str, tuple[str, ...]]
    metric_records: tuple[MetricRecord, ...] = ()
    metric_summary: Mapping[str, Mapping[str, float | int]] = field(
        default_factory=dict
    )
    quality_results: tuple[Mapping[str, Any], ...] = ()

    def to_manifest(self, *, output_root: Path) -> dict[str, Any]:
        return {
            "id": self.scenario.id,
            "name": self.scenario.name,
            "description": self.scenario.description,
            "tags": list(self.scenario.tags),
            "report_group": (
                None
                if self.scenario.report_group is None
                else self.scenario.report_group.to_manifest()
            ),
            "status": self.status,
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "wall_time_s": self.wall_time_s,
            "command": shlex.join(self.command),
            "argv": list(self.command),
            "output_dir": _relpath(self.output_dir, output_root),
            "log_path": _relpath(self.log_path, output_root),
            "warmup_steps": self.scenario.warmup_steps,
            "artifacts": {
                key: [path for path in values] for key, values in self.artifacts.items()
            },
            "metric_summary": {
                key: dict(value) for key, value in self.metric_summary.items()
            },
            "metric_summary_metadata": _metric_summary_metadata(self.metric_records),
            "metric_highlights": _metric_highlights(
                self.metric_records,
                self.metric_summary,
            ),
            "quality_results": [dict(result) for result in self.quality_results],
            "scenario": self.scenario.to_manifest(),
        }


def run_benchmark_suite(
    scenarios: Sequence[BenchmarkScenario],
    *,
    output_root: Path,
    repo_root: Path,
    keep_going: bool = True,
    dry_run: bool = False,
    base_env: Mapping[str, str] | None = None,
    include_environment: bool = True,
    quality_baseline: QualityBaselineConfig | None = None,
    progress: ProgressCallback | None = None,
    progress_interval_s: float = 30.0,
) -> dict[str, Any]:
    """Run selected scenarios and write benchmark artifacts."""
    if progress_interval_s <= 0:
        raise ValueError("progress_interval_s must be > 0")
    output_root = output_root.resolve()
    repo_root = repo_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    environment = collect_environment(repo_root) if include_environment else {}
    environment_path = output_root / "environment.json"
    _write_json(environment_path, environment)

    results: list[ScenarioRunResult] = []
    all_records: list[MetricRecord] = []
    scenario_count = len(scenarios)
    for index, scenario in enumerate(scenarios, start=1):
        _emit_progress(
            progress,
            f"[flashdreams-benchmark] starting scenario {index}/{scenario_count}: "
            f"{scenario.id}",
        )
        result = run_scenario(
            scenario,
            output_root=output_root,
            repo_root=repo_root,
            dry_run=dry_run,
            base_env=base_env,
            quality_baseline=quality_baseline,
            progress=progress,
            progress_interval_s=progress_interval_s,
        )
        _emit_progress(
            progress,
            f"[flashdreams-benchmark] finished scenario {index}/{scenario_count}: "
            f"{scenario.id} status={result.status} "
            f"elapsed={_format_duration(result.wall_time_s)} log={result.log_path}",
        )
        results.append(result)
        all_records.extend(result.metric_records)
        if result.status not in ("pass", "dry_run") and not keep_going:
            _emit_progress(
                progress,
                "[flashdreams-benchmark] stopping after failure because "
                "--no-keep-going is set",
            )
            break

    metrics_ndjson_path = output_root / "metrics.ndjson"
    metrics_csv_path = output_root / "metrics.csv"
    write_metrics_ndjson(all_records, metrics_ndjson_path)
    write_metrics_csv(all_records, metrics_csv_path)

    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "mode": "local-developer",
        "created_at": _utc_now(),
        "dry_run": dry_run,
        "repo_root": str(repo_root),
        "output_root": str(output_root),
        "quality_baseline": (
            None if quality_baseline is None else quality_baseline.to_manifest()
        ),
        "environment_path": _relpath(environment_path, output_root),
        "metrics_ndjson_path": _relpath(metrics_ndjson_path, output_root),
        "metrics_csv_path": _relpath(metrics_csv_path, output_root),
        "report_path": "report.html",
        "scenario_count": len(results),
        "failed_scenario_count": sum(
            1 for result in results if result.status not in ("pass", "dry_run")
        ),
        "environment": environment,
        "scenarios": [
            result.to_manifest(output_root=output_root) for result in results
        ],
    }
    manifest_path = output_root / "manifest.json"
    _write_json(manifest_path, manifest)
    write_html_report(manifest, output_root / "report.html")
    return manifest


def _metric_highlights(
    records: Iterable[MetricRecord],
    metric_summary: Mapping[str, Mapping[str, float | int]],
) -> dict[str, float | int]:
    highlights: dict[str, float | int] = {}
    records_tuple = tuple(records)
    command_wall_s = _first_metric_value(
        records_tuple,
        record_type="lifecycle",
        metric_name="command_wall_s",
    )
    if command_wall_s is not None:
        highlights["command_wall_s"] = command_wall_s

    startup_step_total_s = _first_step_metric_value(records_tuple, "total_s")
    if startup_step_total_s is not None:
        highlights["startup_step_total_s"] = startup_step_total_s

    startup_step_total_wo_finalize_s = _first_step_metric_value(
        records_tuple,
        "total_wo_finalize_s",
    )
    if startup_step_total_wo_finalize_s is not None:
        highlights["startup_step_total_wo_finalize_s"] = (
            startup_step_total_wo_finalize_s
        )

    for metric_name in (
        "total_s",
        "total_wo_finalize_s",
        "quality_score",
        "quality_similarity_score",
        "pai_bench_g_score",
        "pai_bench_long_score",
    ):
        median = _summary_median(metric_summary, metric_name)
        if median is not None:
            highlights[f"{metric_name}_median"] = median
    return highlights


def _metric_summary_metadata(
    records: Iterable[MetricRecord],
) -> dict[str, dict[str, list[str]]]:
    record_types_by_metric: dict[str, set[str]] = {}
    parsers_by_metric: dict[str, set[str]] = {}
    sources_by_metric: dict[str, set[str]] = {}
    for record in records:
        for key, value in record.metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            metric = str(key)
            record_types_by_metric.setdefault(metric, set()).add(record.record_type)
            parser = record.metadata.get("parser")
            if isinstance(parser, str):
                parsers_by_metric.setdefault(metric, set()).add(parser)
            if record.source:
                sources_by_metric.setdefault(metric, set()).add(record.source)

    metadata: dict[str, dict[str, list[str]]] = {}
    for metric, record_types in sorted(record_types_by_metric.items()):
        entry = {"record_types": sorted(record_types)}
        parsers = parsers_by_metric.get(metric)
        if parsers:
            entry["parsers"] = sorted(parsers)
        sources = sources_by_metric.get(metric)
        if sources:
            entry["sources"] = sorted(sources)
        metadata[metric] = entry
    return metadata


def _first_metric_value(
    records: Iterable[MetricRecord],
    *,
    record_type: str,
    metric_name: str,
) -> float | int | None:
    for record in records:
        if record.record_type != record_type:
            continue
        value = record.metrics.get(metric_name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        return value
    return None


def _first_step_metric_value(
    records: Iterable[MetricRecord],
    metric_name: str,
) -> float | int | None:
    step_records = [record for record in records if record.record_type == "step"]
    step_records.sort(
        key=lambda record: (
            record.record_index is None,
            -1 if record.record_index is None else record.record_index,
        )
    )
    for record in step_records:
        value = record.metrics.get(metric_name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        return value
    return None


def _summary_median(
    metric_summary: Mapping[str, Mapping[str, float | int]],
    metric_name: str,
) -> float | int | None:
    stats = metric_summary.get(metric_name)
    if not isinstance(stats, Mapping):
        return None
    value = stats.get("median")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def run_scenario(
    scenario: BenchmarkScenario,
    *,
    output_root: Path,
    repo_root: Path,
    dry_run: bool = False,
    base_env: Mapping[str, str] | None = None,
    quality_baseline: QualityBaselineConfig | None = None,
    progress: ProgressCallback | None = None,
    progress_interval_s: float = 30.0,
) -> ScenarioRunResult:
    """Run one command-backed scenario and collect its artifacts."""
    scenario_output_dir = output_root / "scenarios" / scenario.id
    scenario_output_dir.mkdir(parents=True, exist_ok=True)
    log_path = scenario_output_dir / "command.log"
    context = _scenario_context(
        scenario,
        output_root=output_root,
        repo_root=repo_root,
        scenario_output_dir=scenario_output_dir,
        log_path=log_path,
    )
    env = scenario.rendered_env(context=context, base_env=base_env)
    command = scenario.rendered_command(context=context, env=env)
    cwd = Path(scenario.rendered_cwd(context=context, env=env)).resolve()
    _emit_progress(progress, f"[flashdreams-benchmark] log: {log_path}")
    _emit_progress(progress, f"[flashdreams-benchmark] command: {shlex.join(command)}")

    started_at = _utc_now()
    start = time.perf_counter()
    returncode: int | None = None
    timed_out = False
    if dry_run:
        status = "dry_run"
        _write_command_log_header(
            log_path,
            command=command,
            cwd=cwd,
            dry_run=True,
        )
    else:
        returncode, timed_out = _run_process(
            command,
            cwd=cwd,
            env=env,
            log_path=log_path,
            timeout_s=scenario.timeout_s,
            progress=progress,
            progress_interval_s=progress_interval_s,
            progress_label=f"scenario {scenario.id}",
        )
        if timed_out:
            status = "pass" if scenario.timeout_status == "pass" else "timeout"
        else:
            status = "pass" if returncode == 0 else "fail"
    wall_time_s = time.perf_counter() - start
    finished_at = _utc_now()

    metric_records = [
        lifecycle_record(
            scenario_id=scenario.id,
            wall_time_s=wall_time_s,
            returncode=returncode,
            status=status,
            command=shlex.join(command),
            timed_out=timed_out,
        )
    ]
    metric_records.extend(
        _collect_scenario_metrics(
            scenario,
            scenario_output_dir=scenario_output_dir,
            output_root=output_root,
            log_path=log_path,
        )
    )
    artifacts = _collect_artifacts(
        scenario, scenario_output_dir, output_root=output_root
    )
    quality_results = _run_baseline_quality(
        scenario,
        output_root=output_root,
        scenario_output_dir=scenario_output_dir,
        artifacts=artifacts,
        quality_baseline=quality_baseline,
        dry_run=dry_run,
        progress=progress,
    )
    quality_results.extend(
        _run_quality_commands(
            scenario,
            scenario_output_dir=scenario_output_dir,
            output_root=output_root,
            repo_root=repo_root,
            artifacts=artifacts,
            base_env=base_env,
            dry_run=dry_run,
            progress=progress,
            progress_interval_s=progress_interval_s,
        )
    )
    metric_records.extend(
        _quality_metric_records(quality_results, output_root=output_root)
    )
    artifacts = _collect_artifacts(
        scenario, scenario_output_dir, output_root=output_root
    )
    metric_summary = summarize_records(
        metric_records, warmup_steps=scenario.warmup_steps
    )

    result = ScenarioRunResult(
        scenario=scenario,
        output_dir=scenario_output_dir,
        log_path=log_path,
        command=command,
        status=status,
        returncode=returncode,
        timed_out=timed_out,
        started_at=started_at,
        finished_at=finished_at,
        wall_time_s=wall_time_s,
        artifacts=artifacts,
        metric_records=tuple(metric_records),
        metric_summary=metric_summary,
        quality_results=tuple(quality_results),
    )
    _write_json(
        scenario_output_dir / "scenario_manifest.json",
        result.to_manifest(output_root=output_root),
    )
    return result


def _run_process(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    log_path: Path,
    timeout_s: float | None,
    progress: ProgressCallback | None = None,
    progress_interval_s: float = 30.0,
    progress_label: str | None = None,
) -> tuple[int | None, bool]:
    _write_command_log_header(log_path, command=command, cwd=cwd, dry_run=False)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            env=dict(env),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        start = time.perf_counter()
        deadline = None if timeout_s is None else start + timeout_s
        if progress is None:
            try:
                return process.wait(timeout=timeout_s), False
            except subprocess.TimeoutExpired:
                _terminate_process(process)
                log.write("\n[flashdreams-benchmark] command timed out\n")
                return process.returncode, True
        while True:
            now = time.perf_counter()
            wait_s = progress_interval_s
            if deadline is not None:
                remaining_s = deadline - now
                if remaining_s <= 0:
                    _terminate_process(process)
                    log.write("\n[flashdreams-benchmark] command timed out\n")
                    return process.returncode, True
                wait_s = min(wait_s, remaining_s)
            try:
                return process.wait(timeout=wait_s), False
            except subprocess.TimeoutExpired:
                now = time.perf_counter()
                if deadline is not None and now >= deadline:
                    _terminate_process(process)
                    log.write("\n[flashdreams-benchmark] command timed out\n")
                    return process.returncode, True
                label = progress_label or shlex.join(command)
                _emit_progress(
                    progress,
                    f"[flashdreams-benchmark] still running: {label} "
                    f"elapsed={_format_duration(now - start)} log={log_path}",
                )


def _terminate_process(process: subprocess.Popen[str]) -> None:
    try:
        root = psutil.Process(process.pid)
        processes = [*reversed(root.children(recursive=True)), root]
    except psutil.NoSuchProcess:
        processes = []

    for item in processes:
        try:
            item.kill()
        except psutil.NoSuchProcess:
            pass

    deadline = time.monotonic() + _PROCESS_KILL_TIMEOUT_S
    alive = processes
    while alive:
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0:
            break
        _, alive = psutil.wait_procs(
            alive,
            timeout=min(_PROCESS_WAIT_POLL_S, remaining_s),
        )
        alive = [item for item in alive if not _is_terminated_zombie(item)]
    if alive:
        pids = ", ".join(str(item.pid) for item in alive)
        raise RuntimeError(
            f"Failed to terminate process tree rooted at PID {process.pid}; "
            f"processes still alive: {pids}."
        )
    process.wait(timeout=_PROCESS_KILL_TIMEOUT_S)


def _is_terminated_zombie(process: psutil.Process) -> bool:
    try:
        return process.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


def _collect_scenario_metrics(
    scenario: BenchmarkScenario,
    *,
    scenario_output_dir: Path,
    output_root: Path,
    log_path: Path,
) -> list[MetricRecord]:
    records: list[MetricRecord] = []
    for stats_path in _glob_many(scenario_output_dir, scenario.stats_globs):
        records.extend(
            records_from_stats_file(
                stats_path, scenario_id=scenario.id, source_root=output_root
            )
        )
    records.extend(
        records_from_log(log_path, scenario_id=scenario.id, source_root=output_root)
    )
    return records


def _run_baseline_quality(
    scenario: BenchmarkScenario,
    *,
    output_root: Path,
    scenario_output_dir: Path,
    artifacts: Mapping[str, tuple[str, ...]],
    quality_baseline: QualityBaselineConfig | None,
    dry_run: bool,
    progress: ProgressCallback | None,
) -> list[dict[str, Any]]:
    if quality_baseline is None:
        return []
    if not scenario.quality_baseline_compare:
        return _baseline_video_review(
            scenario,
            output_root=output_root,
            artifacts=artifacts,
            quality_baseline=quality_baseline,
            dry_run=dry_run,
        )
    quality_id = "baseline-clip-compare"
    quality_dir = scenario_output_dir / "quality" / quality_id
    metrics_path = quality_dir / "metrics.json"
    log_path = quality_dir / "command.log"
    if dry_run:
        quality_dir.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "[dry-run] baseline clip quality comparison\n",
            encoding="utf-8",
        )
        return [
            {
                "scenario_id": scenario.id,
                "id": quality_id,
                "status": "dry_run",
                "metrics_path": _relpath(metrics_path, output_root),
                "log_path": _relpath(log_path, output_root),
            }
        ]

    first_video = _first_artifact_path(artifacts.get("videos", ()), output_root)
    _emit_progress(
        progress,
        f"[flashdreams-benchmark] starting quality hook {scenario.id}/{quality_id}",
    )
    start = time.perf_counter()
    effective_quality_baseline = _quality_baseline_for_scenario(
        quality_baseline,
        scenario=scenario,
    )
    result = run_baseline_clip_compare(
        scenario_id=scenario.id,
        candidate_video=first_video,
        quality_dir=quality_dir,
        config=effective_quality_baseline,
    )
    wall_time_s = time.perf_counter() - start
    status = str(result.get("status", "unknown"))
    result.update(
        {
            "scenario_id": scenario.id,
            "id": quality_id,
            "wall_time_s": wall_time_s,
            "metrics_path": _relpath(Path(result["metrics_path"]), output_root),
            "log_path": _relpath(Path(result["log_path"]), output_root),
        }
    )
    for key in ("candidate_video", "baseline_video"):
        value = result.get(key)
        if isinstance(value, Path):
            result[key] = _relpath(value, output_root)
    _emit_progress(
        progress,
        f"[flashdreams-benchmark] finished quality hook "
        f"{scenario.id}/{quality_id} status={status} "
        f"elapsed={_format_duration(wall_time_s)} log={log_path}",
    )
    return [result]


def _baseline_video_review(
    scenario: BenchmarkScenario,
    *,
    output_root: Path,
    artifacts: Mapping[str, tuple[str, ...]],
    quality_baseline: QualityBaselineConfig,
    dry_run: bool,
) -> list[dict[str, Any]]:
    if dry_run:
        return []
    candidate_video = _first_artifact_path(artifacts.get("videos", ()), output_root)
    if candidate_video is None:
        return []
    baseline_video = find_baseline_video(quality_baseline.baseline_dir, scenario.id)
    if baseline_video is None:
        return []
    return [
        {
            "scenario_id": scenario.id,
            "id": "baseline-video-review",
            "status": "available",
            "reason": "quality_baseline_compare is disabled for this scenario",
            "candidate_video": _relpath(candidate_video, output_root),
            "baseline_video": _relpath(baseline_video, output_root),
        }
    ]


def _quality_baseline_for_scenario(
    config: QualityBaselineConfig,
    *,
    scenario: BenchmarkScenario,
) -> QualityBaselineConfig:
    if config.compare_region is not None:
        return config
    return replace(config, compare_region=scenario.quality_compare_region or "full")


def _run_quality_commands(
    scenario: BenchmarkScenario,
    *,
    scenario_output_dir: Path,
    output_root: Path,
    repo_root: Path,
    artifacts: Mapping[str, tuple[str, ...]],
    base_env: Mapping[str, str] | None,
    dry_run: bool,
    progress: ProgressCallback | None,
    progress_interval_s: float,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for quality in scenario.quality_commands:
        quality_dir = scenario_output_dir / "quality" / quality.id
        quality_dir.mkdir(parents=True, exist_ok=True)
        first_video = _first_artifact_path(artifacts.get("videos", ()), output_root)
        context = _scenario_context(
            scenario,
            output_root=output_root,
            repo_root=repo_root,
            scenario_output_dir=scenario_output_dir,
            log_path=scenario_output_dir / "command.log",
        )
        context["quality_dir"] = str(quality_dir)
        context["first_video"] = "" if first_video is None else str(first_video)
        log_path = quality_dir / "command.log"
        metrics_path = quality_dir / "metrics.json"
        try:
            env = _render_quality_env(quality, context=context, base_env=base_env)
            metrics_path = _quality_metrics_path(
                quality,
                quality_dir=quality_dir,
                context=context,
                env=env,
            )
            command = tuple(
                render_template(value, context=context, env=env)
                for value in quality.command
            )
            cwd = Path(
                render_template(
                    quality.cwd or "{repo_root}",
                    context=context,
                    env=env,
                )
            ).resolve()
        except KeyError as exc:
            results.append(
                {
                    "scenario_id": scenario.id,
                    "id": quality.id,
                    "status": "skipped",
                    "reason": str(exc),
                    "metrics_path": _relpath(metrics_path, output_root),
                    "log_path": _relpath(log_path, output_root),
                }
            )
            continue
        if dry_run:
            _write_command_log_header(log_path, command=command, cwd=cwd, dry_run=True)
            status = "dry_run"
            returncode = None
            timed_out = False
            wall_time_s = 0.0
        else:
            _emit_progress(
                progress,
                f"[flashdreams-benchmark] starting quality hook "
                f"{scenario.id}/{quality.id}",
            )
            start = time.perf_counter()
            returncode, timed_out = _run_process(
                command,
                cwd=cwd,
                env=env,
                log_path=log_path,
                timeout_s=quality.timeout_s,
                progress=progress,
                progress_interval_s=progress_interval_s,
                progress_label=f"quality {scenario.id}/{quality.id}",
            )
            wall_time_s = time.perf_counter() - start
            status = "timeout" if timed_out else "pass" if returncode == 0 else "fail"
            metrics_status = _quality_metrics_status(metrics_path)
            if not timed_out and metrics_status is not None:
                status = metrics_status if returncode == 0 else "fail"
            _emit_progress(
                progress,
                f"[flashdreams-benchmark] finished quality hook "
                f"{scenario.id}/{quality.id} status={status} "
                f"elapsed={_format_duration(wall_time_s)} log={log_path}",
            )
        results.append(
            {
                "scenario_id": scenario.id,
                "id": quality.id,
                "status": status,
                "returncode": returncode,
                "timed_out": timed_out,
                "wall_time_s": wall_time_s,
                "command": shlex.join(command),
                "argv": list(command),
                "metrics_path": _relpath(metrics_path, output_root),
                "log_path": _relpath(log_path, output_root),
            }
        )
    return results


def _quality_metrics_status(metrics_path: Path) -> str | None:
    if not metrics_path.is_file():
        return None
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    status = payload.get("status")
    if status in ("pass", "fail", "skipped"):
        return str(status)
    return None


def _quality_metric_records(
    quality_results: Iterable[Mapping[str, Any]],
    *,
    output_root: Path,
) -> list[MetricRecord]:
    records: list[MetricRecord] = []
    for result in quality_results:
        metrics_path_value = result.get("metrics_path")
        scenario_id = result.get("scenario_id")
        if not scenario_id or not isinstance(metrics_path_value, str):
            continue
        metrics_path = output_root / metrics_path_value
        record = record_from_quality_metrics(
            metrics_path,
            scenario_id=str(scenario_id),
            quality_id=str(result.get("id", "")),
            source_root=output_root,
        )
        if record is not None:
            records.append(record)
    return records


def _quality_metrics_path(
    quality: QualityCommandConfig,
    *,
    quality_dir: Path,
    context: Mapping[str, str],
    env: Mapping[str, str],
) -> Path:
    if quality.metrics_path is None:
        return quality_dir / "metrics.json"
    return Path(render_template(quality.metrics_path, context=context, env=env))


def _render_quality_env(
    quality: QualityCommandConfig,
    *,
    context: Mapping[str, str],
    base_env: Mapping[str, str] | None,
) -> dict[str, str]:
    env = dict(base_env or os.environ)
    for key, value in quality.env.items():
        env[str(key)] = render_template(value, context=context, env=env)
    return env


def _collect_artifacts(
    scenario: BenchmarkScenario,
    scenario_output_dir: Path,
    *,
    output_root: Path,
) -> dict[str, tuple[str, ...]]:
    quality_root = scenario_output_dir / "quality"
    videos = _outside_root(
        _glob_many(scenario_output_dir, scenario.video_globs),
        quality_root,
    )
    stats = _outside_root(
        _glob_many(scenario_output_dir, scenario.stats_globs),
        quality_root,
    )
    quality = _collect_quality_artifacts(quality_root)
    video_set = set(videos)
    stats_set = set(stats)
    other = [
        path
        for path in _outside_root(
            _glob_many(scenario_output_dir, scenario.artifact_globs),
            quality_root,
        )
        if path not in video_set and path not in stats_set
    ]
    logs = [scenario_output_dir / "command.log"]
    return {
        "videos": tuple(_relpath(path, output_root) for path in videos),
        "stats": tuple(_relpath(path, output_root) for path in stats),
        "logs": tuple(_relpath(path, output_root) for path in logs if path.exists()),
        "quality": tuple(
            _relpath(path, output_root) for path in quality if path.is_file()
        ),
        "other": tuple(_relpath(path, output_root) for path in other if path.is_file()),
    }


def _outside_root(paths: Iterable[Path], root: Path) -> list[Path]:
    return [path for path in paths if not _is_relative_to(path, root)]


def _collect_quality_artifacts(quality_root: Path) -> list[Path]:
    if not quality_root.exists():
        return []
    artifacts: list[Path] = []
    for path in sorted(quality_root.glob("**/*")):
        if not path.is_file():
            continue
        rel = path.relative_to(quality_root)
        if _is_staged_evaluator_video(rel):
            continue
        if "pai_bench_outputs" in rel.parts and not path.name.endswith(
            "_eval_results.json"
        ):
            continue
        artifacts.append(path)
    return artifacts


def _is_staged_evaluator_video(path: Path) -> bool:
    parts = path.parts
    return len(parts) >= 3 and parts[1] == "staged" and parts[2] == "videos"


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _glob_many(root: Path, patterns: Iterable[str]) -> list[Path]:
    seen: set[Path] = set()
    matches: list[Path] = []
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if path.is_file() and path not in seen:
                seen.add(path)
                matches.append(path)
    return matches


def _first_artifact_path(paths: Sequence[str], output_root: Path) -> Path | None:
    if not paths:
        return None
    return output_root / paths[0]


def _scenario_context(
    scenario: BenchmarkScenario,
    *,
    output_root: Path,
    repo_root: Path,
    scenario_output_dir: Path,
    log_path: Path,
) -> dict[str, str]:
    return {
        "scenario_id": scenario.id,
        "scenario_name": scenario.name,
        "scenario_description": scenario.description,
        "output_dir": str(scenario_output_dir),
        "run_root": str(output_root),
        "repo_root": str(repo_root),
        "log_path": str(log_path),
    }


def _write_command_log_header(
    path: Path,
    *,
    command: Sequence[str],
    cwd: Path,
    dry_run: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prefix = "[dry-run] " if dry_run else ""
    path.write_text(
        f"{prefix}command: {shlex.join(command)}\n"
        f"cwd: {cwd}\n"
        f"started_at: {_utc_now()}\n\n",
        encoding="utf-8",
    )


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8"
    )
    return path


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _emit_progress(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(seconds, 60.0)
    if minutes < 60:
        return f"{int(minutes)}m{remainder:04.1f}s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h{minutes:02d}m{remainder:04.1f}s"


def _relpath(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
