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

"""CPU tests for the local benchmark harness."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from tools.benchmarks import cli as benchmark_cli
from tools.benchmarks import harness as benchmark_harness
from tools.benchmarks import pai_bench_profile
from tools.benchmarks import quality as benchmark_quality
from tools.benchmarks.harness import run_benchmark_suite
from tools.benchmarks.metrics import (
    records_from_log,
    records_from_stats_file,
)
from tools.benchmarks.quality import QualityBaselineConfig
from tools.benchmarks.report import write_html_report
from tools.benchmarks.scenarios import (
    BenchmarkScenario,
    QualityCommandConfig,
    built_in_scenarios,
    load_scenario_file,
)

pytestmark = pytest.mark.ci_cpu


def test_built_in_scenarios_are_selectable() -> None:
    scenarios = built_in_scenarios()

    scenario = scenarios["self-forcing-taehv-smoke"]

    assert scenario.command[:2] == (
        "flashdreams-run",
        "self-forcing-wan2.1-t2v-1.3b-taehv",
    )
    assert scenario.warmup_steps == 1


def test_scenario_renders_placeholders_and_injects_output_dir(tmp_path: Path) -> None:
    scenario = BenchmarkScenario(
        id="demo",
        name="Demo",
        command=("flashdreams-run", "demo-runner", "--flag", "{scenario_id}"),
    )

    command = scenario.rendered_command(
        context={
            "scenario_id": "demo",
            "output_dir": str(tmp_path / "scenario"),
            "run_root": str(tmp_path),
            "repo_root": str(tmp_path / "repo"),
            "log_path": str(tmp_path / "scenario" / "command.log"),
        }
    )

    assert command == (
        "flashdreams-run",
        "demo-runner",
        "--flag",
        "demo",
        "--output-dir",
        str(tmp_path / "scenario"),
    )


def test_load_scenario_file_supports_quality_commands(tmp_path: Path) -> None:
    path = tmp_path / "scenarios.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scenarios": [
                    {
                        "id": "custom",
                        "name": "Custom",
                        "report_group": {
                            "id": "custom-model",
                            "name": "Custom Model",
                        },
                        "command": ["python", "-m", "demo", "{output_dir}"],
                        "output_dir_arg": None,
                        "quality_commands": [
                            {
                                "id": "quality",
                                "command": [
                                    "python",
                                    "-c",
                                    "print('quality')",
                                ],
                                "metrics_path": "{quality_dir}/metrics.json",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    scenarios = load_scenario_file(path)

    assert scenarios["custom"].output_dir_arg is None
    assert scenarios["custom"].report_group is not None
    assert scenarios["custom"].report_group.id == "custom-model"
    assert scenarios["custom"].report_group.name == "Custom Model"
    assert scenarios["custom"].quality_commands[0].id == "quality"


def test_stats_records_normalize_ms_to_seconds(tmp_path: Path) -> None:
    stats_path = tmp_path / "stats_demo.json"
    stats_path.write_text(
        json.dumps(
            [
                {
                    "autoregressive_index": 0,
                    "encode_ms": 12.0,
                    "total_ms_wo_finalize": 20.0,
                    "mem_peak_gib": 5.5,
                }
            ]
        ),
        encoding="utf-8",
    )

    records = records_from_stats_file(
        stats_path, scenario_id="demo", source_root=tmp_path
    )

    assert len(records) == 1
    assert records[0].metrics["encode_s"] == pytest.approx(0.012)
    assert records[0].metrics["total_wo_finalize_s"] == pytest.approx(0.020)
    assert records[0].metrics["mem_peak_gib"] == pytest.approx(5.5)


def test_malformed_stats_records_are_ignored(tmp_path: Path) -> None:
    stats_path = tmp_path / "stats_demo.json"
    stats_path.write_text("{", encoding="utf-8")

    records = records_from_stats_file(
        stats_path, scenario_id="demo", source_root=tmp_path
    )

    assert records == []


def test_log_perf_summary_records_median_and_p90(tmp_path: Path) -> None:
    log_path = tmp_path / "command.log"
    log_path.write_text(
        "[perf][chunk summary] samples=2 warmup=1 window=2; "
        "chunk_total_s 12.0ms/20.0ms med/p90, "
        "pixel_fps 24.0fps/30.0fps med/p90\n",
        encoding="utf-8",
    )

    records = records_from_log(log_path, scenario_id="demo", source_root=tmp_path)

    assert len(records) == 1
    assert records[0].metrics["chunk_total_s_median_s"] == pytest.approx(0.012)
    assert records[0].metrics["chunk_total_s_p90_s"] == pytest.approx(0.020)
    assert records[0].metrics["pixel_fps_median_fps"] == pytest.approx(24.0)
    assert records[0].metadata["parser"] == "perf_summary"
    assert records[0].metadata["label"] == "chunk"


def test_profile_e2e_log_summary_records_parser_provenance(tmp_path: Path) -> None:
    log_path = tmp_path / "command.log"
    log_path.write_text(
        "[profile] e2e latency_median_s=0.25 samples=1\n",
        encoding="utf-8",
    )

    records = records_from_log(log_path, scenario_id="demo", source_root=tmp_path)

    assert len(records) == 1
    record = records[0]
    assert record.record_type == "log_summary"
    assert record.metrics["latency_median_s"] == pytest.approx(0.25)
    assert record.metadata["parser"] == "profile_e2e"


def test_log_interactive_drive_records_session_and_e2e_metrics(tmp_path: Path) -> None:
    log_path = tmp_path / "command.log"
    log_path.write_text(
        "2026-01-01 | INFO | [flashdreams-session] start total_ms=1200.5\n"
        "2026-01-01 | INFO | [flashdreams-session] continue "
        "block_index=1 total_ms=98.0\n"
        "2026-01-01 | INFO | [profile] e2e wall_present_fps=30.0 "
        "avg_adj_control_to_present_ms=110.50 "
        "avg_raw_control_to_present_ms=160.50 samples=150\n",
        encoding="utf-8",
    )

    records = records_from_log(
        log_path, scenario_id="interactive", source_root=tmp_path
    )

    assert len(records) == 3
    assert records[0].record_type == "step"
    assert records[0].record_index == 0
    assert records[0].metrics["total_s"] == pytest.approx(1.2005)
    assert records[1].record_index == 1
    assert records[1].metrics["total_s"] == pytest.approx(0.098)
    assert records[2].record_type == "log_summary"
    assert records[2].metrics["wall_present_fps"] == pytest.approx(30.0)
    assert records[2].metrics["avg_adj_control_to_present_s"] == pytest.approx(0.1105)
    assert records[2].metadata["parser"] == "profile_e2e"
    assert records[2].metadata["samples"] == 150


def test_shipped_one_minute_demo_scenarios_load() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    scenarios = load_scenario_file(
        repo_root / "configs" / "one_minute_demo_benchmarks.json"
    )

    assert set(scenarios) == {
        "lingbot-world-fast-taehv-one-minute",
        "omnidreams-sv-one-minute",
    }


def test_shipped_deterministic_quality_scenarios_load() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    scenarios = load_scenario_file(
        repo_root / "configs" / "deterministic_quality_benchmarks.json"
    )

    assert set(scenarios) == {
        "lingbot-world-fast-taehv-quality-smoke",
        "lingbot-world-fast-taehv-one-minute-review",
        "omnidreams-sv-ci-quality-smoke",
        "omnidreams-sv-one-minute-review",
    }
    omnidreams = scenarios["omnidreams-sv-ci-quality-smoke"]
    assert omnidreams.command[:7] == (
        "uv",
        "run",
        "--project",
        "integrations/omnidreams",
        "python",
        "-m",
        "tools.benchmarks.strict_run",
    )
    assert "omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae" in omnidreams.command
    assert "omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf" not in (
        omnidreams.command
    )
    assert omnidreams.report_group is not None
    assert omnidreams.report_group.id == "omnidreams"
    assert omnidreams.report_group.name == "Omnidreams"
    assert "--pipeline.diffusion-model.seed" in omnidreams.command
    assert _command_value(omnidreams.command, "--total-blocks") == "113"
    assert omnidreams.quality_compare_region == "bottom-half"
    assert omnidreams.quality_baseline_compare is True

    lingbot = scenarios["lingbot-world-fast-taehv-quality-smoke"]
    assert _command_value(lingbot.command, "--total-blocks") == "40"
    assert lingbot.report_group is not None
    assert lingbot.report_group.id == "lingbot"
    assert lingbot.report_group.name == "LingBot"
    assert lingbot.quality_compare_region == "full"
    assert lingbot.quality_baseline_compare is True

    lingbot_review = scenarios["lingbot-world-fast-taehv-one-minute-review"]
    assert _command_value(lingbot_review.command, "--total-blocks") == "81"
    assert lingbot_review.quality_baseline_compare is False

    omnidreams_review = scenarios["omnidreams-sv-one-minute-review"]
    assert _command_value(omnidreams_review.command, "--total-blocks") == "226"
    assert (
        "omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae"
        in omnidreams_review.command
    )
    assert "omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf" not in (
        omnidreams_review.command
    )
    assert "--pipeline.diffusion-model.seed" in omnidreams_review.command
    assert omnidreams_review.quality_baseline_compare is False


def test_run_benchmark_suite_writes_manifest_metrics_and_report(tmp_path: Path) -> None:
    script = (
        "import json, sys; "
        "from pathlib import Path; "
        "out = Path(sys.argv[1]); "
        "out.mkdir(parents=True, exist_ok=True); "
        "(out / 'demo.mp4').write_bytes(b'fake mp4'); "
        "(out / 'stats_demo.json').write_text(json.dumps(["
        "{'autoregressive_index': 0, 'total_ms': 100.0}, "
        "{'autoregressive_index': 1, 'total_ms': 80.0}"
        "])); "
        "print('[perf][chunk summary] samples=2 warmup=1 window=2; "
        "chunk_total_s 80.0ms/80.0ms med/p90')"
    )
    quality_script = (
        "import json, sys; "
        "from pathlib import Path; "
        "metrics = Path(sys.argv[1]); "
        "quality_dir = metrics.parent; "
        "(quality_dir / 'staged' / 'videos').mkdir(parents=True, exist_ok=True); "
        "(quality_dir / 'staged' / 'videos' / 'segment__0.mp4')"
        ".write_bytes(b'scratch'); "
        "(quality_dir / 'staged' / 'prompt_file.json').write_text('[]'); "
        "(quality_dir / 'pai_bench_outputs').mkdir(parents=True, exist_ok=True); "
        "(quality_dir / 'pai_bench_outputs' / 'scores_eval_results.json')"
        ".write_text('{}'); "
        "(quality_dir / 'pai_bench_outputs' / 'cache.tmp').write_text('scratch'); "
        "metrics.write_text(json.dumps({'metrics': {"
        "'quality_score': 0.9, 'pai_bench_long_score': 73.0}}))"
    )
    scenario = BenchmarkScenario(
        id="fake-runner",
        name="Fake runner",
        command=(sys.executable, "-c", script, "{output_dir}"),
        output_dir_arg=None,
        warmup_steps=1,
        quality_commands=(
            QualityCommandConfig(
                id="cheap-quality",
                command=(
                    sys.executable,
                    "-c",
                    quality_script,
                    "{quality_dir}/metrics.json",
                ),
                metrics_path="{quality_dir}/metrics.json",
            ),
        ),
    )

    manifest = run_benchmark_suite(
        [scenario],
        output_root=tmp_path / "run",
        repo_root=tmp_path,
        include_environment=False,
    )

    run_root = Path(manifest["output_root"])
    assert (run_root / "manifest.json").is_file()
    assert (run_root / "metrics.ndjson").is_file()
    assert (run_root / "metrics.csv").is_file()
    assert (run_root / "report.html").is_file()
    scenario_manifest = manifest["scenarios"][0]
    assert scenario_manifest["status"] == "pass"
    assert scenario_manifest["artifacts"]["videos"] == [
        "scenarios/fake-runner/demo.mp4"
    ]
    assert (
        "scenarios/fake-runner/quality/cheap-quality/staged/videos/segment__0.mp4"
        not in scenario_manifest["artifacts"]["videos"]
    )
    assert (
        "scenarios/fake-runner/quality/cheap-quality/staged/videos/segment__0.mp4"
        not in scenario_manifest["artifacts"]["quality"]
    )
    assert (
        "scenarios/fake-runner/quality/cheap-quality/staged/prompt_file.json"
        in scenario_manifest["artifacts"]["quality"]
    )
    scores_path = (
        "scenarios/fake-runner/quality/cheap-quality/pai_bench_outputs/"
        "scores_eval_results.json"
    )
    assert scores_path in scenario_manifest["artifacts"]["quality"]
    assert (
        "scenarios/fake-runner/quality/cheap-quality/pai_bench_outputs/cache.tmp"
        not in scenario_manifest["artifacts"]["quality"]
    )
    assert scenario_manifest["metric_summary"]["total_s"]["median"] == pytest.approx(
        0.080
    )
    assert scenario_manifest["metric_summary"]["quality_score"][
        "median"
    ] == pytest.approx(0.9)
    assert scenario_manifest["metric_summary"]["pai_bench_long_score"][
        "median"
    ] == pytest.approx(73.0)
    assert scenario_manifest["metric_summary_metadata"]["chunk_total_s_median_s"][
        "record_types"
    ] == ["log_summary"]
    assert scenario_manifest["metric_summary_metadata"]["chunk_total_s_median_s"][
        "parsers"
    ] == ["perf_summary"]
    assert scenario_manifest["metric_highlights"][
        "startup_step_total_s"
    ] == pytest.approx(0.100)
    assert scenario_manifest["metric_highlights"]["total_s_median"] == pytest.approx(
        0.080
    )
    assert scenario_manifest["metric_highlights"][
        "quality_score_median"
    ] == pytest.approx(0.9)
    report_html = (run_root / "report.html").read_text(encoding="utf-8")
    detail_report = run_root / "reports" / "fake.html"
    assert detail_report.is_file()
    detail_html = detail_report.read_text(encoding="utf-8")
    assert "Model Reports" in report_html
    assert 'href="reports/fake.html' in report_html
    assert "Scenario Highlights" in report_html
    assert "Startup step" in report_html
    assert "Metric Charts" not in report_html
    assert "Quality Comparisons" not in report_html
    assert "Back to summary" in detail_html
    assert "Command wall time: median" in detail_html
    assert "Metric Charts" in detail_html
    assert "Timing: median" in detail_html
    assert "PAI-Bench scores: median" in detail_html
    assert "PAI-Bench-Long score" in detail_html
    assert 'class="scenario-table"' in detail_html
    assert "<summary>Show command</summary>" in detail_html
    assert 'class="command-text"' in detail_html
    assert "Timing: median vs P90" not in detail_html
    assert ">ms<" in detail_html
    assert "80.00" in detail_html


def test_write_html_report_splits_model_detail_pages(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    manifest = {
        "created_at": "2026-01-01T00:00:00Z",
        "mode": "run",
        "output_root": str(run_root),
        "scenarios": [
            {
                "id": "quality-smoke-a",
                "name": "LingBot scenario",
                "tags": ["quality"],
                "report_group": {
                    "id": "lingbot",
                    "name": "LingBot",
                },
                "status": "pass",
                "wall_time_s": 1.0,
                "command": "python -m lingbot",
                "artifacts": {"videos": ["scenarios/quality-smoke-a/demo.mp4"]},
                "metric_summary": {
                    "quality_score": {"count": 1, "median": 0.98},
                },
                "metric_highlights": {"quality_score_median": 0.98},
            },
            {
                "id": "quality-smoke-b",
                "name": "Omnidreams scenario",
                "tags": ["quality"],
                "report_group": {
                    "id": "omnidreams",
                    "name": "Omnidreams",
                },
                "status": "pass",
                "wall_time_s": 2.0,
                "command": "python -m omnidreams",
                "artifacts": {"videos": ["scenarios/quality-smoke-b/demo.mp4"]},
                "metric_summary": {
                    "quality_score": {"count": 1, "median": 0.97},
                },
                "metric_highlights": {"quality_score_median": 0.97},
            },
        ],
    }

    write_html_report(manifest, run_root / "report.html")

    index_html = (run_root / "report.html").read_text(encoding="utf-8")
    lingbot_html = (run_root / "reports" / "lingbot.html").read_text(encoding="utf-8")
    omnidreams_html = (run_root / "reports" / "omnidreams.html").read_text(
        encoding="utf-8"
    )
    assert 'href="reports/lingbot.html' in index_html
    assert 'href="reports/omnidreams.html' in index_html
    assert "LingBot Benchmark Report" in lingbot_html
    assert "Omnidreams Benchmark Report" in omnidreams_html
    assert "quality-smoke-a" in lingbot_html
    assert "quality-smoke-b" not in lingbot_html
    assert "quality-smoke-b" in omnidreams_html
    assert "quality-smoke-a" not in omnidreams_html
    assert "../scenarios/quality-smoke-a/demo.mp4" in lingbot_html


def test_write_html_report_folds_derived_perf_summary_metrics(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    manifest = {
        "created_at": "2026-01-01T00:00:00Z",
        "mode": "run",
        "output_root": str(run_root),
        "scenarios": [
            {
                "id": "perf-demo",
                "name": "Perf demo",
                "report_group": {"id": "demo", "name": "Demo"},
                "status": "pass",
                "wall_time_s": 1.0,
                "command": "python -m demo",
                "artifacts": {},
                "metric_summary": {
                    "model_step_s": {
                        "count": 2,
                        "median": 0.10,
                        "p90": 0.12,
                        "mean": 0.10,
                        "min": 0.08,
                        "max": 0.12,
                    },
                    "model_step_s_median_s": {"count": 1, "median": 0.10},
                    "model_step_s_p90_s": {"count": 1, "median": 0.12},
                    "chunk_total_s_median_s": {"count": 1, "median": 0.20},
                    "chunk_total_s_p90_s": {"count": 1, "median": 0.24},
                    "generate_s": {"count": 2, "median": 0.777},
                    "generate_s_median_s": {"count": 1, "median": 0.333},
                },
                "metric_summary_metadata": {
                    "model_step_s": {"record_types": ["step"]},
                    "model_step_s_median_s": {
                        "record_types": ["log_summary"],
                        "parsers": ["perf_summary"],
                    },
                    "model_step_s_p90_s": {
                        "record_types": ["log_summary"],
                        "parsers": ["perf_summary"],
                    },
                    "chunk_total_s_median_s": {
                        "record_types": ["log_summary"],
                        "parsers": ["perf_summary"],
                    },
                    "chunk_total_s_p90_s": {
                        "record_types": ["log_summary"],
                        "parsers": ["perf_summary"],
                    },
                    "generate_s": {"record_types": ["step"]},
                    "generate_s_median_s": {
                        "record_types": ["log_summary"],
                        "parsers": ["perf_summary"],
                    },
                },
            }
        ],
    }

    write_html_report(manifest, run_root / "report.html")

    detail_html = (run_root / "reports" / "demo.html").read_text(encoding="utf-8")
    assert "<code>model_step_s</code>" in detail_html
    assert "model_step_s_median_s" not in detail_html
    assert "model_step_s_p90_s" not in detail_html
    assert "<code>chunk_total_s</code>" in detail_html
    assert "chunk_total_s_median_s" not in detail_html
    assert "chunk_total_s_p90_s" not in detail_html
    assert "<code>generate_s</code>" in detail_html
    assert "generate_s_median_s" not in detail_html
    assert "777.0" in detail_html
    assert "333.0" not in detail_html
    assert "<th>P90</th>" not in detail_html
    assert "<th>Mean</th>" not in detail_html


def test_write_html_report_preserves_custom_derived_like_metric_names(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    manifest = {
        "created_at": "2026-01-01T00:00:00Z",
        "mode": "run",
        "output_root": str(run_root),
        "scenarios": [
            {
                "id": "custom-demo",
                "name": "Custom demo",
                "report_group": {"id": "demo", "name": "Demo"},
                "status": "pass",
                "wall_time_s": 1.0,
                "command": "python -m demo",
                "artifacts": {},
                "metric_summary": {
                    "latency": {"count": 1, "median": 0.31},
                    "latency_median_s": {"count": 1, "median": 0.25},
                },
                "metric_summary_metadata": {
                    "latency": {"record_types": ["summary"]},
                    "latency_median_s": {"record_types": ["summary"]},
                },
            }
        ],
    }

    write_html_report(manifest, run_root / "report.html")

    detail_html = (run_root / "reports" / "demo.html").read_text(encoding="utf-8")
    assert "<code>latency</code>" in detail_html
    assert "<code>latency_median_s</code>" in detail_html


def test_write_html_report_preserves_profile_e2e_suffix_metrics(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    manifest = {
        "created_at": "2026-01-01T00:00:00Z",
        "mode": "run",
        "output_root": str(run_root),
        "scenarios": [
            {
                "id": "profile-demo",
                "name": "Profile demo",
                "report_group": {"id": "demo", "name": "Demo"},
                "status": "pass",
                "wall_time_s": 1.0,
                "command": "python -m demo",
                "artifacts": {},
                "metric_summary": {
                    "latency": {"count": 1, "median": 0.31},
                    "latency_median_s": {"count": 1, "median": 0.25},
                },
                "metric_summary_metadata": {
                    "latency": {
                        "record_types": ["log_summary"],
                        "parsers": ["profile_e2e"],
                    },
                    "latency_median_s": {
                        "record_types": ["log_summary"],
                        "parsers": ["profile_e2e"],
                    },
                },
            }
        ],
    }

    write_html_report(manifest, run_root / "report.html")

    detail_html = (run_root / "reports" / "demo.html").read_text(encoding="utf-8")
    assert "<code>latency</code>" in detail_html
    assert "<code>latency_median_s</code>" in detail_html


def test_run_benchmark_suite_emits_progress_heartbeat(tmp_path: Path) -> None:
    script = "import time; time.sleep(0.08)"
    scenario = BenchmarkScenario(
        id="slow-runner",
        name="Slow runner",
        command=(sys.executable, "-c", script),
        output_dir_arg=None,
    )
    messages: list[str] = []

    run_benchmark_suite(
        [scenario],
        output_root=tmp_path / "run",
        repo_root=tmp_path,
        include_environment=False,
        progress=messages.append,
        progress_interval_s=0.01,
    )

    assert any("starting scenario 1/1: slow-runner" in item for item in messages)
    assert any("still running: scenario slow-runner" in item for item in messages)
    assert any(
        "finished scenario 1/1: slow-runner status=pass" in item for item in messages
    )


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process groups")
def test_scenario_timeout_terminates_descendant_processes(tmp_path: Path) -> None:
    child_marker = tmp_path / "child_survived.txt"
    spawned_marker = tmp_path / "child_spawned.txt"
    child_script = tmp_path / "child.py"
    child_script.write_text(
        "import signal, sys, time\n"
        "from pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "time.sleep(1.0)\n"
        "Path(sys.argv[1]).write_text('alive', encoding='utf-8')\n",
        encoding="utf-8",
    )
    parent_script = tmp_path / "parent.py"
    parent_script.write_text(
        "import subprocess, sys, time\n"
        "from pathlib import Path\n"
        "subprocess.Popen([sys.executable, sys.argv[1], sys.argv[2]])\n"
        "Path(sys.argv[3]).write_text('spawned', encoding='utf-8')\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    scenario = BenchmarkScenario(
        id="timeout-runner",
        name="Timeout runner",
        command=(
            sys.executable,
            str(parent_script),
            str(child_script),
            str(child_marker),
            str(spawned_marker),
        ),
        output_dir_arg=None,
        timeout_s=0.3,
    )

    manifest = run_benchmark_suite(
        [scenario],
        output_root=tmp_path / "run",
        repo_root=tmp_path,
        include_environment=False,
    )

    scenario_manifest = manifest["scenarios"][0]
    assert scenario_manifest["status"] == "timeout"
    assert scenario_manifest["timed_out"] is True
    assert spawned_marker.is_file()
    time.sleep(1.2)
    assert not child_marker.exists()


def test_windows_popen_uses_new_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(benchmark_harness.os, "name", "nt")
    monkeypatch.setattr(
        benchmark_harness.subprocess,
        "CREATE_NEW_PROCESS_GROUP",
        512,
        raising=False,
    )

    assert benchmark_harness._process_group_popen_kwargs() == {"creationflags": 512}


def test_windows_timeout_uses_taskkill_process_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        pid = 1234
        returncode: int | None = None
        killed = False

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            if self.returncode is None:
                raise subprocess.TimeoutExpired("fake", timeout or 0.0)
            return self.returncode

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

    process = FakeProcess()
    commands: list[list[str]] = []

    def fake_run(
        command: list[str],
        **_: object,
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        process.returncode = 1
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(benchmark_harness.os, "name", "nt")
    monkeypatch.setattr(benchmark_harness.subprocess, "run", fake_run)

    benchmark_harness._terminate_process(cast(subprocess.Popen[str], process))

    assert commands == [["taskkill", "/F", "/T", "/PID", "1234"]]
    assert process.killed is False


def test_windows_timeout_uses_powershell_when_taskkill_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        pid = 1234
        returncode: int | None = None
        killed = False

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            if self.returncode is None:
                raise subprocess.TimeoutExpired("fake", timeout or 0.0)
            return self.returncode

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

    process = FakeProcess()
    commands: list[list[str]] = []

    def fake_run(
        command: list[str],
        **_: object,
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[0] == "taskkill":
            return subprocess.CompletedProcess(command, 1)
        process.returncode = 1
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(benchmark_harness.os, "name", "nt")
    monkeypatch.setattr(benchmark_harness.subprocess, "run", fake_run)

    benchmark_harness._terminate_process(cast(subprocess.Popen[str], process))

    assert commands[0] == ["taskkill", "/F", "/T", "/PID", "1234"]
    assert commands[1][0] == "powershell.exe"
    assert "-Command" in commands[1]
    assert commands[1][-1] == "1234"
    assert process.killed is False


def test_windows_timeout_cleanup_failure_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        pid = 1234
        returncode: int | None = None
        killed = False

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            if self.returncode is None:
                raise subprocess.TimeoutExpired("fake", timeout or 0.0)
            return self.returncode

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

    process = FakeProcess()

    def fake_run(
        command: list[str],
        **_: object,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(benchmark_harness.os, "name", "nt")
    monkeypatch.setattr(benchmark_harness.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="Failed to terminate Windows process tree"):
        benchmark_harness._terminate_process(cast(subprocess.Popen[str], process))

    assert process.killed is True


def test_quality_command_can_report_skipped_status(tmp_path: Path) -> None:
    script = (
        "from pathlib import Path; import sys; "
        "Path(sys.argv[1]).mkdir(parents=True, exist_ok=True)"
    )
    quality_script = (
        "import json, sys; "
        "from pathlib import Path; "
        "Path(sys.argv[1]).write_text(json.dumps({'status': 'skipped', 'metrics': {}}))"
    )
    scenario = BenchmarkScenario(
        id="optional-quality",
        name="Optional quality",
        command=(sys.executable, "-c", script, "{output_dir}"),
        output_dir_arg=None,
        quality_commands=(
            QualityCommandConfig(
                id="optional-evaluator",
                command=(
                    sys.executable,
                    "-c",
                    quality_script,
                    "{quality_dir}/metrics.json",
                ),
                metrics_path="{quality_dir}/metrics.json",
            ),
        ),
    )

    manifest = run_benchmark_suite(
        [scenario],
        output_root=tmp_path / "run",
        repo_root=tmp_path,
        include_environment=False,
    )

    scenario_manifest = manifest["scenarios"][0]
    assert scenario_manifest["quality_results"][0]["status"] == "skipped"
    assert "quality_score" not in scenario_manifest["metric_summary"]
    assert "pai_bench_long_score" not in scenario_manifest["metric_summary"]


def test_malformed_quality_metrics_do_not_abort_reporting(tmp_path: Path) -> None:
    script = (
        "from pathlib import Path; import sys; "
        "Path(sys.argv[1]).mkdir(parents=True, exist_ok=True)"
    )
    quality_script = (
        "from pathlib import Path; import sys; "
        "metrics = Path(sys.argv[1]); "
        "metrics.parent.mkdir(parents=True, exist_ok=True); "
        "metrics.write_text('{', encoding='utf-8')"
    )
    scenario = BenchmarkScenario(
        id="bad-quality",
        name="Bad quality",
        command=(sys.executable, "-c", script, "{output_dir}"),
        output_dir_arg=None,
        quality_commands=(
            QualityCommandConfig(
                id="truncated-metrics",
                command=(
                    sys.executable,
                    "-c",
                    quality_script,
                    "{quality_dir}/metrics.json",
                ),
                metrics_path="{quality_dir}/metrics.json",
            ),
        ),
    )

    manifest = run_benchmark_suite(
        [scenario],
        output_root=tmp_path / "run",
        repo_root=tmp_path,
        include_environment=False,
    )

    scenario_manifest = manifest["scenarios"][0]
    assert scenario_manifest["status"] == "pass"
    assert scenario_manifest["quality_results"][0]["status"] == "pass"
    assert "quality_score" not in scenario_manifest["metric_summary"]
    assert (tmp_path / "run" / "manifest.json").is_file()
    assert (tmp_path / "run" / "report.html").is_file()


def test_run_benchmark_suite_compares_mp4_to_quality_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_root = tmp_path / "baseline"
    baseline_video = baseline_root / "scenarios" / "fake-runner" / "demo.mp4"
    baseline_video.parent.mkdir(parents=True)
    baseline_video.write_bytes(b"baseline mp4")
    (baseline_root / "manifest.json").write_text(
        json.dumps(
            {
                "scenarios": [
                    {
                        "id": "fake-runner",
                        "artifacts": {
                            "videos": ["scenarios/fake-runner/demo.mp4"],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    reference = _quality_test_video(frames=4, height=12, width=16)
    candidate = reference.copy()
    candidate[..., 0] = np.clip(candidate[..., 0].astype(np.int16) + 10, 0, 255)

    def fake_read_video_rgb(path: Path) -> np.ndarray:
        if "baseline" in path.parts:
            return reference
        return candidate

    monkeypatch.setattr(benchmark_quality, "_read_video_rgb", fake_read_video_rgb)

    script = (
        "import sys; "
        "from pathlib import Path; "
        "out = Path(sys.argv[1]); "
        "out.mkdir(parents=True, exist_ok=True); "
        "(out / 'demo.mp4').write_bytes(b'candidate mp4')"
    )
    scenario = BenchmarkScenario(
        id="fake-runner",
        name="Fake runner",
        command=(sys.executable, "-c", script, "{output_dir}"),
        output_dir_arg=None,
    )

    manifest = run_benchmark_suite(
        [scenario],
        output_root=tmp_path / "run",
        repo_root=tmp_path,
        include_environment=False,
        quality_baseline=QualityBaselineConfig(
            baseline_dir=baseline_root,
            sample_count=2,
        ),
    )

    scenario_manifest = manifest["scenarios"][0]
    assert manifest["quality_baseline"]["sample_count"] == 2
    assert scenario_manifest["status"] == "pass"
    assert scenario_manifest["quality_results"][0]["status"] == "pass"
    assert scenario_manifest["metric_summary"]["quality_score"]["median"] <= 1.0
    assert (
        scenario_manifest["metric_summary"]["quality_visual_sanity_score"]["median"]
        <= 1.0
    )
    assert scenario_manifest["metric_summary"]["quality_rmse"]["median"] > 0.0
    assert (
        scenario_manifest["metric_summary"]["quality_similarity_score"]["median"] <= 1.0
    )
    assert scenario_manifest["metric_summary"]["quality_ssim_score"]["median"] <= 1.0
    assert "schema_version" not in scenario_manifest["metric_summary"]
    metrics_path = (
        tmp_path
        / "run"
        / "scenarios"
        / "fake-runner"
        / "quality"
        / "baseline-clip-compare"
        / "metrics.json"
    )
    assert metrics_path.is_file()
    quality_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "quality_max_frame_rmse" in quality_payload["diagnostics"]
    assert "quality_max_frame_rmse" not in quality_payload["metrics"]
    report_html = (tmp_path / "run" / "report.html").read_text(encoding="utf-8")
    detail_report = tmp_path / "run" / "reports" / "fake.html"
    assert detail_report.is_file()
    detail_html = detail_report.read_text(encoding="utf-8")
    assert "Quality Guide" in report_html
    assert "Quality Comparisons" not in report_html
    assert "Quality scores: median" in detail_html
    assert "Quality scores: median vs P90" not in detail_html
    assert "Quality Comparisons" in detail_html
    assert "Baseline" in detail_html
    assert "Candidate" in detail_html
    assert "../../baseline/scenarios/fake-runner/demo.mp4" in detail_html
    assert "quality_rmse" in detail_html
    assert "8-bit pixel RMSE against baseline" in detail_html
    assert "Quality score" in detail_html
    assert "Clip similarity score" in detail_html


def test_quality_baseline_uses_scenario_compare_region(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_root = tmp_path / "baseline"
    baseline_video = baseline_root / "scenarios" / "stacked-runner" / "demo.mp4"
    baseline_video.parent.mkdir(parents=True)
    baseline_video.write_bytes(b"baseline mp4")
    (baseline_root / "manifest.json").write_text(
        json.dumps(
            {
                "scenarios": [
                    {
                        "id": "stacked-runner",
                        "artifacts": {
                            "videos": ["scenarios/stacked-runner/demo.mp4"],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    reference = _quality_test_video(frames=4, height=12, width=16)
    candidate = reference.copy()
    candidate[:, :6, :, 0] = np.clip(
        candidate[:, :6, :, 0].astype(np.int16) + 80,
        0,
        255,
    )

    def fake_read_video_rgb(path: Path) -> np.ndarray:
        if "baseline" in path.parts:
            return reference
        return candidate

    monkeypatch.setattr(benchmark_quality, "_read_video_rgb", fake_read_video_rgb)

    script = (
        "import sys; "
        "from pathlib import Path; "
        "out = Path(sys.argv[1]); "
        "out.mkdir(parents=True, exist_ok=True); "
        "(out / 'demo.mp4').write_bytes(b'candidate mp4')"
    )
    scenario = BenchmarkScenario(
        id="stacked-runner",
        name="Stacked runner",
        command=(sys.executable, "-c", script, "{output_dir}"),
        output_dir_arg=None,
        quality_compare_region="bottom-half",
    )

    manifest = run_benchmark_suite(
        [scenario],
        output_root=tmp_path / "run",
        repo_root=tmp_path,
        include_environment=False,
        quality_baseline=QualityBaselineConfig(
            baseline_dir=baseline_root,
            sample_count=2,
        ),
    )

    scenario_manifest = manifest["scenarios"][0]
    assert manifest["quality_baseline"]["compare_region"] == "scenario-default"
    assert scenario_manifest["metric_summary"]["quality_rmse"][
        "median"
    ] == pytest.approx(0.0)
    quality_payload = json.loads(
        (
            tmp_path
            / "run"
            / "scenarios"
            / "stacked-runner"
            / "quality"
            / "baseline-clip-compare"
            / "metrics.json"
        ).read_text(encoding="utf-8")
    )
    assert quality_payload["compare_region"] == "bottom-half"


def test_quality_baseline_can_be_disabled_per_scenario(tmp_path: Path) -> None:
    baseline_root = tmp_path / "baseline"
    baseline_video = baseline_root / "scenarios" / "review-runner" / "review.mp4"
    baseline_video.parent.mkdir(parents=True)
    baseline_video.write_bytes(b"baseline review mp4")
    (baseline_root / "manifest.json").write_text(
        json.dumps(
            {
                "scenarios": [
                    {
                        "id": "review-runner",
                        "artifacts": {
                            "videos": ["scenarios/review-runner/review.mp4"],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    script = (
        "import sys; "
        "from pathlib import Path; "
        "out = Path(sys.argv[1]); "
        "out.mkdir(parents=True, exist_ok=True); "
        "(out / 'review.mp4').write_bytes(b'review mp4')"
    )
    scenario = BenchmarkScenario(
        id="review-runner",
        name="Review runner",
        command=(sys.executable, "-c", script, "{output_dir}"),
        output_dir_arg=None,
        quality_baseline_compare=False,
    )

    manifest = run_benchmark_suite(
        [scenario],
        output_root=tmp_path / "run",
        repo_root=tmp_path,
        include_environment=False,
        quality_baseline=QualityBaselineConfig(
            baseline_dir=baseline_root,
            sample_count=2,
        ),
    )

    scenario_manifest = manifest["scenarios"][0]
    assert len(scenario_manifest["quality_results"]) == 1
    review_result = scenario_manifest["quality_results"][0]
    assert review_result["scenario_id"] == "review-runner"
    assert review_result["id"] == "baseline-video-review"
    assert review_result["status"] == "available"
    assert review_result["reason"] == (
        "quality_baseline_compare is disabled for this scenario"
    )
    assert review_result["candidate_video"] == "scenarios/review-runner/review.mp4"
    assert review_result["baseline_video"] == str(baseline_video.resolve())
    assert "metrics_path" not in review_result
    assert "quality_score" not in scenario_manifest["metric_summary"]
    assert not (
        tmp_path
        / "run"
        / "scenarios"
        / "review-runner"
        / "quality"
        / "baseline-clip-compare"
    ).exists()
    assert scenario_manifest["scenario"]["quality_baseline_compare"] is False
    report_html = (tmp_path / "run" / "report.html").read_text(encoding="utf-8")
    detail_report = tmp_path / "run" / "reports" / "review.html"
    assert detail_report.is_file()
    detail_html = detail_report.read_text(encoding="utf-8")
    assert "Manual review" not in report_html
    assert "Manual review" in detail_html
    assert "Baseline scoring was not run for this scenario." in detail_html
    assert "../../baseline/scenarios/review-runner/review.mp4" in detail_html
    assert "../scenarios/review-runner/review.mp4" in detail_html


def test_cli_quality_profile_adds_pai_bench_long_command() -> None:
    scenario = BenchmarkScenario(
        id="review-runner",
        name="Review runner",
        tags=("one-minute",),
        command=("python", "-m", "demo"),
        output_dir_arg=None,
    )
    args = benchmark_cli._parse_args(
        [
            "--scenario",
            "review-runner",
            "--quality-profile",
            "pai-bench-long",
            "--pai-bench-dimension",
            "motion_smoothness",
            "--pai-bench-dimension",
            "overall_consistency",
        ]
    )

    updated = benchmark_cli._apply_quality_profiles([scenario], args=args)

    assert len(updated) == 1
    quality = updated[0].quality_commands[0]
    assert quality.id == "pai-bench-long"
    assert quality.metrics_path == "{quality_dir}/metrics.json"
    assert "tools.benchmarks.pai_bench_profile" in quality.command
    assert "--runner" in quality.command
    assert "local" in quality.command
    assert "--pai-bench-root" in quality.command
    assert "--pai-bench-repo" in quality.command
    assert "--python" in quality.command
    assert sys.executable in quality.command
    assert "paibench_metric.run_metric" not in quality.command
    assert "--dimensions" in quality.command
    assert (
        "{repo_root}/.cache/flashdreams/evaluators/physical-ai-bench" in quality.command
    )
    assert "--keep-staged-videos" not in quality.command
    assert "motion_smoothness" in quality.command
    assert "overall_consistency" in quality.command


def test_cli_pai_bench_profile_skips_non_pai_scenarios() -> None:
    quality_smoke = BenchmarkScenario(
        id="quality-smoke",
        name="Quality smoke",
        tags=("quality-30s",),
        command=("python", "-m", "demo"),
        output_dir_arg=None,
    )
    one_minute = BenchmarkScenario(
        id="one-minute-review",
        name="One-minute review",
        tags=("one-minute",),
        command=("python", "-m", "demo"),
        output_dir_arg=None,
    )
    args = benchmark_cli._parse_args(
        [
            "--scenario",
            "quality-smoke",
            "--scenario",
            "one-minute-review",
            "--quality-profile",
            "pai-bench-long",
        ]
    )

    updated = benchmark_cli._apply_quality_profiles(
        [quality_smoke, one_minute],
        args=args,
    )

    assert not updated[0].quality_commands
    assert updated[1].quality_commands[0].id == "pai-bench-long"


def test_cli_quality_profile_can_keep_pai_bench_staged_videos() -> None:
    scenario = BenchmarkScenario(
        id="review-runner",
        name="Review runner",
        tags=("one-minute",),
        command=("python", "-m", "demo"),
        output_dir_arg=None,
    )
    args = benchmark_cli._parse_args(
        [
            "--scenario",
            "review-runner",
            "--quality-profile",
            "pai-bench-long",
            "--pai-bench-keep-staged-videos",
        ]
    )

    updated = benchmark_cli._apply_quality_profiles([scenario], args=args)

    assert "--keep-staged-videos" in updated[0].quality_commands[0].command


def test_cli_quality_profile_can_use_upstream_pai_bench_runner() -> None:
    scenario = BenchmarkScenario(
        id="review-runner",
        name="Review runner",
        tags=("one-minute",),
        command=("python", "-m", "demo"),
        output_dir_arg=None,
    )
    args = benchmark_cli._parse_args(
        [
            "--scenario",
            "review-runner",
            "--quality-profile",
            "pai-bench-g",
            "--pai-bench-runner",
            "upstream",
        ]
    )

    updated = benchmark_cli._apply_quality_profiles([scenario], args=args)
    command = updated[0].quality_commands[0].command

    assert "--runner" in command
    assert command[command.index("--runner") + 1] == "upstream"
    assert command[command.index("--python") + 1] == "python"


def test_pai_bench_upstream_runner_warns_on_aarch64(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = pai_bench_profile._parse_args(
        [
            "--profile",
            "pai-bench-g",
            "--runner",
            "upstream",
            "--video",
            "candidate.mp4",
            "--scenario-id",
            "demo",
            "--quality-dir",
            "quality",
            "--output",
            "quality/metrics.json",
            "--pai-bench-root",
            "physical-ai-bench",
            "--dimensions",
            "motion_smoothness",
        ]
    )
    monkeypatch.setattr(pai_bench_profile.platform, "machine", lambda: "aarch64")

    warnings = pai_bench_profile._runner_setup_warnings(args)

    assert warnings
    assert "decord" in warnings[0]


def test_pai_bench_preflight_reports_missing_import(tmp_path: Path) -> None:
    video = tmp_path / "candidate.mp4"
    video.write_bytes(b"fake mp4")
    pai_root = tmp_path / "physical-ai-bench"
    generation_dir = pai_root / "generation"
    pbench_dir = generation_dir / "pbench"
    pbench_dir.mkdir(parents=True)
    (generation_dir / "evaluate.py").write_text(
        "raise AssertionError('evaluator should not launch after preflight failure')\n",
        encoding="utf-8",
    )
    (pbench_dir / "__init__.py").write_text("", encoding="utf-8")
    (pbench_dir / "aesthetic_quality.py").write_text(
        "import missing_clip_for_test\n",
        encoding="utf-8",
    )

    result = pai_bench_profile.main(
        [
            "--profile",
            "pai-bench-g",
            "--video",
            str(video),
            "--scenario-id",
            "demo-scenario",
            "--quality-dir",
            str(tmp_path / "quality"),
            "--output",
            str(tmp_path / "quality" / "metrics.json"),
            "--pai-bench-root",
            str(pai_root),
            "--no-fetch",
            "--dimensions",
            "aesthetic_quality",
        ]
    )

    assert result == 1
    payload = json.loads((tmp_path / "quality" / "metrics.json").read_text())
    assert payload["status"] == "fail"
    assert payload["metrics"] == {}
    assert payload["metadata"]["preflight"]["returncode"] == 1
    assert (
        payload["metadata"]["preflight"]["payload"]["failures"][0]["missing_import"]
        == "missing_clip_for_test"
    )
    assert "missing Python import 'missing_clip_for_test'" in payload["warnings"][0]
    assert (tmp_path / "quality" / "pai_bench_preflight.log").is_file()
    assert not (tmp_path / "quality" / "pai_bench_command.log").exists()
    assert not (tmp_path / "quality" / "staged" / "videos").exists()


def test_pai_bench_profile_normalizes_public_evaluator_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "candidate.mp4"
    video.write_bytes(b"fake mp4")
    pai_root = tmp_path / "physical-ai-bench"
    generation_dir = pai_root / "generation"
    generation_dir.mkdir(parents=True)
    pbench_dir = generation_dir / "pbench"
    pbench_dir.mkdir()
    (pbench_dir / "__init__.py").write_text("", encoding="utf-8")
    (pbench_dir / "motion_smoothness.py").write_text("", encoding="utf-8")
    (pbench_dir / "imaging_quality.py").write_text("", encoding="utf-8")
    fake_runner = generation_dir / "evaluate.py"
    fake_runner.write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        "out = Path(sys.argv[sys.argv.index('--output_path') + 1])\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
        "payload = {\n"
        "  'motion_smoothness': [0.8, [{'video_path': 'demo_000.mp4', 'video_results': 0.8}]],\n"
        "  'imaging_quality': [0.82, [{'video_path': 'demo_000.mp4', 'video_results': 82.0}]],\n"
        "}\n"
        "(out / 'fake_eval_results.json').write_text(json.dumps(payload))\n",
        encoding="utf-8",
    )
    fake_python = tmp_path / "fake_python"
    fake_python.write_text(
        "#!/usr/bin/env python3\n"
        "import os, runpy, sys\n"
        "if len(sys.argv) > 2 and sys.argv[1] == '-c':\n"
        "    code = sys.argv[2]\n"
        "    sys.argv = [sys.argv[0], *sys.argv[3:]]\n"
        "    exec(code)\n"
        "    raise SystemExit(0)\n"
        "assert 'FLASHDREAMS_PAI_BENCH_LOCAL_RUNNER' in os.environ\n"
        "import decord\n"
        "assert decord.VideoReader is not None\n"
        "script_index = next(\n"
        "    index for index, arg in enumerate(sys.argv)\n"
        "    if arg.endswith('evaluate.py')\n"
        ")\n"
        "sys.argv = [sys.argv[script_index], *sys.argv[script_index + 1:]]\n"
        "runpy.run_path(sys.argv[0], run_name='__main__')\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    result = pai_bench_profile.main(
        [
            "--profile",
            "pai-bench-g",
            "--video",
            str(video),
            "--scenario-id",
            "demo-scenario",
            "--prompt",
            "Demo scenario",
            "--quality-dir",
            str(tmp_path / "quality"),
            "--output",
            str(tmp_path / "quality" / "metrics.json"),
            "--pai-bench-root",
            str(pai_root),
            "--python",
            str(fake_python),
            "--no-fetch",
            "--dimensions",
            "motion_smoothness",
            "imaging_quality",
        ]
    )

    assert result == 0
    payload = json.loads((tmp_path / "quality" / "metrics.json").read_text())
    metrics = payload["metrics"]
    assert payload["status"] == "pass"
    assert payload["metadata"]["runner"] == "local"
    assert (
        "FLASHDREAMS_PAI_BENCH_LOCAL_RUNNER"
        in payload["metadata"]["command_manifest"]["env"]
    )
    assert metrics["pai_bench_g_motion_smoothness_score"] == pytest.approx(80.0)
    assert metrics["pai_bench_g_imaging_quality_score"] == pytest.approx(82.0)
    assert metrics["pai_bench_g_score"] == pytest.approx(81.0)
    assert metrics["pai_bench_g_videos_evaluated"] == 1
    prompt_payload = json.loads(
        (tmp_path / "quality" / "staged" / "prompt_file.json").read_text()
    )
    assert prompt_payload == [
        {"video_id": "demo-scenario", "prompt_en": "Demo scenario"}
    ]
    assert not (tmp_path / "quality" / "staged" / "videos").exists()


def test_pai_bench_decord_compat_reads_video_batches(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    torch = pytest.importorskip("torch")
    from tools.benchmarks._pai_bench_compat import decord

    video_path = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        4.0,
        (6, 4),
    )
    assert writer.isOpened()
    for value in (20, 80, 140):
        frame = np.full((4, 6, 3), value, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    decord.bridge.set_bridge("native")
    reader = decord.VideoReader(video_path)
    native_batch = reader.get_batch([0, 2])
    assert len(reader) == 3
    assert reader.get_avg_fps() == pytest.approx(4.0)
    assert native_batch.asnumpy().shape == (2, 4, 6, 3)

    decord.bridge.set_bridge("torch")
    torch_batch = reader.get_batch([1])
    assert isinstance(torch_batch, torch.Tensor)
    assert tuple(torch_batch.shape) == (1, 4, 6, 3)


def test_strict_run_sets_deterministic_env_and_forwards_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.benchmarks import strict_run

    deterministic_calls: list[tuple[bool, bool]] = []
    forwarded_argv: list[str] = []

    def fake_use_deterministic_algorithms(enabled: bool, *, warn_only: bool) -> None:
        deterministic_calls.append((enabled, warn_only))

    def fake_entrypoint() -> None:
        forwarded_argv.extend(sys.argv)

    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(
            use_deterministic_algorithms=fake_use_deterministic_algorithms
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "flashdreams.scripts.cli",
        types.SimpleNamespace(entrypoint=fake_entrypoint),
    )

    original_argv = list(sys.argv)

    assert strict_run.main(["--", "demo-runner", "--total-blocks", "4"]) == 0

    assert deterministic_calls == [(True, True)]
    assert forwarded_argv == ["flashdreams-run", "demo-runner", "--total-blocks", "4"]
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert sys.argv == original_argv


def _command_value(command: tuple[str, ...], option: str) -> str:
    return command[command.index(option) + 1]


def test_baseline_clip_compare_identical_clips_have_perfect_similarity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_video = tmp_path / "baseline" / "demo.mp4"
    candidate_video = tmp_path / "candidate" / "demo.mp4"
    baseline_video.parent.mkdir()
    candidate_video.parent.mkdir()
    baseline_video.write_bytes(b"baseline")
    candidate_video.write_bytes(b"candidate")
    video = _quality_test_video(frames=5, height=10, width=14)

    monkeypatch.setattr(benchmark_quality, "_read_video_rgb", lambda _path: video)

    result = benchmark_quality.run_baseline_clip_compare(
        scenario_id="demo",
        candidate_video=candidate_video,
        quality_dir=tmp_path / "quality",
        config=QualityBaselineConfig(baseline_dir=baseline_video.parent),
    )

    assert result["status"] == "pass"
    payload = json.loads((tmp_path / "quality" / "metrics.json").read_text())
    metrics = payload["metrics"]
    assert metrics["quality_similarity_score"] == pytest.approx(1.0)
    assert metrics["quality_ssim_score"] == pytest.approx(1.0)
    assert metrics["quality_rmse"] == pytest.approx(0.0)
    assert metrics["quality_psnr_db"] == pytest.approx(100.0)


def test_baseline_clip_compare_missing_flip_is_nonfatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flashdreams.quality.clip_compare import compare_video_arrays

    baseline_video = tmp_path / "baseline" / "demo.mp4"
    candidate_video = tmp_path / "candidate" / "demo.mp4"
    baseline_video.parent.mkdir()
    candidate_video.parent.mkdir()
    baseline_video.write_bytes(b"baseline")
    candidate_video.write_bytes(b"candidate")
    video = _quality_test_video(frames=4, height=10, width=14)
    calls: list[bool] = []

    def fake_compare_video_arrays(
        *_args: object,
        compute_flip: bool,
        **_kwargs: object,
    ) -> object:
        calls.append(compute_flip)
        if compute_flip:
            raise ImportError("no flip")
        return compare_video_arrays(
            video,
            video,
            frame_indices=None,
            sample_count=2,
            compute_flip=False,
        )

    monkeypatch.setattr(benchmark_quality, "_read_video_rgb", lambda _path: video)
    monkeypatch.setattr(
        benchmark_quality,
        "_compare_video_arrays",
        fake_compare_video_arrays,
    )

    result = benchmark_quality.run_baseline_clip_compare(
        scenario_id="demo",
        candidate_video=candidate_video,
        quality_dir=tmp_path / "quality",
        config=QualityBaselineConfig(
            baseline_dir=baseline_video.parent,
            sample_count=2,
            compute_flip=True,
        ),
    )

    assert result["status"] == "pass"
    assert calls == [True, False]
    payload = json.loads((tmp_path / "quality" / "metrics.json").read_text())
    assert payload["flip_requested"] is True
    assert payload["flip_computed"] is False
    assert "FLIP unavailable" in payload["warnings"][0]


def _quality_test_video(*, frames: int, height: int, width: int) -> np.ndarray:
    yy, xx = np.mgrid[:height, :width]
    video = np.empty((frames, height, width, 3), dtype=np.uint8)
    for frame_idx in range(frames):
        video[frame_idx, ..., 0] = (xx * 9 + frame_idx * 13) % 256
        video[frame_idx, ..., 1] = (yy * 11 + frame_idx * 7) % 256
        video[frame_idx, ..., 2] = ((xx + yy) * 5 + frame_idx * 17) % 256
    return video
