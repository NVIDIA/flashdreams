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

"""CLI for local FlashDreams runner/demo benchmarks."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from tools.benchmarks.harness import run_benchmark_suite
from tools.benchmarks.pai_bench_profile import (
    DEFAULT_PAI_BENCH_REPO,
    DEFAULT_PAI_BENCH_REVISION,
)
from tools.benchmarks.quality import (
    QualityBaselineConfig,
    parse_quality_frame_indices,
)
from tools.benchmarks.scenarios import (
    BenchmarkScenario,
    QualityCommandConfig,
    built_in_scenarios,
    load_scenario_file,
)

_DEFAULT_PAI_BENCH_DIMENSIONS = (
    "aesthetic_quality",
    "background_consistency",
    "imaging_quality",
    "motion_smoothness",
    "overall_consistency",
    "subject_consistency",
)
_PAI_BENCH_TAGS = frozenset({"one-minute", "pai-bench"})


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repo_root = args.repo_root.resolve()
    scenarios = _load_scenarios(args.scenario_file)

    if args.list_scenarios:
        _print_scenarios(scenarios)
        return 0

    selected = _selected_scenarios(
        scenarios, scenario_ids=args.scenario, all_scenarios=args.all
    )
    selected = _apply_quality_profiles(selected, args=args)
    output_root = args.output_dir or Path("artifacts/benchmarks") / _timestamp_slug()
    manifest = run_benchmark_suite(
        selected,
        output_root=output_root,
        repo_root=repo_root,
        keep_going=args.keep_going,
        dry_run=args.dry_run,
        quality_baseline=_quality_baseline_config(args),
        progress=None if args.quiet else _print_progress,
        progress_interval_s=args.progress_interval_s,
    )
    print(
        f"benchmark report: {Path(manifest['output_root']) / manifest['report_path']}"
    )
    print(f"benchmark manifest: {Path(manifest['output_root']) / 'manifest.json'}")
    failed = int(manifest.get("failed_scenario_count", 0))
    return 1 if failed else 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="Scenario id to run. May be passed multiple times.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run every known scenario, including custom scenarios.",
    )
    parser.add_argument(
        "--scenario-file",
        type=Path,
        action="append",
        default=[],
        help="JSON file containing additional command-backed scenarios.",
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="List built-in and loaded custom scenarios without running them.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Run output directory. Defaults to artifacts/benchmarks/<timestamp>.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root used as the default scenario working directory.",
    )
    parser.add_argument(
        "--keep-going",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue to later scenarios after a scenario command fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render commands, manifests, and report without launching scenario commands.",
    )
    parser.add_argument(
        "--quality-baseline-dir",
        type=Path,
        help=(
            "Previous benchmark run root, or flat directory of <scenario-id>.mp4 "
            "files, used for non-gating MP4 quality comparisons."
        ),
    )
    parser.add_argument(
        "--quality-sample-count",
        type=int,
        default=8,
        help="Number of evenly spaced frames to sample for baseline MP4 comparison.",
    )
    parser.add_argument(
        "--quality-frame-indices",
        help="Comma-separated frame indices to compare instead of even sampling.",
    )
    parser.add_argument(
        "--quality-compare-region",
        choices=("scenario-default", "full", "bottom-half"),
        default="scenario-default",
        help=(
            "Video region to compare. The default uses a scenario's "
            "quality_compare_region when configured, otherwise full. Use "
            "bottom-half for stacked runner MP4s when only the generated "
            "region should be scored."
        ),
    )
    parser.add_argument(
        "--quality-compute-flip",
        action="store_true",
        help="Also compute FLIP scores when flip-evaluator is installed.",
    )
    parser.add_argument(
        "--quality-profile",
        action="append",
        choices=("pai-bench-g", "pai-bench-long"),
        default=[],
        help=(
            "Enable a named optional quality profile. "
            "'pai-bench-g' runs public PAI-Bench-G on the full MP4. "
            "'pai-bench-long' splits the MP4 into local segments and scores "
            "them with public PAI-Bench-G, then reports a long-video average. "
            "PAI-Bench profiles only apply to selected scenarios tagged "
            "one-minute or pai-bench. "
            "Run with `uv run --group pai-bench ...` or pass a prepared "
            "--pai-bench-python for the PAI-Bench evaluator dependencies."
        ),
    )
    parser.add_argument(
        "--pai-bench-root",
        type=Path,
        help=(
            "Path to a public physical-ai-bench checkout. Defaults to "
            "<repo-root>/.cache/flashdreams/evaluators/physical-ai-bench. "
            "The adapter clones the pinned repo there if the path does not exist."
        ),
    )
    parser.add_argument(
        "--pai-bench-repo",
        default=DEFAULT_PAI_BENCH_REPO,
        help="Public PAI-Bench Git repository URL used when cloning.",
    )
    parser.add_argument(
        "--pai-bench-revision",
        default=DEFAULT_PAI_BENCH_REVISION,
        help="Public PAI-Bench revision checked out before evaluation.",
    )
    parser.add_argument(
        "--pai-bench-python",
        default=None,
        help=(
            "Python command used for PAI-Bench. Defaults to this benchmark "
            "process's Python in local runner mode, or 'python' in upstream "
            "runner mode."
        ),
    )
    parser.add_argument(
        "--pai-bench-runner",
        choices=("local", "upstream"),
        default="local",
        help=(
            "PAI-Bench execution mode. local uses FlashDreams compatibility "
            "shims for aarch64-friendly video decoding; upstream runs the "
            "public checkout's normal entrypoint."
        ),
    )
    parser.add_argument(
        "--pai-bench-nproc-per-node",
        type=int,
        default=1,
        help="Value passed to torch.distributed.run --nproc_per_node.",
    )
    parser.add_argument(
        "--pai-bench-dimension",
        action="append",
        default=[],
        help=(
            "PAI-Bench quality dimension to evaluate. May be passed multiple "
            "times. Defaults to the non-I2V quality dimensions."
        ),
    )
    parser.add_argument(
        "--pai-bench-segment-duration-s",
        type=int,
        default=10,
        help="Segment length in seconds for the local PAI-Bench-Long profile.",
    )
    parser.add_argument(
        "--pai-bench-timeout-s",
        type=float,
        default=21600.0,
        help="Timeout for each PAI-Bench quality hook.",
    )
    parser.add_argument(
        "--pai-bench-custom-image-folder",
        type=Path,
        help=(
            "Optional condition-image folder for PAI-Bench I2V dimensions such "
            "as i2v_background and i2v_subject."
        ),
    )
    parser.add_argument(
        "--pai-bench-fetch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fetch the public PAI-Bench repository before checkout when it is a git repo.",
    )
    parser.add_argument(
        "--pai-bench-keep-staged-videos",
        action="store_true",
        help=(
            "Keep copied or segmented PAI-Bench MP4 inputs under "
            "quality/.../staged/videos for evaluator debugging. By default "
            "they are removed after the profile runs."
        ),
    )
    parser.add_argument(
        "--progress-interval-s",
        type=float,
        default=30.0,
        help="Seconds between heartbeat messages while a command is still running.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages; final report paths are still printed.",
    )
    args = parser.parse_args(argv)
    if not args.list_scenarios and not args.all and not args.scenario:
        parser.error("pass --scenario <id>, --all, or --list-scenarios")
    if args.all and args.scenario:
        parser.error("--all cannot be combined with --scenario")
    if args.progress_interval_s <= 0:
        parser.error("--progress-interval-s must be > 0")
    if args.quality_sample_count <= 0:
        parser.error("--quality-sample-count must be > 0")
    if args.quality_frame_indices:
        try:
            parse_quality_frame_indices(args.quality_frame_indices)
        except ValueError as exc:
            parser.error(str(exc))
    if args.pai_bench_segment_duration_s <= 0:
        parser.error("--pai-bench-segment-duration-s must be > 0")
    if args.pai_bench_timeout_s <= 0:
        parser.error("--pai-bench-timeout-s must be > 0")
    if args.pai_bench_nproc_per_node <= 0:
        parser.error("--pai-bench-nproc-per-node must be > 0")
    if args.pai_bench_dimension and not any(
        profile.startswith("pai-bench") for profile in args.quality_profile
    ):
        parser.error("--pai-bench-dimension requires a PAI-Bench quality profile")
    return args


def _quality_baseline_config(args: argparse.Namespace) -> QualityBaselineConfig | None:
    if args.quality_baseline_dir is None:
        return None
    frame_indices = parse_quality_frame_indices(args.quality_frame_indices)
    compare_region = (
        None
        if args.quality_compare_region == "scenario-default"
        else args.quality_compare_region
    )
    return QualityBaselineConfig(
        baseline_dir=args.quality_baseline_dir,
        sample_count=args.quality_sample_count,
        frame_indices=frame_indices,
        compare_region=compare_region,
        compute_flip=args.quality_compute_flip,
    )


def _load_scenarios(paths: list[Path]) -> dict[str, BenchmarkScenario]:
    scenarios = built_in_scenarios()
    for path in paths:
        loaded = load_scenario_file(path)
        duplicates = set(scenarios) & set(loaded)
        if duplicates:
            raise SystemExit(
                f"{path} redefines existing scenario ids: {sorted(duplicates)}"
            )
        scenarios.update(loaded)
    return scenarios


def _selected_scenarios(
    scenarios: dict[str, BenchmarkScenario],
    *,
    scenario_ids: list[str],
    all_scenarios: bool,
) -> list[BenchmarkScenario]:
    if all_scenarios:
        return list(scenarios.values())
    missing = [
        scenario_id for scenario_id in scenario_ids if scenario_id not in scenarios
    ]
    if missing:
        available = ", ".join(sorted(scenarios))
        raise SystemExit(f"unknown scenario id(s): {missing}; available: {available}")
    return [scenarios[scenario_id] for scenario_id in scenario_ids]


def _apply_quality_profiles(
    scenarios: Sequence[BenchmarkScenario],
    *,
    args: argparse.Namespace,
) -> list[BenchmarkScenario]:
    profiles = tuple(dict.fromkeys(args.quality_profile))
    if not profiles:
        return list(scenarios)
    extra_commands: list[QualityCommandConfig] = []
    if "pai-bench-g" in profiles:
        extra_commands.append(_pai_bench_quality_command(args, profile="pai-bench-g"))
    if "pai-bench-long" in profiles:
        extra_commands.append(
            _pai_bench_quality_command(args, profile="pai-bench-long")
        )
    updated: list[BenchmarkScenario] = []
    for scenario in scenarios:
        existing_ids = {command.id for command in scenario.quality_commands}
        additions = [
            command
            for command in extra_commands
            if command.id not in existing_ids
            and _scenario_accepts_quality_profile(scenario, command.id)
        ]
        if additions:
            scenario = replace(
                scenario,
                quality_commands=(*scenario.quality_commands, *additions),
            )
        updated.append(scenario)
    return updated


def _scenario_accepts_quality_profile(
    scenario: BenchmarkScenario,
    profile: str,
) -> bool:
    if profile.startswith("pai-bench"):
        return bool(_PAI_BENCH_TAGS.intersection(scenario.tags))
    return True


def _pai_bench_quality_command(
    args: argparse.Namespace,
    *,
    profile: str,
) -> QualityCommandConfig:
    dimensions = tuple(args.pai_bench_dimension or _DEFAULT_PAI_BENCH_DIMENSIONS)
    pai_bench_root = (
        str(args.pai_bench_root)
        if args.pai_bench_root is not None
        else "{repo_root}/.cache/flashdreams/evaluators/physical-ai-bench"
    )
    command = [
        sys.executable,
        "-m",
        "tools.benchmarks.pai_bench_profile",
        "--profile",
        profile,
        "--runner",
        str(args.pai_bench_runner),
        "--video",
        "{first_video}",
        "--scenario-id",
        "{scenario_id}",
        "--prompt",
        "{scenario_name}",
        "--quality-dir",
        "{quality_dir}",
        "--output",
        "{quality_dir}/metrics.json",
        "--pai-bench-root",
        pai_bench_root,
        "--pai-bench-repo",
        str(args.pai_bench_repo),
        "--pai-bench-revision",
        str(args.pai_bench_revision),
        "--python",
        _pai_bench_python_command(args),
        "--nproc-per-node",
        str(args.pai_bench_nproc_per_node),
        "--segment-duration",
        str(args.pai_bench_segment_duration_s),
        "--dimensions",
        *dimensions,
    ]
    command.append("--fetch" if args.pai_bench_fetch else "--no-fetch")
    if args.pai_bench_keep_staged_videos:
        command.append("--keep-staged-videos")
    if args.pai_bench_custom_image_folder is not None:
        command.extend(
            (
                "--custom-image-folder",
                str(args.pai_bench_custom_image_folder),
            )
        )
    return QualityCommandConfig(
        id=profile,
        command=tuple(command),
        metrics_path="{quality_dir}/metrics.json",
        timeout_s=float(args.pai_bench_timeout_s),
    )


def _pai_bench_python_command(args: argparse.Namespace) -> str:
    if args.pai_bench_python:
        return str(args.pai_bench_python)
    if args.pai_bench_runner == "local":
        return sys.executable
    return "python"


def _print_scenarios(scenarios: dict[str, BenchmarkScenario]) -> None:
    for scenario in scenarios.values():
        tags = ", ".join(scenario.tags)
        suffix = f" [{tags}]" if tags else ""
        print(f"{scenario.id}: {scenario.name}{suffix}")
        if scenario.description:
            print(f"  {scenario.description}")


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")


def _print_progress(message: str) -> None:
    print(message, flush=True)


if __name__ == "__main__":
    sys.exit(main())
