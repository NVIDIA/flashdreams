# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-line entry point for OmniDreams evaluation automation."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from omnidreams.eval.baseline import (
    check_summary_against_baseline,
    format_baseline_check_report,
    load_json,
    write_baseline_check_json,
)
from omnidreams.eval.batches import (
    cases_for_batch,
    parse_byte_size,
    plan_batches,
    read_batch_plan,
    write_batch_plan,
)
from omnidreams.eval.drivinggen import (
    DEFAULT_DRIVINGGEN_REPO,
    DEFAULT_DRIVINGGEN_REVISION,
    ensure_drivinggen_checkout,
    run_fvd_lite,
    run_fvd_reference_lite,
    run_video_metrics,
    stage_drivinggen_video_inputs,
    video_metrics_command,
)
from omnidreams.eval.generation import generate_cases
from omnidreams.eval.hf_assets import list_hf_dataset_files
from omnidreams.eval.manifest import (
    DEFAULT_CAMERA,
    DEFAULT_DATASET_REPO,
    DEFAULT_DATASET_REVISION,
    DEFAULT_DATASET_SUBPATH,
    build_cases_from_repo_files,
    read_cases_jsonl,
    read_staged_cases_jsonl,
    write_cases_jsonl,
)
from omnidreams.eval.report import (
    build_run_summary,
    render_run_summary_markdown,
    write_run_summary_json,
    write_run_summary_markdown,
)
from omnidreams.eval.staging import stage_cases
from omnidreams.eval.validation import validate_generated_run, write_validation_json
from omnidreams.eval.worldlens import (
    DEFAULT_WORLDLENS_CONFIG_NAME,
    DEFAULT_WORLDLENS_METHOD,
    DEFAULT_WORLDLENS_REPO,
    DEFAULT_WORLDLENS_REVISION,
    ensure_worldlens_checkout,
    run_worldlens_evaluation,
    stage_worldlens_video_inputs,
    write_worldlens_consistency_config,
)

DEFAULT_GENERATION_RECIPE = "omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae"


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


def _cmd_discover(args: argparse.Namespace) -> int:
    files = list_hf_dataset_files(
        repo_id=args.dataset_repo,
        revision=args.dataset_revision,
        subpath=args.dataset_subpath,
        token=_token_from_env(args.token_env),
    )
    cases = build_cases_from_repo_files(
        files,
        dataset_repo=args.dataset_repo,
        dataset_revision=args.dataset_revision,
        dataset_subpath=args.dataset_subpath,
        camera=args.camera,
    )
    write_cases_jsonl(cases, args.output)
    print(f"wrote {len(cases)} cases -> {args.output}")
    return 0


def _cmd_plan_batches(args: argparse.Namespace) -> int:
    cases = read_cases_jsonl(args.manifest)
    max_batch_bytes = (
        parse_byte_size(args.max_batch_bytes) if args.max_batch_bytes else None
    )
    batches = plan_batches(
        cases,
        batch_size=args.batch_size,
        max_batch_bytes=max_batch_bytes,
    )
    write_batch_plan(args.output, batches)
    print(f"wrote {len(batches)} batches -> {args.output}")
    return 0


def _cmd_stage_batch(args: argparse.Namespace) -> int:
    cases = read_cases_jsonl(args.manifest)
    batches = read_batch_plan(args.batch_plan)
    batch = _select_batch(batches, args.batch_id)
    staged = stage_cases(
        cases_for_batch(cases, batch),
        batch_id=batch.batch_id,
        scratch_root=args.scratch_root,
        output_manifest=args.output,
        token=_token_from_env(args.token_env),
        force=args.force,
    )
    print(f"staged {len(staged)} cases for {batch.batch_id} -> {args.output}")
    return 0


def _cmd_setup_evaluator(args: argparse.Namespace) -> int:
    checkout = ensure_drivinggen_checkout(
        cache_dir=args.cache_dir,
        repo_url=args.evaluator_repo,
        revision=args.evaluator_revision,
        fetch=not args.no_fetch,
        patch_checkout=not args.no_patch,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(checkout.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"prepared DrivingGen {checkout.resolved_commit} -> {checkout.path}")
    return 0


def _cmd_setup_worldlens(args: argparse.Namespace) -> int:
    checkout = ensure_worldlens_checkout(
        cache_dir=args.cache_dir,
        repo_url=args.evaluator_repo,
        revision=args.evaluator_revision,
        fetch=not args.no_fetch,
        install_config=not args.no_config,
        config_name=args.config_name,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(checkout.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"prepared WorldLens {checkout.resolved_commit} -> {checkout.path}")
    if checkout.config_name is not None:
        print(f"installed WorldLens config: {checkout.config_name}")
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    staged_cases = read_staged_cases_jsonl(args.staged_manifest)
    try:
        results = generate_cases(
            staged_cases,
            run_root=args.run_root,
            recipe=args.recipe,
            total_blocks=args.total_blocks,
            flashdreams_run=args.flashdreams_run,
            force=args.force,
            dry_run=args.dry_run,
            stream_logs=args.stream_logs,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    for result in results:
        if args.dry_run:
            print(shlex.join(result.command))
        else:
            print(
                f"{result.uuid}: generated={result.generated_video_path} "
                f"log={result.log_path}"
            )
    print(
        f"{'planned' if args.dry_run else 'processed'} {len(results)} generation jobs"
    )
    return 0


def _cmd_prepare_drivinggen(args: argparse.Namespace) -> int:
    staged_cases = read_staged_cases_jsonl(args.staged_manifest)
    stage_drivinggen_video_inputs(
        staged_cases,
        generated_root=args.run_root / "generated",
        drivinggen_root=args.drivinggen_root,
        split=args.split,
        model_name=args.model_name,
        exp_id=args.exp_id,
        force=args.force,
    )
    print(f"prepared DrivingGen layout for {len(staged_cases)} cases")
    return 0


def _cmd_prepare_worldlens(args: argparse.Namespace) -> int:
    staged_cases = read_staged_cases_jsonl(args.staged_manifest)
    if args.write_config:
        config_path = write_worldlens_consistency_config(
            args.worldlens_root,
            config_name=args.config_name,
        )
        print(f"wrote WorldLens config -> {config_path}")
    manifest_path = stage_worldlens_video_inputs(
        staged_cases,
        generated_root=args.run_root / "generated",
        worldlens_root=args.worldlens_root,
        method_name=args.method_name,
        generation_index=args.generation_index,
        camera_name=args.camera_name,
        force=args.force,
    )
    print(f"prepared WorldLens layout for {len(staged_cases)} cases -> {manifest_path}")
    return 0


def _cmd_validate_generation(args: argparse.Namespace) -> int:
    results = validate_generated_run(args.run_root, uuid=args.uuid)
    if args.output is not None:
        write_validation_json(results, args.output)
    failures = 0
    for result in results:
        failures += 0 if result.ok else 1
        frames = result.runner_written_frames or result.expected_frames_from_steps
        duration = (frames / args.fps) if frames is not None and args.fps > 0 else None
        duration_text = f"{duration:.3f}s" if duration is not None else "unknown"
        status = "ok" if result.ok else "failed"
        print(
            f"{result.uuid}: {status} frames={frames or 'unknown'} "
            f"duration={duration_text} total_blocks={result.total_blocks or 'unknown'} "
            f"hdmap_frames={result.hdmap_frames or 'unknown'}"
        )
        for issue in result.issues:
            print(f"  - {issue}")
    print(f"validated {len(results)} generation job(s), failures={failures}")
    return 1 if failures else 0


def _cmd_drivinggen_video_metrics(args: argparse.Namespace) -> int:
    command = video_metrics_command(
        drivinggen_root=args.drivinggen_root,
        split=args.split,
        model_name=args.model_name,
        exp_id=args.exp_id,
        metric=args.metric,
        python=args.python,
    )
    extra_env = {}
    if args.i3d_checkpoint is not None:
        extra_env["DRIVINGGEN_I3D_CKPT"] = str(args.i3d_checkpoint)
    if args.inception_checkpoint is not None:
        extra_env["DRIVINGGEN_INCEPTION_CKPT"] = str(args.inception_checkpoint)
    log_path = None
    if not args.stream_logs:
        log_path = args.log_file or (
            args.drivinggen_root
            / "cache"
            / "eval_logs"
            / args.split
            / f"{args.model_name}-{args.exp_id}-{args.metric}.log"
        )
    if args.dry_run:
        prefix = ""
        if extra_env:
            prefix = " ".join(
                f"{name}={shlex.quote(value)}"
                for name, value in sorted(extra_env.items())
            )
            prefix += " "
        suffix = f" > {shlex.quote(str(log_path))} 2>&1" if log_path else ""
        print(prefix + shlex.join(command) + suffix)
        return 0
    try:
        run_video_metrics(
            drivinggen_root=args.drivinggen_root,
            split=args.split,
            model_name=args.model_name,
            exp_id=args.exp_id,
            metric=args.metric,
            python=args.python,
            log_path=log_path,
            extra_env=extra_env,
        )
    except subprocess.CalledProcessError as exc:
        if log_path is not None:
            print(
                f"DrivingGen metric command failed; see log: {log_path}",
                file=sys.stderr,
            )
        return int(exc.returncode or 1)
    if log_path is not None:
        print(f"wrote DrivingGen metric log -> {log_path}")
    return 0


def _cmd_drivinggen_fvd_lite(args: argparse.Namespace) -> int:
    log_path = None
    if not args.stream_logs:
        log_path = args.log_file or (
            args.drivinggen_root
            / "cache"
            / "eval_logs"
            / args.split
            / f"{args.model_name}-{args.exp_id}-fvd-lite.log"
        )
    output_json = args.output_json or (
        args.drivinggen_root
        / "cache"
        / "eval_logs"
        / args.split
        / f"{args.model_name}-{args.exp_id}-fvd-lite.json"
    )
    if args.dry_run:
        print(
            "drivinggen-fvd-lite "
            f"--drivinggen-root {shlex.quote(str(args.drivinggen_root))} "
            f"--split {shlex.quote(args.split)} "
            f"--model-name {shlex.quote(args.model_name)} "
            f"--exp-id {shlex.quote(args.exp_id)}"
        )
        if log_path is not None:
            print(f"log: {log_path}")
        print(f"json: {output_json}")
        return 0
    try:
        payload = run_fvd_lite(
            drivinggen_root=args.drivinggen_root,
            split=args.split,
            model_name=args.model_name,
            exp_id=args.exp_id,
            log_path=log_path,
            output_json=output_json,
            force=args.force,
        )
    except Exception as exc:
        if log_path is not None:
            print(f"DrivingGen FVD-lite failed; see log: {log_path}", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 1
    value = payload.get("value")
    print(f"DrivingGen FVD-lite fvd2048_100f={value} -> {output_json}")
    if log_path is not None:
        print(f"wrote DrivingGen FVD-lite log -> {log_path}")
    return 0


def _cmd_drivinggen_fvd_reference(args: argparse.Namespace) -> int:
    label = args.label or f"{args.split_a}_vs_{args.split_b}"
    log_path = None
    if not args.stream_logs:
        log_path = args.log_file or (
            args.drivinggen_root
            / "cache"
            / "eval_logs"
            / label
            / "reference-fvd-lite.log"
        )
    output_json = args.output_json or (
        args.drivinggen_root / "cache" / "eval_logs" / label / "reference-fvd-lite.json"
    )
    if args.dry_run:
        print(
            "drivinggen-fvd-reference "
            f"--drivinggen-root {shlex.quote(str(args.drivinggen_root))} "
            f"--split-a {shlex.quote(args.split_a)} "
            f"--split-b {shlex.quote(args.split_b)}"
        )
        if log_path is not None:
            print(f"log: {log_path}")
        print(f"json: {output_json}")
        return 0
    try:
        payload = run_fvd_reference_lite(
            drivinggen_root=args.drivinggen_root,
            split_a=args.split_a,
            split_b=args.split_b,
            log_path=log_path,
            output_json=output_json,
            force=args.force,
        )
    except Exception as exc:
        if log_path is not None:
            print(
                f"DrivingGen reference FVD-lite failed; see log: {log_path}",
                file=sys.stderr,
            )
        print(str(exc), file=sys.stderr)
        return 1
    value = payload.get("value")
    print(f"DrivingGen reference FVD-lite fvd2048_100f={value} -> {output_json}")
    if log_path is not None:
        print(f"wrote DrivingGen reference FVD-lite log -> {log_path}")
    return 0


def _cmd_worldlens_evaluate(args: argparse.Namespace) -> int:
    log_path = None
    if not args.stream_logs:
        log_path = args.log_file or (
            args.worldlens_root
            / "cache"
            / "eval_logs"
            / args.split
            / f"{args.method_name}-worldlens.log"
        )
    output_json = args.output_json or (
        args.worldlens_root
        / "cache"
        / "eval_logs"
        / args.split
        / f"{args.method_name}-worldlens.json"
    )
    exp_root = args.exp_root or (args.worldlens_root / "tools" / "exp")
    generated_data_path: Path | str
    generated_data_path = args.generated_data_path or "generated_results"
    command = [
        args.python,
        "tools/evaluate.py",
        "--config-name",
        args.config_name,
        f"modality={args.modality}",
        f"method_name={args.method_name}",
        f"generated_data_path={generated_data_path}",
    ]
    command.extend(args.hydra_override)
    if args.dry_run:
        prefix = f"WORLDBENCH_EXP_ROOT={shlex.quote(str(exp_root))} "
        suffix = f" > {shlex.quote(str(log_path))} 2>&1" if log_path else ""
        print(prefix + shlex.join(command) + suffix)
        print(f"json: {output_json}")
        return 0
    try:
        payload = run_worldlens_evaluation(
            worldlens_root=args.worldlens_root,
            modality=args.modality,
            method_name=args.method_name,
            config_name=args.config_name,
            generated_data_path=generated_data_path,
            python=args.python,
            exp_root=exp_root,
            log_path=log_path,
            output_json=output_json,
            hydra_overrides=args.hydra_override,
        )
    except subprocess.CalledProcessError as exc:
        if log_path is not None:
            print(f"WorldLens evaluation failed; see log: {log_path}", file=sys.stderr)
        return int(exc.returncode or 1)
    except Exception as exc:
        if log_path is not None:
            print(f"WorldLens evaluation failed; see log: {log_path}", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 1
    print(f"WorldLens metrics -> {output_json}")
    metric_results_path = payload.get("metric_results_path")
    if metric_results_path is not None:
        print(f"WorldLens metric_results: {metric_results_path}")
    if log_path is not None:
        print(f"wrote WorldLens log -> {log_path}")
    return 0


def _cmd_summarize_run(args: argparse.Namespace) -> int:
    summary = build_run_summary(args.run_root)
    output_json = args.output_json or (args.run_root / "evaluation-summary.json")
    output_md = args.output_md or (args.run_root / "evaluation-summary.md")
    write_run_summary_json(summary, output_json)
    write_run_summary_markdown(summary, output_md)
    if args.print_markdown:
        print(render_run_summary_markdown(summary), end="")
    else:
        print(f"wrote evaluation summary JSON -> {output_json}")
        print(f"wrote evaluation summary Markdown -> {output_md}")
    return 0


def _cmd_check_baseline(args: argparse.Namespace) -> int:
    summary = load_json(args.summary)
    baseline = load_json(args.baseline)
    if not isinstance(summary, dict):
        print("summary JSON must be an object", file=sys.stderr)
        return 1
    if not isinstance(baseline, dict):
        print("baseline JSON must be an object", file=sys.stderr)
        return 1
    try:
        report = check_summary_against_baseline(
            summary,
            baseline,
            summary_path=args.summary,
            baseline_path=args.baseline,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.output_json is not None:
        write_baseline_check_json(report, args.output_json)
        if not args.quiet:
            print(f"wrote baseline check JSON -> {args.output_json}")
    if not args.quiet:
        print(format_baseline_check_report(report), end="")
    return 0 if report["passed"] else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="omnidreams-eval")
    sub = parser.add_subparsers(required=True)

    discover = sub.add_parser("discover", help="discover model-ready HF scenes")
    _add_dataset_args(discover)
    discover.add_argument("--output", type=Path, required=True)
    discover.set_defaults(func=_cmd_discover)

    plan = sub.add_parser("plan-batches", help="create byte-capped batch plan")
    plan.add_argument("--manifest", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--batch-size", type=int, default=None)
    plan.add_argument("--max-batch-bytes", default=None)
    plan.set_defaults(func=_cmd_plan_batches)

    stage = sub.add_parser("stage-batch", help="download and prepare one batch")
    stage.add_argument("--manifest", type=Path, required=True)
    stage.add_argument("--batch-plan", type=Path, required=True)
    stage.add_argument("--batch-id", required=True)
    stage.add_argument("--scratch-root", type=Path, required=True)
    stage.add_argument("--output", type=Path, required=True)
    stage.add_argument("--token-env", default="HF_TOKEN")
    stage.add_argument("--force", action="store_true")
    stage.set_defaults(func=_cmd_stage_batch)

    setup_eval = sub.add_parser("setup-evaluator", help="clone/pin DrivingGen")
    setup_eval.add_argument("--cache-dir", type=Path, required=True)
    setup_eval.add_argument("--output", type=Path, required=True)
    setup_eval.add_argument("--evaluator-repo", default=DEFAULT_DRIVINGGEN_REPO)
    setup_eval.add_argument("--evaluator-revision", default=DEFAULT_DRIVINGGEN_REVISION)
    setup_eval.add_argument("--no-fetch", action="store_true")
    setup_eval.add_argument(
        "--no-patch",
        action="store_true",
        help="leave the cloned DrivingGen checkout unmodified",
    )
    setup_eval.set_defaults(func=_cmd_setup_evaluator)

    setup_worldlens = sub.add_parser(
        "setup-worldlens",
        help="clone/pin WorldLens and install OmniDreams consistency config",
    )
    setup_worldlens.add_argument("--cache-dir", type=Path, required=True)
    setup_worldlens.add_argument("--output", type=Path, required=True)
    setup_worldlens.add_argument("--evaluator-repo", default=DEFAULT_WORLDLENS_REPO)
    setup_worldlens.add_argument(
        "--evaluator-revision", default=DEFAULT_WORLDLENS_REVISION
    )
    setup_worldlens.add_argument("--config-name", default=DEFAULT_WORLDLENS_CONFIG_NAME)
    setup_worldlens.add_argument("--no-fetch", action="store_true")
    setup_worldlens.add_argument(
        "--no-config",
        action="store_true",
        help="do not write the OmniDreams WorldLens consistency Hydra config",
    )
    setup_worldlens.set_defaults(func=_cmd_setup_worldlens)

    generate = sub.add_parser("generate", help="run FlashDreams for staged cases")
    generate.add_argument("--staged-manifest", type=Path, required=True)
    generate.add_argument("--run-root", type=Path, required=True)
    generate.add_argument(
        "--recipe",
        default=DEFAULT_GENERATION_RECIPE,
        help=(
            "FlashDreams runner recipe. Defaults to the non-perf recipe for "
            "stable unattended evaluation; pass a perf recipe explicitly when "
            "benchmarking optimized runtime paths."
        ),
    )
    generate.add_argument("--total-blocks", type=int, default=20)
    generate.add_argument("--flashdreams-run", default="flashdreams-run")
    generate.add_argument("--force", action="store_true")
    generate.add_argument("--dry-run", action="store_true")
    generate.add_argument(
        "--stream-logs",
        action="store_true",
        help="stream flashdreams-run output to the terminal instead of per-case log files",
    )
    generate.set_defaults(func=_cmd_generate)

    dg_stage = sub.add_parser(
        "prepare-drivinggen",
        help="stage generated/reference frames in DrivingGen's expected layout",
    )
    dg_stage.add_argument("--staged-manifest", type=Path, required=True)
    dg_stage.add_argument("--run-root", type=Path, required=True)
    dg_stage.add_argument("--drivinggen-root", type=Path, required=True)
    _add_drivinggen_run_args(dg_stage)
    dg_stage.add_argument("--force", action="store_true")
    dg_stage.set_defaults(func=_cmd_prepare_drivinggen)

    wl_stage = sub.add_parser(
        "prepare-worldlens",
        help="stage generated/reference videos in WorldLens' video_submission layout",
    )
    wl_stage.add_argument("--staged-manifest", type=Path, required=True)
    wl_stage.add_argument("--run-root", type=Path, required=True)
    wl_stage.add_argument("--worldlens-root", type=Path, required=True)
    _add_worldlens_run_args(wl_stage)
    wl_stage.add_argument("--generation-index", type=int, default=0)
    wl_stage.add_argument("--camera-name", default="CAM_FRONT")
    wl_stage.add_argument(
        "--write-config",
        action="store_true",
        help="write the OmniDreams WorldLens consistency Hydra config before staging",
    )
    wl_stage.add_argument(
        "--force",
        action="store_true",
        help="replace existing staged video links/files",
    )
    wl_stage.set_defaults(func=_cmd_prepare_worldlens)

    validate = sub.add_parser(
        "validate-generation",
        help="validate generated artifacts, runner logs, and frame schedule",
    )
    validate.add_argument("--run-root", type=Path, required=True)
    validate.add_argument("--uuid", default=None)
    validate.add_argument("--fps", type=float, default=30.0)
    validate.add_argument("--output", type=Path, default=None)
    validate.set_defaults(func=_cmd_validate_generation)

    dg_metrics = sub.add_parser(
        "drivinggen-video-metrics", help="run DrivingGen video metrics"
    )
    dg_metrics.add_argument("--drivinggen-root", type=Path, required=True)
    _add_drivinggen_run_args(dg_metrics)
    dg_metrics.add_argument("--metric", default="fvd")
    dg_metrics.add_argument(
        "--python",
        default="python",
        help="Python executable used to run DrivingGen; point this at the evaluator environment",
    )
    dg_metrics.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="write DrivingGen stdout/stderr to this file",
    )
    dg_metrics.add_argument(
        "--stream-logs",
        action="store_true",
        help="stream DrivingGen output to the terminal instead of a log file",
    )
    dg_metrics.add_argument(
        "--i3d-checkpoint",
        type=Path,
        default=None,
        help="override DRIVINGGEN_I3D_CKPT for FVD",
    )
    dg_metrics.add_argument(
        "--inception-checkpoint",
        type=Path,
        default=None,
        help="override DRIVINGGEN_INCEPTION_CKPT for FID-style metrics",
    )
    dg_metrics.add_argument("--dry-run", action="store_true")
    dg_metrics.set_defaults(func=_cmd_drivinggen_video_metrics)

    dg_fvd_lite = sub.add_parser(
        "drivinggen-fvd-lite",
        help="run DrivingGen StyleGAN-V FVD without importing non-FVD metric stacks",
    )
    dg_fvd_lite.add_argument("--drivinggen-root", type=Path, required=True)
    _add_drivinggen_run_args(dg_fvd_lite)
    dg_fvd_lite.add_argument("--log-file", type=Path, default=None)
    dg_fvd_lite.add_argument("--output-json", type=Path, default=None)
    dg_fvd_lite.add_argument("--stream-logs", action="store_true")
    dg_fvd_lite.add_argument(
        "--force",
        action="store_true",
        help="rebuild the generated-frame FVD directory",
    )
    dg_fvd_lite.add_argument("--dry-run", action="store_true")
    dg_fvd_lite.set_defaults(func=_cmd_drivinggen_fvd_lite)

    dg_fvd_reference = sub.add_parser(
        "drivinggen-fvd-reference",
        help=(
            "diagnostic only: run StyleGAN-V FVD between two DrivingGen "
            "reference splits; do not use for model regression scoring"
        ),
        description=(
            "Diagnostic only. This compares two DrivingGen reference splits, "
            "which measures scenario/dataset diversity. Do not use it for "
            "model regression scoring; use drivinggen-fvd-lite on the same "
            "fixed split instead."
        ),
    )
    dg_fvd_reference.add_argument("--drivinggen-root", type=Path, required=True)
    dg_fvd_reference.add_argument("--split-a", required=True)
    dg_fvd_reference.add_argument("--split-b", required=True)
    dg_fvd_reference.add_argument(
        "--label",
        default=None,
        help="output label under cache/eval_logs; defaults to <split-a>_vs_<split-b>",
    )
    dg_fvd_reference.add_argument("--log-file", type=Path, default=None)
    dg_fvd_reference.add_argument("--output-json", type=Path, default=None)
    dg_fvd_reference.add_argument("--stream-logs", action="store_true")
    dg_fvd_reference.add_argument(
        "--force",
        action="store_true",
        help="rebuild split-specific reference FVD directories",
    )
    dg_fvd_reference.add_argument("--dry-run", action="store_true")
    dg_fvd_reference.set_defaults(func=_cmd_drivinggen_fvd_reference)

    wl_eval = sub.add_parser(
        "worldlens-evaluate",
        help="run WorldLens as a separate evaluator on staged OmniDreams videos",
    )
    wl_eval.add_argument("--worldlens-root", type=Path, required=True)
    _add_worldlens_run_args(wl_eval)
    wl_eval.add_argument("--modality", default="videogen")
    wl_eval.add_argument(
        "--generated-data-path",
        type=Path,
        default=None,
        help=(
            "WorldLens generated_data_path override. Defaults to generated_results "
            "relative to --worldlens-root."
        ),
    )
    wl_eval.add_argument(
        "--python",
        default="python",
        help="Python executable used to run WorldLens; point this at the evaluator environment",
    )
    wl_eval.add_argument(
        "--exp-root",
        type=Path,
        default=None,
        help="WORLDBENCH_EXP_ROOT; defaults to <worldlens-root>/tools/exp",
    )
    wl_eval.add_argument("--log-file", type=Path, default=None)
    wl_eval.add_argument("--output-json", type=Path, default=None)
    wl_eval.add_argument("--stream-logs", action="store_true")
    wl_eval.add_argument(
        "--hydra-override",
        action="append",
        default=[],
        help="extra Hydra override, for example videogen.dimensions=...",
    )
    wl_eval.add_argument("--dry-run", action="store_true")
    wl_eval.set_defaults(func=_cmd_worldlens_evaluate)

    summarize = sub.add_parser(
        "summarize-run",
        help="write JSON and Markdown summaries for an evaluation run",
    )
    summarize.add_argument("--run-root", type=Path, required=True)
    summarize.add_argument("--output-json", type=Path, default=None)
    summarize.add_argument("--output-md", type=Path, default=None)
    summarize.add_argument(
        "--print-markdown",
        action="store_true",
        help="also print the Markdown report to stdout",
    )
    summarize.set_defaults(func=_cmd_summarize_run)

    check_baseline = sub.add_parser(
        "check-baseline",
        help="compare an evaluation summary against a checked-in baseline JSON",
    )
    check_baseline.add_argument("--summary", type=Path, required=True)
    check_baseline.add_argument("--baseline", type=Path, required=True)
    check_baseline.add_argument("--output-json", type=Path, default=None)
    check_baseline.add_argument(
        "--quiet",
        action="store_true",
        help="only write --output-json and exit status; do not print the table",
    )
    check_baseline.set_defaults(func=_cmd_check_baseline)

    return parser


def _add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-repo", default=DEFAULT_DATASET_REPO)
    parser.add_argument("--dataset-revision", default=DEFAULT_DATASET_REVISION)
    parser.add_argument("--dataset-subpath", default=DEFAULT_DATASET_SUBPATH)
    parser.add_argument("--camera", default=DEFAULT_CAMERA)
    parser.add_argument("--token-env", default="HF_TOKEN")


def _add_drivinggen_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--split", default="omnidreams_eval")
    parser.add_argument("--model-name", default="omnidreams")
    parser.add_argument("--exp-id", default="default")


def _add_worldlens_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--split", default="omnidreams_eval")
    parser.add_argument("--method-name", default=DEFAULT_WORLDLENS_METHOD)
    parser.add_argument("--config-name", default=DEFAULT_WORLDLENS_CONFIG_NAME)


def _select_batch(batches: list, batch_id: str):
    for batch in batches:
        if batch.batch_id == batch_id:
            return batch
    raise KeyError(f"batch id not found: {batch_id}")


def _token_from_env(name: str) -> str | None:
    return os.environ.get(name) if name else None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
