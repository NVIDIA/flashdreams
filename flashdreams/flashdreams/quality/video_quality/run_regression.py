# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run manifest-driven video-quality regression checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import operator
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from flashdreams.quality.clip_compare import read_video_rgb
from flashdreams.quality.video_quality.manifest import (
    Threshold,
    VideoQualityCase,
    load_manifest,
)
from flashdreams.quality.video_quality.metrics import (
    VideoMetricsInput,
    compute_video_metrics,
    synthetic_video,
)

_OPS = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    started_at = time.time()
    manifest = load_manifest(args.manifest)
    selected_cases = manifest.select_cases(suite=args.suite, case_id=args.case_id)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    case_results: list[dict[str, Any]] = [
        _run_case(
            case,
            suite=args.suite,
            evaluate_only=args.evaluate_only,
            asset_root=args.asset_root,
            output_dir=output_dir / case.id,
            hf_revision_override=args.hf_revision,
            dump_clips=args.dump_clips,
        )
        for case in selected_cases
    ]

    failed_cases = [
        case for case in case_results if case["decision"]["status"] != "pass"
    ]
    run_manifest = {
        "schema_version": 1,
        "manifest_path": str(args.manifest),
        "suite": args.suite,
        "case_id": args.case_id,
        "evaluate_only": args.evaluate_only,
        "github": _github_metadata(),
        "started_at_unix": started_at,
        "duration_s": time.time() - started_at,
        "output_dir": str(output_dir),
        "case_count": len(case_results),
        "failed_case_count": len(failed_cases),
        "decision": "fail" if failed_cases else "pass",
        "cases": case_results,
    }
    _write_json(output_dir / "manifest.json", run_manifest)

    print(
        f"video-quality regression {run_manifest['decision']}: "
        f"{len(case_results)} case(s), {len(failed_cases)} failure(s); "
        f"manifest={output_dir / 'manifest.json'}"
    )
    return 1 if failed_cases and args.fail_on_regression else 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/video_quality_cases.yml"),
        help="Path to the video-quality case manifest.",
    )
    parser.add_argument(
        "--suite",
        default="calibration",
        help="Suite label to run: calibration, per_commit, nightly, quarantine, or vlm_experimental.",
    )
    parser.add_argument("--case-id", help="Optional single case id to run.")
    parser.add_argument(
        "--hf-revision",
        help="Override Hugging Face dataset revision for asset-backed cases.",
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        help="Optional local root for resolving manifest asset paths before HF download.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/video_quality"),
        help="Directory for run manifest and per-case metrics.",
    )
    parser.add_argument(
        "--evaluate-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate known-good/known-bad clips without model inference.",
    )
    parser.add_argument(
        "--dump-clips",
        action="store_true",
        help="Copy or synthesize evaluated clips into the output directory when possible.",
    )
    parser.add_argument(
        "--fail-on-regression",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit non-zero when case decisions fail.",
    )
    return parser.parse_args(argv)


def _run_case(
    case: VideoQualityCase,
    *,
    suite: str,
    evaluate_only: bool,
    asset_root: Path | None,
    output_dir: Path,
    hf_revision_override: str | None,
    dump_clips: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    if evaluate_only:
        clip_specs = _evaluate_only_clip_specs(case)
    else:
        if case.assets.generated is None:
            raise NotImplementedError(
                f"{case.id}: generation mode needs a generated asset or a runner plugin"
            )
        clip_specs = [("generated", "generated", case.assets.generated, ())]

    clip_results: list[dict[str, Any]] = []
    for role, clip_id, asset, expected_failures in clip_specs:
        metrics_input = _resolve_video(
            case,
            asset,
            asset_key=clip_id,
            asset_root=asset_root,
            hf_revision_override=hf_revision_override,
        )
        metrics = compute_video_metrics(
            metrics_input.frames,
            fps=metrics_input.fps,
            windows=case.windows,
            metric_groups=case.metrics,
        )
        threshold_failures = _threshold_failures(metrics, case.thresholds)
        decision = _clip_decision(
            role=role,
            threshold_failures=threshold_failures,
            expected_failures=expected_failures,
        )
        clip_output_dir = output_dir / clip_id
        clip_output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = clip_output_dir / "metrics.json"
        _write_json(metrics_path, metrics)
        if dump_clips:
            _dump_clip(
                metrics_input, asset=asset, output_path=clip_output_dir / "clip.mp4"
            )
        clip_results.append(
            {
                "id": clip_id,
                "role": role,
                "asset": asset,
                "metrics_path": str(metrics_path),
                "metrics": metrics,
                "threshold_failures": threshold_failures,
                "expected_failures": list(expected_failures),
                "decision": decision,
            }
        )

    failed_clips = [
        clip for clip in clip_results if clip["decision"]["status"] != "pass"
    ]
    result = {
        "id": case.id,
        "description": case.description,
        "suites": list(case.suites),
        "hf_dataset": case.hf_dataset,
        "hf_revision": hf_revision_override or case.hf_revision,
        "metrics": list(case.metrics),
        "generation": case.generation,
        "windows": {name: asdict(window) for name, window in case.windows.items()},
        "thresholds": [asdict(threshold) for threshold in case.thresholds],
        "clips": clip_results,
        "decision": {
            "status": "fail" if failed_clips else "pass",
            "failed_clip_ids": [clip["id"] for clip in failed_clips],
            "gates_merge": suite == "per_commit" and "per_commit" in case.suites,
        },
    }
    _write_json(output_dir / "case_manifest.json", result)
    return result


def _evaluate_only_clip_specs(
    case: VideoQualityCase,
) -> list[tuple[str, str, str, tuple[str, ...]]]:
    specs: list[tuple[str, str, str, tuple[str, ...]]] = []
    if case.assets.known_good is not None:
        specs.append(("known_good", "known_good", case.assets.known_good, ()))
    for known_bad in case.assets.known_bad:
        specs.append(
            ("known_bad", known_bad.id, known_bad.path, known_bad.expected_failures)
        )
    if not specs:
        raise ValueError(
            f"{case.id}: evaluate-only mode needs known_good or known_bad assets"
        )
    return specs


def _resolve_video(
    case: VideoQualityCase,
    asset: str,
    *,
    asset_key: str,
    asset_root: Path | None,
    hf_revision_override: str | None,
) -> VideoMetricsInput:
    if asset.startswith("synthetic://"):
        return _resolve_synthetic(case, asset.removeprefix("synthetic://"))

    path = _resolve_local_path(asset, asset_root=asset_root)
    if path is None:
        path = _download_hf_asset(
            case,
            asset,
            hf_revision_override=hf_revision_override,
        )
    expected_sha = case.assets.sha256.get(asset_key) or case.assets.sha256.get(asset)
    if expected_sha is not None:
        _validate_sha256(path, expected_sha)
    return VideoMetricsInput(frames=read_video_rgb(path), fps=_asset_fps(case))


def _resolve_synthetic(case: VideoQualityCase, pattern: str) -> VideoMetricsInput:
    source = case.source
    return synthetic_video(
        pattern,
        frames=int(source.get("frames", 16)),
        height=int(source.get("height", 64)),
        width=int(source.get("width", 64)),
        fps=float(source.get("fps", 8)),
        seed=int(source.get("seed", case.generation.get("seed", 0))),
    )


def _resolve_local_path(asset: str, *, asset_root: Path | None) -> Path | None:
    path = Path(asset).expanduser()
    candidates = [path] if path.is_absolute() else []
    if asset_root is not None and not path.is_absolute():
        candidates.append(asset_root / path)
    if not path.is_absolute():
        candidates.append(Path.cwd() / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _download_hf_asset(
    case: VideoQualityCase, asset: str, *, hf_revision_override: str | None
) -> Path:
    dataset = case.hf_dataset
    if dataset is None:
        raise FileNotFoundError(
            f"{case.id}: asset {asset!r} was not local and no hf_dataset is set"
        )
    revision = hf_revision_override or case.hf_revision
    if revision is None:
        raise ValueError(f"{case.id}: hf_revision is required to download {asset!r}")
    try:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - import-time gate
        raise ImportError(
            "Hugging Face asset resolution needs huggingface-hub installed."
        ) from exc
    return Path(
        hf_hub_download(
            repo_id=dataset,
            filename=asset,
            repo_type="dataset",
            revision=revision,
        )
    )


def _asset_fps(case: VideoQualityCase) -> float | None:
    for key in ("fps", "output_fps"):
        value = case.source.get(key, case.generation.get(key))
        if value is not None:
            return float(value)
    return None


def _threshold_failures(
    metrics: dict[str, float | int | bool | None], thresholds: tuple[Threshold, ...]
) -> list[dict[str, Any]]:
    failures = []
    for threshold in thresholds:
        actual = metrics.get(threshold.metric)
        if actual is None:
            failures.append(
                {
                    "id": threshold.id,
                    "metric": threshold.metric,
                    "op": threshold.op,
                    "expected": threshold.value,
                    "actual": actual,
                    "severity": threshold.severity,
                    "reason": "metric_missing",
                }
            )
            continue
        if not _OPS[threshold.op](actual, threshold.value):
            failures.append(
                {
                    "id": threshold.id,
                    "metric": threshold.metric,
                    "op": threshold.op,
                    "expected": threshold.value,
                    "actual": actual,
                    "severity": threshold.severity,
                    "reason": "threshold_failed",
                }
            )
    return failures


def _clip_decision(
    *,
    role: str,
    threshold_failures: list[dict[str, Any]],
    expected_failures: tuple[str, ...],
) -> dict[str, Any]:
    failed_ids = {str(failure["id"]) for failure in threshold_failures}
    if role == "known_bad":
        expected_ids = set(expected_failures)
        observed_expected = (
            expected_ids <= failed_ids if expected_ids else bool(failed_ids)
        )
        unexpected_failure_ids = sorted(failed_ids - expected_ids)
        missing_expected_failure_ids = sorted(expected_ids - failed_ids)
        return {
            "status": (
                "pass" if observed_expected and not unexpected_failure_ids else "fail"
            ),
            "calibration_expected_failure_observed": observed_expected,
            "unexpected_failure_ids": unexpected_failure_ids,
            "missing_expected_failure_ids": missing_expected_failure_ids,
        }
    return {
        "status": "pass" if not threshold_failures else "fail",
        "failed_threshold_ids": sorted(failed_ids),
    }


def _dump_clip(
    metrics_input: VideoMetricsInput, *, asset: str, output_path: Path
) -> None:
    if asset.startswith("synthetic://"):
        try:
            import mediapy as media  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - import-time gate
            raise ImportError("Dumping synthetic clips needs mediapy.") from exc
        media.write_video(
            str(output_path),
            metrics_input.frames,
            fps=int(metrics_input.fps or 8),
        )
        return

    source = Path(asset)
    if source.exists():
        shutil.copy2(source, output_path)


def _validate_sha256(path: Path, expected_sha256: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected_sha256:
        raise ValueError(
            f"sha256 mismatch for {path}: expected {expected_sha256}, got {actual}"
        )


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


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
