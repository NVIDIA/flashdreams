# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Crop OmniDreams sweep videos and write per-output quality metrics.

OmniDreams sweep artifacts store ``video.mp4`` as a vertical stack where the
HDMap/conditioning visualization is on top and the generated camera video is on
the bottom. This script crops the bottom half once per output directory and
evaluates only that cropped copy.

For generated-only comparison videos, pass ``--input-is-generated`` to skip the
crop step while keeping the same ``metrics.json`` schema for the VLM evaluator.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from flashdreams.eval.evaluate import (
    FULL_REFERENCE_METRICS,
    format_per_sample_metrics,
    run_single,
)
from flashdreams.eval.metrics import MetricRegistry

CROP_FILTER = "crop=iw:floor(ih/2):0:ih-floor(ih/2)"
DEFAULT_ROOT = Path("outputs/omnidreams-quality-sweep")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)
        f.write("\n")
    tmp.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def find_source_videos(root: Path, video_name: str) -> list[Path]:
    return sorted(p for p in root.rglob(video_name) if p.is_file())


def crop_bottom_half_video(
    src: Path,
    dst: Path,
    *,
    crf: int,
    preset: str,
    overwrite: bool,
) -> bool:
    if dst.exists() and not overwrite and dst.stat().st_mtime >= src.stat().st_mtime:
        return False

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH")

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".tmp" + dst.suffix)
    if tmp.exists():
        tmp.unlink()

    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-vf",
        CROP_FILTER,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        str(tmp),
    ]
    subprocess.run(cmd, check=True)
    tmp.replace(dst)
    return True


def build_metrics(args: argparse.Namespace, device: str) -> dict[str, Any]:
    full_ref = [m for m in args.metrics if m in FULL_REFERENCE_METRICS]
    if full_ref:
        raise ValueError(
            "This sweep evaluator has no ground-truth path; use only no-reference "
            f"metrics. Full-reference metrics requested: {full_ref}"
        )
    if "dover" in args.metrics and not args.dover_config:
        raise ValueError("--dover-config is required when using the dover metric")

    metrics: dict[str, Any] = {}
    for name in args.metrics:
        if name == "clipiqa":
            metrics[name] = MetricRegistry.get(
                name, device=device, model=args.clipiqa_model
            )
        elif name == "musiq":
            metrics[name] = MetricRegistry.get(
                name, device=device, model=args.musiq_model
            )
        elif name == "dover":
            metrics[name] = MetricRegistry.get(
                name, device=device, config_path=args.dover_config
            )
        else:
            metrics[name] = MetricRegistry.get(name, device=device)
    return metrics


def make_success_payload(
    *,
    root: Path,
    src: Path,
    cropped: Path,
    metrics_path: Path,
    result: dict[str, Any],
    metric_names: list[str],
    crop_created: bool,
    crop_seconds: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": "ok",
        "evaluated_at_utc": _utc_now(),
        "output_dir": str(src.parent),
        "relative_output_dir": str(src.parent.relative_to(root)),
        "source_video": str(src),
        "cropped_video": str(cropped),
        "metrics_json": str(metrics_path),
        "crop": {
            "filter": CROP_FILTER,
            "created": crop_created,
            "seconds": crop_seconds,
        },
        "metrics": {name: result[name] for name in metric_names if name in result},
        "result": result,
        "config": config,
    }


def make_failure_payload(
    *,
    root: Path,
    src: Path,
    cropped: Path,
    metrics_path: Path,
    error: BaseException,
    config: dict[str, Any],
) -> dict[str, Any]:
    try:
        relative_output_dir = str(src.parent.relative_to(root))
    except ValueError:
        relative_output_dir = str(src.parent)
    return {
        "status": "failed",
        "evaluated_at_utc": _utc_now(),
        "output_dir": str(src.parent),
        "relative_output_dir": relative_output_dir,
        "source_video": str(src),
        "cropped_video": str(cropped),
        "metrics_json": str(metrics_path),
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
        "config": config,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crop stacked OmniDreams videos to the generated bottom half and "
            "write one metrics.json beside each output."
        )
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--video-name", default="video.mp4")
    parser.add_argument("--cropped-name", default="video_generated_bottom.mp4")
    parser.add_argument("--metrics-json-name", default="metrics.json")
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Aggregate progress JSON path. Defaults to <root>/metrics_summary.json.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["niqe", "musiq", "clipiqa"],
        choices=["niqe", "musiq", "clipiqa", "dover", "psnr", "ssim", "lpips"],
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--clipiqa-model", default="clipiqa")
    parser.add_argument("--musiq-model", default="musiq")
    parser.add_argument("--dover-config", default=None)
    parser.add_argument("--ffmpeg-crf", type=int, default=18)
    parser.add_argument("--ffmpeg-preset", default="veryfast")
    parser.add_argument(
        "--input-is-generated",
        action="store_true",
        help=(
            "Treat --video-name as the generated video itself and skip the "
            "bottom-half crop step."
        ),
    )
    parser.add_argument("--overwrite-crops", action="store_true")
    parser.add_argument("--overwrite-metrics", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    args = parse_args()
    root = args.root.expanduser().resolve()
    summary_path = args.summary_path or root / "metrics_summary.json"
    device = _resolve_device(args.device)

    videos = find_source_videos(root, args.video_name)
    if args.limit is not None:
        videos = videos[: args.limit]
    if not videos:
        raise ValueError(f"No {args.video_name!r} files found under {root}")

    config = vars(args).copy()
    config["root"] = str(root)
    config["summary_path"] = str(summary_path)
    config["device"] = device

    print(f"Found {len(videos)} source video(s) under {root}")
    print(f"Metrics: {args.metrics}")
    print(f"Device: {device}")

    metrics = build_metrics(args, device)

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    iterator = videos
    if tqdm is not None:
        iterator = tqdm(videos, desc="Evaluating OmniDreams outputs", unit="video")

    started = time.time()
    records: list[dict[str, Any]] = []
    status_counts = {"ok": 0, "failed": 0, "skipped": 0}

    for src in iterator:
        cropped = src.with_name(args.cropped_name)
        metrics_path = src.parent / args.metrics_json_name

        if metrics_path.exists() and not args.overwrite_metrics:
            status_counts["skipped"] += 1
            records.append(
                {
                    "status": "skipped",
                    "source_video": str(src),
                    "metrics_json": str(metrics_path),
                }
            )
            continue

        try:
            crop_t0 = time.time()
            if args.input_is_generated:
                cropped = src
                crop_created = False
            else:
                crop_created = crop_bottom_half_video(
                    src,
                    cropped,
                    crf=args.ffmpeg_crf,
                    preset=args.ffmpeg_preset,
                    overwrite=args.overwrite_crops,
                )
            crop_seconds = time.time() - crop_t0

            name = str(src.parent.relative_to(root))
            result = run_single(
                pred_path=str(cropped),
                gt_path=None,
                name=name,
                metrics=metrics,
                metric_names=args.metrics,
                max_frames=args.max_frames,
                resize_to_pred=False,
                scale=None,
                device=device,
                batch_size=args.batch_size,
                save_comparison=None,
            )
            payload = make_success_payload(
                root=root,
                src=src,
                cropped=cropped,
                metrics_path=metrics_path,
                result=result,
                metric_names=args.metrics,
                crop_created=crop_created,
                crop_seconds=crop_seconds,
                config=config,
            )
            _write_json(metrics_path, payload)
            status_counts["ok"] += 1
            records.append(
                {
                    "status": "ok",
                    "source_video": str(src),
                    "cropped_video": str(cropped),
                    "metrics_json": str(metrics_path),
                    "metrics": payload["metrics"],
                }
            )
            print(f"{name}: {format_per_sample_metrics(result, args.metrics)}")
        except Exception as exc:
            payload = make_failure_payload(
                root=root,
                src=src,
                cropped=cropped,
                metrics_path=metrics_path,
                error=exc,
                config=config,
            )
            _write_json(metrics_path, payload)
            status_counts["failed"] += 1
            records.append(
                {
                    "status": "failed",
                    "source_video": str(src),
                    "metrics_json": str(metrics_path),
                    "error": payload["error"],
                }
            )
            if not args.keep_going:
                raise
            print(f"{src}: failed: {type(exc).__name__}: {exc}", file=sys.stderr)

    summary = {
        "status": "ok" if status_counts["failed"] == 0 else "failed",
        "evaluated_at_utc": _utc_now(),
        "root": str(root),
        "elapsed_seconds": time.time() - started,
        "status_counts": status_counts,
        "records": records,
        "config": config,
    }
    _write_json(summary_path, summary)
    print(f"Summary saved to {summary_path}")
    print(f"Status counts: {status_counts}")
    return 0 if status_counts["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
