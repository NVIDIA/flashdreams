# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FlashDreams video evaluation script.

Computes quality metrics on generated videos.  Inference and evaluation are
DECOUPLED: run inference first, then evaluate with this script.

Metrics
-------
Full-reference (require GT):  psnr, ssim, lpips
No-reference image quality:   niqe, musiq, clipiqa
No-reference video quality:   dover  (requires --dover_config)
FVD is handled separately.

Inputs can be:
  - A directory of video files (.mp4 / .avi / .mov / .mkv / .webm)
  - A directory of image sub-folders (one folder of PNGs per clip)

Usage examples
--------------
# No-reference quality on generated videos (no GT needed):
python -m flashdreams.eval.evaluate \\
    --pred_dir outputs/batch_mads_sv_perf_generated \\
    --metrics niqe musiq clipiqa \\
    --output_dir eval_results/

# Full-reference metrics comparing generated vs. GT:
python -m flashdreams.eval.evaluate \\
    --pred_dir outputs/batch_mads_sv_perf_generated \\
    --gt_dir   outputs/batch_mads_sv_perf \\
    --metrics  psnr ssim lpips \\
    --output_dir eval_results/

# All image-quality metrics (full + no-reference):
python -m flashdreams.eval.evaluate \\
    --pred_dir outputs/batch_mads_sv_perf_generated \\
    --gt_dir   outputs/batch_mads_sv_perf \\
    --metrics  psnr ssim lpips niqe musiq clipiqa \\
    --output_dir eval_results/

# With DOVER video quality (requires dover package + config):
python -m flashdreams.eval.evaluate \\
    --pred_dir outputs/batch_mads_sv_perf_generated \\
    --metrics  dover \\
    --dover_config /path/to/dover.yml \\
    --output_dir eval_results/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch as th

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

FULL_REFERENCE_METRICS = {"psnr", "ssim", "lpips"}
NO_REFERENCE_METRICS = {"niqe", "musiq", "clipiqa", "dover"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _natural_sort_key(s: str):
    import re

    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"([0-9]+)", s)]


def load_frames_from_folder(
    folder: str, max_frames: Optional[int] = None
) -> np.ndarray:
    """Load frames from a folder of PNG/JPG images using parallel I/O."""
    from PIL import Image

    files = sorted(
        [
            f
            for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ],
        key=_natural_sort_key,
    )
    if max_frames:
        files = files[:max_frames]
    paths = [os.path.join(folder, f) for f in files]

    def _load(p):
        return np.array(Image.open(p).convert("RGB"))

    with ThreadPoolExecutor(max_workers=8) as ex:
        frames = list(ex.map(_load, paths))
    return np.stack(frames, axis=0)


def load_frames_from_video(path: str, max_frames: Optional[int] = None) -> np.ndarray:
    """Load frames from a video file (decord when available, else OpenCV)."""
    from ._video_decode import get_video_frame_batch, get_video_frame_count

    total = get_video_frame_count(path)
    indices = list(range(min(max_frames, total) if max_frames else total))
    return get_video_frame_batch(path, indices)


def load_frames(path: str, max_frames: Optional[int] = None) -> np.ndarray:
    if os.path.isdir(path):
        return load_frames_from_folder(path, max_frames)
    elif os.path.isfile(path):
        return load_frames_from_video(path, max_frames)
    raise ValueError(f"Path does not exist: {path}")


def resize_frames(frames: np.ndarray, h: int, w: int) -> np.ndarray:
    from PIL import Image

    return np.stack(
        [
            np.array(Image.fromarray(frames[i]).resize((w, h), Image.BICUBIC))
            for i in range(len(frames))
        ]
    )


def get_sample_pairs(pred_dir: str, gt_dir: Optional[str]) -> List[tuple]:
    """Return list of (pred_path, gt_path_or_None, name) tuples.

    Matches by filename stem (without extension).  Both video files and
    image sub-folders are supported in the same directory.  Non-video files
    (JSON, txt, etc.) are silently skipped.
    """

    def _is_valid(item: str) -> bool:
        full = os.path.join(pred_dir, item)
        return (
            os.path.isdir(full) or os.path.splitext(item)[1].lower() in VIDEO_EXTENSIONS
        )

    pred_items = sorted(
        [item for item in os.listdir(pred_dir) if _is_valid(item)],
        key=_natural_sort_key,
    )

    if gt_dir is None:
        return [
            (os.path.join(pred_dir, item), None, os.path.splitext(item)[0])
            for item in pred_items
        ]

    gt_items = os.listdir(gt_dir)
    gt_map: Dict[str, str] = {}
    for item in gt_items:
        stem = os.path.splitext(item)[0]
        gt_map[stem] = os.path.join(gt_dir, item)
        gt_map[stem.lower()] = os.path.join(gt_dir, item)

    pairs = []
    for item in pred_items:
        stem = os.path.splitext(item)[0]
        pred_path = os.path.join(pred_dir, item)
        gt_path = gt_map.get(stem) or gt_map.get(stem.lower())
        if gt_path:
            pairs.append((pred_path, gt_path, stem))
        else:
            print(f"  [warn] no GT match for {item}")
    return pairs


def save_comparison_images(
    pred_frames: np.ndarray,
    gt_frames: np.ndarray,
    name: str,
    output_dir: str,
    max_frames: int = 5,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    os.makedirs(output_dir, exist_ok=True)
    for i in range(min(len(pred_frames), max_frames)):
        pred_img = Image.fromarray(pred_frames[i].astype(np.uint8))
        gt_img = Image.fromarray(gt_frames[i].astype(np.uint8))
        w, h = pred_img.size
        diff = np.clip(
            np.abs(pred_frames[i].astype(np.int16) - gt_frames[i].astype(np.int16)) * 5,
            0,
            255,
        ).astype(np.uint8)
        concat = Image.new("RGB", (w * 3, h))
        concat.paste(pred_img, (0, 0))
        concat.paste(gt_img, (w, 0))
        concat.paste(Image.fromarray(diff), (w * 2, 0))
        draw = ImageDraw.Draw(concat)
        font = ImageFont.load_default()
        for label, x in [("Pred", 10), ("GT", w + 10), ("Diff x5", w * 2 + 10)]:
            draw.text((x, 10), label, fill=(255, 255, 0), font=font)
        concat.save(os.path.join(output_dir, f"{name}_frame{i:04d}.png"))


def run_single(
    pred_path: str,
    gt_path: Optional[str],
    name: str,
    metrics: Dict[str, Any],
    metric_names: List[str],
    max_frames: Optional[int],
    resize_to_pred: bool,
    scale: Optional[int],
    device: str,
    batch_size: Optional[int],
    save_comparison: Optional[str],
) -> Dict[str, Any]:
    timings: Dict[str, float] = {}

    t0 = time.time()
    pred_frames = load_frames(pred_path, max_frames)
    timings["load_pred"] = time.time() - t0
    pred_orig_res = (pred_frames.shape[1], pred_frames.shape[2])
    pred_orig_frames = pred_frames.shape[0]

    full_ref = [m for m in metric_names if m in FULL_REFERENCE_METRICS]
    need_gt = bool(full_ref)

    gt_frames = None
    gt_orig_res = gt_orig_frames = gt_final_res = None

    if need_gt:
        if gt_path is None:
            raise ValueError(f"GT required for metrics: {full_ref}")
        t0 = time.time()
        gt_frames = load_frames(gt_path, max_frames)
        timings["load_gt"] = time.time() - t0
        gt_orig_res = (gt_frames.shape[1], gt_frames.shape[2])
        gt_orig_frames = gt_frames.shape[0]

        if pred_frames.shape[1:3] != gt_frames.shape[1:3]:
            if resize_to_pred:
                gt_frames = resize_frames(gt_frames, *pred_frames.shape[1:3])
            elif scale:
                th_, tw_ = gt_frames.shape[1] * scale, gt_frames.shape[2] * scale
                gt_frames = resize_frames(gt_frames, th_, tw_)
                if pred_frames.shape[1:3] != (th_, tw_):
                    pred_frames = resize_frames(pred_frames, th_, tw_)
            else:
                gt_frames = resize_frames(gt_frames, *pred_frames.shape[1:3])

        min_frames = min(pred_frames.shape[0], gt_frames.shape[0])
        pred_frames = pred_frames[:min_frames]
        gt_frames = gt_frames[:min_frames]
        gt_final_res = (gt_frames.shape[1], gt_frames.shape[2])

        if save_comparison:
            save_comparison_images(pred_frames, gt_frames, name, save_comparison)
    else:
        min_frames = pred_frames.shape[0]

    pred_gpu = gt_gpu = None
    if device == "cuda":
        pred_gpu = (
            th.from_numpy(np.transpose(pred_frames, (0, 3, 1, 2)).copy())
            .float()
            .to(device)
        )
        if gt_frames is not None:
            gt_gpu = (
                th.from_numpy(np.transpose(gt_frames, (0, 3, 1, 2)).copy())
                .float()
                .to(device)
            )

    result: Dict[str, Any] = {
        "name": name,
        # Preserve the original basename (with extension if any) so downstream
        # tools — e.g. build_eval_dashboard.py — can construct an accurate
        # artifact URL without having to assume ``.mp4``. ``name`` above is
        # the stem only and loses the extension for video files.
        "pred_filename": os.path.basename(pred_path),
        "num_frames": min_frames,
        "pred_original_frames": pred_orig_frames,
        "pred_original_resolution": pred_orig_res,
        "pred_final_resolution": (pred_frames.shape[1], pred_frames.shape[2]),
        "gt_original_frames": gt_orig_frames,
        "gt_original_resolution": gt_orig_res,
        "gt_final_resolution": gt_final_res,
        "timings": timings,
    }

    for mname in metric_names:
        metric = metrics[mname]
        is_no_ref = mname in NO_REFERENCE_METRICS

        t0 = time.time()

        if mname == "dover":
            if not (
                os.path.isfile(pred_path)
                and pred_path.lower().endswith(tuple(VIDEO_EXTENSIONS))
            ):
                raise ValueError(f"DOVER requires a video file path, got: {pred_path}")
            score = metric.update(pred_path, target=None)
            frame_values = [score]
            detail = metric.get_detailed_scores()
            if detail["aesthetic"]:
                result["dover_aesthetic"] = detail["aesthetic"][-1]
                result["dover_technical"] = detail["technical"][-1]

        elif hasattr(metric, "compute_batch"):
            pred_in = (
                pred_gpu if device == "cuda" and pred_gpu is not None else pred_frames
            )
            gt_in = gt_gpu if device == "cuda" and gt_gpu is not None else gt_frames
            if is_no_ref:
                frame_values = metric.compute_batch(
                    pred_in, target=None, batch_size=batch_size
                )
            else:
                frame_values = metric.compute_batch(
                    pred_in, gt_in, batch_size=batch_size
                )

        else:
            pred_in = (
                pred_gpu if device == "cuda" and pred_gpu is not None else pred_frames
            )
            gt_in = gt_gpu if device == "cuda" and gt_gpu is not None else gt_frames
            frame_values = []
            for t in range(min_frames):
                p = pred_in[t : t + 1] if isinstance(pred_in, th.Tensor) else pred_in[t]
                g = (
                    (gt_in[t : t + 1] if isinstance(gt_in, th.Tensor) else gt_in[t])
                    if gt_in is not None
                    else None
                )
                frame_values.append(
                    metric.compute(p, g) if not is_no_ref else metric.compute(p)
                )

        timings[f"metric_{mname}"] = time.time() - t0
        result[mname] = float(np.mean(frame_values))
        result[f"{mname}_values"] = frame_values

    return result


def format_per_sample_metrics(row: Dict[str, Any], metric_names: List[str]) -> str:
    """Format the metric values from a per-sample ``row`` dict for log output.

    Only keys explicitly named in ``metric_names`` (plus DOVER sub-scores when
    ``dover`` is requested) are formatted, and we additionally skip non-numeric
    values so a stray string field (e.g. ``pred_filename``) cannot break the
    ``f"{v:.4f}"`` conversion.
    """
    keys = list(metric_names)
    if "dover" in metric_names:
        keys += ["dover_aesthetic", "dover_technical"]
    return "  ".join(
        f"{k.upper()}: {row[k]:.4f}"
        for k in keys
        if k in row and isinstance(row[k], (int, float))
    )


def aggregate(
    results: List[Dict], metric_names: List[str], pred_dir: str, gt_dir: Optional[str]
) -> Dict:
    agg: Dict[str, Any] = {
        "pred_dir": pred_dir,
        "gt_dir": gt_dir,
        "num_samples": len(results),
        "total_frames": sum(r["num_frames"] for r in results),
    }
    for mname in metric_names:
        vals = [r[mname] for r in results]
        agg[f"{mname}_mean"] = float(np.mean(vals))
        agg[f"{mname}_std"] = float(np.std(vals))
        agg[f"{mname}_min"] = float(np.min(vals))
        agg[f"{mname}_max"] = float(np.max(vals))
    if "dover" in metric_names:
        for sub in ("aesthetic", "technical"):
            vals = [r[f"dover_{sub}"] for r in results if f"dover_{sub}" in r]
            if vals:
                agg[f"DOVER_{sub}_mean"] = float(np.mean(vals))
                agg[f"DOVER_{sub}_std"] = float(np.std(vals))
    return agg


def print_results(results: Dict) -> None:
    from tabulate import tabulate

    agg = results["aggregate"]
    print(f"\n{'=' * 60}")
    print("              FLASHDREAMS EVALUATION RESULTS")
    print(f"{'=' * 60}\n")

    info = [
        ["Prediction dir", agg["pred_dir"]],
        ["GT dir", agg["gt_dir"] or "(no-reference mode)"],
        ["Samples", agg["num_samples"]],
        ["Total frames", agg.get("total_frames", "N/A")],
    ]
    print(tabulate(info, headers=["Info", "Value"], tablefmt="simple"))
    print()

    rows = []
    dover_rows = []
    for k, v in agg.items():
        if not k.endswith("_mean"):
            continue
        if k.startswith("DOVER_") and k != "dover_mean":
            label = k.replace("_mean", "").replace("DOVER_", "DOVER ")
            dover_rows.append(
                [
                    f"  {label}",
                    f"{v:.4f}",
                    f"±{agg.get(k.replace('_mean', '_std'), 0):.4f}",
                ]
            )
            continue
        mname = k.replace("_mean", "").upper()
        rows.append(
            [
                mname,
                f"{v:.4f}",
                f"±{agg.get(k.replace('_mean', '_std'), 0):.4f}",
                f"{agg.get(k.replace('_mean', '_min'), 0):.4f}",
                f"{agg.get(k.replace('_mean', '_max'), 0):.4f}",
            ]
        )
    if rows:
        print(
            tabulate(
                rows, headers=["Metric", "Mean", "Std", "Min", "Max"], tablefmt="simple"
            )
        )
    if dover_rows:
        print("\nDOVER breakdown:")
        print(
            tabulate(
                dover_rows, headers=["Component", "Mean", "Std"], tablefmt="simple"
            )
        )
    print(f"\n{'=' * 60}\n")


def save_results(
    results: Dict,
    output_dir: str,
    tag: Optional[str],
    output_file: Optional[str] = None,
) -> str:
    if output_file:
        parent = os.path.dirname(output_file)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to: {output_file}")
        return output_file
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"eval_{tag}_{ts}.json" if tag else f"eval_{ts}.json"
    path = os.path.join(output_dir, fname)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to: {path}")
    return path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FlashDreams evaluation — compute video quality metrics on generated outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--pred_dir",
        required=True,
        help="Directory of generated videos or image folders.",
    )
    p.add_argument(
        "--gt_dir",
        default=None,
        help="Directory of ground-truth videos or image folders (optional for no-reference metrics).",
    )
    p.add_argument(
        "--metrics",
        nargs="+",
        default=["niqe", "musiq", "clipiqa"],
        choices=["psnr", "ssim", "lpips", "niqe", "musiq", "clipiqa", "dover"],
        help="Metrics to compute.",
    )
    p.add_argument("--lpips_net", default="vgg", choices=["alex", "vgg", "squeeze"])
    p.add_argument(
        "--clipiqa_model",
        default="clipiqa",
        choices=["clipiqa", "clipiqa+", "clipiqa+_vitL14_512", "clipiqa+_rn50_512"],
    )
    p.add_argument(
        "--musiq_model",
        default="musiq",
        choices=["musiq", "musiq-ava", "musiq-paq2piq", "musiq-spaq"],
    )
    p.add_argument(
        "--dover_config",
        default=None,
        help="Path to dover.yml config (required for DOVER metric).",
    )
    p.add_argument(
        "--output_dir",
        default="./eval_results",
        help="Directory to write JSON results.",
    )
    p.add_argument(
        "--tag", default=None, help="Optional tag appended to output filename."
    )
    p.add_argument(
        "--resize_to_pred",
        action="store_true",
        help="Resize GT to match pred resolution.",
    )
    p.add_argument(
        "--scale",
        type=int,
        default=None,
        help="Upscale GT by this factor before comparison.",
    )
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument(
        "--save_comparison",
        default=None,
        help="Save pred/GT/diff side-by-side images to this directory.",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Write results to this exact file path instead of --output_dir/eval_<tag>_<ts>.json. "
        "Useful in CI where downstream jobs need a fixed artifact path.",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    from .metrics import MetricRegistry

    args = parse_args()

    print(f"\n{'#' * 60}")
    print("FlashDreams Evaluation")
    print(f"{'#' * 60}")
    print(f"Time:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Pred:    {args.pred_dir}")
    print(f"GT:      {args.gt_dir or '(none — no-reference mode)'}")
    print(f"Metrics: {args.metrics}")
    print(f"Device:  {args.device}")

    full_ref = [m for m in args.metrics if m in FULL_REFERENCE_METRICS]
    if full_ref and not args.gt_dir:
        raise ValueError(f"--gt_dir required for full-reference metrics: {full_ref}")
    if "dover" in args.metrics and not args.dover_config:
        raise ValueError("--dover_config required when using the dover metric")

    pairs = get_sample_pairs(args.pred_dir, args.gt_dir)
    if not pairs:
        raise ValueError("No matching samples found.")
    if args.max_samples:
        pairs = pairs[: args.max_samples]
    print(f"\nFound {len(pairs)} sample(s)\n{'─' * 60}")

    metrics: Dict[str, Any] = {}
    for mname in args.metrics:
        if mname == "lpips":
            metrics[mname] = MetricRegistry.get(
                mname, device=args.device, net=args.lpips_net
            )
        elif mname == "clipiqa":
            metrics[mname] = MetricRegistry.get(
                mname, device=args.device, model=args.clipiqa_model
            )
        elif mname == "musiq":
            metrics[mname] = MetricRegistry.get(
                mname, device=args.device, model=args.musiq_model
            )
        elif mname == "dover":
            metrics[mname] = MetricRegistry.get(
                mname, device=args.device, config_path=args.dover_config
            )
        else:
            metrics[mname] = MetricRegistry.get(mname, device=args.device)

    from tqdm import tqdm

    need_gt = bool(full_ref)
    start = time.time()
    per_sample: List[Dict] = []

    for pred_path, gt_path, name in tqdm(
        pairs, desc="Evaluating", unit="video", dynamic_ncols=True
    ):
        tqdm.write(f"  {name}")
        result = run_single(
            pred_path=pred_path,
            gt_path=gt_path if need_gt else None,
            name=name,
            metrics=metrics,
            metric_names=args.metrics,
            max_frames=args.max_frames,
            resize_to_pred=args.resize_to_pred,
            scale=args.scale,
            device=args.device,
            batch_size=args.batch_size,
            save_comparison=args.save_comparison,
        )

        row = {
            k: v
            for k, v in result.items()
            if not k.endswith("_values") and k != "timings"
        }
        per_sample.append(row)

        metric_str = format_per_sample_metrics(row, args.metrics)
        tqdm.write(f"  -> {metric_str}")
        if args.verbose:
            tqdm.write(f"     timings: {result['timings']}")
        tqdm.write("")

    agg = aggregate(per_sample, args.metrics, args.pred_dir, args.gt_dir)
    agg["total_evaluation_time"] = time.time() - start

    final = {"aggregate": agg, "per_sample": per_sample, "config": vars(args)}
    print_results(final)
    save_results(final, args.output_dir, args.tag, output_file=args.output)
    print(f"Done in {agg['total_evaluation_time']:.1f}s")


if __name__ == "__main__":
    main()
