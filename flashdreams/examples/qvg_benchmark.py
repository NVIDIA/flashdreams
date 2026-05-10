# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-file QVG reproduction benchmark runner and metric aggregator."""

from __future__ import annotations

import argparse
import inspect
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import mediapy as media
except ImportError:  # pragma: no cover - unit tests can avoid video IO.
    media = None

try:
    import cv2
except ImportError:  # pragma: no cover - labels/resizing degrade gracefully.
    cv2 = None


REPO_ROOT = Path(__file__).resolve().parents[2]
QVG_BENCHMARK_ASSETS = REPO_ROOT / "flashdreams/examples/qvg_benchmark_assets"
DEFAULT_QVG_PROMPTS = QVG_BENCHMARK_ASSETS / "prompts/qvg_prompt_matrix_extra.txt"
DEFAULT_OFFICIAL_CONFIG = QVG_BENCHMARK_ASSETS / "configs/official_self_forcing_dmd_shift8.yaml"
PRIMARY_RANGE = "all"
FIRST_DECODED_CHUNK_RANGE = "first_decoded_chunk"
FIRST_DECODED_CHUNK_FRAMES = 3
VBENCH_METRICS = (
    "background_consistency",
    "image_quality",
    "subject_consistency",
    "aesthetic_quality",
)
VBENCH_DIMENSIONS = {
    "background_consistency": "background_consistency",
    "image_quality": "imaging_quality",
    "subject_consistency": "subject_consistency",
    "aesthetic_quality": "aesthetic_quality",
}
TOLERANCES = {
    "compression_ratio": 0.2,
    "psnr_db": 0.5,
    "ssim_rgb_gaussian": 0.01,
    "lpips": 0.01,
    "vbench": 0.01,
}
MISSING_VBENCH_MESSAGE = (
    "VBench is not installed in this environment. Install/run it in an isolated "
    "benchmark environment, then rerun with --vbench_py pointing at that Python."
)


@dataclass(frozen=True)
class FrameRange:
    name: str
    start: int
    end: int
    role: str

    @property
    def frame_count(self) -> int:
        return self.end - self.start


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value):
        return None
    return value


def _mean_std(values: list[float | None]) -> tuple[float | None, float | None]:
    if not values or any(value is None for value in values):
        return None, None
    array = np.asarray(values, dtype=np.float64)
    return float(np.mean(array)), float(np.std(array))


def _format_score_delta(score: float | None, delta: float | None) -> str | None:
    if score is None or delta is None:
        return None
    return f"{score:.4f} ({delta:+.4f})"


def _read_video(path: Path) -> np.ndarray:
    if media is not None:
        try:
            frames = media.read_video(path)
            if frames is not None and len(frames) > 0:
                return np.asarray(frames)
        except RuntimeError:
            pass
    if cv2 is None:
        raise RuntimeError("Reading videos requires mediapy or opencv-python.")
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"Could not decode video '{path}'")
    return np.stack(frames)


def _write_video(path: Path, frames: np.ndarray, *, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if media is not None:
        try:
            media.write_video(path, frames, fps=fps)
            return
        except RuntimeError:
            pass
    if cv2 is None:
        raise RuntimeError("Writing videos requires mediapy or opencv-python.")
    height, width = frames.shape[1:3]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def crop_video(input_path: Path, output_path: Path, *, end: int, fps: float = 16.0) -> Path:
    frames = _read_video(input_path)[:end]
    _write_video(output_path, frames, fps=fps)
    return output_path


def _build_metric_ranges(
    *,
    frame_count: int,
    generated_frame_start: int,
    post_compression_start: int,
    first_compressed_frame_start: int,
) -> tuple[dict[str, FrameRange], list[str]]:
    unavailable: list[str] = []
    if frame_count < generated_frame_start + FIRST_DECODED_CHUNK_FRAMES:
        raise ValueError("first_decoded_chunk requires at least 3 paired frames")
    ranges = {
        "all": FrameRange("all", 0, frame_count, "primary"),
        "first_decoded_chunk": FrameRange(
            "first_decoded_chunk",
            generated_frame_start,
            generated_frame_start + FIRST_DECODED_CHUNK_FRAMES,
            "diagnostic",
        ),
        "post_compression": FrameRange(
            "post_compression",
            min(post_compression_start, frame_count),
            frame_count,
            "diagnostic",
        ),
        "first_compressed_decoded_chunk": FrameRange(
            "first_compressed_decoded_chunk",
            min(first_compressed_frame_start, frame_count),
            min(first_compressed_frame_start + FIRST_DECODED_CHUNK_FRAMES, frame_count),
            "diagnostic",
        ),
        "last12": FrameRange("last12", max(0, frame_count - 12), frame_count, "diagnostic"),
    }
    for name, frame_range in list(ranges.items()):
        if frame_range.frame_count <= 0:
            unavailable.append(name)
            del ranges[name]
    return ranges, unavailable


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return 99.0
    return float(20 * np.log10(255.0 / np.sqrt(mse)))


def _ssim_rgb_gaussian(a: np.ndarray, b: np.ndarray) -> float:
    if cv2 is None:
        return _ssim_simple(a, b)
    scores = []
    for x, y in zip(a, b, strict=True):
        x = x.astype(np.float64)
        y = y.astype(np.float64)
        for channel in range(3):
            scores.append(_ssim_channel_gaussian(x[..., channel], y[..., channel]))
    return float(np.mean(scores))


def _ssim_channel_gaussian(x: np.ndarray, y: np.ndarray) -> float:
    assert cv2 is not None
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    kernel = (11, 11)
    sigma = 1.5
    mu_x = cv2.GaussianBlur(x, kernel, sigma)
    mu_y = cv2.GaussianBlur(y, kernel, sigma)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x2 = cv2.GaussianBlur(x * x, kernel, sigma) - mu_x2
    sigma_y2 = cv2.GaussianBlur(y * y, kernel, sigma) - mu_y2
    sigma_xy = cv2.GaussianBlur(x * y, kernel, sigma) - mu_xy
    ssim = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    )
    return float(np.mean(ssim))


def _ssim_simple(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu_a = a.mean()
    mu_b = b.mean()
    var_a = a.var()
    var_b = b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)))


def _metrics_for_slice(
    baseline: np.ndarray,
    candidate: np.ndarray,
    frame_range: FrameRange,
) -> dict[str, object]:
    base = baseline[frame_range.start : frame_range.end]
    cand = candidate[frame_range.start : frame_range.end]
    diff = np.abs(base.astype(np.int16) - cand.astype(np.int16))
    return {
        "frame_indices": [frame_range.start, frame_range.end],
        "frame_count": frame_range.frame_count,
        "psnr_db": _psnr(base, cand),
        "ssim_rgb_gaussian": _ssim_rgb_gaussian(base, cand),
        "mean_abs_diff": float(np.mean(diff)),
        "max_abs_diff": int(np.max(diff)),
    }


def _lpips_video(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    net: str,
    batch_size: int,
    resize_short_side: int | None,
) -> tuple[float, str | None]:
    try:
        import lpips
        import torch
        import torch.nn.functional as F
    except ImportError:
        return math.nan, "lpips package is not installed"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = lpips.LPIPS(net=net).to(device).eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(baseline), batch_size):
            base = baseline[start : start + batch_size]
            cand = candidate[start : start + batch_size]
            base_t = torch.from_numpy(base).permute(0, 3, 1, 2).float() / 127.5 - 1.0
            cand_t = torch.from_numpy(cand).permute(0, 3, 1, 2).float() / 127.5 - 1.0
            if resize_short_side is not None:
                h, w = base_t.shape[-2:]
                scale = resize_short_side / min(h, w)
                size = (round(h * scale), round(w * scale))
                base_t = F.interpolate(base_t, size=size, mode="bilinear", align_corners=False)
                cand_t = F.interpolate(cand_t, size=size, mode="bilinear", align_corners=False)
            score = loss_fn(base_t.to(device), cand_t.to(device))
            values.extend(score.flatten().detach().cpu().tolist())
    return float(np.mean(values)), None


def _add_lpips_range_metrics(
    *,
    metrics: dict[str, Any],
    baseline: np.ndarray,
    candidate: np.ndarray,
    ranges: dict[str, FrameRange],
    selected_ranges: list[str],
    net: str,
    batch_size: int,
    resize_short_side: int | None,
) -> None:
    for name in selected_ranges:
        frame_range = ranges[name]
        score, warning = _lpips_video(
            baseline[frame_range.start : frame_range.end],
            candidate[frame_range.start : frame_range.end],
            net=net,
            batch_size=batch_size,
            resize_short_side=resize_short_side,
        )
        if warning is not None:
            metrics.setdefault("warnings", []).append(warning)
            continue
        metrics["ranges"][name]["lpips"] = score


def _add_primary_aliases(metrics: dict[str, Any]) -> None:
    primary = metrics["ranges"][PRIMARY_RANGE]
    metrics["primary_psnr_db"] = primary.get("psnr_db")
    metrics["primary_ssim_rgb_gaussian"] = primary.get("ssim_rgb_gaussian")
    metrics["primary_lpips"] = primary.get("lpips")


def compare_videos(
    baseline_path: Path,
    candidate_path: Path,
    *,
    output_json: Path | None,
    contact_sheet: Path | None,
    lpips_enabled: bool,
    lpips_batch_size: int,
    lpips_net: str = "alex",
    max_frames: int | None = None,
    generated_frame_start: int = 0,
    post_compression_start: int = 93,
    first_compressed_frame_start: int = 93,
) -> dict[str, object]:
    baseline = _read_video(baseline_path)
    candidate = _read_video(candidate_path)
    frame_count = min(len(baseline), len(candidate))
    if max_frames is not None:
        frame_count = min(frame_count, max_frames)
    baseline = baseline[:frame_count]
    candidate = candidate[:frame_count]
    ranges, unavailable = _build_metric_ranges(
        frame_count=frame_count,
        generated_frame_start=generated_frame_start,
        post_compression_start=post_compression_start,
        first_compressed_frame_start=first_compressed_frame_start,
    )
    metrics: dict[str, Any] = {
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "frame_count": frame_count,
        "ranges": {
            name: _metrics_for_slice(baseline, candidate, frame_range)
            for name, frame_range in ranges.items()
        },
        "unavailable_ranges": unavailable,
    }
    if lpips_enabled:
        _add_lpips_range_metrics(
            metrics=metrics,
            baseline=baseline,
            candidate=candidate,
            ranges=ranges,
            selected_ranges=[PRIMARY_RANGE],
            net=lpips_net,
            batch_size=lpips_batch_size,
            resize_short_side=256,
        )
    _add_primary_aliases(metrics)
    if contact_sheet is not None:
        _write_contact_sheet(contact_sheet, baseline, candidate)
        metrics["contact_sheet"] = str(contact_sheet)
    if output_json is not None:
        _write_json(output_json, metrics)
    return metrics


def _write_contact_sheet(path: Path, baseline: np.ndarray, candidate: np.ndarray) -> None:
    indices = [0, 1, 2, 30, 60, 92, 93, 94, 95, 104]
    indices = [idx for idx in indices if idx < len(baseline)]
    rows = []
    for idx in indices:
        base = baseline[idx]
        cand = candidate[idx]
        diff = np.clip(np.abs(base.astype(np.int16) - cand.astype(np.int16)) * 4, 0, 255).astype(np.uint8)
        rows.append(np.concatenate([base, cand, diff], axis=1))
    sheet = np.concatenate(rows, axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    if cv2 is not None:
        cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    else:
        assert media is not None
        media.write_image(path, sheet)


def _import_vbench_class() -> Any:
    if importlib.util.find_spec("vbench") is None:
        raise RuntimeError(MISSING_VBENCH_MESSAGE)
    from vbench import VBench

    return VBench


def _parse_indices(value: str | None) -> list[int] | None:
    if value is None:
        return None
    return [int(part) for part in value.split(",") if part]


def _resolve_prompts(
    *,
    prompts: list[str] | None,
    prompts_file: Path | None,
    prompt_indices: str | None,
    video_count: int,
) -> list[str]:
    if prompts is None:
        if prompts_file is None:
            raise ValueError("Pass --prompts or --prompts_file")
        prompts = [line.strip() for line in prompts_file.read_text().splitlines() if line.strip()]
        indices = _parse_indices(prompt_indices)
        if indices is not None:
            prompts = [prompts[index] for index in indices]
    if len(prompts) != video_count:
        raise ValueError(f"Expected {video_count} prompts, got {len(prompts)}")
    return list(prompts)


def normalize_vbench_payload(
    payload: dict[str, Any],
    *,
    videos: list[Path],
    prompts: list[str],
) -> dict[str, Any]:
    scores: dict[str, float | None] = {}
    per_video = [
        {"video": str(video), "prompt": prompt, "scores": {}}
        for video, prompt in zip(videos, prompts, strict=True)
    ]
    for canonical, raw_name in VBENCH_DIMENSIONS.items():
        raw = payload.get(raw_name, payload.get(canonical))
        mean_value: float | None = None
        raw_per_video: list[Any] = []
        if isinstance(raw, list):
            mean_value = _as_float(raw[0]) if raw else None
            if len(raw) > 1 and isinstance(raw[1], list):
                raw_per_video = raw[1]
        elif isinstance(raw, dict):
            mean_value = _as_float(raw.get("mean", raw.get("score")))
            raw_per_video = raw.get("per_video", [])
        else:
            mean_value = _as_float(raw)
        if canonical == "image_quality" and mean_value is not None and mean_value > 1:
            mean_value /= 100.0
        scores[canonical] = mean_value
        for index, entry in enumerate(raw_per_video[: len(per_video)]):
            value = None
            if isinstance(entry, dict):
                value = _as_float(entry.get("video_results", entry.get("score")))
            else:
                value = _as_float(entry)
            if canonical == "image_quality" and value is not None and value > 1:
                value /= 100.0
            per_video[index]["scores"][canonical] = value
    return {"scores": scores, "per_video": per_video}


def run_or_normalize_vbench(
    *,
    videos: list[Path],
    prompts: list[str],
    output_dir: Path,
    name: str,
    output_json: Path | None = None,
    raw_results_json: Path | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_json or output_dir / f"{name}_normalized.json"
    if raw_results_json is not None:
        raw_payload = _load_json(raw_results_json)
    else:
        VBench = _import_vbench_class()
        input_dir = output_dir / "input_videos"
        input_dir.mkdir(parents=True, exist_ok=True)
        vbench_videos: list[Path] = []
        for index, video in enumerate(videos):
            target = input_dir / f"{index:04d}_{video.name}"
            if target.exists() or target.is_symlink():
                target.unlink()
            try:
                target.symlink_to(video)
            except OSError:
                shutil.copy2(video, target)
            vbench_videos.append(target)

        if len(vbench_videos) == 1:
            videos_path = str(vbench_videos[0])
            prompt_list: list[str] | dict[str, str] = [prompts[0]]
        else:
            videos_path = str(input_dir)
            prompt_list = {
                video.name: prompt
                for video, prompt in zip(vbench_videos, prompts, strict=True)
            }
        if len(inspect.signature(VBench).parameters) >= 3:
            evaluator = VBench(device, str(output_dir), str(output_dir))
        else:
            evaluator = VBench(device, str(output_dir))
        if len(inspect.signature(evaluator.build_full_dimension_list).parameters) >= 1:
            evaluator.build_full_dimension_list(list(VBENCH_DIMENSIONS.values()))
        else:
            evaluator.build_full_dimension_list()
        evaluator.evaluate(
            videos_path=videos_path,
            name=name,
            prompt_list=prompt_list,
            dimension_list=list(VBENCH_DIMENSIONS.values()),
            mode="custom_input",
        )
        raw_results_json = _find_latest_vbench_json(output_dir)
        raw_payload = _load_json(raw_results_json)
    normalized = normalize_vbench_payload(raw_payload, videos=videos, prompts=prompts)
    normalized["raw_results_json"] = None if raw_results_json is None else str(raw_results_json)
    _write_json(output_json, normalized)
    return normalized


def _find_latest_vbench_json(output_dir: Path) -> Path:
    candidates = [
        path
        for path in output_dir.rglob("*.json")
        if "normalized" not in path.name
        and not path.name.endswith("_inputs.json")
        and path.parent.name != "input"
    ]
    if not candidates:
        raise FileNotFoundError(f"No VBench JSON found under {output_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _extract_fidelity_metrics(compare: dict[str, Any], *, range_name: str) -> dict[str, Any]:
    selected = compare.get("ranges", {}).get(range_name, {})
    return {
        "frame_indices": selected.get("frame_indices"),
        "psnr_db": _as_float(selected.get("psnr_db", compare.get("primary_psnr_db"))),
        "ssim_rgb_gaussian": _as_float(selected.get("ssim_rgb_gaussian", compare.get("primary_ssim_rgb_gaussian"))),
        "lpips": _as_float(selected.get("lpips", compare.get("primary_lpips"))),
    }


def _summarize_fidelity(compares: list[dict[str, Any]], *, range_name: str) -> dict[str, Any]:
    extracted = [_extract_fidelity_metrics(compare, range_name=range_name) for compare in compares]
    out: dict[str, Any] = {"sample_count": len(extracted)}
    for key in ("psnr_db", "ssim_rgb_gaussian", "lpips"):
        values = [item[key] for item in extracted]
        mean, std = _mean_std(values)
        out[key] = mean
        out[f"{key}_std"] = std
        out[f"{key}_values"] = values
    frame_indices = [item.get("frame_indices") for item in extracted]
    out["frame_indices"] = frame_indices[0] if frame_indices and all(v == frame_indices[0] for v in frame_indices) else frame_indices
    return out


def _extract_vbench_scores(payload: dict[str, Any]) -> dict[str, float | None]:
    scores = payload.get("scores", {})
    out = {}
    for name in VBENCH_METRICS:
        value = scores.get(name)
        if value is None and name == "image_quality":
            value = scores.get("imaging_quality")
        out[name] = _as_float(value)
    return out


def _extract_vbench_per_video(payload: dict[str, Any]) -> dict[str, list[float | None]]:
    out: dict[str, list[float | None]] = {name: [] for name in VBENCH_METRICS}
    for entry in payload.get("per_video", []):
        scores = entry.get("scores", {})
        for name in VBENCH_METRICS:
            value = scores.get(name)
            if value is None and name == "image_quality":
                value = scores.get("imaging_quality")
            out[name].append(_as_float(value))
    return out


def _extract_ratio_from_stats(path: Path | None) -> float | None:
    if path is None or not path.exists():
        return None
    payload = _load_json(path)
    if isinstance(payload, dict):
        return _as_float(payload.get("kv_cache_compression_ratio"))
    if isinstance(payload, list):
        for item in reversed(payload):
            if isinstance(item, dict) and "kv_cache_compression_ratio" in item:
                return _as_float(item["kv_cache_compression_ratio"])
    return None


def _summary_from_paths(paths: list[Path], extractor: Any) -> dict[str, Any]:
    values = [extractor(path) for path in paths]
    mean, std = _mean_std(values)
    return {"value": mean, "std": std, "values": values, "sample_count": len(values)}


def _wall_seconds(path: Path | None) -> float | None:
    if path is None or not path.exists():
        return None
    payload = _load_json(path)
    if isinstance(payload, dict):
        return _as_float(payload.get("wall_seconds", payload.get("generation_wall_seconds")))
    return None


def _generation_seconds(path: Path | None) -> float | None:
    if path is None or not path.exists():
        return None
    payload = _load_json(path)
    if isinstance(payload, dict):
        return _as_float(payload.get("generation_seconds"))
    return None


def _fps_from_generation(path: Path | None) -> float | None:
    if path is None or not path.exists():
        return None
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return None
    seconds = _as_float(payload.get("generation_seconds"))
    frames = _as_float(payload.get("generated_frames"))
    if seconds is not None and frames is not None and seconds > 0:
        return frames / seconds
    return _as_float(payload.get("generation_fps"))


def _fps_from_wall(path: Path | None) -> float | None:
    if path is None or not path.exists():
        return None
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return None
    seconds = _as_float(payload.get("wall_seconds", payload.get("generation_wall_seconds")))
    frames = _as_float(payload.get("generated_frames"))
    if seconds is None or frames is None or seconds <= 0:
        return None
    return frames / seconds


def _flashdreams_generation_seconds(stats_path: Path | None) -> float | None:
    if stats_path is None or not stats_path.exists():
        return None
    payload = _load_json(stats_path)
    if not isinstance(payload, list):
        return None
    total_ms = 0.0
    found = False
    for item in payload:
        if isinstance(item, dict):
            value = _as_float(item.get("total_ms"))
            if value is not None:
                total_ms += value
                found = True
    if not found:
        return None
    return total_ms / 1000.0


def _merge_generation_stats(
    *,
    run_stats_path: Path,
    generation_seconds: float | None,
    generated_frames: int,
    timing_scope: str,
    generation_stats_path: Path | None = None,
) -> None:
    if generation_seconds is None or generation_seconds <= 0 or not run_stats_path.exists():
        return
    payload = _load_json(run_stats_path)
    if not isinstance(payload, dict):
        return
    payload["generation_seconds"] = generation_seconds
    payload["generation_fps"] = generated_frames / generation_seconds
    payload["generation_timing_scope"] = timing_scope
    if generation_stats_path is not None:
        payload["generation_stats_path"] = str(generation_stats_path)
    _write_json(run_stats_path, payload)


def _metric_pass_higher(flashdreams: float | None, official: float | None, tolerance: float) -> bool:
    return flashdreams is not None and official is not None and flashdreams >= official - tolerance


def _metric_pass_lower(flashdreams: float | None, official: float | None, tolerance: float) -> bool:
    return flashdreams is not None and official is not None and flashdreams <= official + tolerance


def _comparison_entry(name: str, official: float | None, flashdreams: float | None, *, tolerance: float, higher_is_better: bool, official_std: float | None = None, flashdreams_std: float | None = None) -> dict[str, object]:
    passed = _metric_pass_higher(flashdreams, official, tolerance) if higher_is_better else _metric_pass_lower(flashdreams, official, tolerance)
    entry: dict[str, object] = {
        "metric": name,
        "official": official,
        "flashdreams": flashdreams,
        "flashdreams_minus_official": None if flashdreams is None or official is None else flashdreams - official,
        "tolerance": tolerance,
        "higher_is_better": higher_is_better,
        "pass": passed,
    }
    if official_std is not None:
        entry["official_std"] = official_std
    if flashdreams_std is not None:
        entry["flashdreams_std"] = flashdreams_std
    return entry


def _compression_entry(official: float | None, flashdreams: float | None) -> dict[str, object]:
    passed = official is not None and flashdreams is not None and flashdreams >= 6.5 and abs(flashdreams - official) <= TOLERANCES["compression_ratio"]
    return {
        "metric": "compression_ratio",
        "official": official,
        "flashdreams": flashdreams,
        "flashdreams_minus_official": None if official is None or flashdreams is None else flashdreams - official,
        "minimum_flashdreams": 6.5,
        "tolerance": TOLERANCES["compression_ratio"],
        "higher_is_better": True,
        "pass": passed,
    }


def _vbench_entry(name: str, official: float | None, flashdreams: float | None, official_bf16: float | None, flashdreams_bf16: float | None) -> dict[str, object]:
    if official is None or flashdreams is None:
        passed = False
        mode = "missing"
        extra: dict[str, object] = {}
    elif official_bf16 is None or flashdreams_bf16 is None:
        passed = False
        mode = "missing_bf16_baseline"
        extra = {
            "official_bf16": official_bf16,
            "flashdreams_bf16": flashdreams_bf16,
        }
    else:
        official_delta = official - official_bf16
        flash_delta = flashdreams - flashdreams_bf16
        passed = flash_delta >= official_delta - TOLERANCES["vbench"]
        mode = "bf16_delta"
        extra = {
            "official_bf16": official_bf16,
            "official_qvg": official,
            "official_int2_delta": official_delta,
            "official_display": _format_score_delta(official_bf16, official_delta),
            "flashdreams_bf16": flashdreams_bf16,
            "flashdreams_qvg": flashdreams,
            "flashdreams_int2_delta": flash_delta,
            "flashdreams_display": _format_score_delta(flashdreams_bf16, flash_delta),
        }
    return {
        "metric": name,
        "official": official,
        "flashdreams": flashdreams,
        "tolerance": TOLERANCES["vbench"],
        "higher_is_better": True,
        "comparison_mode": mode,
        "pass": passed,
        **extra,
    }


def _perf_entry(metric: str, official_bf16: float | None, official_qvg: float | None, flashdreams_bf16: float | None, flashdreams_qvg: float | None, *, higher_is_better: bool, source: str) -> dict[str, object]:
    official_delta = None if official_bf16 is None or official_qvg is None else official_qvg - official_bf16
    flash_delta = None if flashdreams_bf16 is None or flashdreams_qvg is None else flashdreams_qvg - flashdreams_bf16
    return {
        "metric": metric,
        "official_bf16": official_bf16,
        "official_qvg": official_qvg,
        "official_int2_delta": official_delta,
        "official_display": _format_score_delta(official_bf16, official_delta),
        "flashdreams_bf16": flashdreams_bf16,
        "flashdreams_qvg": flashdreams_qvg,
        "flashdreams_int2_delta": flash_delta,
        "flashdreams_display": _format_score_delta(flashdreams_bf16, flash_delta),
        "higher_is_better": higher_is_better,
        "source": source,
    }


def _series_value(series: dict[str, list[float | None]], name: str, index: int) -> float | None:
    values = series.get(name, [])
    return values[index] if index < len(values) else None


def aggregate_report(
    *,
    official_compares: list[dict[str, Any]],
    flashdreams_compares: list[dict[str, Any]],
    official_vbench: dict[str, Any],
    flashdreams_vbench: dict[str, Any],
    official_bf16_vbench: dict[str, Any],
    flashdreams_bf16_vbench: dict[str, Any],
    official_compression_ratio: float,
    flashdreams_stats: list[Path],
    official_bf16_run_stats: list[Path],
    official_qvg_run_stats: list[Path],
    flashdreams_bf16_run_stats: list[Path],
    flashdreams_qvg_run_stats: list[Path],
    quant_label: str,
    fidelity_range: str = PRIMARY_RANGE,
) -> dict[str, object]:
    if len(official_compares) != len(flashdreams_compares):
        raise ValueError("official and FlashDreams comparison JSON counts must match")
    sample_count = len(official_compares)
    official_fidelity = _summarize_fidelity(official_compares, range_name=fidelity_range)
    flash_fidelity = _summarize_fidelity(flashdreams_compares, range_name=fidelity_range)
    flash_ratio_summary = _summary_from_paths(flashdreams_stats, _extract_ratio_from_stats)
    official_v = _extract_vbench_scores(official_vbench)
    flash_v = _extract_vbench_scores(flashdreams_vbench)
    official_bf16_v = _extract_vbench_scores(official_bf16_vbench)
    flash_bf16_v = _extract_vbench_scores(flashdreams_bf16_vbench)
    metric_entries: dict[str, dict[str, object]] = {
        "compression_ratio": _compression_entry(official_compression_ratio, _as_float(flash_ratio_summary["value"])),
        "psnr_db": _comparison_entry("psnr_db", official_fidelity["psnr_db"], flash_fidelity["psnr_db"], tolerance=TOLERANCES["psnr_db"], higher_is_better=True, official_std=official_fidelity["psnr_db_std"], flashdreams_std=flash_fidelity["psnr_db_std"]),
        "ssim_rgb_gaussian": _comparison_entry("ssim_rgb_gaussian", official_fidelity["ssim_rgb_gaussian"], flash_fidelity["ssim_rgb_gaussian"], tolerance=TOLERANCES["ssim_rgb_gaussian"], higher_is_better=True, official_std=official_fidelity["ssim_rgb_gaussian_std"], flashdreams_std=flash_fidelity["ssim_rgb_gaussian_std"]),
        "lpips": _comparison_entry("lpips", official_fidelity["lpips"], flash_fidelity["lpips"], tolerance=TOLERANCES["lpips"], higher_is_better=False, official_std=official_fidelity["lpips_std"], flashdreams_std=flash_fidelity["lpips_std"]),
    }
    for name in VBENCH_METRICS:
        metric_entries[name] = _vbench_entry(name, official_v[name], flash_v[name], official_bf16_v[name], flash_bf16_v[name])

    gen_off_bf16 = _summary_from_paths(official_bf16_run_stats, _generation_seconds)
    gen_off_qvg = _summary_from_paths(official_qvg_run_stats, _generation_seconds)
    gen_fd_bf16 = _summary_from_paths(flashdreams_bf16_run_stats, _generation_seconds)
    gen_fd_qvg = _summary_from_paths(flashdreams_qvg_run_stats, _generation_seconds)
    gen_fps_off_bf16 = _summary_from_paths(official_bf16_run_stats, _fps_from_generation)
    gen_fps_off_qvg = _summary_from_paths(official_qvg_run_stats, _fps_from_generation)
    gen_fps_fd_bf16 = _summary_from_paths(flashdreams_bf16_run_stats, _fps_from_generation)
    gen_fps_fd_qvg = _summary_from_paths(flashdreams_qvg_run_stats, _fps_from_generation)
    wall_off_bf16 = _summary_from_paths(official_bf16_run_stats, _wall_seconds)
    wall_off_qvg = _summary_from_paths(official_qvg_run_stats, _wall_seconds)
    wall_fd_bf16 = _summary_from_paths(flashdreams_bf16_run_stats, _wall_seconds)
    wall_fd_qvg = _summary_from_paths(flashdreams_qvg_run_stats, _wall_seconds)
    fps_off_bf16 = _summary_from_paths(official_bf16_run_stats, _fps_from_wall)
    fps_off_qvg = _summary_from_paths(official_qvg_run_stats, _fps_from_wall)
    fps_fd_bf16 = _summary_from_paths(flashdreams_bf16_run_stats, _fps_from_wall)
    fps_fd_qvg = _summary_from_paths(flashdreams_qvg_run_stats, _fps_from_wall)
    performance = {
        "generation_seconds": _perf_entry(
            "generation_seconds",
            _as_float(gen_off_bf16["value"]),
            _as_float(gen_off_qvg["value"]),
            _as_float(gen_fd_bf16["value"]),
            _as_float(gen_fd_qvg["value"]),
            higher_is_better=False,
            source="model_generation_only_excludes_checkpoint_load_and_video_write",
        ),
        "generation_fps": _perf_entry(
            "generation_fps",
            _as_float(gen_fps_off_bf16["value"]),
            _as_float(gen_fps_off_qvg["value"]),
            _as_float(gen_fps_fd_bf16["value"]),
            _as_float(gen_fps_fd_qvg["value"]),
            higher_is_better=True,
            source="generated_frames / generation_seconds",
        ),
        "end_to_end_wall_seconds": _perf_entry(
            "end_to_end_wall_seconds",
            _as_float(wall_off_bf16["value"]),
            _as_float(wall_off_qvg["value"]),
            _as_float(wall_fd_bf16["value"]),
            _as_float(wall_fd_qvg["value"]),
            higher_is_better=False,
            source="end_to_end_command_wall_clock",
        ),
        "end_to_end_fps": _perf_entry(
            "end_to_end_fps",
            _as_float(fps_off_bf16["value"]),
            _as_float(fps_off_qvg["value"]),
            _as_float(fps_fd_bf16["value"]),
            _as_float(fps_fd_qvg["value"]),
            higher_is_better=True,
            source="generated_frames / end_to_end_wall_seconds",
        ),
    }

    official_v_series = _extract_vbench_per_video(official_vbench)
    flash_v_series = _extract_vbench_per_video(flashdreams_vbench)
    official_bf16_v_series = _extract_vbench_per_video(official_bf16_vbench)
    flash_bf16_v_series = _extract_vbench_per_video(flashdreams_bf16_vbench)
    flash_ratio_values = [_as_float(value) for value in flash_ratio_summary["values"]]
    rows = []
    for index in range(sample_count):
        row_metrics = {
            "compression_ratio": _compression_entry(official_compression_ratio, flash_ratio_values[index] if index < len(flash_ratio_values) else None),
            "psnr_db": _comparison_entry("psnr_db", _extract_fidelity_metrics(official_compares[index], range_name=fidelity_range)["psnr_db"], _extract_fidelity_metrics(flashdreams_compares[index], range_name=fidelity_range)["psnr_db"], tolerance=TOLERANCES["psnr_db"], higher_is_better=True),
            "ssim_rgb_gaussian": _comparison_entry("ssim_rgb_gaussian", _extract_fidelity_metrics(official_compares[index], range_name=fidelity_range)["ssim_rgb_gaussian"], _extract_fidelity_metrics(flashdreams_compares[index], range_name=fidelity_range)["ssim_rgb_gaussian"], tolerance=TOLERANCES["ssim_rgb_gaussian"], higher_is_better=True),
            "lpips": _comparison_entry("lpips", _extract_fidelity_metrics(official_compares[index], range_name=fidelity_range)["lpips"], _extract_fidelity_metrics(flashdreams_compares[index], range_name=fidelity_range)["lpips"], tolerance=TOLERANCES["lpips"], higher_is_better=False),
        }
        for name in VBENCH_METRICS:
            row_metrics[name] = _vbench_entry(
                name,
                _series_value(official_v_series, name, index),
                _series_value(flash_v_series, name, index),
                _series_value(official_bf16_v_series, name, index),
                _series_value(flash_bf16_v_series, name, index),
            )
        row_performance = {
            "generation_seconds": _perf_entry(
                "generation_seconds",
                _generation_seconds(official_bf16_run_stats[index]) if index < len(official_bf16_run_stats) else None,
                _generation_seconds(official_qvg_run_stats[index]) if index < len(official_qvg_run_stats) else None,
                _generation_seconds(flashdreams_bf16_run_stats[index]) if index < len(flashdreams_bf16_run_stats) else None,
                _generation_seconds(flashdreams_qvg_run_stats[index]) if index < len(flashdreams_qvg_run_stats) else None,
                higher_is_better=False,
                source="model_generation_only_excludes_checkpoint_load_and_video_write",
            ),
            "generation_fps": _perf_entry(
                "generation_fps",
                _fps_from_generation(official_bf16_run_stats[index]) if index < len(official_bf16_run_stats) else None,
                _fps_from_generation(official_qvg_run_stats[index]) if index < len(official_qvg_run_stats) else None,
                _fps_from_generation(flashdreams_bf16_run_stats[index]) if index < len(flashdreams_bf16_run_stats) else None,
                _fps_from_generation(flashdreams_qvg_run_stats[index]) if index < len(flashdreams_qvg_run_stats) else None,
                higher_is_better=True,
                source="generated_frames / generation_seconds",
            ),
            "end_to_end_wall_seconds": _perf_entry(
                "end_to_end_wall_seconds",
                _wall_seconds(official_bf16_run_stats[index]) if index < len(official_bf16_run_stats) else None,
                _wall_seconds(official_qvg_run_stats[index]) if index < len(official_qvg_run_stats) else None,
                _wall_seconds(flashdreams_bf16_run_stats[index]) if index < len(flashdreams_bf16_run_stats) else None,
                _wall_seconds(flashdreams_qvg_run_stats[index]) if index < len(flashdreams_qvg_run_stats) else None,
                higher_is_better=False,
                source="end_to_end_command_wall_clock",
            ),
            "end_to_end_fps": _perf_entry(
                "end_to_end_fps",
                _fps_from_wall(official_bf16_run_stats[index]) if index < len(official_bf16_run_stats) else None,
                _fps_from_wall(official_qvg_run_stats[index]) if index < len(official_qvg_run_stats) else None,
                _fps_from_wall(flashdreams_bf16_run_stats[index]) if index < len(flashdreams_bf16_run_stats) else None,
                _fps_from_wall(flashdreams_qvg_run_stats[index]) if index < len(flashdreams_qvg_run_stats) else None,
                higher_is_better=True,
                source="generated_frames / end_to_end_wall_seconds",
            ),
        }
        missing = [name for name, entry in row_metrics.items() if entry["official"] is None or entry["flashdreams"] is None]
        rows.append(
            {
                "row": f"prompt{index}",
                "index": index,
                "metrics": row_metrics,
                "performance_metrics": row_performance,
                "missing_metrics": missing,
                "all_required_metrics_pass": not missing and all(bool(entry["pass"]) for entry in row_metrics.values()),
            }
        )
    missing = [name for name, entry in metric_entries.items() if entry["official"] is None or entry["flashdreams"] is None]
    passed = not missing and all(bool(entry["pass"]) for entry in metric_entries.values())
    average_row = {
        "row": "average",
        "metrics": metric_entries,
        "performance_metrics": performance,
        "missing_metrics": missing,
        "all_required_metrics_pass": passed,
    }
    return {
        "quant_label": quant_label,
        "fidelity_range": fidelity_range,
        "fidelity_frame_indices": official_fidelity.get("frame_indices"),
        "fidelity_sample_count": sample_count,
        "tolerances": TOLERANCES,
        "metrics": metric_entries,
        "average_metrics": metric_entries,
        "performance_metrics": performance,
        "per_prompt_metrics": rows,
        "metric_rows": [*rows, average_row],
        "missing_metrics": missing,
        "all_required_metrics_pass": passed,
    }


def write_visual_grid(
    *,
    official_bf16_video: Path,
    official_qvg_video: Path,
    flashdreams_bf16_video: Path,
    flashdreams_qvg_video: Path,
    output_path: Path,
    max_frames: int | None = None,
    fps: float = 16.0,
) -> dict[str, object]:
    labels = ("Official BF16", "Official INT2", "FlashDreams BF16", "FlashDreams INT2")
    videos = [
        _read_video(official_bf16_video),
        _read_video(official_qvg_video),
        _read_video(flashdreams_bf16_video),
        _read_video(flashdreams_qvg_video),
    ]
    frames = min(len(video) for video in videos)
    if max_frames is not None:
        frames = min(frames, max_frames)
    videos = [video[:frames] for video in videos]
    height = min(video.shape[1] for video in videos)
    width = min(video.shape[2] for video in videos)
    resized = []
    for video in videos:
        if video.shape[1] != height or video.shape[2] != width:
            if cv2 is None:
                raise RuntimeError("OpenCV is required to resize grid videos")
            video = np.stack([cv2.resize(frame, (width, height)) for frame in video])
        if cv2 is not None:
            video = _label_video_tile(video, labels[len(resized)])
        resized.append(video)
    top = np.concatenate([resized[0], resized[1]], axis=2)
    bottom = np.concatenate([resized[2], resized[3]], axis=2)
    grid = np.concatenate([top, bottom], axis=1)
    _write_video(output_path, grid, fps=fps)
    return {
        "path": str(output_path),
        "layout": [[labels[0], labels[1]], [labels[2], labels[3]]],
        "frames": frames,
        "fps": fps,
        "tile_shape": [height, width],
        "grid_shape": [height * 2, width * 2],
    }


def _label_video_tile(video: np.ndarray, label: str) -> np.ndarray:
    assert cv2 is not None
    labelled = np.ascontiguousarray(video.copy())
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.55, min(labelled.shape[1], labelled.shape[2]) / 720.0)
    thickness = max(1, int(round(scale * 2)))
    margin = max(8, int(round(scale * 12)))
    text_size, baseline = cv2.getTextSize(label, font, scale, thickness)
    box_width = text_size[0] + margin * 2
    box_height = text_size[1] + baseline + margin * 2
    for frame in labelled:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (box_width, box_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, dst=frame)
        cv2.putText(
            frame,
            label,
            (margin, margin + text_size[1]),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return labelled


def _run_timed(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    stats_path: Path,
    repo: str,
    variant: str,
    prompt_index: int,
    generated_frames: int,
    generation_stats_path: Path | None = None,
) -> None:
    start = time.time()
    subprocess.run(command, cwd=cwd, env=env, check=True)
    end = time.time()
    seconds = end - start
    payload: dict[str, object] = {
        "repo": repo,
        "variant": variant,
        "prompt_index": prompt_index,
        "generated_frames": generated_frames,
        "wall_seconds": seconds,
        "end_to_end_fps": generated_frames / seconds if seconds > 0 else None,
        "timing_scope": "end_to_end_command_wall_clock",
        "wall_clock_start_seconds": start,
        "wall_clock_end_seconds": end,
    }
    if generation_stats_path is not None and generation_stats_path.exists():
        generation_payload = _load_json(generation_stats_path)
        if isinstance(generation_payload, dict):
            generation_seconds = _as_float(generation_payload.get("generation_seconds"))
            if generation_seconds is not None:
                payload["generation_seconds"] = generation_seconds
                payload["generation_fps"] = generated_frames / generation_seconds
            payload["generation_timing_scope"] = generation_payload.get("timing_scope")
            payload["generation_stats_path"] = str(generation_stats_path)
    _write_json(stats_path, payload)


def _prompt_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _copy_or_move(source: Path, target: Path) -> None:
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(target))


def _run_vbench_condition(
    *,
    args: argparse.Namespace,
    videos: list[Path],
    prompts: list[str],
    output_dir: Path,
    name: str,
) -> dict[str, Any]:
    if args.vbench_py:
        input_json = output_dir / f"{name}_inputs.json"
        output_json = output_dir / f"{name}_normalized.json"
        _write_json(input_json, {"videos": [str(video) for video in videos], "prompts": prompts})
        command = [
            args.vbench_py,
            str(Path(__file__).resolve()),
            "--vbench_only",
            "--vbench_inputs_json",
            str(input_json),
            "--vbench_output_dir",
            str(output_dir),
            "--vbench_name",
            name,
            "--vbench_output_json",
            str(output_json),
        ]
        subprocess.run(command, cwd=args.fd_dir, env=os.environ.copy(), check=True)
        return _load_json(output_json)
    return run_or_normalize_vbench(videos=videos, prompts=prompts, output_dir=output_dir, name=name)


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    fd_dir = args.fd_dir
    qvg_dir = args.qvg_dir
    fd_asset_dir = fd_dir / "flashdreams/examples/qvg_benchmark_assets"
    matrix_out = args.output_dir
    indices = args.prompt_indices
    frame_count = args.frame_count
    prompts = _prompt_lines(args.prompts)
    for index in indices:
        if index < 0 or index >= len(prompts):
            raise IndexError(f"prompt index {index} out of range for {args.prompts}")

    dirs = {
        "official_bf16": matrix_out / "official_bf16",
        "official_int2": matrix_out / "official_int2",
        "official_bf16_crop": matrix_out / f"official_bf16_first{frame_count}",
        "official_int2_crop": matrix_out / f"official_int2_first{frame_count}",
        "fd_bf16": matrix_out / "flashdreams_bf16",
        "fd_int2": matrix_out / "flashdreams_int2",
        "fd_stats": matrix_out / "flashdreams_stats",
        "run_stats": matrix_out / "run_stats",
        "prompts": matrix_out / "prompts",
        "compare": matrix_out / "compare",
        "vbench": matrix_out / "vbench",
        "grids": matrix_out / "grids",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    base_env = os.environ.copy()
    base_env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    official_env = base_env.copy()
    official_compat_dir = args.official_compat_dir or fd_asset_dir / "official_compat"
    official_pythonpath = [
        str(official_compat_dir),
        "experiments/Self-Forcing",
        ".",
    ]
    if base_env.get("PYTHONPATH"):
        official_pythonpath.append(base_env["PYTHONPATH"])
    official_env["PYTHONPATH"] = ":".join(official_pythonpath)
    official_env["DUMP_KV_LEVEL"] = "0"
    if args.chunkwise_official_noise:
        official_env["QVG_CHUNKWISE_INITIAL_NOISE"] = "1"
    for index in indices:
        prompt_path = dirs["prompts"] / f"prompt{index}.txt"
        prompt_path.write_text(prompts[index] + "\n")
        for variant, quant_args, output_dir in (
            ("bf16", ["--quant_type", "none"], dirs["official_bf16"] / f"prompt{index}"),
            (
                "int2",
                [
                    "--quant_type",
                    "triton-nstages-kmeans-int2",
                    "--cache_num_k_centroids",
                    "256",
                    "--cache_num_v_centroids",
                    "256",
                    "--kmeans_max_iters",
                    "2",
                    "--quant_block_size",
                    "64",
                    "--num_prq_stages",
                    "1",
                ],
                dirs["official_int2"] / f"prompt{index}",
            ),
        ):
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "0-0_ema.mp4"
            generation_stats_path = dirs["run_stats"] / f"official_{variant}_prompt{index}_generation.json"
            if not output_path.exists():
                command = [
                    args.official_py,
                    "-m",
                    "torch.distributed.run",
                    "--nproc_per_node=1",
                    "--standalone",
                    "experiments/Self-Forcing/inference.py",
                    "--config_path",
                    str(args.official_config_path),
                    "--checkpoint_path",
                    str(args.official_checkpoint_path),
                    "--data_path",
                    str(prompt_path),
                    "--output_folder",
                    str(output_dir),
                    "--num_samples",
                    "1",
                    "--num_output_frames",
                    str(args.num_output_frames),
                    "--local_attn_size",
                    str(args.local_attn_size),
                    "--use_ema",
                    "--save_with_index",
                    "--seed",
                    str(args.seed),
                    "--timing_output_path",
                    str(generation_stats_path),
                    *quant_args,
                ]
                _run_timed(
                    command,
                    cwd=qvg_dir,
                    env=official_env,
                    stats_path=dirs["run_stats"] / f"official_{variant}_prompt{index}.json",
                    repo="official",
                    variant=variant,
                    prompt_index=index,
                    generated_frames=frame_count,
                    generation_stats_path=generation_stats_path,
                )
            _merge_generation_stats(
                run_stats_path=dirs["run_stats"] / f"official_{variant}_prompt{index}.json",
                generation_seconds=_generation_seconds(generation_stats_path),
                generated_frames=frame_count,
                timing_scope="official_pipeline_profile_excludes_text_encode_and_video_write",
                generation_stats_path=generation_stats_path,
            )

    fd_env = base_env.copy()
    if args.hf_token is not None:
        fd_env["HF_TOKEN"] = args.hf_token
    elif "HF_TOKEN" not in fd_env:
        token_path = fd_dir / "credentials/hf_token.secret"
        if token_path.exists():
            fd_env["HF_TOKEN"] = token_path.read_text().strip()
    for index in indices:
        bf16_tag = f"{args.tag_prefix}_p{index}_window{args.local_attn_size}_{args.total_blocks}blocks"
        qvg_tag = (
            f"{args.tag_prefix}_p{index}_int2_prerope_s1b64_i2_"
            f"window{args.local_attn_size}_{args.total_blocks}blocks"
        )
        fd_bf16_video = fd_dir / "outputs" / f"causal_wan21_self_forcing_t2v_{bf16_tag}_1gpus.mp4"
        fd_bf16_stats = fd_dir / "outputs" / f"stats_causal_wan21_self_forcing_t2v_{bf16_tag}_1gpus.json"
        fd_qvg_video = fd_dir / "outputs" / f"qvg_wan21_self_forcing_qvg_int2_{args.total_blocks}blocks_seed{args.seed}_nocompile_{qvg_tag}_t2v_1gpus.mp4"
        fd_qvg_stats = fd_dir / "outputs" / f"stats_qvg_wan21_self_forcing_qvg_int2_{args.total_blocks}blocks_seed{args.seed}_nocompile_{qvg_tag}_t2v_1gpus.json"
        if not (dirs["fd_bf16"] / fd_bf16_video.name).exists() and not fd_bf16_video.exists():
            _run_timed(
                [
                    "uv",
                    "run",
                    "--package",
                    "flashdreams",
                    "--extra",
                    "examples",
                    "torchrun",
                    "--standalone",
                    "--nproc_per_node=1",
                    "flashdreams/examples/run_causal_wan21.py",
                    "--config_name",
                    "self_forcing",
                    "--total_blocks",
                    str(args.total_blocks),
                    "--no_compile",
                    "--seed",
                    str(args.seed),
                    "--window_size_t",
                    str(args.local_attn_size),
                    "--prompt_or_txt_path",
                    str(args.prompts),
                    "--prompt_index",
                    str(index),
                    "--output_tag",
                    bf16_tag,
                ],
                cwd=fd_dir,
                env=fd_env,
                stats_path=dirs["run_stats"] / f"flashdreams_bf16_prompt{index}.json",
                repo="flashdreams",
                variant="bf16",
                prompt_index=index,
                generated_frames=frame_count,
            )
        if not (dirs["fd_int2"] / fd_qvg_video.name).exists() and not fd_qvg_video.exists():
            _run_timed(
                [
                    "uv",
                    "run",
                    "--package",
                    "flashdreams",
                    "--extra",
                    "examples",
                    "torchrun",
                    "--standalone",
                    "--nproc_per_node=1",
                    "flashdreams/examples/run_qvg_wan21.py",
                    "--config_name",
                    "self_forcing_qvg_int2",
                    "--total_blocks",
                    str(args.total_blocks),
                    "--no_compile",
                    "--seed",
                    str(args.seed),
                    "--window_size_t",
                    str(args.local_attn_size),
                    "--prompt_or_txt_path",
                    str(args.prompts),
                    "--prompt_index",
                    str(index),
                    "--output_tag",
                    qvg_tag,
                    "--qvg_scale_dtype",
                    "bfloat16",
                    "--qvg_kmeans_seed",
                    str(args.seed),
                    "--qvg_kmeans_max_iters",
                    "2",
                    "--qvg_quant_block_size",
                    "64",
                    "--qvg_cache_num_k_centroids",
                    "256",
                    "--qvg_cache_num_v_centroids",
                    "256",
                    "--qvg_num_prq_stages",
                    "1",
                    "--qvg_compress_every_n_chunks",
                    "8",
                    "--qvg_protected_recent_chunks",
                    "0",
                ],
                cwd=fd_dir,
                env=fd_env,
                stats_path=dirs["run_stats"] / f"flashdreams_int2_prompt{index}.json",
                repo="flashdreams",
                variant="int2",
                prompt_index=index,
                generated_frames=frame_count,
            )
        fd_bf16_stats_target = dirs["fd_stats"] / fd_bf16_stats.name
        fd_qvg_stats_target = dirs["fd_stats"] / fd_qvg_stats.name
        _copy_or_move(fd_bf16_video, dirs["fd_bf16"] / fd_bf16_video.name)
        _copy_or_move(fd_qvg_video, dirs["fd_int2"] / fd_qvg_video.name)
        _copy_or_move(fd_bf16_stats, fd_bf16_stats_target)
        _copy_or_move(fd_qvg_stats, fd_qvg_stats_target)
        _merge_generation_stats(
            run_stats_path=dirs["run_stats"] / f"flashdreams_bf16_prompt{index}.json",
            generation_seconds=_flashdreams_generation_seconds(fd_bf16_stats_target),
            generated_frames=frame_count,
            timing_scope="flashdreams_sum_per_ar_total_ms_excludes_checkpoint_load_and_video_write",
            generation_stats_path=fd_bf16_stats_target,
        )
        _merge_generation_stats(
            run_stats_path=dirs["run_stats"] / f"flashdreams_int2_prompt{index}.json",
            generation_seconds=_flashdreams_generation_seconds(fd_qvg_stats_target),
            generated_frames=frame_count,
            timing_scope="flashdreams_sum_per_ar_total_ms_excludes_checkpoint_load_and_video_write",
            generation_stats_path=fd_qvg_stats_target,
        )

    official_compares = []
    fd_compares = []
    official_bf16_videos = []
    official_int2_videos = []
    fd_bf16_videos = []
    fd_int2_videos = []
    fd_stats = []
    for index in indices:
        official_bf16_crop = dirs["official_bf16_crop"] / f"prompt{index}_bf16_first{frame_count}.mp4"
        official_int2_crop = dirs["official_int2_crop"] / f"prompt{index}_int2_first{frame_count}.mp4"
        crop_video(dirs["official_bf16"] / f"prompt{index}/0-0_ema.mp4", official_bf16_crop, end=frame_count)
        crop_video(dirs["official_int2"] / f"prompt{index}/0-0_ema.mp4", official_int2_crop, end=frame_count)
        fd_bf16 = next(dirs["fd_bf16"].glob(f"*p{index}_window{args.local_attn_size}_{args.total_blocks}blocks*.mp4"))
        fd_int2 = next(dirs["fd_int2"].glob(f"*p{index}_int2_prerope*.mp4"))
        fd_stat = next(dirs["fd_stats"].glob(f"*p{index}_int2_prerope*.json"))
        official_compare = compare_videos(
            official_bf16_crop,
            official_int2_crop,
            output_json=dirs["compare"] / f"official_prompt{index}_int2_vs_bf16_fullvideo.json",
            contact_sheet=dirs["compare"] / f"official_prompt{index}_int2_vs_bf16_contact.jpg",
            lpips_enabled=not args.no_lpips,
            lpips_batch_size=args.lpips_batch_size,
        )
        fd_compare = compare_videos(
            fd_bf16,
            fd_int2,
            output_json=dirs["compare"] / f"flashdreams_prompt{index}_int2_vs_bf16_fullvideo.json",
            contact_sheet=dirs["compare"] / f"flashdreams_prompt{index}_int2_vs_bf16_contact.jpg",
            lpips_enabled=not args.no_lpips,
            lpips_batch_size=args.lpips_batch_size,
        )
        official_compares.append(official_compare)
        fd_compares.append(fd_compare)
        official_bf16_videos.append(official_bf16_crop)
        official_int2_videos.append(official_int2_crop)
        fd_bf16_videos.append(fd_bf16)
        fd_int2_videos.append(fd_int2)
        fd_stats.append(fd_stat)

    selected_prompts = [prompts[index] for index in indices]
    official_bf16_vbench = _run_vbench_condition(
        args=args,
        videos=official_bf16_videos,
        prompts=selected_prompts,
        output_dir=dirs["vbench"] / "official_bf16",
        name=f"official_bf16_{args.name}",
    )
    official_int2_vbench = _run_vbench_condition(
        args=args,
        videos=official_int2_videos,
        prompts=selected_prompts,
        output_dir=dirs["vbench"] / "official_int2",
        name=f"official_int2_{args.name}",
    )
    fd_bf16_vbench = _run_vbench_condition(
        args=args,
        videos=fd_bf16_videos,
        prompts=selected_prompts,
        output_dir=dirs["vbench"] / "flashdreams_bf16",
        name=f"flashdreams_bf16_{args.name}",
    )
    fd_int2_vbench = _run_vbench_condition(
        args=args,
        videos=fd_int2_videos,
        prompts=selected_prompts,
        output_dir=dirs["vbench"] / "flashdreams_int2",
        name=f"flashdreams_int2_{args.name}",
    )
    report = aggregate_report(
        official_compares=official_compares,
        flashdreams_compares=fd_compares,
        official_vbench=official_int2_vbench,
        flashdreams_vbench=fd_int2_vbench,
        official_bf16_vbench=official_bf16_vbench,
        flashdreams_bf16_vbench=fd_bf16_vbench,
        official_compression_ratio=args.official_compression_ratio,
        flashdreams_stats=fd_stats,
        official_bf16_run_stats=[dirs["run_stats"] / f"official_bf16_prompt{index}.json" for index in indices],
        official_qvg_run_stats=[dirs["run_stats"] / f"official_int2_prompt{index}.json" for index in indices],
        flashdreams_bf16_run_stats=[dirs["run_stats"] / f"flashdreams_bf16_prompt{index}.json" for index in indices],
        flashdreams_qvg_run_stats=[dirs["run_stats"] / f"flashdreams_int2_prompt{index}.json" for index in indices],
        quant_label=(
            f"int2_s1b64_i2_window{args.local_attn_size}_"
            f"{args.name}_seed{args.seed}"
        ),
    )
    grids = []
    for row_index, prompt_index in enumerate(indices):
        grids.append(
            write_visual_grid(
                official_bf16_video=official_bf16_videos[row_index],
                official_qvg_video=official_int2_videos[row_index],
                flashdreams_bf16_video=fd_bf16_videos[row_index],
                flashdreams_qvg_video=fd_int2_videos[row_index],
                output_path=dirs["grids"] / f"{args.grid_stem}_prompt{prompt_index}.mp4",
                max_frames=frame_count,
            )
        )
    report["visual_grid"] = {"count": len(grids), "grids": grids}
    output_json = matrix_out / f"qvgbench_{args.name}_8metric_report.json"
    _write_json(output_json, report)
    return report


def _parse_prompt_indices(value: str) -> list[int]:
    return [int(part) for part in value.replace(",", " ").split()]


def _run_vbench_only(args: argparse.Namespace) -> None:
    payload = _load_json(args.vbench_inputs_json)
    videos = [Path(video) for video in payload["videos"]]
    prompts = list(payload["prompts"])
    run_or_normalize_vbench(
        videos=videos,
        prompts=prompts,
        output_dir=args.vbench_output_dir,
        name=args.vbench_name,
        output_json=args.vbench_output_json,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the QVG reproduction benchmark.")
    parser.add_argument("--fd_dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--qvg_dir", type=Path, default=Path("/lustre/fs12/portfolios/nvr/projects/nvr_torontoai_videogen/users/junchenl/Quant-VideoGen"))
    parser.add_argument("--official_py", default="/lustre/fs12/portfolios/nvr/projects/nvr_torontoai_videogen/users/junchenl/Self-Forcing/.venv/bin/python")
    parser.add_argument("--official_config_path", type=Path, default=DEFAULT_OFFICIAL_CONFIG)
    parser.add_argument("--official_checkpoint_path", type=Path, default=Path("ckpts/Self-Forcing/self_forcing_dmd.pt"))
    parser.add_argument("--official_compat_dir", type=Path, default=None)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_QVG_PROMPTS)
    parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "outputs/qvgbench_prompt_matrix_extra_48")
    parser.add_argument("--name", default="prompt_matrix_extra")
    parser.add_argument("--tag_prefix", default="qvgmatrix_extra")
    parser.add_argument("--grid_stem", default="qvgmatrix_extra_grid")
    parser.add_argument("--prompt_indices", type=_parse_prompt_indices, default=[0, 1, 2, 3])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total_blocks", type=int, default=16)
    parser.add_argument("--frame_count", type=int, default=189)
    parser.add_argument("--num_output_frames", type=int, default=48)
    parser.add_argument("--local_attn_size", type=int, default=180)
    parser.add_argument("--official_compression_ratio", type=float, default=6.60)
    parser.add_argument("--lpips_batch_size", type=int, default=16)
    parser.add_argument("--no_lpips", action="store_true")
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--vbench_py", default=os.environ.get("VBENCH_PY"))
    parser.add_argument(
        "--chunkwise_official_noise",
        "--chunkwise-official-noise",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--vbench_only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--vbench_inputs_json", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--vbench_output_dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--vbench_name", help=argparse.SUPPRESS)
    parser.add_argument("--vbench_output_json", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.vbench_only:
        _run_vbench_only(args)
        return
    report = run_benchmark(args)
    print(json.dumps({"all_required_metrics_pass": report["all_required_metrics_pass"]}, indent=2))


if __name__ == "__main__":
    main()
