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

"""Local MP4 quality comparison helpers for benchmark reports."""

from __future__ import annotations

import json
import math
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

CompareRegion = Literal["full", "bottom-half"]
RGBVideo = npt.NDArray[np.uint8]


@dataclass(frozen=True, kw_only=True)
class QualityBaselineConfig:
    """Settings for comparing generated MP4s against a previous benchmark run."""

    baseline_dir: Path
    sample_count: int = 8
    frame_indices: tuple[int, ...] | None = None
    compare_region: CompareRegion | None = None
    compute_flip: bool = False

    def __post_init__(self) -> None:
        if self.sample_count <= 0:
            raise ValueError("quality sample_count must be positive")
        if self.compare_region not in (None, "full", "bottom-half"):
            raise ValueError(
                "quality compare_region must be 'full', 'bottom-half', or None"
            )

    def to_manifest(self) -> dict[str, Any]:
        return {
            "baseline_dir": str(self.baseline_dir),
            "sample_count": self.sample_count,
            "frame_indices": (
                None if self.frame_indices is None else list(self.frame_indices)
            ),
            "compare_region": self.compare_region or "scenario-default",
            "compute_flip": self.compute_flip,
        }


def parse_quality_frame_indices(value: str | None) -> tuple[int, ...] | None:
    """Parse a comma-separated frame-index list without importing video deps."""

    if value is None or not value.strip():
        return None
    indices = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not indices:
        return None
    if any(index < 0 for index in indices):
        raise ValueError(f"Frame indices must be non-negative, got {indices}")
    return indices


def run_baseline_clip_compare(
    *,
    scenario_id: str,
    candidate_video: Path | None,
    quality_dir: Path,
    config: QualityBaselineConfig,
) -> dict[str, Any]:
    """Compare one scenario MP4 against its matching baseline MP4.

    The baseline directory can be either a previous benchmark run root
    containing ``manifest.json`` or a flat directory with ``<scenario-id>.mp4``.
    """

    quality_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = quality_dir / "metrics.json"
    log_path = quality_dir / "command.log"
    result: dict[str, Any] = {
        "id": "baseline-clip-compare",
        "status": "skipped",
        "metrics_path": metrics_path,
        "log_path": log_path,
    }
    compare_region = config.compare_region or "full"
    log_lines = [
        "baseline clip quality comparison",
        f"scenario_id: {scenario_id}",
        f"baseline_dir: {config.baseline_dir}",
        f"compare_region: {compare_region}",
        f"sample_count: {config.sample_count}",
        f"frame_indices: {config.frame_indices}",
        f"compute_flip: {config.compute_flip}",
    ]

    if candidate_video is None:
        reason = "scenario did not produce an MP4 artifact"
        result["reason"] = reason
        log_lines.append(f"skipped: {reason}")
        _write_log(log_path, log_lines)
        return result

    baseline_video = find_baseline_video(config.baseline_dir, scenario_id)
    if baseline_video is None:
        reason = f"no baseline MP4 found for scenario {scenario_id!r}"
        result["reason"] = reason
        log_lines.append(f"candidate_video: {candidate_video}")
        log_lines.append(f"skipped: {reason}")
        _write_log(log_path, log_lines)
        return result

    result["candidate_video"] = candidate_video
    result["baseline_video"] = baseline_video
    log_lines.append(f"candidate_video: {candidate_video}")
    log_lines.append(f"baseline_video: {baseline_video}")

    warnings: list[str] = []
    try:
        reference = _read_video_rgb(baseline_video)
        candidate = _read_video_rgb(candidate_video)
        if compare_region == "bottom-half":
            reference = _bottom_half(reference)
            candidate = _bottom_half(candidate)
        try:
            comparison = _compare_video_arrays(
                reference,
                candidate,
                frame_indices=config.frame_indices,
                sample_count=config.sample_count,
                compute_flip=config.compute_flip,
            )
        except ImportError as exc:
            if not config.compute_flip:
                raise
            warning = f"FLIP unavailable; continuing without FLIP: {exc}"
            warnings.append(warning)
            log_lines.append(f"warning: {warning}")
            comparison = _compare_video_arrays(
                reference,
                candidate,
                frame_indices=config.frame_indices,
                sample_count=config.sample_count,
                compute_flip=False,
            )
    except Exception as exc:  # noqa: BLE001 - report quality errors without masking runs.
        result["status"] = "fail"
        result["reason"] = str(exc)
        log_lines.append(f"failed: {exc}")
        log_lines.append(traceback.format_exc())
        _write_log(log_path, log_lines)
        return result

    metrics, diagnostics = _quality_metrics(
        comparison=comparison,
        reference=reference,
        candidate=candidate,
    )
    payload = {
        "schema_version": 1,
        "baseline_video": str(baseline_video),
        "candidate_video": str(candidate_video),
        "compare_region": compare_region,
        "sample_count": config.sample_count,
        "frame_indices": list(comparison.frame_indices),
        "reference_frame_count": comparison.reference_frame_count,
        "candidate_frame_count": comparison.candidate_frame_count,
        "flip_requested": config.compute_flip,
        "flip_computed": comparison.mean_flip is not None,
        "metrics": metrics,
        "diagnostics": diagnostics,
    }
    if warnings:
        payload["warnings"] = warnings
        result["warnings"] = warnings
    metrics_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    result["status"] = "pass"
    log_lines.append(_format_clip_comparison(comparison))
    log_lines.append(f"quality_score={metrics['quality_score']:.4f}")
    log_lines.append(
        f"quality_similarity_score={metrics['quality_similarity_score']:.4f}"
    )
    log_lines.append(
        f"quality_visual_sanity_score={metrics['quality_visual_sanity_score']:.4f}"
    )
    _write_log(log_path, log_lines)
    return result


def find_baseline_video(baseline_dir: Path, scenario_id: str) -> Path | None:
    """Find the MP4 for ``scenario_id`` in a previous run or flat baseline dir."""

    baseline_dir = baseline_dir.expanduser().resolve()
    manifest_video = _baseline_from_manifest(
        baseline_dir / "manifest.json", scenario_id
    )
    if manifest_video is not None:
        return manifest_video

    flat_candidate = baseline_dir / f"{scenario_id}.mp4"
    if flat_candidate.is_file():
        return flat_candidate

    for directory in (
        baseline_dir / "scenarios" / scenario_id,
        baseline_dir / scenario_id,
    ):
        for path in sorted(directory.glob("*.mp4")):
            if path.is_file():
                return path
    return None


def _baseline_from_manifest(manifest_path: Path, scenario_id: str) -> Path | None:
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    scenarios = manifest.get("scenarios", [])
    if not isinstance(scenarios, list):
        return None
    for scenario in scenarios:
        if not isinstance(scenario, dict) or scenario.get("id") != scenario_id:
            continue
        artifacts = scenario.get("artifacts", {})
        if not isinstance(artifacts, dict):
            continue
        videos = artifacts.get("videos", [])
        if not isinstance(videos, list):
            continue
        for video in videos:
            if not isinstance(video, str):
                continue
            path = manifest_path.parent / video
            if path.is_file():
                return path
    return None


def _quality_metrics(
    *,
    comparison: Any,
    reference: RGBVideo,
    candidate: RGBVideo,
) -> tuple[dict[str, float | int], dict[str, Any]]:
    sampled_reference = _sampled_frames(reference, comparison.frame_indices)
    sampled_candidate = _sampled_frames(candidate, comparison.frame_indices)
    visual_metrics, visual_diagnostics = _visual_sanity_metrics(sampled_candidate)
    ssim_score = _mean_global_ssim(sampled_reference, sampled_candidate)
    mean_abs_score = _bounded_similarity(comparison.mean_abs)
    rmse_score = _bounded_similarity(comparison.rmse)
    psnr_score = _psnr_score(comparison.psnr_db)
    frame_count_score = _frame_count_score(
        comparison.reference_frame_count,
        comparison.candidate_frame_count,
    )
    similarity_score = _weighted_score(
        (
            (ssim_score, 0.45),
            (rmse_score, 0.30),
            (psnr_score, 0.15),
            (frame_count_score, 0.10),
        )
    )
    quality_score = _weighted_score(
        (
            (similarity_score, 0.70),
            (visual_metrics["quality_visual_sanity_score"], 0.30),
        )
    )
    metrics: dict[str, float | int] = {
        "quality_score": quality_score,
        "quality_similarity_score": similarity_score,
        "quality_visual_sanity_score": visual_metrics["quality_visual_sanity_score"],
        "quality_temporal_score": visual_metrics["quality_temporal_score"],
        "quality_ssim_score": ssim_score,
        "quality_mean_abs": comparison.mean_abs,
        "quality_rmse": comparison.rmse,
        "quality_psnr_db": _finite_or_cap(comparison.psnr_db),
    }
    if comparison.mean_flip is not None:
        metrics["quality_mean_flip"] = comparison.mean_flip
        metrics["quality_flip_score"] = max(0.0, min(1.0, 1.0 - comparison.mean_flip))
    max_frame_flip = comparison.max_frame_flip
    if max_frame_flip is not None:
        metrics["quality_max_frame_flip"] = max_frame_flip
    diagnostics: dict[str, Any] = {
        "quality_mean_abs_score": mean_abs_score,
        "quality_rmse_score": rmse_score,
        "quality_psnr_score": psnr_score,
        "quality_frame_count_score": frame_count_score,
        "quality_max_frame_mean_abs": comparison.max_frame_mean_abs,
        "quality_max_frame_rmse": comparison.max_frame_rmse,
        "visual_sanity": visual_diagnostics,
        "frame_metrics": [
            {
                "frame_index": frame.frame_index,
                "mean_abs": frame.mean_abs,
                "rmse": frame.rmse,
                "psnr_db": _finite_or_cap(frame.psnr_db),
                "mean_flip": frame.mean_flip,
            }
            for frame in comparison.frame_metrics
        ],
    }
    return metrics, diagnostics


def _sampled_frames(video: RGBVideo, frame_indices: tuple[int, ...]) -> RGBVideo:
    return _as_uint8_rgb(video)[list(frame_indices)]


def _visual_sanity_metrics(
    frames: RGBVideo,
) -> tuple[dict[str, float], dict[str, float]]:
    rgb = _as_uint8_rgb(frames).astype(np.float32) / 255.0
    luma = _luma(rgb)
    luma_std = float(np.std(luma))
    rgb_range = np.max(rgb, axis=-1) - np.min(rgb, axis=-1)
    channel_max = np.max(rgb, axis=-1)
    saturation = np.divide(
        rgb_range,
        np.maximum(channel_max, 1.0e-6),
        out=np.zeros_like(rgb_range),
        where=channel_max > 1.0e-6,
    )
    saturation_mean = float(np.mean(saturation))
    grey_pixel_ratio = float(np.mean(rgb_range <= 0.025))
    laplacian_variance = _laplacian_variance(luma)
    high_frequency_energy = _high_frequency_energy(luma)
    fft_axis_energy_ratio = _fft_axis_energy_ratio(luma)
    temporal_delta_mean_abs, temporal_jitter_ratio = _temporal_diagnostics(luma)

    luma_std_score = _clamp01((luma_std - 0.015) / 0.120)
    saturation_score = _clamp01(saturation_mean / 0.080)
    grey_penalty = 1.0 - _clamp01((grey_pixel_ratio - 0.980) / 0.020)
    blank_score = _clamp01(
        ((0.70 * luma_std_score) + (0.30 * saturation_score)) * grey_penalty
    )
    sharpness_score = _clamp01(laplacian_variance / 0.0025)
    stripe_score = 1.0 - _clamp01((fft_axis_energy_ratio - 0.55) / 0.35)
    temporal_score = 1.0 - _clamp01((temporal_jitter_ratio - 0.35) / 1.15)
    visual_sanity_score = _weighted_score(
        (
            (blank_score, 0.35),
            (sharpness_score, 0.30),
            (stripe_score, 0.20),
            (temporal_score, 0.15),
        )
    )
    return (
        {
            "quality_visual_sanity_score": visual_sanity_score,
            "quality_temporal_score": temporal_score,
        },
        {
            "blank_score": blank_score,
            "sharpness_score": sharpness_score,
            "stripe_artifact_score": stripe_score,
            "temporal_score": temporal_score,
            "luma_std": luma_std,
            "saturation_mean": saturation_mean,
            "grey_pixel_ratio": grey_pixel_ratio,
            "laplacian_variance": laplacian_variance,
            "high_frequency_energy": high_frequency_energy,
            "fft_axis_energy_ratio": fft_axis_energy_ratio,
            "temporal_delta_mean_abs": temporal_delta_mean_abs,
            "temporal_jitter_ratio": temporal_jitter_ratio,
        },
    )


def _mean_global_ssim(reference: RGBVideo, candidate: RGBVideo) -> float:
    ref_luma = _luma(_as_uint8_rgb(reference).astype(np.float32))
    cand_luma = _luma(_as_uint8_rgb(candidate).astype(np.float32))
    axes = (1, 2)
    mu_ref = np.mean(ref_luma, axis=axes)
    mu_cand = np.mean(cand_luma, axis=axes)
    ref_centered = ref_luma - mu_ref[:, None, None]
    cand_centered = cand_luma - mu_cand[:, None, None]
    var_ref = np.mean(np.square(ref_centered), axis=axes)
    var_cand = np.mean(np.square(cand_centered), axis=axes)
    covariance = np.mean(ref_centered * cand_centered, axis=axes)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    numerator = (2.0 * mu_ref * mu_cand + c1) * (2.0 * covariance + c2)
    denominator = (np.square(mu_ref) + np.square(mu_cand) + c1) * (
        var_ref + var_cand + c2
    )
    scores = np.divide(
        numerator,
        denominator,
        out=np.ones_like(numerator, dtype=np.float32),
        where=denominator > 0.0,
    )
    return _clamp01(float(np.mean(scores)))


def _luma(rgb: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def _laplacian_variance(luma: npt.NDArray[np.float32]) -> float:
    if luma.shape[1] < 3 or luma.shape[2] < 3:
        return 0.0
    center = luma[:, 1:-1, 1:-1]
    laplacian = (
        -4.0 * center
        + luma[:, :-2, 1:-1]
        + luma[:, 2:, 1:-1]
        + luma[:, 1:-1, :-2]
        + luma[:, 1:-1, 2:]
    )
    return float(np.var(laplacian))


def _high_frequency_energy(luma: npt.NDArray[np.float32]) -> float:
    if luma.shape[1] < 2 or luma.shape[2] < 2:
        return 0.0
    grad_y = np.diff(luma, axis=1)
    grad_x = np.diff(luma, axis=2)
    return float(0.5 * (np.mean(np.abs(grad_y)) + np.mean(np.abs(grad_x))))


def _fft_axis_energy_ratio(luma: npt.NDArray[np.float32]) -> float:
    row_profile = np.mean(luma, axis=(0, 2))
    col_profile = np.mean(luma, axis=(0, 1))
    return float(
        max(
            _periodic_energy_ratio(row_profile),
            _periodic_energy_ratio(col_profile),
        )
    )


def _periodic_energy_ratio(profile: npt.NDArray[np.float32]) -> float:
    profile = np.asarray(profile, dtype=np.float32)
    profile = profile - float(np.mean(profile))
    total = float(np.sum(np.square(profile)))
    if total <= 1.0e-12 or profile.size < 4:
        return 0.0
    spectrum = np.abs(np.fft.rfft(profile)) ** 2
    if spectrum.size <= 2:
        return 0.0
    return float(np.max(spectrum[2:]) / np.sum(spectrum[1:]))


def _temporal_diagnostics(luma: npt.NDArray[np.float32]) -> tuple[float, float]:
    if luma.shape[0] < 2:
        return 0.0, 0.0
    frame_deltas = np.mean(np.abs(np.diff(luma, axis=0)), axis=(1, 2))
    delta_mean = float(np.mean(frame_deltas))
    if frame_deltas.shape[0] < 2 or delta_mean <= 0.003:
        return delta_mean, 0.0
    jitter = float(np.mean(np.abs(np.diff(frame_deltas))))
    return delta_mean, jitter / max(delta_mean, 0.003)


def _frame_count_score(reference_count: int, candidate_count: int) -> float:
    max_count = max(reference_count, candidate_count)
    if max_count <= 0:
        return 0.0
    return _clamp01(1.0 - abs(reference_count - candidate_count) / max_count)


def _psnr_score(psnr_db: float) -> float:
    if math.isinf(psnr_db):
        return 1.0
    return _clamp01((psnr_db - 20.0) / 20.0)


def _weighted_score(values: tuple[tuple[float, float], ...]) -> float:
    weighted_sum = sum(_clamp01(value) * weight for value, weight in values)
    weight_sum = sum(weight for _, weight in values)
    if weight_sum <= 0:
        return 0.0
    return _clamp01(weighted_sum / weight_sum)


def _bounded_similarity(error_255: float) -> float:
    return _clamp01(1.0 - float(error_255) / 255.0)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _finite_or_cap(value: float) -> float:
    if math.isinf(value):
        return 100.0
    return value


def _read_video_rgb(path: Path) -> Any:
    from flashdreams.quality.clip_compare import read_video_rgb  # noqa: PLC0415

    return read_video_rgb(path)


def _bottom_half(video: Any) -> Any:
    from flashdreams.quality.clip_compare import bottom_half  # noqa: PLC0415

    return bottom_half(video)


def _compare_video_arrays(
    reference: Any,
    candidate: Any,
    *,
    frame_indices: tuple[int, ...] | None,
    sample_count: int,
    compute_flip: bool,
) -> Any:
    from flashdreams.quality.clip_compare import compare_video_arrays  # noqa: PLC0415

    return compare_video_arrays(
        reference,
        candidate,
        frame_indices=frame_indices,
        sample_count=sample_count,
        compute_flip=compute_flip,
    )


def _format_clip_comparison(comparison: Any) -> str:
    from flashdreams.quality.clip_compare import format_clip_comparison  # noqa: PLC0415

    return format_clip_comparison(comparison)


def _as_uint8_rgb(video: Any) -> RGBVideo:
    array = np.asarray(video)
    if array.ndim != 4 or array.shape[-1] < 3:
        raise ValueError(f"Expected video shape [T,H,W,C>=3], got {array.shape}")
    rgb = array[..., :3]
    if rgb.dtype == np.uint8:
        return rgb
    if np.issubdtype(rgb.dtype, np.floating) and np.nanmax(rgb) <= 1.0:
        rgb = rgb * 255.0
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _write_log(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
