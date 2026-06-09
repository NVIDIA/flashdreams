# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic video-quality metrics for regression sentinels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import numpy.typing as npt

from flashdreams.quality.video_quality.manifest import Window

RGBVideo = npt.NDArray[np.uint8]
CORE_METRIC_GROUPS = frozenset({"decode_metadata", "grey_blank", "sharpness", "stripe"})


@dataclass(frozen=True)
class VideoMetricsInput:
    """Video frames plus optional timing metadata."""

    frames: RGBVideo
    fps: float | None = None


def synthetic_video(
    pattern: str,
    *,
    frames: int = 16,
    height: int = 64,
    width: int = 64,
    fps: float = 8,
    seed: int = 0,
) -> VideoMetricsInput:
    """Generate a tiny deterministic video for metric validation."""
    if frames <= 0 or height <= 0 or width <= 0:
        raise ValueError("frames, height, and width must be positive")

    if pattern == "grey_blank":
        video = np.full((frames, height, width, 3), 128, dtype=np.uint8)
    elif pattern == "blurry_gradient":
        video = _blurry_gradient(frames=frames, height=height, width=width)
    elif pattern == "horizontal_stripes":
        video = _axis_stripes(frames=frames, height=height, width=width, axis="row")
    elif pattern == "vertical_stripes":
        video = _axis_stripes(frames=frames, height=height, width=width, axis="col")
    elif pattern == "textured_motion":
        video = _textured_motion(frames=frames, height=height, width=width, seed=seed)
    else:
        raise ValueError(f"Unknown synthetic video pattern: {pattern!r}")

    return VideoMetricsInput(frames=video, fps=float(fps))


def compute_video_metrics(
    video: RGBVideo,
    *,
    fps: float | None = None,
    windows: Mapping[str, Window] | None = None,
    metric_groups: tuple[str, ...] | None = None,
) -> dict[str, float | int | bool | None]:
    """Compute cheap deterministic quality metrics for an RGB video."""
    selected_groups = _selected_metric_groups(metric_groups)
    frames = _as_uint8_rgb(video)
    frame_count, height, width, _ = frames.shape
    metrics: dict[str, float | int | bool | None] = {
        "decode_ok": True,
        "frame_count": int(frame_count),
        "fps": float(fps) if fps else None,
        "duration_s": float(frame_count / fps) if fps else None,
        "height": int(height),
        "width": int(width),
    }
    metrics.update(_content_metrics(frames, metric_groups=selected_groups))

    if windows and fps:
        window_metrics: dict[str, dict[str, float]] = {}
        for name, window in windows.items():
            start = max(0, int(round(window.start_s * fps)))
            end = min(frame_count, int(round(window.end_s * fps)))
            if end <= start:
                continue
            values = _content_metrics(frames[start:end], metric_groups=selected_groups)
            numeric_values = {
                key: value for key, value in values.items() if isinstance(value, float)
            }
            window_metrics[name] = numeric_values
            for key, value in numeric_values.items():
                metrics[f"{key}_{name}"] = value

        head = window_metrics.get("head")
        tail = window_metrics.get("tail")
        if head and tail:
            head_sharpness = head.get("laplacian_variance", 0.0)
            tail_sharpness = tail.get("laplacian_variance", 0.0)
            metrics["sharpness_tail_head_ratio"] = (
                tail_sharpness / head_sharpness if head_sharpness > 0 else None
            )

    return metrics


def _selected_metric_groups(metric_groups: tuple[str, ...] | None) -> frozenset[str]:
    if not metric_groups:
        return CORE_METRIC_GROUPS
    selected_groups = frozenset(metric_groups)
    unknown_groups = selected_groups - CORE_METRIC_GROUPS
    if unknown_groups:
        raise ValueError(f"Unsupported metric group(s): {sorted(unknown_groups)}")
    return selected_groups | {"decode_metadata"}


def _content_metrics(
    frames: RGBVideo, *, metric_groups: frozenset[str]
) -> dict[str, float]:
    rgb = frames.astype(np.float32) / 255.0
    luma = _luma(rgb)
    metrics: dict[str, float] = {}

    if "grey_blank" in metric_groups:
        rgb_range = np.max(rgb, axis=-1) - np.min(rgb, axis=-1)
        channel_max = np.max(rgb, axis=-1)
        saturation = np.divide(
            rgb_range,
            np.maximum(channel_max, 1.0e-6),
            out=np.zeros_like(rgb_range),
            where=channel_max > 1.0e-6,
        )
        metrics.update(
            {
                "luma_std": float(np.std(luma)),
                "rgb_channel_std": float(np.mean(np.std(rgb, axis=(0, 1, 2)))),
                "saturation_mean": float(np.mean(saturation)),
                "grey_pixel_ratio": float(np.mean(rgb_range <= 0.025)),
            }
        )

    if "sharpness" in metric_groups:
        laplacian = _laplacian(luma)
        grad_y = np.diff(luma, axis=1)
        grad_x = np.diff(luma, axis=2)
        metrics.update(
            {
                "laplacian_variance": float(np.var(laplacian)),
                "high_frequency_energy": float(
                    0.5 * (np.mean(np.abs(grad_y)) + np.mean(np.abs(grad_x)))
                ),
            }
        )

    if "stripe" in metric_groups:
        row_profile = np.mean(luma, axis=(0, 2))
        col_profile = np.mean(luma, axis=(0, 1))
        row_fft = _periodic_energy_ratio(row_profile)
        col_fft = _periodic_energy_ratio(col_profile)
        metrics.update(
            {
                "fft_axis_energy_ratio": float(max(row_fft, col_fft)),
                "row_autocorr_peak": _autocorr_peak(row_profile),
                "col_autocorr_peak": _autocorr_peak(col_profile),
            }
        )

    return metrics


def _luma(rgb: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def _laplacian(luma: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    if luma.shape[1] < 3 or luma.shape[2] < 3:
        return np.zeros_like(luma)
    center = luma[:, 1:-1, 1:-1]
    return (
        -4.0 * center
        + luma[:, :-2, 1:-1]
        + luma[:, 2:, 1:-1]
        + luma[:, 1:-1, :-2]
        + luma[:, 1:-1, 2:]
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
    periodic_energy = float(np.max(spectrum[2:]))
    return periodic_energy / float(np.sum(spectrum[1:]))


def _autocorr_peak(profile: npt.NDArray[np.float32]) -> float:
    profile = np.asarray(profile, dtype=np.float32)
    if _periodic_energy_ratio(profile) < 0.5:
        return 0.0
    profile = profile - float(np.mean(profile))
    denom = float(np.dot(profile, profile))
    if denom <= 1.0e-12 or profile.size < 4:
        return 0.0
    corr = np.correlate(profile, profile, mode="full")[profile.size - 1 :] / denom
    upper = max(2, profile.size // 2)
    if upper <= 2:
        return 0.0
    return float(np.max(np.abs(corr[2:upper])))


def _textured_motion(*, frames: int, height: int, width: int, seed: int) -> RGBVideo:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    y = np.linspace(0, 80, height, dtype=np.float32)[:, None]
    x = np.linspace(0, 120, width, dtype=np.float32)[None, :]
    video = np.empty((frames, height, width, 3), dtype=np.uint8)
    for idx in range(frames):
        rolled = np.roll(base, shift=idx * 3, axis=1).astype(np.float32)
        rolled[..., 0] = (rolled[..., 0] + x + idx * 4) % 256
        rolled[..., 1] = (rolled[..., 1] + y + idx * 7) % 256
        video[idx] = np.clip(rolled, 0, 255).astype(np.uint8)
    return video


def _blurry_gradient(*, frames: int, height: int, width: int) -> RGBVideo:
    x = np.linspace(0, 255, width, dtype=np.float32)[None, :]
    y = np.linspace(0, 255, height, dtype=np.float32)[:, None]
    video = np.empty((frames, height, width, 3), dtype=np.uint8)
    for idx in range(frames):
        phase = idx * 8.0
        red = np.broadcast_to((x + phase) % 256, (height, width))
        green = np.broadcast_to((y + phase * 0.5) % 256, (height, width))
        blue = (red * 0.35 + green * 0.65) % 256
        video[idx] = np.stack([red, green, blue], axis=-1).astype(np.uint8)
    return video


def _axis_stripes(*, frames: int, height: int, width: int, axis: str) -> RGBVideo:
    yy = np.arange(height)[:, None]
    xx = np.arange(width)[None, :]
    band_source = yy if axis == "row" else xx
    bands = ((band_source // 4) % 2).astype(np.float32)
    bands = np.broadcast_to(bands, (height, width))
    video = np.empty((frames, height, width, 3), dtype=np.uint8)
    for idx in range(frames):
        shifted = np.roll(bands, shift=idx % 4, axis=0 if axis == "row" else 1)
        red = np.where(shifted > 0, 240, 30)
        green = np.where(shifted > 0, 35, 220)
        blue = np.where(shifted > 0, 35, 220)
        video[idx] = np.stack([red, green, blue], axis=-1).astype(np.uint8)
    return video


def _as_uint8_rgb(video: np.ndarray) -> RGBVideo:
    if video.ndim != 4 or video.shape[-1] < 3:
        raise ValueError(f"Expected video shape [T,H,W,C>=3], got {video.shape}")
    rgb = video[..., :3]
    if rgb.dtype == np.uint8:
        return rgb
    if np.issubdtype(rgb.dtype, np.floating) and np.nanmax(rgb) <= 1.0:
        rgb = rgb * 255.0
    return np.clip(rgb, 0, 255).astype(np.uint8)
