# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frame-sampled RGB clip comparison for golden-output regression tests."""

from __future__ import annotations

import importlib
import importlib.util
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import numpy.typing as npt

RGBVideo = npt.NDArray[np.uint8]


@dataclass(frozen=True)
class ClipFrameMetrics:
    """Per-sampled-frame image-difference metrics in RGB uint8 space."""

    frame_index: int
    mean_abs: float
    rmse: float
    psnr_db: float
    mean_flip: float | None = None


@dataclass(frozen=True)
class ClipComparisonThresholds:
    """Bounds used by :func:`assert_clip_within_thresholds`.

    ``mean_abs`` and ``rmse`` are measured in 8-bit RGB intensity units
    (0..255). ``mean_flip`` is optional and only computed when one of the
    FLIP thresholds is non-``None``.
    """

    max_mean_abs: float = 4.0
    max_rmse: float = 8.0
    min_psnr_db: float = 30.0
    max_frame_mean_abs: float = 8.0
    max_frame_rmse: float = 14.0
    max_mean_flip: float | None = 0.070
    max_frame_flip: float | None = 0.080
    require_same_frame_count: bool = True


@dataclass(frozen=True)
class ClipComparisonResult:
    """Aggregate and per-frame metrics for one reference/candidate comparison."""

    reference_frame_count: int
    candidate_frame_count: int
    frame_indices: tuple[int, ...]
    mean_abs: float
    rmse: float
    psnr_db: float
    frame_metrics: tuple[ClipFrameMetrics, ...]
    mean_flip: float | None = None

    @property
    def max_frame_mean_abs(self) -> float:
        return max((m.mean_abs for m in self.frame_metrics), default=0.0)

    @property
    def max_frame_rmse(self) -> float:
        return max((m.rmse for m in self.frame_metrics), default=0.0)

    @property
    def max_frame_flip(self) -> float | None:
        values = [m.mean_flip for m in self.frame_metrics if m.mean_flip is not None]
        return max(values) if values else None


def read_video_rgb(path: Path | str) -> RGBVideo:
    """Read an RGB video file as ``uint8 [T, H, W, 3]``."""
    try:
        import mediapy as media  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - import-time gate
        raise ImportError(
            "Clip comparison needs mediapy to read videos. Install the dev or "
            "runner extras before running quality regression tests."
        ) from exc

    video = media.read_video(str(path))
    if video.ndim != 4 or video.shape[-1] < 3:
        raise ValueError(f"Expected RGB/RGBA video [T,H,W,C], got shape {video.shape}")
    return _as_uint8_rgb(video[..., :3])


def parse_frame_indices(value: str | None) -> tuple[int, ...] | None:
    """Parse a comma-separated frame-index list from an env var style string."""
    if value is None or not value.strip():
        return None
    indices = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not indices:
        return None
    if any(idx < 0 for idx in indices):
        raise ValueError(f"Frame indices must be non-negative, got {indices}")
    return indices


def select_frame_indices(
    reference_frame_count: int,
    candidate_frame_count: int,
    *,
    frame_indices: Iterable[int] | None = None,
    sample_count: int = 8,
) -> tuple[int, ...]:
    """Return explicit or evenly-spaced indices valid for both clips."""
    if reference_frame_count <= 0:
        raise ValueError("reference clip has no frames")
    if candidate_frame_count <= 0:
        raise ValueError("candidate clip has no frames")

    max_common = min(reference_frame_count, candidate_frame_count)
    if frame_indices is not None:
        indices = tuple(dict.fromkeys(frame_indices))
        if not indices:
            raise ValueError("frame_indices was empty")
        out_of_bounds = [idx for idx in indices if idx >= max_common]
        if out_of_bounds:
            raise ValueError(
                f"Frame indices {out_of_bounds} exceed common frame count "
                f"{max_common}; reference has {reference_frame_count}, "
                f"candidate has {candidate_frame_count}."
            )
        return indices

    sample_count = max(1, min(sample_count, max_common))
    indices_np = np.linspace(0, max_common - 1, num=sample_count, dtype=np.int64)
    return tuple(int(idx) for idx in dict.fromkeys(indices_np.tolist()))


def bottom_half(video: RGBVideo) -> RGBVideo:
    """Return the lower half of a ``[T,H,W,3]`` video.

    Omnidreams runner MP4s stack the HDMap condition over the generated frames,
    so this helper extracts the generated region.
    """
    _validate_video(video, name="video")
    return video[:, video.shape[1] // 2 :, :, :]


def compare_video_arrays(
    reference: RGBVideo,
    candidate: RGBVideo,
    *,
    frame_indices: Iterable[int] | None = None,
    sample_count: int = 8,
    compute_flip: bool = False,
) -> ClipComparisonResult:
    """Compare sampled frames from two RGB clips."""
    reference = _as_uint8_rgb(reference)
    candidate = _as_uint8_rgb(candidate)
    _validate_compatible_spatial_shape(reference, candidate)

    indices = select_frame_indices(
        reference.shape[0],
        candidate.shape[0],
        frame_indices=frame_indices,
        sample_count=sample_count,
    )
    ref = reference[list(indices)].astype(np.float32)
    cand = candidate[list(indices)].astype(np.float32)
    diff = cand - ref
    mean_abs = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(np.square(diff))))
    psnr_db = _psnr_db(rmse)

    frame_metrics = []
    flip_scores = (
        _compute_flip_scores(reference, candidate, indices) if compute_flip else None
    )
    for metric_pos, frame_idx in enumerate(indices):
        frame_diff = diff[metric_pos]
        frame_rmse = float(np.sqrt(np.mean(np.square(frame_diff))))
        frame_metrics.append(
            ClipFrameMetrics(
                frame_index=frame_idx,
                mean_abs=float(np.mean(np.abs(frame_diff))),
                rmse=frame_rmse,
                psnr_db=_psnr_db(frame_rmse),
                mean_flip=flip_scores[metric_pos] if flip_scores is not None else None,
            )
        )

    return ClipComparisonResult(
        reference_frame_count=int(reference.shape[0]),
        candidate_frame_count=int(candidate.shape[0]),
        frame_indices=indices,
        mean_abs=mean_abs,
        rmse=rmse,
        psnr_db=psnr_db,
        frame_metrics=tuple(frame_metrics),
        mean_flip=float(np.mean(flip_scores)) if flip_scores is not None else None,
    )


def assert_clip_within_thresholds(
    reference: RGBVideo,
    candidate: RGBVideo,
    *,
    thresholds: ClipComparisonThresholds = ClipComparisonThresholds(),
    frame_indices: Iterable[int] | None = None,
    sample_count: int = 8,
) -> ClipComparisonResult:
    """Compare two clips and raise ``AssertionError`` when any bound is crossed."""
    compute_flip = (
        thresholds.max_mean_flip is not None or thresholds.max_frame_flip is not None
    )
    result = compare_video_arrays(
        reference,
        candidate,
        frame_indices=frame_indices,
        sample_count=sample_count,
        compute_flip=compute_flip,
    )
    failures = _threshold_failures(result, thresholds)
    if failures:
        details = "\n".join(f"- {failure}" for failure in failures)
        raise AssertionError(
            f"Clip quality regression detected:\n{details}\n"
            f"{format_clip_comparison(result)}"
        )
    return result


def format_clip_comparison(result: ClipComparisonResult) -> str:
    """Return a compact human-readable metric summary."""
    parts = [
        "Compared "
        f"{len(result.frame_indices)} sampled frame(s) "
        f"{result.frame_indices} from reference "
        f"{result.reference_frame_count} frame(s) and candidate "
        f"{result.candidate_frame_count} frame(s):",
        f"mean_abs={result.mean_abs:.4f}",
        f"rmse={result.rmse:.4f}",
        f"psnr={result.psnr_db:.2f} dB",
        f"max_frame_mean_abs={result.max_frame_mean_abs:.4f}",
        f"max_frame_rmse={result.max_frame_rmse:.4f}",
    ]
    if result.mean_flip is not None:
        max_frame_flip = result.max_frame_flip
        parts.append(f"mean_flip={result.mean_flip:.4f}")
        parts.append(
            f"max_frame_flip={max_frame_flip:.4f}"
            if max_frame_flip is not None
            else "n/a"
        )
    worst = max(result.frame_metrics, key=lambda m: m.rmse, default=None)
    if worst is not None:
        parts.append(
            f"worst_frame={worst.frame_index} "
            f"(mean_abs={worst.mean_abs:.4f}, rmse={worst.rmse:.4f}, "
            f"psnr={worst.psnr_db:.2f} dB)"
        )
    return " ".join(parts)


def _threshold_failures(
    result: ClipComparisonResult, thresholds: ClipComparisonThresholds
) -> list[str]:
    failures: list[str] = []
    if (
        thresholds.require_same_frame_count
        and result.reference_frame_count != result.candidate_frame_count
    ):
        failures.append(
            "frame count changed: "
            f"reference={result.reference_frame_count}, "
            f"candidate={result.candidate_frame_count}"
        )
    if result.mean_abs > thresholds.max_mean_abs:
        failures.append(
            f"mean_abs {result.mean_abs:.4f} > {thresholds.max_mean_abs:.4f}"
        )
    if result.rmse > thresholds.max_rmse:
        failures.append(f"rmse {result.rmse:.4f} > {thresholds.max_rmse:.4f}")
    if result.psnr_db < thresholds.min_psnr_db:
        failures.append(
            f"psnr {result.psnr_db:.2f} dB < {thresholds.min_psnr_db:.2f} dB"
        )
    if result.max_frame_mean_abs > thresholds.max_frame_mean_abs:
        failures.append(
            "max_frame_mean_abs "
            f"{result.max_frame_mean_abs:.4f} > "
            f"{thresholds.max_frame_mean_abs:.4f}"
        )
    if result.max_frame_rmse > thresholds.max_frame_rmse:
        failures.append(
            f"max_frame_rmse {result.max_frame_rmse:.4f} > "
            f"{thresholds.max_frame_rmse:.4f}"
        )
    if (
        thresholds.max_mean_flip is not None
        and result.mean_flip is not None
        and result.mean_flip > thresholds.max_mean_flip
    ):
        failures.append(
            f"mean_flip {result.mean_flip:.4f} > {thresholds.max_mean_flip:.4f}"
        )
    max_frame_flip = result.max_frame_flip
    if (
        thresholds.max_frame_flip is not None
        and max_frame_flip is not None
        and max_frame_flip > thresholds.max_frame_flip
    ):
        failures.append(
            f"max_frame_flip {max_frame_flip:.4f} > {thresholds.max_frame_flip:.4f}"
        )
    return failures


def _compute_flip_scores(
    reference: RGBVideo, candidate: RGBVideo, indices: tuple[int, ...]
) -> tuple[float, ...]:
    if importlib.util.find_spec("flip_evaluator") is None:
        raise ImportError(
            "FLIP clip comparison requires flip-evaluator. Install the "
            "omnidreams dev extra or set max_mean_flip/max_frame_flip to None."
        )
    try:
        flip_evaluator = importlib.import_module("flip_evaluator")
    except ImportError as exc:
        raise ImportError(
            "FLIP clip comparison requires a working flip-evaluator install. "
            "Install the omnidreams dev extra or set max_mean_flip/max_frame_flip to None."
        ) from exc

    scores = []
    for idx in indices:
        _, mean_flip, _ = flip_evaluator.evaluate(
            reference[idx].astype(np.float32) / 255.0,
            candidate[idx].astype(np.float32) / 255.0,
            "LDR",
            inputsRGB=True,
            computeMeanError=True,
        )
        scores.append(float(mean_flip))
    return tuple(scores)


def _validate_compatible_spatial_shape(
    reference: RGBVideo, candidate: RGBVideo
) -> None:
    _validate_video(reference, name="reference")
    _validate_video(candidate, name="candidate")
    if reference.shape[1:] != candidate.shape[1:]:
        raise ValueError(
            f"Reference/candidate spatial shapes differ: "
            f"{reference.shape[1:]} != {candidate.shape[1:]}"
        )


def _validate_video(video: np.ndarray, *, name: str) -> None:
    if video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError(f"{name} must have shape [T,H,W,3], got {video.shape}")


def _as_uint8_rgb(video: np.ndarray) -> RGBVideo:
    _validate_video(video, name="video")
    if video.dtype == np.uint8:
        return video
    if np.issubdtype(video.dtype, np.floating):
        if np.nanmin(video) >= 0.0 and np.nanmax(video) <= 1.0:
            video = video * 255.0
        return np.clip(video, 0, 255).astype(np.uint8)
    return np.clip(video, 0, 255).astype(np.uint8)


def _psnr_db(rmse: float) -> float:
    if rmse == 0.0:
        return math.inf
    return 20.0 * math.log10(255.0 / rmse)
