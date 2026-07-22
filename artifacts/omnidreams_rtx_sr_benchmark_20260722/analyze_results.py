"""Summarize matched OmniDreams native and RTX VSR benchmark outputs."""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
SLUG = "omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae"
BASELINE_VIDEO = ROOT / "baseline" / f"{SLUG}.mp4"
RTX_VIDEO = ROOT / "rtx_super_resolution" / f"{SLUG}.mp4"
COMPARISONS = ROOT / "comparisons"
WARMUP_CHUNKS = 3


def _read_video(path: Path, *, baseline_canvas: bool = False) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if baseline_canvas:
            frame = frame[frame.shape[0] // 2 :, :, :]
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return np.stack(frames)


def _ssim_luma(a: np.ndarray, b: np.ndarray) -> float:
    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float64)
    b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float64)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    sigma_a = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a * mu_a
    sigma_b = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b * mu_b
    sigma_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    score = ((2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)) / (
        (mu_a * mu_a + mu_b * mu_b + c1) * (sigma_a + sigma_b + c2)
    )
    return float(np.mean(score))


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return math.inf if mse == 0 else 10.0 * math.log10((255.0**2) / mse)


def _sharpness(frame: np.ndarray) -> tuple[float, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sobel_energy = float(np.mean(np.sqrt(gx * gx + gy * gy)))
    return laplacian_variance, sobel_energy


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _label(frame: np.ndarray, text: str) -> np.ndarray:
    result = frame.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 72), (0, 0, 0), -1)
    cv2.putText(
        result,
        text,
        (24, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.25,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return result


def _write_comparisons(lanczos: np.ndarray, rtx: np.ndarray) -> dict[str, object]:
    COMPARISONS.mkdir(parents=True, exist_ok=True)
    frame_index = 36
    baseline_frame = lanczos[frame_index]
    rtx_frame = rtx[frame_index]

    # Full-frame panels are reduced equally for a readable 2560-pixel canvas.
    baseline_full = cv2.resize(baseline_frame, (1280, 704), interpolation=cv2.INTER_AREA)
    rtx_full = cv2.resize(rtx_frame, (1280, 704), interpolation=cv2.INTER_AREA)
    full = np.hstack(
        [_label(baseline_full, "Native + Lanczos 2x"), _label(rtx_full, "RTX Super Resolution 2x")]
    )
    cv2.imwrite(str(COMPARISONS / "full_frame_comparison.png"), full)

    # High-resolution coordinates: left-side car, lane boundary, and hillside.
    x, y, width, height = 0, 480, 1120, 720
    baseline_crop = baseline_frame[y : y + height, x : x + width]
    rtx_crop = rtx_frame[y : y + height, x : x + width]
    close = np.hstack(
        [_label(baseline_crop, "Native + Lanczos 2x"), _label(rtx_crop, "RTX Super Resolution 2x")]
    )
    cv2.imwrite(str(COMPARISONS / "close_up_comparison.png"), close)
    return {
        "frame_index": frame_index,
        "timestamp_seconds": frame_index / 30.0,
        "crop_xywh_at_2x": [x, y, width, height],
    }


def _performance() -> dict[str, object]:
    baseline_stats = json.loads((ROOT / "baseline" / f"stats_{SLUG}.json").read_text())
    rtx_stats = json.loads(
        (ROOT / "rtx_super_resolution" / f"stats_{SLUG}.json").read_text()
    )
    baseline_measured = baseline_stats[WARMUP_CHUNKS:]
    rtx_measured = rtx_stats[WARMUP_CHUNKS:]
    baseline_total = [float(item["total_ms"]) for item in baseline_measured]
    rtx_model_total = [float(item["total_ms"]) for item in rtx_measured]
    postprocess = [float(item["postprocess"]["elapsed_ms"]) for item in rtx_measured]
    rtx_effective = [model + post for model, post in zip(rtx_model_total, postprocess)]
    frames = [int(item["postprocess"]["input_frames"]) for item in rtx_measured]
    baseline_fps = [count / (elapsed / 1000.0) for count, elapsed in zip(frames, baseline_total)]
    rtx_fps = [count / (elapsed / 1000.0) for count, elapsed in zip(frames, rtx_effective)]
    pp_per_frame = [elapsed / count for elapsed, count in zip(postprocess, frames)]
    baseline_median = float(np.median(baseline_total))
    rtx_median = float(np.median(rtx_effective))
    return {
        "warmup_chunks_excluded": WARMUP_CHUNKS,
        "measured_chunk_indices": [int(item["autoregressive_index"]) for item in baseline_measured],
        "frames_per_measured_chunk": frames,
        "baseline_total_ms": _distribution(baseline_total),
        "rtx_model_total_ms": _distribution(rtx_model_total),
        "rtx_postprocess_ms": _distribution(postprocess),
        "rtx_postprocess_ms_per_frame": _distribution(pp_per_frame),
        "rtx_effective_total_ms": _distribution(rtx_effective),
        "baseline_effective_fps": _distribution(baseline_fps),
        "rtx_effective_fps": _distribution(rtx_fps),
        "median_effective_latency_overhead_percent": (rtx_median / baseline_median - 1.0) * 100.0,
        "baseline_reserved_gib_median": float(np.median([item["mem_reserved_gib"] for item in baseline_measured])),
        "rtx_reserved_gib_median": float(np.median([item["mem_reserved_gib"] for item in rtx_measured])),
    }


def main() -> None:
    baseline = _read_video(BASELINE_VIDEO, baseline_canvas=True)
    rtx = _read_video(RTX_VIDEO)
    if len(baseline) != len(rtx):
        raise RuntimeError(f"frame count mismatch: baseline={len(baseline)}, rtx={len(rtx)}")

    lanczos = np.stack(
        [cv2.resize(frame, (2560, 1408), interpolation=cv2.INTER_LANCZOS4) for frame in baseline]
    )
    downsampled_rtx = np.stack(
        [cv2.resize(frame, (1280, 704), interpolation=cv2.INTER_AREA) for frame in rtx]
    )

    high_psnr = [_psnr(a, b) for a, b in zip(lanczos, rtx)]
    high_ssim = [_ssim_luma(a, b) for a, b in zip(lanczos, rtx)]
    down_psnr = [_psnr(a, b) for a, b in zip(baseline, downsampled_rtx)]
    down_ssim = [_ssim_luma(a, b) for a, b in zip(baseline, downsampled_rtx)]
    baseline_sharpness = [_sharpness(frame) for frame in lanczos]
    rtx_sharpness = [_sharpness(frame) for frame in rtx]
    baseline_temporal = [
        float(np.mean(np.abs(lanczos[i].astype(np.float32) - lanczos[i - 1].astype(np.float32))))
        for i in range(1, len(lanczos))
    ]
    rtx_temporal = [
        float(np.mean(np.abs(rtx[i].astype(np.float32) - rtx[i - 1].astype(np.float32))))
        for i in range(1, len(rtx))
    ]

    result = {
        "inputs": {
            "baseline_video": str(BASELINE_VIDEO.relative_to(ROOT)),
            "rtx_video": str(RTX_VIDEO.relative_to(ROOT)),
            "frame_count": int(len(baseline)),
            "native_resolution": [1280, 704],
            "rtx_resolution": [2560, 1408],
        },
        "performance": _performance(),
        "quality": {
            "rtx_vs_lanczos_2x_psnr_db": _distribution(high_psnr),
            "rtx_vs_lanczos_2x_ssim_luma": _distribution(high_ssim),
            "rtx_downsampled_vs_native_psnr_db": _distribution(down_psnr),
            "rtx_downsampled_vs_native_ssim_luma": _distribution(down_ssim),
            "lanczos_2x_laplacian_variance": _distribution([v[0] for v in baseline_sharpness]),
            "rtx_laplacian_variance": _distribution([v[0] for v in rtx_sharpness]),
            "lanczos_2x_sobel_energy": _distribution([v[1] for v in baseline_sharpness]),
            "rtx_sobel_energy": _distribution([v[1] for v in rtx_sharpness]),
            "lanczos_temporal_frame_delta_mae": _distribution(baseline_temporal),
            "rtx_temporal_frame_delta_mae": _distribution(rtx_temporal),
        },
        "visual_comparison": _write_comparisons(lanczos, rtx),
    }
    (ROOT / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
