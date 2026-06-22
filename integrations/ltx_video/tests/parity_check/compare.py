"""Frame-level quality comparison between streaming and reference videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_video_frames(path: Path) -> np.ndarray:
    try:
        import mediapy as media

        return media.read_video(str(path))
    except ImportError:
        import imageio.v3 as iio

        return iio.imread(str(path))


def compute_psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(255.0**2 / mse))


def compute_ssim_frame(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu_a, mu_b = a.mean(), b.mean()
    sigma_a, sigma_b = a.var(), b.var()
    sigma_ab = ((a - mu_a) * (b - mu_b)).mean()
    num = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (sigma_a + sigma_b + c2)
    return float(num / den) if den else 1.0


def compare_videos(streaming: Path, reference: Path) -> dict:
    stream_frames = load_video_frames(streaming)
    ref_frames = load_video_frames(reference)
    n = min(len(stream_frames), len(ref_frames))
    stream_frames = stream_frames[:n]
    ref_frames = ref_frames[:n]

    if stream_frames.shape[1:3] != ref_frames.shape[1:3]:
        from PIL import Image

        resized = []
        for frame in stream_frames:
            img = Image.fromarray(frame).resize(
                (ref_frames.shape[2], ref_frames.shape[1]), Image.Resampling.BILINEAR
            )
            resized.append(np.array(img))
        stream_frames = np.stack(resized)

    psnrs = [compute_psnr(s, r) for s, r in zip(stream_frames, ref_frames, strict=True)]
    ssims = [
        compute_ssim_frame(s, r) for s, r in zip(stream_frames, ref_frames, strict=True)
    ]
    return {
        "frames_compared": n,
        "psnr_mean": float(np.mean(psnrs)),
        "psnr_min": float(np.min(psnrs)),
        "ssim_mean": float(np.mean(ssims)),
        "ssim_min": float(np.min(ssims)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--streaming", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("parity_results.json"))
    args = parser.parse_args()

    result = compare_videos(args.streaming, args.reference)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
