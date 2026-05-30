#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark PyTorch vs native FP8 OmniDreams LightVAE encoders.

The native preset's FP8 LightVAE state must be provided through
``OMNIDREAMS_LIGHTVAE_FP8_STATE_PATH`` or ``--fp8-state-path``.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

BASELINE_CONFIG = "omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf"
NATIVE_CONFIG = "omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-native-perf"
FP8_STATE_ENV = "OMNIDREAMS_LIGHTVAE_FP8_STATE_PATH"


@dataclass(frozen=True)
class EncoderRow:
    case: str
    implementation: str
    shape: str
    dtype: str
    mae: float | None
    rmse: float | None
    max_abs: float | None
    rel_l2: float | None
    p50_ms: float
    p95_ms: float
    speedup: float | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-config", default=BASELINE_CONFIG)
    parser.add_argument("--native-config", default=NATIVE_CONFIG)
    parser.add_argument(
        "--baseline-eager",
        action="store_true",
        help="Disable compile/cuda_graph on the baseline encoder for quick local checks.",
    )
    parser.add_argument("--fp8-state-path", default=os.environ.get(FP8_STATE_ENV))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/native_vae"))
    parser.add_argument(
        "--native-build-root",
        type=Path,
        default=None,
        help="Optional build root used to preload the native extension before setup.",
    )
    parser.add_argument("--native-max-jobs", type=int, default=None)
    parser.add_argument(
        "--source-video",
        type=Path,
        default=None,
        help="Optional video input; random input is used when omitted.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--latent-t", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def _import_configs(fp8_state_path: str | None) -> dict[str, Any]:
    if fp8_state_path:
        os.environ[FP8_STATE_ENV] = fp8_state_path
    from omnidreams.config import OMNIDREAMS_CONFIGS

    return OMNIDREAMS_CONFIGS


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _gpu_name(device: torch.device) -> str:
    if device.type != "cuda":
        return str(device)
    props = torch.cuda.get_device_properties(device)
    return f"{props.name} sm_{props.major}{props.minor}"


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_ms(
    fn: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    iters: int,
) -> tuple[list[float], torch.Tensor]:
    output = fn()
    _sync(device)
    for _ in range(warmup):
        output = fn()
    _sync(device)

    times: list[float] = []
    for _ in range(iters):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = fn()
            end.record()
            end.synchronize()
            times.append(float(start.elapsed_time(end)))
        else:
            import time

            start_t = time.perf_counter()
            output = fn()
            times.append((time.perf_counter() - start_t) * 1000.0)
    _sync(device)
    return times, output


def _p95(values: list[float]) -> float:
    if len(values) < 2:
        return values[0]
    return statistics.quantiles(values, n=20, method="inclusive")[18]


def _metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    cand = candidate.detach().float()
    ref = reference.detach().float()
    diff = cand - ref
    rmse = diff.square().mean().sqrt()
    ref_norm = torch.linalg.vector_norm(ref).clamp_min(1.0e-12)
    return {
        "mae": float(diff.abs().mean().item()),
        "rmse": float(rmse.item()),
        "max_abs": float(diff.abs().max().item()),
        "rel_l2": float((torch.linalg.vector_norm(diff) / ref_norm).item()),
    }


def _shape(tensor: torch.Tensor) -> str:
    return "x".join(str(dim) for dim in tensor.shape)


def _load_video_prefix_btchw(
    path: Path,
    *,
    frames: int,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    try:
        import cv2  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required to load benchmark source video frames"
        ) from exc

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {path}")
    images: list[torch.Tensor] = []
    while len(images) < frames:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != (height, width):
            rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
        images.append(torch.from_numpy(rgb).permute(2, 0, 1).contiguous())
    cap.release()
    if len(images) < frames:
        raise RuntimeError(f"{path} has {len(images)} readable frames; need {frames}")
    video = (
        torch.stack(images, dim=0).unsqueeze(0).to(device=device, dtype=torch.float32)
    )
    return (video / 127.5 - 1.0).contiguous()


def _make_row(
    *,
    case: str,
    implementation: str,
    output: torch.Tensor,
    dtype: torch.dtype,
    times: list[float],
    metrics: dict[str, float] | None,
    speedup: float | None,
) -> EncoderRow:
    return EncoderRow(
        case=case,
        implementation=implementation,
        shape=_shape(output),
        dtype=str(dtype).replace("torch.", ""),
        mae=None if metrics is None else metrics["mae"],
        rmse=None if metrics is None else metrics["rmse"],
        max_abs=None if metrics is None else metrics["max_abs"],
        rel_l2=None if metrics is None else metrics["rel_l2"],
        p50_ms=statistics.median(times),
        p95_ms=_p95(times),
        speedup=speedup,
    )


def _bench_pair(
    *,
    case: str,
    baseline_name: str,
    native_name: str,
    baseline_fn: Callable[[], torch.Tensor],
    native_fn: Callable[[], torch.Tensor],
    baseline_dtype: torch.dtype,
    native_dtype: torch.dtype,
    device: torch.device,
    warmup: int,
    iters: int,
) -> list[EncoderRow]:
    baseline_times, baseline_out = _time_ms(
        baseline_fn, device=device, warmup=warmup, iters=iters
    )
    native_times, native_out = _time_ms(
        native_fn, device=device, warmup=warmup, iters=iters
    )
    speedup = statistics.median(baseline_times) / statistics.median(native_times)
    return [
        _make_row(
            case=case,
            implementation=baseline_name,
            output=baseline_out,
            dtype=baseline_dtype,
            times=baseline_times,
            metrics=None,
            speedup=None,
        ),
        _make_row(
            case=case,
            implementation=native_name,
            output=native_out,
            dtype=native_dtype,
            times=native_times,
            metrics=_metrics(native_out, baseline_out),
            speedup=speedup,
        ),
    ]


def _run(args: argparse.Namespace, configs: dict[str, Any]) -> list[EncoderRow]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device requested but torch.cuda.is_available() is false"
        )
    torch.manual_seed(args.seed)

    baseline_cfg = configs[args.baseline_config]
    native_cfg = configs[args.native_config]
    if args.baseline_eager:
        from flashdreams.infra.config import derive_config

        baseline_cfg = derive_config(
            baseline_cfg,
            image_encoder=dict(use_compile=False, use_cuda_graph=False),
            encoder=dict(use_compile=False, use_cuda_graph=False),
        )

    baseline_encoder = baseline_cfg.encoder.setup().to(device)
    native_encoder = native_cfg.encoder.setup().to(device)

    first_pixel_t = 1 + (args.latent_t - 1) * 4
    steady_pixel_t = args.latent_t * 4
    source_t = first_pixel_t + steady_pixel_t
    if args.source_video is None:
        source = torch.randn(
            (1, source_t, 3, args.height, args.width),
            device=device,
            dtype=torch.float32,
        ).clamp_(-1, 1)
    else:
        source = _load_video_prefix_btchw(
            args.source_video.expanduser().resolve(),
            frames=source_t,
            height=args.height,
            width=args.width,
            device=device,
        )

    first_source = source[:, :first_pixel_t]
    steady_source = source[:, first_pixel_t : first_pixel_t + steady_pixel_t]

    def baseline_encode_first() -> torch.Tensor:
        cache = baseline_encoder.initialize_autoregressive_cache()
        return baseline_encoder(
            first_source.to(baseline_encoder.config.dtype), cache=cache
        )

    def native_encode_first() -> torch.Tensor:
        cache = native_encoder.initialize_autoregressive_cache()
        return native_encoder(first_source.to(native_encoder.config.dtype), cache=cache)

    rows = _bench_pair(
        case="lightvae_encoder_first_chunk",
        baseline_name=args.baseline_config,
        native_name=args.native_config,
        baseline_fn=baseline_encode_first,
        native_fn=native_encode_first,
        baseline_dtype=baseline_encoder.config.dtype,
        native_dtype=native_encoder.config.dtype,
        device=device,
        warmup=args.warmup,
        iters=args.iters,
    )

    baseline_cache = baseline_encoder.initialize_autoregressive_cache()
    native_cache = native_encoder.initialize_autoregressive_cache()
    baseline_encoder(
        first_source.to(baseline_encoder.config.dtype), cache=baseline_cache
    )
    native_encoder(first_source.to(native_encoder.config.dtype), cache=native_cache)

    rows.extend(
        _bench_pair(
            case="lightvae_encoder_steady_chunk",
            baseline_name=args.baseline_config,
            native_name=args.native_config,
            baseline_fn=lambda: baseline_encoder(
                steady_source.to(baseline_encoder.config.dtype),
                cache=baseline_cache,
            ),
            native_fn=lambda: native_encoder(
                steady_source.to(native_encoder.config.dtype),
                cache=native_cache,
            ),
            baseline_dtype=baseline_encoder.config.dtype,
            native_dtype=native_encoder.config.dtype,
            device=device,
            warmup=args.warmup,
            iters=args.iters,
        )
    )
    return rows


def _write_csv(path: Path, rows: list[EncoderRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(EncoderRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def _write_markdown(
    path: Path, rows: list[EncoderRow], args: argparse.Namespace
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    lines = [
        "# Native LightVAE FP8 Encoder Before/After Report",
        "",
        f"- commit: `{_git_commit()}`",
        f"- device: `{_gpu_name(device)}`",
        f"- seed: `{args.seed}`",
        f"- warmup: `{args.warmup}`",
        f"- iterations: `{args.iters}`",
        f"- resolution: `{args.height}x{args.width}`",
        f"- source_video: `{args.source_video or 'random'}`",
        f"- baseline_config: `{args.baseline_config}`",
        f"- native_config: `{args.native_config}`",
        f"- baseline_eager: `{args.baseline_eager}`",
        "",
        "| case | implementation | shape | dtype | MAE | RMSE | max abs | rel L2 | p50 ms | p95 ms | speedup |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.case,
                    row.implementation,
                    row.shape,
                    row.dtype,
                    _fmt(row.mae),
                    _fmt(row.rmse),
                    _fmt(row.max_abs),
                    _fmt(row.rel_l2),
                    _fmt(row.p50_ms),
                    _fmt(row.p95_ms),
                    _fmt(row.speedup),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    if not args.fp8_state_path:
        raise SystemExit(
            f"--fp8-state-path or {FP8_STATE_ENV} is required for the native FP8 encoder"
        )
    configs = _import_configs(args.fp8_state_path)
    if args.native_build_root is not None:
        from omnidreams.native import omnidreams_singleview

        ext = omnidreams_singleview.load_extension(
            build_root=args.native_build_root,
            max_jobs=args.native_max_jobs,
            verbose=False,
        )
        if ext is None:
            raise RuntimeError(omnidreams_singleview.extension_load_error())

    rows = _run(args, configs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "native_lightvae_encoder_before_after.csv"
    md_path = args.output_dir / "native_lightvae_encoder_before_after.md"
    _write_csv(csv_path, rows)
    _write_markdown(md_path, rows, args)
    print(f"Wrote {md_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
