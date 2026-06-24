# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DiT/chunk runtime benchmark: baseline diffusers vs FlashDreams Helios streaming."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Any

import torch

PROMPT = "A coastal road at dusk, waves breaking on rocky cliffs, cinematic wide shot"
CHUNK_FRAMES = 33
N_WARMUP = 3
N_MEASURE = 10
STEP_TO_MEASURE = 6


def measure(fn, n_warmup: int = N_WARMUP, n_measure: int = N_MEASURE) -> list[float]:
    for _ in range(n_warmup):
        fn()
        torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(n_measure):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return times


def run_baseline() -> dict[str, Any]:
    from diffusers import AutoencoderKLWan, HeliosPyramidPipeline

    torch.backends.cuda.enable_flash_sdp(True)
    vae = AutoencoderKLWan.from_pretrained(
        "BestWishYsh/Helios-Distilled",
        subfolder="vae",
        torch_dtype=torch.float32,
    ).to("cuda")
    pipe = HeliosPyramidPipeline.from_pretrained(
        "BestWishYsh/Helios-Distilled",
        vae=vae,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    pe, npe = pipe.encode_prompt(
        prompt=[PROMPT],
        negative_prompt=["worst quality"],
        device="cuda",
        do_classifier_free_guidance=False,
    )

    def step() -> None:
        with torch.no_grad():
            pipe(
                prompt_embeds=pe,
                negative_prompt_embeds=npe,
                num_frames=CHUNK_FRAMES,
                height=384,
                width=640,
                pyramid_num_inference_steps_list=[2, 2, 2],
                guidance_scale=1.0,
                is_amplify_first_chunk=False,
                output_type="pt",
            )

    return {"mode": "baseline", "times_ms": measure(step)}


def run_flashdreams(mode: str) -> dict[str, Any]:
    from flashdreams.infra.config import derive_config
    from helios.config import PIPELINE_HELIOS_DISTILLED_T2V_14B
    from helios.pipeline import HeliosStreamingPipeline

    cfg = derive_config(
        PIPELINE_HELIOS_DISTILLED_T2V_14B,
        device="cuda:0",
        compile=(mode == "compiled"),
        amplify_first_chunk=False,
    )
    pipe = HeliosStreamingPipeline(cfg)
    cache = pipe.initialize_cache(text=[PROMPT])
    for s in range(STEP_TO_MEASURE):
        pipe.generate(s, cache, width=640, height=384)
        pipe.finalize(s, cache)

    def step() -> None:
        pipe.generate(STEP_TO_MEASURE, cache, width=640, height=384)

    return {"mode": mode, "times_ms": measure(step)}


def summarise(r: dict[str, Any], baseline_mean: float | None = None) -> dict[str, Any]:
    ts = r["times_ms"]
    s: dict[str, Any] = {
        "mode": r["mode"],
        "chunk_time_ms_mean": round(statistics.mean(ts), 1),
        "chunk_time_ms_p50": round(statistics.median(ts), 1),
        "chunk_time_ms_p95": round(sorted(ts)[int(len(ts) * 0.95)], 1),
        "fps": round(CHUNK_FRAMES / (statistics.mean(ts) / 1000), 1),
    }
    if baseline_mean:
        s["speedup_vs_baseline"] = round(baseline_mean / s["chunk_time_ms_mean"], 2)
    return s


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["baseline", "streaming", "compiled", "all"],
        default="all",
    )
    parser.add_argument("--output", default="helios_benchmark_results.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for Helios benchmark")

    results: dict[str, Any] = {}
    if args.mode in ("baseline", "all"):
        r = run_baseline()
        results["baseline"] = summarise(r)
        print("Baseline:", results["baseline"])
    if args.mode in ("streaming", "all"):
        r = run_flashdreams("streaming")
        bm = results.get("baseline", {}).get("chunk_time_ms_mean")
        results["streaming"] = summarise(r, bm)
        print("Streaming:", results["streaming"])
    if args.mode in ("compiled", "all"):
        r = run_flashdreams("compiled")
        bm = results.get("baseline", {}).get("chunk_time_ms_mean")
        results["compiled"] = summarise(r, bm)
        print("Compiled:", results["compiled"])

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
