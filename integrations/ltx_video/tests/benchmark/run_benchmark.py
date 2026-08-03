"""DiT runtime benchmark: baseline vs streaming vs optimized."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Any

import torch


PROMPT = "A coastal road at dusk, waves breaking on rocky cliffs, cinematic wide shot"
N_WARMUP = 3
N_MEASURE = 10
STEP_TO_MEASURE = 5


def measure_dit_time(fn, n_warmup: int = N_WARMUP, n_measure: int = N_MEASURE) -> list[float]:
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
    from diffusers import LTXPipeline

    pipe = LTXPipeline.from_pretrained(
        "Lightricks/LTX-Video", torch_dtype=torch.bfloat16
    ).to("cuda")
    torch.backends.cuda.enable_flash_sdp(True)

    pe, pm, npe, npm = pipe.encode_prompt(
        prompt=PROMPT,
        negative_prompt="worst quality",
        device="cuda",
        do_classifier_free_guidance=True,
    )
    pipe.scheduler.set_timesteps(50, device="cuda")
    t = pipe.scheduler.timesteps[STEP_TO_MEASURE]
    latents = torch.randn(
        1,
        pipe.transformer.config.in_channels,
        4,
        64,
        96,
        device="cuda",
        dtype=torch.bfloat16,
    )
    latent_input = torch.cat([latents] * 2)
    enc_hs = torch.cat([npe, pe])
    enc_mask = torch.cat([npm, pm])
    t_batch = t.expand(2)

    def step() -> None:
        with torch.no_grad():
            pipe.transformer(
                hidden_states=latent_input,
                encoder_hidden_states=enc_hs,
                encoder_attention_mask=enc_mask,
                timestep=t_batch,
            )

    return {"mode": "baseline", "times_ms": measure_dit_time(step)}


def run_flashdreams(mode: str) -> dict[str, Any]:
    from flashdreams.infra.config import derive_config
    from ltx_video.config import PIPELINE_LTX_T2V_2B, PIPELINE_LTX_T2V_2B_OPTIMIZED
    from ltx_video.pipeline import LTXVideoStreamingPipeline

    base = PIPELINE_LTX_T2V_2B if mode == "streaming" else PIPELINE_LTX_T2V_2B_OPTIMIZED
    pipe = LTXVideoStreamingPipeline(derive_config(base, device="cuda"))
    cache = pipe.initialize_cache(text=[PROMPT])

    for s in range(STEP_TO_MEASURE):
        pipe.generate(s, cache, width=768, height=512)
        pipe.finalize(s, cache)
        cache = pipe.initialize_cache(text=[PROMPT])

    def step() -> None:
        pipe.generate(STEP_TO_MEASURE, cache, width=768, height=512)

    return {"mode": mode, "times_ms": measure_dit_time(step)}


def summarise(result: dict[str, Any], baseline_mean: float | None = None) -> dict[str, Any]:
    ts = result["times_ms"]
    summary: dict[str, Any] = {
        "mode": result["mode"],
        "dit_runtime_ms_mean": round(statistics.mean(ts), 1),
        "dit_runtime_ms_p50": round(statistics.median(ts), 1),
        "dit_runtime_ms_p95": round(sorted(ts)[int(len(ts) * 0.95)], 1),
    }
    if baseline_mean:
        summary["speedup_vs_baseline"] = round(
            baseline_mean / summary["dit_runtime_ms_mean"], 2
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["baseline", "streaming", "optimized", "all"],
        default="all",
    )
    parser.add_argument("--output", default="benchmark_results.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for benchmark")

    results: dict[str, Any] = {}

    if args.mode in ("baseline", "all"):
        r = run_baseline()
        results["baseline"] = summarise(r)
        print("Baseline:", results["baseline"])

    bm = results.get("baseline", {}).get("dit_runtime_ms_mean")

    if args.mode in ("streaming", "all"):
        r = run_flashdreams("streaming")
        results["streaming"] = summarise(r, bm)
        print("Streaming:", results["streaming"])

    if args.mode in ("optimized", "all"):
        r = run_flashdreams("optimized")
        results["optimized"] = summarise(r, bm)
        print("Optimized:", results["optimized"])

    with open(args.output, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
