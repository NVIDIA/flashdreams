#!/usr/bin/env python3
"""Offline full `generate_chunk` for Nsight Compute � the exact per-chunk GPU work
the WebRTC server does (encode -> DiT -> [VAE decode] -> finalize), with a random
HDMap standing in for the Ludus render (same kernels). Used to measure server-side
SM cycles per chunk.

  --decode true   -> PIXEL server work (encode + DiT + VAE decode + finalize)
  --decode false  -> TOKEN server work (encode + DiT + finalize; decode skipped)

The profiled chunk is bracketed by cudaProfilerStart/Stop so `ncu` (with
--profile-from-start off) captures only that one steady-state chunk, after warmup.
cuda-graph / compile are turned OFF for clean per-kernel capture.

    source env.sh
    sudo -E env HOME=$NCU_HOME HF_HOME=$HF_HOME \
      ncu --profile-from-start off \
      --metrics gpu__time_duration.sum,sm__cycles_active.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active \
      --target-processes all -f -o ./ncu_gen_pixel \
      python ncu_generate_probe.py --decode true
"""
import argparse
import sys

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decode", default="true")   # true=pixel(full), false=token(no decode)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()
    decode = str(args.decode).lower() in ("1", "true", "yes")

    H, W, V = 704, 1280, 1
    dev, dt = torch.device("cuda"), torch.bfloat16

    from flashdreams.infra.config import derive_config
    from omnidreams.config import OMNIDREAMS_CONFIGS
    from omnidreams.pipeline import OmnidreamsPipeline

    NAME = "omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf"
    cfg = OMNIDREAMS_CONFIGS[NAME]
    # Turn off graphs/compile so ncu sees clean per-kernel launches.
    cfg = derive_config(
        cfg,
        image_encoder=dict(use_compile=False, use_cuda_graph=False),
        encoder=dict(use_compile=False, use_cuda_graph=False),
        decoder=dict(use_compile=False, use_cuda_graph=False),
        diffusion_model=dict(transformer=dict(compile_network=False, use_cuda_graph=False)),
    )
    print(f"[gen] building pipeline (decode={decode}) ...", flush=True)
    pipe = cfg.setup().to(dev)
    assert isinstance(pipe, OmnidreamsPipeline)

    image = torch.randn(1, V, 1, 3, H, W, device=dev, dtype=dt)  # first frame [-1,1]
    cache = pipe.initialize_cache(text=[["a driving scene on a highway"]], image=image)
    pipe.release_oneshot_encoders()   # free the Cosmos-Reason1 text encoder VRAM

    prof_at = args.warmup
    video = None
    for i in range(args.warmup + 1):
        n = pipe.get_num_frames(i)                                   # 5 then 8
        hdmap = torch.randn(1, V, n, 3, H, W, device=dev, dtype=dt)  # HDMap cond [-1,1]
        torch.cuda.synchronize()
        if i == prof_at:
            torch.cuda.cudart().cudaProfilerStart()
        video = pipe.generate(autoregressive_index=i, hdmap=hdmap, cache=cache, decode=decode)
        pipe.finalize(autoregressive_index=i, cache=cache)
        torch.cuda.synchronize()
        if i == prof_at:
            torch.cuda.cudart().cudaProfilerStop()
        print(f"[gen] chunk {i} frames={n} decode={decode}", flush=True)

    print(f"[gen] profiled chunk {prof_at}: video="
          f"{None if video is None else tuple(video.shape)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
