#!/usr/bin/env python3
"""Per-chunk HW SM-cycle probe (offline, ncu). Wraps each chunk's generate in an
NVTX range so ncu can attribute kernels to a chunk; run under ncu with a SINGLE
metric (sm__cycles_active.sum) so application-replay needs only ONE pass (the app
runs once, no re-run). Post-process groups kernels by NVTX chunk range and sums.

  ncu --profile-from-start off --nvtx --replay-mode application \
      --metrics sm__cycles_active.sum --csv --page raw \
      --log-file /scratch/nisjain/hw_run/hw.csv \
      /scratch/nisjain/venv/bin/python3 hw_smcycles_probe.py --chunks 1000 --mode token

mode: pixel (decode=True) | token (decode=False, == token/sas gen; SAS encode is
separate via sas_encode_probe.py). SM cycles of generate are identical for
token-raw / sas-2s / sas-4s (same decode=False generate); pixel differs by +VAE-decode.
"""
from __future__ import annotations

import argparse
import sys

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--mode", choices=["pixel", "token"], default="token")
    args = ap.parse_args()
    decode = (args.mode == "pixel")

    H, W, V = 704, 1280, 1
    dev, dt = torch.device("cuda"), torch.bfloat16
    from omnidreams.config import OMNIDREAMS_CONFIGS
    # graphs OFF: ncu cannot see kernels inside a CUDA graph. Derive a config
    # with cuda_graph/compile disabled so every kernel is individually launched.
    NAME = "omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf"
    cfg = OMNIDREAMS_CONFIGS[NAME]
    for attr in ("cuda_graph", "use_cuda_graph", "compile", "use_compile"):
        if hasattr(cfg, attr):
            try:
                setattr(cfg, attr, False)
            except Exception:
                pass
    print(f"[hw] building pipeline ({NAME}, graphs OFF for ncu), mode={args.mode}", flush=True)
    pipe = OMNIDREAMS_CONFIGS[NAME].setup().to(dev)
    # best-effort: disable any graph runner post-setup
    for a in ("cuda_graph", "use_cuda_graph", "_cuda_graph"):
        if hasattr(pipe, a):
            try:
                setattr(pipe, a, None)
            except Exception:
                pass

    image = torch.randn(1, V, 1, 3, H, W, device=dev, dtype=dt)
    cache = pipe.initialize_cache(text=[["a driving scene on a highway"]], image=image)
    pipe.release_oneshot_encoders()

    def hdmap(i):
        n = pipe.get_num_frames(i)
        return n, torch.randn(1, V, n, 3, H, W, device=dev, dtype=dt)

    for i in range(args.warmup):
        _, hd = hdmap(i)
        pipe.generate(autoregressive_index=i, hdmap=hd, cache=cache, decode=decode)
        pipe.finalize(autoregressive_index=i, cache=cache)
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    for j in range(args.chunks):
        i = args.warmup + j
        _, hd = hdmap(i)
        torch.cuda.nvtx.range_push(f"chunk{j:04d}")
        pipe.generate(autoregressive_index=i, hdmap=hd, cache=cache, decode=decode)
        pipe.finalize(autoregressive_index=i, cache=cache)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
    print(f"[hw] profiled {args.chunks} chunks mode={args.mode}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
