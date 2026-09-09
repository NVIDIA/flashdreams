#!/usr/bin/env python3
"""Offline SW throughput for all 4 stream modes over N chunks (no browser drive).

Server-side before-network throughput, matching the manager's METRIC bracket:
  gen_ms   = time around runtime.generate_chunk() == pipe.generate(decode=)+finalize
             (excludes nvenc; excludes ack-wait -- exactly the mentor definition)
  pixel    before-network = gen(decode=True)  + nvenc_ms
  token    before-network = gen(decode=False) + ~0        (raw fp16, no codec)
  sas-2s   before-network = gen(decode=False) + sas2_ms
  sas-4s   before-network = gen(decode=False) + sas4_ms

token-raw / sas-2s / sas-4s share ONE decode=False generate; only the appended
encode differs. Pixel is a separate decode=True pass, used to CALIBRATE offline
gen_ms against the live drive (must be ~664 ms/chunk = graphs ON). nvenc_ms is
taken from the live drive (same NVENC chip, ~5.6 ms, 0.8%); pass --nvenc-ms to override.

  source /scratch/nisjain/env.sh
  python sw_throughput_offline.py --chunks 1000 --out /scratch/nisjain/sw_run
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--out", default="/scratch/nisjain/sw_run")
    ap.add_argument("--nvenc-ms", type=float, default=5.595)  # from live pixel drive
    ap.add_argument("--calib-gen-ms", type=float, default=664.14)  # live pixel gen_ms
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    H, W, V = 704, 1280, 1
    dev, dt = torch.device("cuda"), torch.bfloat16
    from omnidreams.config import OMNIDREAMS_CONFIGS
    from flashdreams.infra.sas import build_preset_quantizer
    NAME = "omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf"  # graphs ON

    print(f"[sw] building pipeline ({NAME}) + SAS quantizers ...", flush=True)
    pipe = OMNIDREAMS_CONFIGS[NAME].setup().to(dev)
    q2 = build_preset_quantizer("int4-2s")
    q4 = build_preset_quantizer("int4-4s")

    image = torch.randn(1, V, 1, 3, H, W, device=dev, dtype=dt)
    cache_t = pipe.initialize_cache(text=[["a driving scene on a highway"]], image=image)
    cache_p = pipe.initialize_cache(text=[["a driving scene on a highway"]], image=image)
    pipe.release_oneshot_encoders()

    def hdmap(i):
        n = pipe.get_num_frames(i)
        return n, torch.randn(1, V, n, 3, H, W, device=dev, dtype=dt)

    def warm(cache, decode):
        for i in range(args.warmup):
            _, hd = hdmap(i)
            pipe.generate(autoregressive_index=i, hdmap=hd, cache=cache, decode=decode)
            pipe.finalize(autoregressive_index=i, cache=cache)
        torch.cuda.synchronize()

    csv_path = os.path.join(args.out, "sw_per_chunk.csv")
    cols = ["chunk", "frames",
            "token_gen_ms", "sas2_ms", "sas4_ms",
            "raw_bytes", "sas2_bytes", "sas4_bytes",
            "pixel_gen_ms"]
    rows = {}

    # -------- PASS 1: token family (decode=False) -> gen + sas2/sas4 encode --------
    print("[sw] pass1 token/sas (decode=False) ...", flush=True)
    warm(cache_t, decode=False)
    for j in range(args.chunks):
        i = args.warmup + j
        n, hd = hdmap(i)
        torch.cuda.synchronize(); t0 = time.time()
        pipe.generate(autoregressive_index=i, hdmap=hd, cache=cache_t, decode=False)
        pipe.finalize(autoregressive_index=i, cache=cache_t)
        torch.cuda.synchronize()
        token_gen_ms = (time.time() - t0) * 1e3

        latent = cache_t.clean_latent
        raw_bytes = latent.numel() * 2
        torch.cuda.synchronize(); t1 = time.time()
        p2 = q2.compress(latent); torch.cuda.synchronize(); t2 = time.time()
        p4 = q4.compress(latent); torch.cuda.synchronize(); t3 = time.time()
        rows[j] = {"chunk": j, "frames": int(n),
                   "token_gen_ms": round(token_gen_ms, 3),
                   "sas2_ms": round((t2 - t1) * 1e3, 3),
                   "sas4_ms": round((t3 - t2) * 1e3, 3),
                   "raw_bytes": raw_bytes, "sas2_bytes": len(p2), "sas4_bytes": len(p4),
                   "pixel_gen_ms": ""}
        if j % 100 == 0:
            print(f"[sw] p1 {j}/{args.chunks} token_gen={token_gen_ms:.1f} "
                  f"sas2={rows[j]['sas2_ms']:.2f}ms/{len(p2)}B", flush=True)

    # -------- PASS 2: pixel (decode=True) -> gen (calibration + pixel throughput) --
    print("[sw] pass2 pixel (decode=True) ...", flush=True)
    warm(cache_p, decode=True)
    for j in range(args.chunks):
        i = args.warmup + j
        _, hd = hdmap(i)
        torch.cuda.synchronize(); t0 = time.time()
        pipe.generate(autoregressive_index=i, hdmap=hd, cache=cache_p, decode=True)
        pipe.finalize(autoregressive_index=i, cache=cache_p)
        torch.cuda.synchronize()
        rows[j]["pixel_gen_ms"] = round((time.time() - t0) * 1e3, 3)
        if j % 100 == 0:
            print(f"[sw] p2 {j}/{args.chunks} pixel_gen={rows[j]['pixel_gen_ms']:.1f}", flush=True)

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for j in range(args.chunks):
            w.writerow(rows[j])

    # -------- summary (skip first 20 for warmup drift) --------
    import statistics as st
    steady = [rows[j] for j in range(20, args.chunks)] if args.chunks > 40 else list(rows.values())
    def mean(k): return st.mean(r[k] for r in steady)
    FPC = 8
    pix_gen = mean("pixel_gen_ms"); tok_gen = mean("token_gen_ms")
    sas2 = mean("sas2_ms"); sas4 = mean("sas4_ms")
    def fps(ms): return FPC / (ms / 1000.0)
    print("\n=== CALIBRATION (offline pixel gen vs live) ===")
    print(f"  offline pixel_gen_ms = {pix_gen:.2f}   live = {args.calib_gen_ms:.2f}   "
          f"delta = {100*(pix_gen-args.calib_gen_ms)/args.calib_gen_ms:+.1f}%")
    print("\n=== SW before-network throughput (excl ack-wait), steady mean ===")
    print(f"  pixel   : gen {pix_gen:.1f} + nvenc {args.nvenc_ms:.2f}  -> {fps(pix_gen+args.nvenc_ms):.3f} fps")
    print(f"  token   : gen {tok_gen:.1f} + ~0                 -> {fps(tok_gen):.3f} fps")
    print(f"  sas-2s  : gen {tok_gen:.1f} + sas2 {sas2:.2f}       -> {fps(tok_gen+sas2):.3f} fps")
    print(f"  sas-4s  : gen {tok_gen:.1f} + sas4 {sas4:.2f}       -> {fps(tok_gen+sas4):.3f} fps")
    print(f"\n[sw] wrote {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
