#!/usr/bin/env python3
"""Offline SERVER throughput � timed full generate loop (black-box wall-clock,
sync included), graphs ON (production-like). Same per-chunk work as the WebRTC
server (encode -> DiT -> [VAE decode] -> finalize), random HDMap in place of the
Ludus render.

  --decode true   -> PIXEL server work (encode + DiT + VAE decode + finalize)
  --decode false  -> TOKEN server work (encode + DiT + finalize; decode skipped)

Reports frames/s, ms/frame, chunks/s over the measured window (after warmup).

    source env.sh
    python throughput_probe.py --decode true  --chunks 300
"""
import argparse
import sys
import time

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decode", default="true")
    ap.add_argument("--chunks", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=6)
    args = ap.parse_args()
    decode = str(args.decode).lower() in ("1", "true", "yes")

    H, W, V = 704, 1280, 1
    dev, dt = torch.device("cuda"), torch.bfloat16

    from omnidreams.config import OMNIDREAMS_CONFIGS
    from omnidreams.pipeline import OmnidreamsPipeline

    NAME = "omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf"
    cfg = OMNIDREAMS_CONFIGS[NAME]   # perf config: cuda-graph/compile ON (realistic)
    print(f"[tp] building pipeline (decode={decode}) ...", flush=True)
    pipe = cfg.setup().to(dev)
    assert isinstance(pipe, OmnidreamsPipeline)

    image = torch.randn(1, V, 1, 3, H, W, device=dev, dtype=dt)
    cache = pipe.initialize_cache(text=[["a driving scene on a highway"]], image=image)
    pipe.release_oneshot_encoders()

    def one(i):
        n = pipe.get_num_frames(i)
        hdmap = torch.randn(1, V, n, 3, H, W, device=dev, dtype=dt)
        pipe.generate(autoregressive_index=i, hdmap=hdmap, cache=cache, decode=decode)
        pipe.finalize(autoregressive_index=i, cache=cache)
        return n

    # Warmup (captures cuda graphs / compiles), discarded from timing.
    for i in range(args.warmup):
        one(i)
    torch.cuda.synchronize()

    # Measured window: wall-clock across `chunks`, sync included at the end.
    frames = 0
    t0 = time.time()
    for j in range(args.chunks):
        frames += one(args.warmup + j)
    torch.cuda.synchronize()          # include the final sync � nothing skipped
    wall = time.time() - t0

    fps = frames / wall
    print(f"[tp] decode={decode} chunks={args.chunks} frames={frames} "
          f"wall={wall:.2f}s  ->  {fps:.2f} fps   {wall/frames*1e3:.2f} ms/frame   "
          f"{args.chunks/wall:.2f} chunks/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
