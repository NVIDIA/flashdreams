#!/usr/bin/env python3
"""Minimal, clean VAE-decode workload for Nsight Compute (ncu) profiling.

Loads the lighttae decoder and runs ONE decode of a small latent clip, so ncu
sees only the decode kernels (no SAS/mp4/metrics noise). Used to measure the
server-side decode HW load (SM cycles active, achieved occupancy, SM / DRAM
throughput) to compare against the client's WebGPU decode.

    source env.sh
    sudo -E ncu --set basic --target-processes all -f -o ./ncu_decode \
        python ncu_decode_probe.py [latents.npy] [n_frames]

If no .npy is given, a synthetic [n_frames,16,88,160] fp16 latent is used.
"""
import sys

import numpy as np
import torch


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "-" else None
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    if path:
        lat = torch.from_numpy(np.ascontiguousarray(np.load(path)))
        if lat.dim() == 4:
            lat = lat.unsqueeze(0)          # [N,C,H,W] -> [1,N,C,H,W]
        lat = lat[:, :n]
    else:
        lat = torch.from_numpy((np.random.randn(1, n, 16, 88, 160) * 0.4).astype(np.float16))

    from flashdreams.recipes.taehv import (
        AVAILABLE_TAEHV_CHECKPOINT_PATHS,
        TeahvVAEDecoderConfig,
    )

    dec = (
        TeahvVAEDecoderConfig(
            checkpoint_path=AVAILABLE_TAEHV_CHECKPOINT_PATHS["lighttae"],
            dtype=torch.bfloat16,
            use_cuda_graph=False,   # clean per-kernel launches for ncu
            use_compile=False,
        )
        .setup()
        .to("cuda", torch.bfloat16)
        .eval()
    )
    lat = lat.to("cuda", torch.bfloat16)
    cache = dec.initialize_autoregressive_cache()

    # Warm the allocator / any lazy init OUTSIDE the region ncu cares about is not
    # possible with a single fresh cache, so just do the one measured decode.
    # Bracket ONLY the decode so `ncu --profile-from-start off` profiles just the
    # decode kernels (not model-load / init). Robust vs NVTX-range filtering.
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    with torch.no_grad():
        rgb = dec(input=lat, autoregressive_index=0, cache=cache)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print(f"decoded {tuple(lat.shape)} -> {tuple(rgb.shape)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
