# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# THROWAWAY spike (sub-task 1a) -- NOT product code. Delete before any PR.
#
# Validates multi-chunk continuity of the cache-as-IO decode against the native
# TAEHV streaming decoder over N consecutive chunks:
#   [A] the cache-as-IO WRAPPER (torch)  -- proves the dict<->list cache logic
#   [B] the exported ONNX (onnxruntime)  -- proves the export (if onnxruntime present)
# Both seed from the native decoder's post-chunk-0 cache and then THREAD their own
# cache forward. High, non-drifting per-chunk PSNR = faithful + continuous.
#
# Run on the GPU box (after export_streaming.py):
#   uv run python spike/1a-streaming-cache/validate_continuity.py

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from export_streaming import (  # noqa: E402
    CL,
    DEVICE,
    DTYPE,
    HL,
    WL,
    B,
    StreamingWrapper,
    T,
    V,
    build_decoder,
    memblocks_of,
)

N = 6  # number of consecutive chunks to stream


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    return float("inf") if rmse == 0.0 else 20.0 * float(np.log10(2.0 / rmse))


@torch.no_grad()
def main() -> None:
    torch.manual_seed(0)
    decoder = build_decoder()
    mbs = memblocks_of(decoder)
    latents = [
        torch.randn(B, V, T, CL, HL, WL, device=DEVICE, dtype=DTYPE) for _ in range(N)
    ]

    # --- Native torch streaming reference (carried cache across all chunks) ---
    cache_t = decoder.initialize_autoregressive_cache()
    rgb_native: list[np.ndarray] = []
    seed_cache: list[torch.Tensor] = []
    for k in range(N):
        rgb = decoder(input=latents[k], autoregressive_index=k, cache=cache_t)
        rgb_native.append(rgb.detach().float().cpu().numpy())
        if k == 0:
            seed_cache = [cache_t.dec_state[id(mb)].detach().clone() for mb in mbs]
    print(
        f"native: chunk0 rgb {rgb_native[0].shape} (first-chunk trim), "
        f"steady rgb {rgb_native[1].shape}"
    )

    # --- [A] cache-as-IO WRAPPER (torch), threaded, vs native ---
    print("\n[A] wrapper (torch) cache-as-IO vs native streaming decode:")
    wrapper = StreamingWrapper(decoder).eval()
    cache_w = [c.clone() for c in seed_cache]
    for k in range(1, N):
        out = wrapper(latents[k], *cache_w)
        rgb_w = out[0].detach().float().cpu().numpy()
        cache_w = list(out[1:])
        if rgb_w.shape != rgb_native[k].shape:
            print(f"  chunk {k}: SHAPE MISMATCH {rgb_w.shape} vs {rgb_native[k].shape}")
            continue
        print(f"  chunk {k}: PSNR {psnr(rgb_w, rgb_native[k]):.1f} dB")
    print("  → expect ~exact (very high, flat) = dict<->list cache logic is faithful")

    # --- [B] exported ONNX (onnxruntime), threaded, vs native ---
    print("\n[B] exported ONNX (onnxruntime) cache-as-IO vs native streaming decode:")
    try:
        import onnxruntime as ort
    except Exception as exc:  # noqa: BLE001
        print(f"  onnxruntime not available ({exc}); skipping ONNX check.")
        print(
            "  (install: uv pip install onnxruntime-gpu — [A] already proves the cache logic)"
        )
        return

    onnx_path = HERE / "vae_decoder_streaming.onnx"
    if not onnx_path.exists():
        print(f"  {onnx_path.name} missing — run export_streaming.py first.")
        return
    sess = ort.InferenceSession(
        str(onnx_path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    print(f"  providers: {sess.get_providers()}")
    cache_o = [c.detach().float().cpu().numpy() for c in seed_cache]
    for k in range(1, N):
        feeds = {"latent": latents[k].detach().float().cpu().numpy()}
        for i, c in enumerate(cache_o):
            feeds[f"cache_in_{i}"] = c
        outs = sess.run(None, feeds)
        rgb_o, cache_o = outs[0], outs[1:]
        if rgb_o.shape != rgb_native[k].shape:
            print(f"  chunk {k}: SHAPE MISMATCH {rgb_o.shape} vs {rgb_native[k].shape}")
            continue
        print(f"  chunk {k}: PSNR {psnr(rgb_o, rgb_native[k]):.1f} dB")
    print("  → high, non-drifting PSNR across chunks = export is faithful + continuous")


if __name__ == "__main__":
    main()
