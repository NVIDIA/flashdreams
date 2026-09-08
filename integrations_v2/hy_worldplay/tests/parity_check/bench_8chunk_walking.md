# HY-WorldPlay WAN-5B I2V — native vs vendor perf (8-chunk, "a person walking")

Machine: single **GB300** (AArch64 / SBSA), driver 595.71.05, torch 2.11.0+cu130.
Both legs: cuDNN SDPA + `torch.compile` (Inductor). Native runs with
`use_cuda_graph=False` — see the CUDA-graph corruption fix in
`config.py` (graph capture is unsafe on the per-chunk memory-prefill path).

**Config:** `num_chunk=8`, `pose=w-31`, `seed=0`, 704×1280, prompt
`"a person walking"`, warmup-discard 5 (post-warmup medians over chunks 5–7).
Inputs: `HY-WorldPlay/assets/img/{1.png, 2.png, 5.jpeg, 6.jpeg, 10.png}`.

Generated with `bench_pairs.sh` (drives `bench.sh` — upstream `wan/generate.py`
via `run.sh` + the native plugin — once per image). Per-image artifacts
(native + vendor MP4 + stats JSON) under `outputs/bench_pairs_walking/<stem>/`.

> **Metric basis:** `DiT (diffuse)` is the per-AR-step (per-chunk) median
> reported by `bench_summary.py`, i.e. all denoising forwards for one chunk.
> This is a different basis from PR #231's per-forward numbers; compare
> ratios, not absolute ms, against that PR.

## Per-stage medians (across the 5 images)

| stage | native | vendor | speedup |
|-------|--------|--------|---------|
| DiT (diffuse) | 5085 ms | 27939 ms | **5.49×** |
| VAE decode | 2712 ms | 3195 ms | 1.18× |
| **DiT + VAE / chunk** | **7797 ms** | **31135 ms** | **3.99×** |

## Per-input results

| image | DiT nat/ven (ms) | VAE nat/ven (ms) | ratio (DiT+VAE) | mean `\|Δ\|` |
|-------|------------------|------------------|-----------------|--------------|
| `1.png`  | 5121 / 27939 | 2712 / 3192 | 3.97× | 27.8 |
| `2.png`  | 4941 / 28021 | 2711 / 3196 | 4.08× | 22.6 |
| `5.jpeg` | 4962 / 27909 | 2712 / 3195 | 4.05× | 25.7 |
| `6.jpeg` | 5407 / 27932 | 2712 / 3187 | 3.83× | 16.6 |
| `10.png` | 5085 / 28007 | 2712 / 3200 | 4.00× | 24.0 |
| **median** | **5085 / 27939** | **2712 / 3195** | **4.00×** | **24.0** |

`mean |Δ|` is the per-pixel mean absolute difference (uint8 / 255) between the
native and vendor MP4s — cumulative bf16 autoregressive drift across 8 chunks,
not a per-frame error bar. Peak GPU memory ≈ 45 GiB both legs.

Parity is measured with the Wan 2.2 VAE patchify channel-order fix (#338) on the
native leg; before that fix a ~2px VAE-decode checkerboard inflated the median
to 36.0. Perf (DiT / VAE timings, speedups) is unchanged by that fix — it is an
einops axis swap with no compute cost — so only the `mean |Δ|` column moved.

## Visual check

All 5 native rollouts (regenerated with #338) are checkerboard-free; chunk-2 /
chunk-5 boundaries (the former CUDA-graph speckle sites) also render coherently.
Both the `use_cuda_graph=False` and the VAE patchify-order fixes hold across all
inputs.
