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
| `1.png`  | 5121 / 27939 | 2712 / 3192 | 3.97× | 36.0 |
| `2.png`  | 4941 / 28021 | 2711 / 3196 | 4.08× | 28.4 |
| `5.jpeg` | 4962 / 27909 | 2712 / 3195 | 4.05× | 36.4 |
| `6.jpeg` | 5407 / 27932 | 2712 / 3187 | 3.83× | 21.4 |
| `10.png` | 5085 / 28007 | 2712 / 3200 | 4.00× | 46.8 |
| **median** | **5085 / 27939** | **2712 / 3195** | **4.00×** | **36.0** |

`mean |Δ|` is the per-pixel mean absolute difference (uint8 / 255) between the
native and vendor MP4s — cumulative bf16 autoregressive drift across 8 chunks,
not a per-frame error bar. Peak GPU memory ≈ 45 GiB both legs.

## Visual check

All 5 native rollouts were spot-checked at the chunk-2 and chunk-5 boundaries
(the sites of the pre-fix CUDA-graph speckle corruption): every chunk renders
coherently — the subject walks through its scene with no shatter artifacts.
The `use_cuda_graph=False` fix holds across all inputs.
