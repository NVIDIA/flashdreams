# HY-WorldPlay — task status

PR **#155 is merged into `NVIDIA/flashdreams` main** ✅ (squash `9222500`).
Tracking issue for follow-ups: **#203**.

## Done (in #155)

- Native HY-WorldPlay WAN-5B I2V integration (Wan 2.2 TI2V-5B backbone + action/PRoPE/memory conditioners, KV cache, 4-step distilled Euler).
- New `integrations/wan22` workspace member.
- Review fixes: lint/ty CI green · `--example-data` default demo → `data_local/` · dropped `cli.py`.
- Parity: mean |Δ| 15.65/255 vs vendor (acceptance ≤20).

## Issue #203 — code follow-ups: 4 PRs, all checks green, auto-merge armed (pending 1 review)

Each is a small, self-contained PR off `main`, CPU+GPU verified (RTX 6000 Ada), all checks green (cpu/gpu/docs/OSRB/REUSE). Auto-merge armed — they land on approval.

| Item | PR | GPU verification | Outcome |
|---|---|---|---|
| Pose JSON default | **#222** | real `--example-data` `num_chunk=1` rollout → valid 13-frame 704×1280 MP4 | done |
| VAE `.pth` transform | **#223** | native `.pth` vs diffusers → **bit-identical** (196/196 fp32, max \|Δ\|=0) | **default flipped** to `.pth`; diffusers kept as opt-in fallback |
| DiT native checkpoint | **#224** | native vs diffusers → **bit-identical** (825/825 fp32, max \|Δ\|=0) | remap is **load-bearing** (HY distilled ckpt is diffusers-keyed) → kept; native path is a proven-equivalent option. Also **fixes diffusers DiT 404** (sharded index). |

Key correction vs the original plan: the **DiT remap can't be deleted** — `hy_worldplay/_checkpoint.py` layers the distilled-ckpt rewrites on top of it. Only the VAE remap became truly optional (hence the VAE default flip, not the DiT).

### Bug spun off during verification — FIXED (#227)

- Base HY pipeline couldn't load **without `--ckpt-path`** (base Wan ckpt lacks zero-init HY keys + strict `load_state_dict`). **Fixed in #227**: `HyWorldPlayWanDiTNetwork.load_state_dict` tolerates only the HY zero-init keys when absent. Verified end-to-end (base rollout → valid mp4) + CPU tests.

### CI status

All four PRs (`#222` `#223` `#224` `#227`) are **fully green** (cpu/gpu/docs/OSRB/REUSE) with **auto-merge armed** — only a review approval is outstanding.

## Issue #203 — perf / docs: re-bench + model card DONE; MR pending (on GB300, branch `wenqing/hy-worldplay-perf-handoff`)

One follow-up MR. Full command-level steps in `HANDOFF.md`.

| Item | Status |
|---|---|
| Re-bench `num_chunk=8`, `warmup_chunks=5`, DiT + VAE enc/dec scope, both legs cuDNN SDPA + torch.compile | **DONE** — matched bench ran on a single GB300, corroborated across 6 inputs |
| Curated samples | **DONE** — native mp4s for all 5 `data_local/*`; hero (`6`) + 3 gallery (`2`,`1`,`cat_surf`) transcoded to web mp4 in `docs/source/_static/videos/hy_worldplay/` (3.4 MB total) |
| Model-card page (mirror `lingbot_world.rst` + `_static/performance/`) | **DONE** — `docs/source/models/hy_worldplay.rst` authored (lingbot style: hero, install, running, variants list-table, native sample grid, perf chart), registered in `models/index.rst` + the `index.rst` toctree, **builds clean under `sphinx-build -W`** |

### Re-bench result (704×1280, seed 0, pose `w-31`, warmup-discard 5, DiT+VAE scope, both legs cuDNN SDPA + `torch.compile`)

Post-warmup medians (chunks 5–7):

| stage | native | vendor | |
|---|--:|--:|--:|
| DiT (diffuse) | 632 ms | 1206 ms | **1.91×** |
| VAE decode | 383 ms | 372 ms | parity |
| DiT+VAE / chunk | 1015 ms | 1578 ms | **1.55×** |

- **Reverses the original expectation** ("native wins big on VAE, DiT closer"): native wins big on **DiT**; VAE decode is a tie.
- These are the **production-config** numbers (use_cuda_graph=True, after bug 2's fix). The steady-state post-warmup chunks are memory-engaged and run eager either way, so CUDA graphs only accelerate the discarded warmup chunks — the reported medians are graph-independent (verified: 632 ms graphs-off vs 631.7 ms graphs-on).
- Vendor forced onto cuDNN SDPA via `HY_VENDOR_SDPA=1` (now the `bench.sh` default); vendor as-shipped uses sageattention.
- Artifacts: `tests/parity_check/outputs/test/{native,vendor}/*.mp4` + `outputs/test/bench.md` (gitignored); chart data `docs/source/_static/performance/hy_worldplay/perf-0530.md` (committed).

### Full `data_local/*` batch (native vs vendor, same config; 5 images)

Post-warmup (chunks 5–7) medians, ms. Native = production config (CUDA graphs on).

| image | DiT nat/ven | VAE nat/ven | DiT+VAE nat/ven | ratio | parity \|Δ\| |
|---|--:|--:|--:|--:|--:|
| 1.png | 630 / 1208 | 382 / 374 | 1012 / 1583 | 1.56× | 40.8 |
| 2.png | 632 / 1210 | 382 / 374 | 1014 / 1584 | 1.56× | 31.1 |
| 6.jpeg | 628 / 1208 | 381 / 374 | 1008 / 1582 | 1.57× | 21.1 |
| cat_surf.jpg | 632 / 1207 | 381 / 375 | 1014 / 1582 | 1.56× | 51.8 |
| jensen_alaska.jpg | 633 / 1207 | 382 / 375 | 1015 / 1582 | 1.56× | 48.6 |
| **median** | **632 / 1208** | **382 / 374** | **1014 / 1582** | **1.56×** | — |

- **Perf is input-independent** (native DiT 628–633 ms across all 5), corroborating the `perf-0530.md` headline (1015 ms) on 6 distinct inputs total.
- Parity 21–52/255 is benign cumulative AR drift (bug 3); highest on off-aspect inputs (`cat_surf` 625×350 upscaled, `jensen_alaska` 900×1200 portrait cropped).
- Artifacts in `integrations/hy_worldplay/tests/parity_check/outputs/<stem>/{native,vendor}/hy-worldplay-wan-i2v-5b.mp4` + per-image `bench.md` + aggregated `outputs/SUMMARY.md` (stems: `1 2 6 cat_surf jensen_alaska test`). `outputs/` is gitignored (local only — not committed). Re-run the whole set with `bash outputs/run_all.sh`. Cleanest first frames for the gallery: `6.jpeg`, `2.png`, `1.png`.

### Bugs found at `num_chunk≥4` (never reachable on the old 44 GiB card)

1. **FIXED** (`a5f7b74`) — `prefill_memory_kv_cache` asserted `isinstance(self.network, HyWorldPlayWanDiTNetwork)`; under `compile_network=True` that's a `torch.compile` `OptimizedModule`. Now unwraps `_orig_mod`.
2. **FIXED** (`eb121aa`) — `use_cuda_graph=True` (production default) + the data-dependent memory prepend (`torch.cat` lengthens the attention sequence once memory engages) → `cudaErrorIllegalAddress` at AR chunk 4 (graph captured pre-memory can't replay the longer post-memory sequence). Fix: `predict_flow` flags memory-engaged steps and `_select_network` routes them to the wrapper's eager `drain`; pre-memory chunks keep the graph. Verified num_chunk=8 graphs-ON end-to-end. (Graph-accelerating the memory-engaged steady state would need fixed-size in-place memory KV buffers — future opt.)
3. **INVESTIGATED — benign** — parity mean |Δ| 29/255 at `num_chunk=8` (vs 15.65 at `num_chunk=2`) is **cumulative bf16 autoregressive drift, not a bug**. Per-frame |Δ| ramps smoothly: chunk-0 ≈14, chunk-1 ≈18.6 (reproduces the README's documented 12.91 / 18.21 at num_chunk=2), rising to ~49 by chunk 7 — **no discontinuity at memory engagement** (frame ~64). The num_chunk=8 *mean* is higher only because it averages in more later high-drift chunks; the per-chunk curve is unchanged. Memory-frame selection is identical native-vs-vendor for `w-31` (verified cloud-independent at the real chunk indices). Latent risk for non-trivial poses: the FOV point cloud is unseeded global RNG on both sides (`generate_points_in_sphere(generator=None)` / vendor bare `torch.rand`) → could flip selection on rotation/strafe poses; worth seeding native's generator for reproducibility.

### Harness fixes (`2119225`)

`bench.sh` defaults `HY_VENDOR_SDPA=1`; `run.sh` adds `torchvision==0.26.*` to vendor heavy-deps (upstream HEAD's hyvideo import now needs it).

## Next actions

1. Land #222 / #223 / #224 / #227 — all green + auto-merge armed, just need one review approval.
2. Open the perf/docs MR off this branch (`wenqing/hy-worldplay-perf-handoff`) → `main`: the `_action.py` CUDA-graph + compile fixes, the bench-harness fixes, `perf-0530.md`, and the `hy_worldplay.rst` model card + sample videos.
3. (model-card media) The gallery currently uses **committed local** mp4s in `docs/source/_static/videos/hy_worldplay/` (3.4 MB, web-transcoded). If matching LingBot's external-hosting convention is preferred, re-host on `research.nvidia.com/.../assets/hy_worldplay/` and swap the `<source>` srcs.
4. (optional) Seed the FOV Monte-Carlo point cloud (`generate_points_in_sphere` generator) for reproducibility on rotation/strafe poses (bug 3 latent risk).
5. (future opt) Graph-accelerate the memory-engaged steady state via fixed-size in-place memory KV buffers.
