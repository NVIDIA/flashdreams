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

## Issue #203 — perf / docs: IN PROGRESS (on GB300, branch `wenqing/hy-worldplay-perf-handoff`)

One follow-up MR. Full command-level steps in `HANDOFF.md`.

| Item | Status |
|---|---|
| Re-bench `num_chunk=8`, `warmup_chunks=5`, DiT + VAE enc/dec scope, both legs cuDNN SDPA + torch.compile | **DONE** — matched bench ran on a single GB300 |
| Curated samples | not started (deferred pending parity/graph follow-ups) |
| Model-card page (mirror `lingbot_world.rst` + `_static/performance/`) | perf data + draft done; held as `.rst.draft` until samples land |
| (optional) mgpu perf | not needed — 1× GB300 fits `num_chunk=8` |

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
- Artifacts: `tests/parity_check/outputs/bench/bench.md`; chart data `docs/source/_static/performance/hy_worldplay/perf-0530.md`.

### Bugs found at `num_chunk≥4` (never reachable on the old 44 GiB card)

1. **FIXED** (`a5f7b74`) — `prefill_memory_kv_cache` asserted `isinstance(self.network, HyWorldPlayWanDiTNetwork)`; under `compile_network=True` that's a `torch.compile` `OptimizedModule`. Now unwraps `_orig_mod`.
2. **FIXED** (`eb121aa`) — `use_cuda_graph=True` (production default) + the data-dependent memory prepend (`torch.cat` lengthens the attention sequence once memory engages) → `cudaErrorIllegalAddress` at AR chunk 4 (graph captured pre-memory can't replay the longer post-memory sequence). Fix: `predict_flow` flags memory-engaged steps and `_select_network` routes them to the wrapper's eager `drain`; pre-memory chunks keep the graph. Verified num_chunk=8 graphs-ON end-to-end. (Graph-accelerating the memory-engaged steady state would need fixed-size in-place memory KV buffers — future opt.)
3. **OPEN** — parity drifted to mean |Δ| 29.2/255 at `num_chunk=8` (vs 15.65 at `num_chunk=2`, bar ≤20; unchanged graphs on/off). Likely the memory-prefill / FOV-selection path diverging over long rollouts. Perf is independent (same FLOPs).

### Harness fixes (`2119225`)

`bench.sh` defaults `HY_VENDOR_SDPA=1`; `run.sh` adds `torchvision==0.26.*` to vendor heavy-deps (upstream HEAD's hyvideo import now needs it).

## Next actions

1. Land #222 / #223 / #224 / #227 — all green + auto-merge armed, just need one review approval.
2. Investigate bug 3 (long-rollout parity drift at num_chunk=8).
3. Generate curated samples → finalize + go-live on `hy_worldplay.rst` → build docs.
