# HY-WorldPlay — task status

PR **#155 is merged into `NVIDIA/flashdreams` main** ✅ (squash `9222500`).
Tracking issue for follow-ups: **#203**.

## Done (in #155)

- Native HY-WorldPlay WAN-5B I2V integration (Wan 2.2 TI2V-5B backbone + action/PRoPE/memory conditioners, KV cache, 4-step distilled Euler).
- New `integrations/wan22` workspace member.
- Review fixes: lint/ty CI green · `--example-data` default demo → `data_local/` · dropped `cli.py`.
- Parity: mean |Δ| 15.65/255 vs vendor (acceptance ≤20).

## Issue #203 — code follow-ups: PRs OPEN, CPU+GPU verified (RTX 6000 Ada)

Each is a small, self-contained PR, rebased onto `main`, lint-clean (ruff 0.12.7). Awaiting review + CI.

| Item | PR | GPU verification | Outcome |
|---|---|---|---|
| Pose JSON default | **#222** | real `--example-data` `num_chunk=1` rollout → valid 13-frame 704×1280 MP4 | done |
| VAE `.pth` transform | **#223** | native `.pth` vs diffusers → **bit-identical** (196/196 fp32, max \|Δ\|=0) | **default flipped** to `.pth`; diffusers kept as opt-in fallback |
| DiT native checkpoint | **#224** | native vs diffusers → **bit-identical** (825/825 fp32, max \|Δ\|=0) | remap is **load-bearing** (HY distilled ckpt is diffusers-keyed) → kept; native path is a proven-equivalent option. Also **fixes diffusers DiT 404** (sharded index). |

Key correction vs the original plan: the **DiT remap can't be deleted** — `hy_worldplay/_checkpoint.py` layers the distilled-ckpt rewrites on top of it. Only the VAE remap became truly optional (hence the VAE default flip, not the DiT).

### Bug spun off during verification — FIXED

- Base HY pipeline couldn't load **without `--ckpt-path`** (base Wan ckpt lacks zero-init HY keys + strict `load_state_dict`). **Fixed in #227**: `HyWorldPlayWanDiTNetwork.load_state_dict` tolerates only the HY zero-init keys when absent. Verified end-to-end (base rollout → valid mp4) + CPU tests.

### CI status — all four PRs

`#222` `#223` `#224` are **fully green** (cpu/gpu/docs/OSRB/REUSE). `#227` just opened (awaiting `/ok to test`).

## Issue #203 — perf / docs: NOT STARTED (needs a larger-memory GPU / GB300)

These travel together as one follow-up MR. Full command-level steps in `HANDOFF.md`.

| Item | Est. |
|---|---|
| Re-bench `num_chunk=8`, `warmup_chunks=5`, DiT + VAE enc/dec scope, both legs cuDNN SDPA + torch.compile | 1–2 hrs (config bump; bench wall-clock) |
| Generate curated samples | 1–2 hrs |
| Model-card page (mirror `docs/source/models/lingbot_world.rst` + `_static/performance/`) | 2–4 hrs |
| (optional) mgpu perf — only if 1× GB300 can't reach `num_chunk=8` and vendor supports mgpu | 3–5 hrs |

Prereq: larger-mem GPU + checkpoints (~40 GB vendor + ~20 GB native DiT + ~15 GB diffusers/VAE). Pre-staging ckpts cuts wall-clock roughly in half.

## Next actions

1. Get #222 / #223 / #224 reviewed + merged (all green). `/ok to test` + merge #227 (base-load fix).
2. GB300 perf MR: re-bench (`num_chunk=8`) → samples → model-card page (`HANDOFF.md`).
