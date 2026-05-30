# HANDOFF — HY-WorldPlay perf + model-card MR (GB300)

Audience: a fresh Claude Code agent on a GB300 (or any ≥80 GB GPU) box.
You do **not** need the prior chat history — everything actionable is here.
Companion: `tasks.md` (status). Tracking issue: NVIDIA/flashdreams **#203**.
Parent PR **#155 is merged into `main`**.

## Where things stand

The three code follow-ups are **done — CPU- and GPU-verified — and open as PRs**
(awaiting CI). Nothing left to verify on them; they just need review/merge:

| PR | Branch | What | Verified |
|---|---|---|---|
| **#222** | `wenqing/hy-worldplay-pose-json-default` | pose parser `==`→`>=` prefix-slice; `--example-data` auto-downloads the pose JSON | real `--example-data num_chunk=1` rollout → valid 13-frame 704×1280 mp4 |
| **#223** | `wenqing/hy-worldplay-vae-pth-transform` | native `.pth` VAE transform; **default flipped** to the `.pth` | native vs diffusers weights **bit-identical** (196/196 fp32, max \|Δ\|=0) |
| **#224** | `wenqing/hy-worldplay-dit-native-ckpt` | native DiT path; **diffusers-DiT 404 fix** | native vs diffusers weights **bit-identical** (825/825 fp32, max \|Δ\|=0) |

Verification method was **weight-equality** (load both checkpoints, compare tensors
after each transform) — stronger than a decode smoke: identical weights ⇒ identical
output. Manual tests codify it: `test_wan22_vae_pth_and_diffusers_weights_identical`,
`test_native_dit_matches_diffusers_weights`.

Two corrections from the original plan (already reflected in the PRs):
- **DiT remap is load-bearing** — the HY *distilled* checkpoint is diffusers-keyed and
  routes through `wan22_ti2v_5b_dit_state_dict_transform` (`hy_worldplay/_checkpoint.py`),
  so it can't be deleted. Native path is a proven-equivalent option, not a replacement.
- Only the **VAE** default was flipped (its remap was truly optional).

## ⚠️ Known bugs you'll hit (don't rabbit-hole)

1. **Base HY pipeline can't load without `--ckpt-path`** — the base Wan ckpt lacks the
   zero-init HY keys (`o_prope`, `action_embedding`) and `Wan21Transformer` does a
   *strict* `load_state_dict`. So **always pass `--ckpt-path <distilled model.pt>`**.
   (Flagged on #203 for a separate fix; not your problem here.)
2. **diffusers DiT single-file URL 404s** — fixed in #224 (points at the sharded
   `.safetensors.index.json`). If you're on a branch without #224, use `--ckpt-path`.

## The remaining work: perf re-bench + samples + model-card MR (#203)

This is why the GB300 was wanted. The 6000 Ada capped both legs at `num_chunk=2`, so
the "discard first 5 AR steps" methodology never fit. A ≥80 GB GPU unlocks it. New MR
off `main`, linked to #203.

### Prereqs
```bash
# distilled HY checkpoint (gated repo — needs access):
hf download tencent/HY-WorldPlay --include "wan_distilled_model/*" --local-dir /path/to/models
export CKPT_PATH=/path/to/models/wan_distilled_model/model.pt
export HF_TOKEN=<token>
```
Base Wan 2.2 ckpts auto-download. On the original box these were all cached, plus the
distilled `model.pt` lived at
`integrations/hy_worldplay/tests/parity_check/HY-WorldPlay/hf_models/wan_distilled_model/model.pt`.

### 4.1 Provision the bench (vendor tree + heavy deps) — one-time
```bash
cd integrations/hy_worldplay/tests/parity_check
bash run.sh            # clones HY-WorldPlay, builds the vendor sub-venv, downloads vendor ckpts (~40 GB)
( cd HY-WorldPlay && git log -1 --oneline )   # sanity
```

### 4.2 Matched native-vs-vendor bench at `num_chunk=8`, warmup-discard 5
```bash
cd integrations/hy_worldplay/tests/parity_check
NUM_CHUNK=8 WARMUP_CHUNKS=5 POSE="w-31" SEED=0 CKPT_PATH="$CKPT_PATH" \
    IMAGE_PATH="$PWD/../../../../data_local/cat_surf.jpg" \
    bash bench.sh
# bench.sh sets HY_VENDOR_PROFILE=1, HY_VENDOR_NOISE_MODE=1, USE_KV_CACHE_TRUE=1 and
# scopes expandable_segments to the vendor leg automatically.
# POSE: num_chunk*4-1 = 31 motions. WARMUP_CHUNKS=5 = the manager spec (was forced to 0
# on the 44 GiB 6000 Ada). Defaults: NUM_CHUNK=2, WARMUP_CHUNKS=0, POSE=w-7.
```
Requirements the Slack thread pinned (verify all hold):
- **Scope = DiT + VAE enc/dec**, not DiT-only. Both legs instrumented: native via the
  pipeline `EventProfiler` (`enable_sync_and_profile=True` → per-chunk encode/diffuse/
  decode in `stats_*.json`); vendor via `vendor_profile_patch.py` (`HY_VENDOR_PROFILE=1`).
- **Both legs cuDNN SDPA + torch.compile** (native network-level; vendor block-level
  `@torch.compile`, already upstream).
- Keep `expandable_segments:True` scoped to the vendor leg (breaks native CUDA graphs).

### 4.3 Per-stage medians (bench.sh does this)
`bench.sh` runs `bench_summary.py --warmup-chunks "$WARMUP_CHUNKS" …` and writes per-stage
(encode/diffuse/decode) medians + end-to-end to `bench.md` in `OUTPUT_DIR`. Just read it.
Expected: native wins big on VAE (streaming AR vs vendor's non-streaming `AutoencoderKLWan`);
DiT closer. Same stack both legs → apples-to-apples.

### 4.4 (optional) multi-GPU — only if 1× GB300 can't hit the target and vendor supports it
Vendor shards `num_chunk=4` across 4 GPUs; mirror lingbot's 4-GPU ulysses report. Skip otherwise.

### 4.5 Curated sample MP4s
```bash
NUM_CHUNK=8 CKPT_PATH="$CKPT_PATH" bash bench_batch.sh   # per-image mp4 + stats, aggregates bench_all.md
```
Pick 3–6 (varied first frames + camera moves); optional sidecar `<stem>.txt` prompts via
`bench_pairs.sh`. These become the model-card hero + gallery.

### 4.6 Author the model-card page
Mirror LingBot exactly:
- Template: `docs/source/models/lingbot_world.rst` (title + `.. raw:: html` hero video,
  Installation, Running, Variants `list-table`, CLI args, sample `<video>` grid, perf).
- Create `docs/source/models/hy_worldplay.rst`.
- Perf data → `docs/source/_static/performance/hy_worldplay/perf-<MMDD>.md`
  (mirrors `_static/performance/lingbot_world/perf-0521.md`).
- Sample videos: follow how `lingbot_world.rst` references its `<video src=...>` paths.
- Register in `docs/source/models/index.rst` (toctree + the `:doc:` bullet list).

### 4.7 Build + verify docs
```bash
uv run --group docs sphinx-build -b html docs/source /tmp/docs-out
# open /tmp/docs-out/models/hy_worldplay.html — hero + gallery play, perf table renders
```
The `docs` CI job (`.github/workflows/doc.yml`) rebuilds on the MR.

## Gotchas learned the hard way

- **Always `--ckpt-path <distilled>`** for any rollout (base path is broken — see bug #1).
- **CI-pinned ruff is `0.12.7`** — `uvx ruff` defaults to newer and re-sorts imports
  differently (touches unrelated files). Always `uvx ruff@0.12.7 …`.
- `uv run`/`uv sync` builds `block-sparse-attn` (CUDA ext) — needs `CUDA_HOME` set.
- `expandable_segments:True` is **incompatible with CUDA graphs** — vendor leg only.
- The first AR chunk's `diffuse` time is dominated by `torch.compile` autotune (cold) —
  not steady-state; that's exactly why `WARMUP_CHUNKS=5` discards the early chunks.
- Bench env-var overrides: `IMAGE_PATH`/`IMAGES_DIR`, `NUM_CHUNK`, `WARMUP_CHUNKS`, `POSE`,
  `SEED`, `OUTPUT_DIR`, `CKPT_PATH`.

## Full history (optional)

Raw transcript (on the original box):
`~/.claude/projects/-localhome-local-wenqingw-projs-flashdreams/1e82f3c3-e07a-466a-8480-4dd18b916773.jsonl`
You don't need it — this file + `tasks.md` + the PRs are the handoff. `claude --resume`
will **not** reliably pick it up cross-machine.
