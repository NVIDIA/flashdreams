<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# HY-WorldPlay parity check

Self-contained benchmark of upstream
[HY-WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay) WAN-5B
I2V model. Phase 1 of the integration ships **no** patch on top of
upstream — the parity check is a faithful re-execution of upstream's
own `wan/generate.py` so we can verify that the bundled `flashdreams`
plugin (which delegates to the same `WanRunner.predict()` call)
produces bit-identical output.

## Run

From this directory — i.e.

```
/workspace/flashdreams/integrations/hy_worldplay/tests/parity_check/
```

run:

```bash
bash run.sh
```

Single-GPU defaults are used; override via env vars:

```bash
NUM_GPU=4 NUM_CHUNK=4 POSE='w-16' bash run.sh
```

Other tunables (defaults shown):

| env var | default | meaning |
| --- | --- | --- |
| `NUM_GPU` | `1` | torchrun `--nproc_per_node` |
| `NUM_CHUNK` | `1` | autoregressive chunk count (each chunk = 4 latents) |
| `POSE` | `w-4` | camera trajectory (must total `NUM_CHUNK * 4` latents) |
| `SEED` | `0` | RNG seed |
| `PROMPT` | `"First-person view ... ancient Athens ..."` | text prompt |
| `IMAGE_PATH` | `${REPO_DIR}/assets/img/test.png` | first-frame I2V input |
| `OUTPUT_DIR` | `${REPO_DIR}/outputs/parity` | benchmark output dir |
| `USE_KV_CACHE_TRUE` | `0` | when `1`, swap vendor's `wan/generate.py` for the `use_kv_cache=True` monkey-patch (phase 2b.6 acceptance baseline -- see below) |

### Re-baselining against vendor's `use_kv_cache=True` code path

Phase 2b.6 closes the native HY-WorldPlay runner by validating
parity against vendor's *cache-prefill* code path
(`use_kv_cache=True`) rather than the single-forward-pass default.
Set `USE_KV_CACHE_TRUE=1` to swap the default `wan/generate.py`
invocation for `run_vendor_use_kv_cache.py`, which runtime-monkey-
patches `WanPipeline.__setattr__` so `self.use_kv_cache = False`
(vendor's line 707 inside `predict`) is silently coerced to `True`:

```bash
USE_KV_CACHE_TRUE=1 \
    NUM_CHUNK=2 POSE='w-8' SEED=0 \
    OUTPUT_DIR="${PWD}/HY-WorldPlay/outputs/parity_use_kv_cache_true" \
    bash run.sh
```

The output MP4 lands in `${OUTPUT_DIR}`. Diff against the native
HY runner's output via the inline `imageio` snippet above (or the
ad-hoc `tmp/hy_parity_diff.py` script if it's still present) and
confirm `mean |Δ| ≤ 5 / 255`.

This mode is the **2b.6 acceptance baseline**. The default
(no env var) mode keeps producing the phase-1
`use_kv_cache=False` baseline so older parity numbers remain
comparable.

The script is idempotent: on first run it clones upstream, downloads
`tencent/HY-WorldPlay`'s `wan_transformer/` and `wan_distilled_model/`
checkpoints into `HY-WorldPlay/hf_models/`, and runs the benchmark.
Subsequent runs skip whatever's already in place and just re-run the
benchmark.

## Outputs

Written under `HY-WorldPlay/outputs/parity/` by default:

- `<pose>_<sanitized_prompt>.mp4` — generated video (16 fps)
- `err.txt` — error log (only created on failures)

To compare against the native `flashdreams` plugin output, run the
same inputs through the plugin. Stay inside this directory so `uv run`
resolves to this sub-venv (which has the `hy_worldplay` workspace
member installed alongside the upstream deps):

```bash
uv run flashdreams-run hy-worldplay-wan-i2v-5b \
    --image-path "${IMAGE_PATH}" \
    --ckpt-path HY-WorldPlay/hf_models/wan_distilled_model/model.pt \
    --num-chunk 1 --pose 'w-4' \
    --seed 0 --output-dir outputs/plugin
```

For the matched-input, side-by-side perf + parity workflow used at
PR review time, use `bench.sh` instead — it drives upstream's
`wan/generate.py` via `run.sh` *and* the native plugin in one shot,
then emits a `bench.md` summary. See the parent
[`README.md`](../../README.md) "PR perf + visual sample" section.

The two MP4s should be equivalent (same checkpoint, same pipeline,
same RNG seed). They are **not** bit-for-bit identical because the
plugin and the upstream script run as separate processes against
separate venvs, so they accumulate independent CUDA-stream-ordering
noise, independent autotune-cache state, and independent H.264 encoder
nondeterminism. Compare numerically, not via `cmp`:

```bash
uv run python - <<'PY'
import numpy as np, imageio.v3 as iio
from pathlib import Path
a = iio.imread(next(Path("HY-WorldPlay/outputs/parity").glob("*.mp4")))
b = iio.imread("/workspace/flashdreams/outputs/hy-worldplay-wan-i2v-5b.mp4")
assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
d = np.abs(a.astype(np.int16) - b.astype(np.int16))
print(f"mean |d| : {d.mean():.4f}  (uint8 / 255)")
print(f"max  |d| : {d.max()}        (uint8 / 255)")
print(f"frames with mean |d| > 5: {(d.mean(axis=(1,2,3))>5).sum()}/{a.shape[0]}")
PY
```

### Parity caveats — drift budget and accepted bar

Phase-1 parity is measured numerically against the upstream
`wan/generate.py` output. On the reference single-GPU `--num-chunk 1
--pose w-4 --seed 0` benchmark, **on the same torch version**, the
observed drift is:

| comparison | mean \|Δ\| (uint8) | max \|Δ\| | frames mean\|Δ\|>5 |
| --- | --- | --- | --- |
| **plugin vs upstream (same torch)** | **3.41** | 130 | **0** |
| upstream vs itself (torch 2.11 vs 2.12) | 3.76 | 138 | 0 |
| plugin vs plugin (two runs, same venv) | 0.00 | 0 | 0 |

So the plugin reproduces upstream more tightly than upstream reproduces
itself across a torch minor bump, and **zero frames cross the
"visually noticeable" mean-delta-of-5 threshold**. The plugin itself
is bit-deterministic across runs in the same venv.

Two prerequisites for this bar:

1. **`torch` version must match between the parity venv and the
   flashdreams main venv.** The parity `pyproject.toml` pins
   `torch==2.11.*` to mirror `flashdreams/uv.lock`. When flashdreams
   bumps torch, bump this pin in lockstep — otherwise drift jumps from
   ~3.4 to ~5 (we measured a 3.76 contribution from the 2.11 -> 2.12
   minor alone).
2. **`DEFAULT_PROMPT` in `hy_worldplay/runner.py` must byte-match
   upstream's `wan/generate.py` `--input` argparse default.** An early
   version had a trailing `.` that shifted the UMT5 tokenisation by
   one token and added ~2 of drift on its own. Now guarded by
   `tests/test_smoke.py::test_default_prompt_byte_matches_upstream`,
   so a regression fails CPU-only pytest before it ever reaches a GPU
   run.

True bit-for-bit parity would require eliminating the two-venv split
entirely (see the phase-1.5 plan in
`../../README.md` — "Staging plan").

## Isolation

Deps are pinned in this directory's `pyproject.toml` and live in
`./.venv/`. Because `uv run` walks upward looking for a project, calls
from inside `HY-WorldPlay/` resolve to *this* venv, not the surrounding
flashdreams one.

This sub-venv intentionally **doubles as the plugin run-venv** in
phase 1: it lists `flashdreams-hy-worldplay` as a path source so the
`flashdreams.runner_configs` entry-point registers without a separate
install step, and `flashdreams-run hy-worldplay-wan-i2v-5b` works
here directly. The upstream + plugin runs share an identical dep stack
(same torch / cuBLAS / sageattention / accelerate), which is required
for the parity comparison below. Outside of this directory, use
`uv run --project integrations/hy_worldplay/tests/parity_check ...`
to target the same venv from elsewhere in the repo.

This collapsing also keeps HY-WorldPlay's heavy upstream deps
(sageattention, cloudpickle, ...) out of the repo-root `uv.lock` —
they only appear in this sub-venv's lockfile. The collapse goes away
in phase 2b when the slug becomes a `flashdreams-run` subcommand and
the run path moves back into the main flashdreams venv.

## Files tracked here

- `README.md` — this file
- `run.sh` — clone + setup + benchmark, idempotent
- `pyproject.toml` — isolated venv definition (materialized via `uv sync`)
- `.gitignore` — ignores the cloned `HY-WorldPlay/` tree, `./.venv/`, caches

## Runtime requirements

- NVIDIA GPU with CUDA support (single-GPU runs use ~25 GB; 4-GPU
  runs spread the same memory budget across SP).
- `HF_TOKEN` exported with read access to `tencent/HY-WorldPlay`.
- ~30 GB free disk for the upstream tree + WAN-5B checkpoints +
  Wan2.2 base model HF cache.
- (Optional) `sageattention` installed; HY-WorldPlay's WAN pipeline
  flags it as required but the upstream code falls back to PyTorch
  SDPA when `--use_sageattn false` (the script's default).
