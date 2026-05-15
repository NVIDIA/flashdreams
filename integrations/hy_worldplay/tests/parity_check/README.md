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

A `changes.patch` slot is wired up in `run.sh` so that a future patch
(e.g. `EventProfiler`-based per-chunk timing mirroring the
`self_forcing` parity check) can be dropped in without touching the
script.

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

The script is idempotent: on first run it clones upstream, downloads
`tencent/HY-WorldPlay`'s `wan_transformer/` and `wan_distilled_model/`
checkpoints into `HY-WorldPlay/hf_models/`, and runs the benchmark.
Subsequent runs skip whatever's already in place and just re-run the
benchmark.

## Outputs

Written under `HY-WorldPlay/outputs/parity/` by default:

- `<pose>_<sanitized_prompt>.mp4` — generated video (16 fps)
- `err.txt` — error log (only created on failures)

To compare against the `flashdreams` plugin output, run the same
inputs through the wrapper:

```bash
uv run python -m hy_worldplay.cli \
    --image-path "${IMAGE_PATH}" \
    --ar-model-path HY-WorldPlay/hf_models/wan_transformer \
    --ckpt-path HY-WorldPlay/hf_models/wan_distilled_model/model.pt \
    --hy-worldplay-repo-root HY-WorldPlay \
    --num-chunk 1 --pose 'w-4' \
    --seed 0 --output-dir outputs/wrapper
```

The two MP4s should be identical (same checkpoint, same pipeline,
same RNG seed). A binary `cmp` is the simplest verification:

```bash
cmp HY-WorldPlay/outputs/parity/*.mp4 outputs/wrapper/hy-worldplay-wan-i2v-5b.mp4
```

> **Note:** ffmpeg-encoded MP4s embed an encoder-version stamp in the
> container header, so `cmp` may flag a few bytes there even when the
> video frames are identical. Fall back to a per-frame PSNR / SSIM
> check (e.g. via `mediapy.read_video`) for a robust comparison.

## Isolation

Deps are pinned in this directory's `pyproject.toml` and live in
`./.venv/`. Because `uv run` walks upward looking for a project, calls
from inside `HY-WorldPlay/` resolve to *this* venv, not the surrounding
flashdreams one.

## Files tracked here

- `README.md` — this file
- `run.sh` — clone + setup + (patch) + benchmark, idempotent
- `pyproject.toml` — isolated venv definition (materialized via `uv sync`)
- `.gitignore` — ignores the cloned `HY-WorldPlay/` tree, `./.venv/`, caches

`changes.patch` is intentionally **not** present in phase 1 (no
upstream edits required). Add it later when introducing
`EventProfiler` timing or any other in-tree instrumentation.

## Runtime requirements

- NVIDIA GPU with CUDA support (single-GPU runs use ~25 GB; 4-GPU
  runs spread the same memory budget across SP).
- `HF_TOKEN` exported with read access to `tencent/HY-WorldPlay`.
- ~30 GB free disk for the upstream tree + WAN-5B checkpoints +
  Wan2.2 base model HF cache.
- (Optional) `sageattention` installed; HY-WorldPlay's WAN pipeline
  flags it as required but the upstream code falls back to PyTorch
  SDPA when `--use_sageattn false` (the script's default).
