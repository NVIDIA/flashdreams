<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Caption model — offline training pipeline

Trains the latent-native live-caption model that runs in the browser on the
video-token path (serving side: `omnidreams.webrtc.caption_artifacts`). The
model consumes the same latents the VAE decoder does — never pixels — and emits
a short ego-motion scene caption.

## Phase 0 (locked)

- **Caption bank (6 classes):** `driving_straight`, `turning_left`,
  `turning_right`, `slowing_or_stopping`, `stopped`, `reversing`. Starting set;
  revisable after teacher labeling.
- **Window K = 5 chunks** (10 latent frames ≈ 1.33 s).
- **Labels:** **control-derived**. The road-keeping capture keeps the ego
  on-road, so the control state is exact ground truth for the 6 classes →
  plain cross-entropy, no VLM teacher needed. (Cosmos-Reason1-7B labeling is
  retained but *optional* — for a cross-check, or later content classes that
  ego motion can't provide.)

## Phases

1. **capture** (`capture.py`) — drive the world model with a **road-keeping**
   WSAD track; persist `(latent-window, RGB-window, metadata)` shards, each
   carrying a control-derived `label`. A detail guard drops off-road windows.
2. **label_teacher** (`label_teacher.py`, *optional*) — Cosmos-Reason1-7B VLM
   labeling. Superseded by control labels for the 6 ego-motion classes (an
   early VLM pass mislabeled off-road turn footage); kept for cross-checks /
   future content classes.
3. **dataset** (`dataset.py`) — read shards + control labels, split **by
   session**, class weights, latents pre-loaded into RAM.
4. **train** (`train.py`) — the caption CNN (`model.py`, ~230K params): weighted
   cross-entropy, AdamW, per-epoch val (top-1 + per-class recall + confusion),
   best-val checkpoint.
5. **export** (`export.py`) — trained checkpoint → ONNX + `.spec.json`, straight
   into the served cache dir with `--to-cache`.

## Data layout (out of git)

Default data root `~/data/omnidreams-caption/`:

```
raw/         # capture shards: raw-000000.tar ... + manifest.jsonl
labeled/     # Phase 2: manifest joined with teacher labels
checkpoints/ # Phase 4
exports/     # Phase 5 staging before the cache dir
```

Keep the data root outside the repo (add it to your global gitignore); shards
are large and regenerable.

## Capture shard format (WebDataset-compatible)

One sample per rolling window, keyed by a zero-padded id:

| Field | Contents |
|---|---|
| `NNNNNNN.latents.npy` | `[K*T, 16, 88, 160]` fp16 — the latent window, narrowed to fp16 exactly as the `raw_f16` token codec streams it |
| `NNNNNNN.rgb.MMM.jpg` | evenly-spaced decoded RGB frames spanning the window (visual QA / optional teacher) |
| `NNNNNNN.json` | metadata: `label` + `label_source: "control"`, `label_per_chunk`, `chunk_index_range`, `wsad_per_chunk`, scene, seed |

`manifest.jsonl` has one line per sample (`__key__`, `session_id`, `label`,
`chunk_index_range`) for indexing and the Phase-3 split.

> `label` is the per-window ego-motion class from the controls. Because the
> road-keeping plan keeps the ego on-road, the control state matches the actual
> motion, so `label` is training ground truth (not just a weak hint). Windows
> that still drift off-road are dropped by the `--min-detail` guard.

## Running the capture (GPU host)

Constructs the omnidreams runtime (loads checkpoints, builds CUDA graphs), so
this must run on the inference box. Labels are written from the controls — no
teacher run needed.

**Pilot first — verify on-road + calibrate the off-road guard.** Capture with
the guard off (`--min-detail 0`), view a few RGB frames per `label` to confirm
the ego stays on-road and turns go the right way, then pick a `--min-detail`
threshold from the window detail-score distribution:

```bash
uv run --project integrations/omnidreams \
  omnidreams-caption-capture \
  --out DATA/raw --session-id pilot --limit 300 --loop
# tune --turn-burst if turns drift off-road (shorter) or are too subtle (longer)
```

**Full capture** — across scenes / variants / seeds for diversity, guard on:

```bash
omnidreams-caption-capture --out DATA/raw \
  --session-id default-rain-s2 --scene-variant rain --seed 2 \
  --limit 1500 --loop --min-detail <calibrated> --max-per-shard 256
```

Key flags: `--turn-burst` (chunks per turn arc, default 3), `--straight-run`
(default 4), `--min-detail` (off-road guard; 0 = off), `--window-chunks`
(K, default 5), `--emit-stride` (default 4), `--rgb-frames` (default 8),
`--session-id` (unique per run), `--plan` (JSON override).

## Phase 2 — teacher labeling (optional; separate env)

`label_teacher.py` runs Cosmos-Reason1-7B via **vLLM**, whose torch/transformers
pins conflict with the omnidreams runtime deps — so it imports nothing from
omnidreams and must run in a **dedicated venv**, invoked by path:

```bash
# One-time: dedicated vLLM env (reuses the already-cached Cosmos snapshot).
uv venv .venv-teacher && . .venv-teacher/bin/activate
uv pip install "vllm>=0.8.5" "qwen-vl-utils>=0.0.11"

# Smoke test on 50 windows first — check letters_found is mostly 6.
HF_HOME=<same cache omnidreams-prepare used> \
  python integrations/omnidreams/omnidreams/caption/label_teacher.py \
    --shards-dir ~/data/omnidreams-caption/raw \
    --out ~/data/omnidreams-caption/labeled --limit 50

# Full run (resumable: re-run to continue; skips already-labeled keys).
HF_HOME=<...> python .../caption/label_teacher.py \
  --shards-dir ~/data/omnidreams-caption/raw \
  --out ~/data/omnidreams-caption/labeled
```

Writes `labeled/labels.jsonl` (one line per window: `teacher_hard`,
`teacher_soft` 6-way dist, `teacher_conf`, `letters_found`) plus
`labeled/label_config.json` (model/revision/prompt provenance). Defaults to the
cached pinned revision (`DEFAULT_REVISION`); pass `--revision main` for the
published card checkpoint (requires a ~14 GB download).

## Phases 3–5 — train & export (omnidreams env, torch-only)

These need only torch/onnx (no world-model runtime, no vLLM). Run via
`uv run --project integrations/omnidreams`. The model is tiny and latents load
into RAM, so training is minutes.

```bash
# Train — holds out one session for validation (default: the last).
uv run --project integrations/omnidreams \
  python -m omnidreams.caption.train \
    --shards-dir DATA/raw --out DATA/checkpoints --val-session s6

# Export the best checkpoint straight into the served cache dir.
uv run --project integrations/omnidreams \
  python -m omnidreams.caption.export \
    --checkpoint DATA/checkpoints/caption_cnn_best.pt --to-cache --version caption-cnn-v1
```

Training writes `checkpoints/caption_cnn_best.pt` + `metrics.json` (best val
accuracy, per-class recall, confusion). Export writes
`caption_model.<precision>.onnx` + `.spec.json` into
`~/.cache/flashdreams/omnidreams-caption-models/`, which the next token-stream
session advertises to the browser client (input window K=5, 6-class bank). A
hard browser reload then shows live, changing captions from the trained model.

**Quick chain check (synthetic, no capture needed):**

```bash
uv run --project integrations/omnidreams python - <<'PY'
import numpy as np, tempfile, subprocess, io, json, random
from pathlib import Path
from omnidreams.caption.shards import ShardWriter
from omnidreams.caption.model import CAPTION_CLASSES
d=Path(tempfile.mkdtemp())/"raw"; 
def npy(a):
    import numpy as np, io; b=io.BytesIO(); np.save(b,a); return b.getvalue()
with ShardWriter(d, prefix="s1") as w:
    for i in range(120):
        lab=CAPTION_CLASSES[i%6]
        w.write(f"s1-{i:05d}", {"latents.npy": npy(np.random.randn(10,16,88,160).astype("float16")),
                 "json": b"{}"}, meta={"session_id":"s1" if i<90 else "s2","label":lab})
print("synthetic shards at", d)
PY
# then: python -m omnidreams.caption.train --shards-dir <that dir> --out /tmp/ck --val-session s2 --epochs 2 --device cpu
```
