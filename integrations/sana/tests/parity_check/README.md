<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SANA-WM Correctness Parity

This directory contains correctness checks for the FlashDreams
`SANA-WM_bidirectional` and `SANA-WM_streaming` integrations. The checks compare
against the pinned upstream [`NVlabs/Sana`](https://github.com/NVlabs/Sana)
entrypoints on matched demo inputs, seeds, resolution, and precision settings.

Do not use `run.sh` timings as performance data. Performance benchmarks live in
[`../performance/`](../performance/README.md).

Shared harness files live one level up in `integrations/sana/tests/`:

- `common.sh` - upstream checkout, patch, and venv setup helpers.
- `pyproject.toml` and `uv.lock` - isolated `../.venv` dependency environment.
- `changes_bidirectional.patch` and `changes_streaming.patch` - upstream Sana
  instrumentation patches.
- `compat/` - small upstream import compatibility shims.
- `run_flashdreams_*.py` - FlashDreams entrypoint wrappers shared with the
  performance harness.

`run.sh` runs `uv sync` from `integrations/sana/tests/`, reuses or clones the
upstream checkout at `../Sana`, pins it to `6298508`, and applies patches
idempotently. The patches add instrumentation and do not change generation
algorithms.

## Run Checks

```bash
cd integrations/sana/tests/parity_check
bash run.sh
```

`run.sh` defaults to `SANA_WM_VARIANT=bidirectional`. Set
`SANA_WM_VARIANT=streaming` to run the streaming frame-parity and continuity
path:

```bash
SANA_WM_VARIANT=streaming bash run.sh
```

Shared defaults:

- `SANA_REPO=../Sana`
- upstream pin `6298508`
- demo input `${SANA_REPO}/asset/sana_wm/demo_0.*`
- `STAGE1_PRECISION=bf16`
- `REFINER_PRECISION=$STAGE1_PRECISION`
- `QUANT_BACKEND=auto`

Bidirectional defaults:

- `OUTPUT_DIR=outputs/parity`
- `NUM_FRAMES=121`
- `FPS=16`
- `STEP=60`
- `CFG_SCALE=5.0`
- `SEED=42`
- `NO_REFINER=1` to isolate Stage-1 plus SANA VAE decode

Streaming defaults:

- `OUTPUT_DIR=outputs/parity/streaming`
- `NUM_FRAMES=241`
- `FPS=16`
- `CFG_SCALE=1.0`
- `FLOW_SHIFT=8.0`
- `SEED=42`
- `NO_REFINER=0`; `NO_REFINER=1` is rejected for streaming parity
- `STREAMING_ACTION=w-80,dw-40,w-80,aw-40`
- `TRANSLATION_SPEED=0.025`
- `ROTATION_SPEED_DEG=0.6`
- `STREAMING_DENOISING_STEP_LIST=1000,960,889,727,0`
- `STREAMING_NUM_FRAME_PER_BLOCK=3`
- `STREAMING_NUM_CACHED_BLOCKS=2`
- `STREAMING_REFINER_BLOCK_SIZE=3`
- `STREAMING_REFINER_KV_MAX_FRAMES=11`
- `STREAMING_SINK_SIZE=1`
- `STREAMING_NO_COMPILE=1`
- `STREAMING_OUTPUT_MODE=mp4`
- `STREAMING_SAMPLE_FRAME_STRIDE=1`

Bidirectional outputs are written under `outputs/parity/`:

- `upstream/frames.npy`
- `upstream/stats.json`
- `flashdreams/frames.npy`
- `flashdreams/stats.json`
- `parity.json`

Streaming outputs are written under `outputs/parity/streaming/`:

- `upstream/frames.npz`
- `upstream/stats.json`
- `flashdreams/frames.npy`
- `flashdreams/stats.json`
- `parity.json`
- `continuity.json`

For bidirectional parity, set `NO_REFINER=0` to compare the full Stage-1 plus
LTX-2 refiner path. Set `COMPILE_STAGE1=1` only when explicitly checking the
compiled Stage-1 path.

Streaming parity always uses the full upstream streaming stack. It saves every
decoded frame and checks both frame parity and chunk-boundary continuity. For
this correctness workflow only, `STREAMING_NO_COMPILE=1` is the default to keep
parity failures easier to debug. Do not carry that setting into performance
runs unless intentionally measuring no-compile streaming behavior.
