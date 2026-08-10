<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SANA-WM Performance Benchmarks

This directory contains timing benchmarks for the FlashDreams
`SANA-WM_bidirectional` and `SANA-WM_streaming` integrations against the pinned
upstream [`NVlabs/Sana`](https://github.com/NVlabs/Sana) entrypoints.

Do not use these latency rows as quality evidence without a matching
correctness pass from [`../parity_check/`](../parity_check/README.md).

Shared harness files live one level up in `integrations/sana/tests/`:

- `common.sh` - upstream checkout, patch, and venv setup helpers.
- `pyproject.toml` and `uv.lock` - isolated `../.venv` dependency environment.
- `changes_bidirectional.patch` and `changes_streaming.patch` - upstream Sana
  instrumentation patches.
- `compat/` - small upstream import compatibility shims.
- `run_flashdreams_*.py` - FlashDreams entrypoint wrappers shared with parity.

`bench.sh` runs `uv sync` from `integrations/sana/tests/`, reuses or clones the
upstream checkout at `../Sana`, pins it to `6298508`, and applies patches
idempotently. The patches add instrumentation and do not change generation
algorithms.

The base isolated environment covers BF16 and FlashDreams FP8/FP4 rows.
Only upstream `SANA-WM_streaming` FP8/FP4 comparison rows opt into the `quant`
extra, which builds TransformerEngine from the pinned git source against the
local CUDA/PyTorch stack. A new checkout needs network access for `uv sync`,
GitHub access for the pinned upstream checkout, and a CUDA build toolchain plus
TransformerEngine source access only when running those upstream streaming
low-precision comparisons.

## Run Benchmarks

```bash
cd integrations/sana/tests/performance
bash bench.sh
```

`bench.sh` defaults to both sibling variants:
`SANA_WM_VARIANT=both BENCH_SIDE=both WARMUP_RUNS=1 MEASURED_RUNS=3
BENCH_PRECISIONS=bf16,fp8,fp4 NO_REFINER=0`.

For bidirectional, the warmup/measured counts mean one live warmup generation
and three measured generations inside each measured process. For streaming,
those counts are process-level warmup/measured runs because steady state is
measured from chunks inside each process. Streaming performance benchmarks
compile the refiner on both sides by default; leave `STREAMING_NO_COMPILE`
unset unless intentionally measuring no-compile behavior.

Default outputs are written under `outputs/bench/` in this directory.

Bidirectional precision-row outputs:

- `outputs/bench/bench.json`, `bench.md`, and `perf.md` - precision sweep
  summary files.
- `outputs/bench/<precision>/bench.json` - machine-readable inputs, medians,
  p90s, memory, and stage timings for that precision row.
- `outputs/bench/<precision>/bench.md` - human-readable report.
- `outputs/bench/<precision>/perf.md` - chart-ready data.

Streaming precision-row outputs:

- `outputs/bench/streaming/<precision>/upstream/run_<N>/stats.json`
- `outputs/bench/streaming/<precision>/upstream/run_<N>/command.txt`
- `outputs/bench/streaming/<precision>/flashdreams/run_<N>/stats.json`
- `outputs/bench/streaming/<precision>/flashdreams/run_<N>/command.txt`
- `outputs/bench/streaming/<precision>/bench.json`
- `outputs/bench/streaming/<precision>/bench.md`
- `outputs/bench/streaming/<precision>/perf.md`

## Metrics

The bidirectional headline metric is steady-state in-process generation latency
per generated clip. Warmup generations run on the live model and are excluded
from the headline metric. Model construction, checkpoint loading, video
writing, and frame dumps are outside the timing boundary. With the default
`NO_REFINER=0`, the timed work covers conditioning, Stage-1 DiT, LTX-2 refiner,
and SANA VAE decode. Set `NO_REFINER=1` only for a diagnostic Stage-1 plus SANA
VAE decode benchmark.

The streaming headline metric is steady-state generation milliseconds per
produced chunk. Warmup runs and the first decoded chunk are excluded; full-clip
wall time remains in `bench.json` and `bench.md` as supporting data.

`bench.md` also reports stage timings, memory, and frame-normalized diagnostic
breakdowns. Those rows explain the benchmark result; they are not second
headline metrics.

## Model-Card Data

Each model-card chart file should have GPU/device as the first column and
implementation as the series columns, for example:

```markdown
| device | official | flashdreams |
| --- | ---: | ---: |
| GPU | 77826.74 | 76938.72 |
```

When copying chart data into the docs tree, use a date-stamped file name such
as `perf-MMDD-bf16.md`.

Bidirectional upstream comparisons are BF16-only because upstream
`SANA-WM_bidirectional` does not support FP8/FP4 precision flags. The default
bidirectional precision sweep therefore runs BF16 as upstream-vs-FlashDreams,
then records FP8 and FP4 as FlashDreams-only diagnostic rows. Use
`outputs/bench/bf16/perf.md` from the BF16 bidirectional row as the model-card
chart data.

Run only the BF16 benchmark used by the `SANA-WM_bidirectional` model card with:

```bash
SANA_WM_VARIANT=bidirectional BENCH_PRECISIONS=bf16 bash bench.sh
```

Run only the streaming comparison precision sweep with:

```bash
SANA_WM_VARIANT=streaming BENCH_SIDE=both bash bench.sh
```

Omit `SANA_WM_VARIANT` to regenerate both model-card datasets in one command.
The default run uses the BF16/FP8/FP4 precision sweep for streaming and for
FlashDreams-only bidirectional diagnostics, while the official bidirectional
comparison stays BF16-only:

```bash
bash bench.sh
```

Set `DEVICE_LABEL` to label chart data with a device name that is not just
`GPU`:

```bash
DEVICE_LABEL="GPU model name" bash bench.sh
```

On shared or mixed-GPU hosts, set `CUDA_VISIBLE_DEVICES` to the benchmark GPU's
UUID before running the harness. This keeps PyTorch's NVML and CUDA runtime
device enumeration consistent without changing the benchmark backend policy:

```bash
CUDA_VISIBLE_DEVICES=GPU-... bash bench.sh
```

Set `FORCE_CUDNN_SDPA=1` only for backend-isolation probes. It is not a default
benchmark setting.

FP8 and FP4 are passed through the precision flags supported by each measured
side. For upstream this means streaming only; for FlashDreams this also includes
bidirectional diagnostic rows. FlashDreams low-precision rows use the base
isolated environment; upstream streaming low-precision rows sync the isolated
venv with the `quant` extra before launch. `bench.sh` lets TransformerEngine
use its upstream CUDA-version-aware default architecture set unless
`NVTE_CUDA_ARCHS` is set explicitly. The script records that setting in the
isolated venv and rebuilds TransformerEngine once when it changes, so stale
extension wheels from an earlier architecture setting are not reused. If
TransformerEngine, hardware, or upstream code rejects a precision, earlier
precision outputs remain in place and the failing command is recorded in the
corresponding `command.txt`.

The scripts do not set allocator overrides or GPU wait loops. If GPU contention
matters in your environment, handle it outside the harness.
