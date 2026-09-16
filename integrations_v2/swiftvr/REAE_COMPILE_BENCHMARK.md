<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SwiftVR ReAE compile benchmark

This report records the encoder/decoder compile experiment and provides a
matched procedure for repeating it on another GPU. The measurements below are
from one GB300 and should not be treated as performance guarantees for other
hardware.

## Result

SwiftVR now exposes independent `compile_reae_encoder` and
`compile_reae_decoder` controls. The compiled functions are bound once to the
resident pipeline, while each stream keeps its own causal encoder and decoder
state.

The selected `swiftvr-2x-compiled` preset compiles the transformer and ReAE
decoder. It does not compile the encoder: compiling both ReAE directions
together did not improve throughput over the selected path on the measured
stack.

| Path | Decision | Reason |
| --- | --- | --- |
| Eager | Default fallback | Fast startup and lowest memory use |
| ReAE encoder compiled | Not selected | Only 4.3% lower latency with 21.5 s extra preparation |
| ReAE decoder compiled | Useful component | 14.0% lower latency in isolation |
| ReAE encoder + decoder compiled | Not selected | No gain over transformer + decoder |
| Transformer compiled | Useful component | 1.8% lower latency in isolation |
| Transformer + ReAE decoder compiled | Selected opt-in | 17.5% lower latency; visual validation passed |

The configuration chart below predates the compiler-only decoder layout
optimization. A later isolated sweep reduced compiled decoder compute from
18.93 ms to 9.83 ms; see
[Compiled decoder layout follow-up](#compiled-decoder-layout-follow-up).

## GB300 configuration

| Setting | Value |
| --- | --- |
| Base commit | `origin/main` at `37e208298d105e54fa5e24edb4fe2b05ed516918` plus the compile change containing this report |
| SwiftVR checkpoint | `H-oliday/SwiftVR` revision `743ed2530c550764905400f38eb6cc41af5abc80` |
| GPU | NVIDIA GB300, 256703 MiB |
| Driver | 595.71.05 |
| PyTorch / CUDA / cuDNN | 2.12.1+cu130 / 13.0 / 9.20.0 |
| Precision | BF16 |
| Input | 24 real RGB frames, uint8 THWC, 1280x704 |
| Output | 2560x1408, 2x |
| Chunk / attention window / overlap | 8 frames / 16x16 / 0 |
| Warmup / samples | 5 chunks / 20 chunks |
| Timing | CUDA events with synchronization, fresh process and compiler cache per candidate |

The 24-frame input tensor had SHA-256
`f36d724c383358ee9d962caf8521f63a80ca33675d2b7a7a06b4d102b9f1fceb`.
It was center-cropped from a 1280x720 moving driving clip. The benchmark is the
SwiftVR streaming stage only; it excludes OmniDreams, presentation, and video
encoding.

## Configuration and performance chart

| Candidate | Transformer | Encoder | Decoder | Median | p90 | Effective FPS | Cold prepare | Peak allocation |
| --- | :---: | :---: | :---: | ---: | ---: | ---: | ---: | ---: |
| Eager | off | off | off | 96.47 ms | 96.86 ms | 82.9 | 3.2 s | 18.65 GiB |
| Encoder | off | on | off | 92.31 ms | 92.66 ms | 86.7 | 24.7 s | 18.65 GiB |
| Decoder | off | off | on | 82.96 ms | 83.26 ms | 96.4 | 121.2 s | 20.45 GiB |
| Encoder + decoder | off | on | on | 79.91 ms | 80.57 ms | 100.1 | 145.5 s | 20.45 GiB |
| Transformer | on | off | off | 94.69 ms | 95.34 ms | 84.5 | 30.0 s | 18.65 GiB |
| Transformer + decoder | on | off | on | **79.60 ms** | **79.97 ms** | **100.5** | 142.0 s | 20.45 GiB |

## Compiled decoder layout follow-up

The compiled decoder now preserves `channels_last` through its ordinary
Conv2d regions and uses `channels_last_3d` for SwiftVR's temporal Conv3d
boundaries. The eager path is unchanged because explicit layout conversions
made it slower.

| Decoder candidate | Eager median | Compiled median | Compiled decision |
| --- | ---: | ---: | --- |
| Contiguous | **32.03 ms** | 18.93 ms | Baseline |
| Conv2d `channels_last` | 44.17 ms | 20.54 ms | Reject |
| Conv3d `channels_last_3d` | 32.87 ms | 10.91 ms | Useful |
| Combined | 41.15 ms | **9.83 ms** | Selected |

The selected result reproduced in a second fresh-cache run. Direct compiled
decoder outputs were bit-identical across layouts. Through the complete
24-frame pipeline, the selected layout differed from the previous compiled
decoder by 0.000048 MAE and 69.13 dB PSNR, with no visible normal-scale
difference in frames 1, 12, or 24.

Run the isolated layout sweep with `scripts/benchmark_reae_layout.py`. Keep
baseline and candidate in separate fresh processes and set a fresh
`TORCHINDUCTOR_CACHE_DIR` for every compiled case.

## Output validation

The quality comparison isolated decoder compilation by comparing the compiled
transformer against the compiled transformer plus decoder on the same 24 input
frames and weights.

| Metric | Result |
| --- | ---: |
| Maximum absolute difference | 0.06836 |
| Mean absolute difference | 0.001012 |
| RMSE | 0.001946 |
| PSNR | 54.22 dB |
| Temporal-delta MAE | 0.001083 |
| Per-frame mean-error range | 0.000917-0.001147 |

Normal-scale side-by-side inspection showed no visible difference through
frame 24 and no increasing temporal drift. A 10x absolute-difference view showed
only low-amplitude changes around texture and edges. The registered
`swiftvr-2x-compiled` postprocessor also passed a 24-frame integration smoke:
it returned exactly 24 finite, non-black 2560x1408 frames through buffered
startup and flush.

## Repeat on RTX PRO 6000

Use the same code revision, checkpoint, input tensor, and environment settings
where possible. Record `git rev-parse HEAD`, the input SHA-256, and the output
JSON files with the result.

Install the integration and confirm the GPU stack:

```bash
uv sync --package flashdreams-swiftvr --extra dev \
  --extra interactive-drive --inexact
nvidia-smi --query-gpu=name,driver_version,memory.total \
  --format=csv,noheader
uv run --no-sync python -c \
  'import torch; print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version())'
```

The input must be a `torch.uint8` RGB tensor shaped `[T, 704, 1280, 3]`, with
`T` a positive multiple of eight. For the strictest comparison, copy the
24-frame tensor identified by the checksum above. Otherwise use a real moving
clip, report its checksum, and use the same tensor for every candidate.

Run each candidate in a fresh process and fresh TorchInductor cache:

```bash
export SWIFTVR_BENCH_INPUT=/path/to/input-24x704x1280.pt
export SWIFTVR_BENCH_RESULTS=/tmp/swiftvr-compile-rtx6000
mkdir -p "$SWIFTVR_BENCH_RESULTS"

run_swiftvr_case() {
  local swiftvr_label="$1"
  shift
  local swiftvr_cache
  swiftvr_cache="$(mktemp -d "/tmp/swiftvr-${swiftvr_label}.XXXXXX")"
  TORCHINDUCTOR_CACHE_DIR="$swiftvr_cache" uv run --no-sync python \
    integrations_v2/swiftvr/scripts/benchmark_reae_compile.py \
    --label "$swiftvr_label" \
    --input-tensor "$SWIFTVR_BENCH_INPUT" \
    --output-dir "$SWIFTVR_BENCH_RESULTS" "$@"
}

run_swiftvr_case eager
run_swiftvr_case encoder --compile-encoder
run_swiftvr_case decoder --compile-decoder
run_swiftvr_case encoder-decoder --compile-encoder --compile-decoder
run_swiftvr_case transformer --compile-transformer
run_swiftvr_case transformer-decoder \
  --compile-transformer --compile-decoder
```

Each run writes raw timing samples, stack metadata, peak CUDA allocation, and
the restored tensor to `$SWIFTVR_BENCH_RESULTS`. Allow roughly 4 GiB for the
six 24-frame outputs.

Compare the isolated decoder-compile outputs numerically:

```bash
uv run --no-sync python - <<'PY'
import math
import os
from pathlib import Path

import torch

root = Path(os.environ["SWIFTVR_BENCH_RESULTS"])
reference = torch.load(root / "transformer.pt", weights_only=True).float()
candidate = torch.load(root / "transformer-decoder.pt", weights_only=True).float()
delta = candidate - reference
rmse = delta.square().mean().sqrt().item()
temporal = ((candidate[:, 1:] - candidate[:, :-1]) -
            (reference[:, 1:] - reference[:, :-1])).abs().mean().item()
print({
    "max_abs": delta.abs().max().item(),
    "mean_abs": delta.abs().mean().item(),
    "rmse": rmse,
    "psnr_db": -20 * math.log10(rmse),
    "temporal_delta_mae": temporal,
})
PY
```

Finally, exercise the production integration with a fixed world-model seed:

```bash
uv run --no-sync flashdreams-run-v2 \
  interactive-drive-omnidreams \
  --mode mp4 \
  --output-path /tmp/interactive-drive-swiftvr-compiled.mp4 \
  --stats-path /tmp/interactive-drive-swiftvr-compiled.json -- \
  --total-blocks 100 --no-ui --world-model-seed 42 \
  --width 1280 --height 704 \
  --postprocess-preset swiftvr-2x-compiled
```

For the RTX PRO 6000 result, report the same configuration/performance table,
startup time, first steady chunk, median, p90, effective FPS, peak allocation,
frame count, and whether side-by-side playback shows flicker or drift. Keep the
regular `swiftvr-2x` preset as the fallback if compilation is slower or its
startup cost is unsuitable on that stack.
