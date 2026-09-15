<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SwiftVR inference performance recommendations

This note records the current SwiftVR performance profile and a prioritized set
of experiments. The measurements are diagnostic results from one GB300 system,
not performance or quality guarantees for other GPUs.

## Summary

The validated opt-in path is transformer plus ReAE decoder compilation. It
reduced steady eight-frame processing latency from 96.47 ms to 79.60 ms at
2560x1408 output, a 17.5% latency reduction and a throughput increase from 82.9
to 100.5 FPS. The tradeoff is approximately 142 seconds of cold preparation
with fresh compiler caches on the measured stack.

The ReAE decoder provided most of the gain: decoder-only compilation reduced
latency by 14.0%. Transformer-only and encoder-only compilation improved it by
1.8% and 4.3%, respectively. Encoder plus decoder matched the selected path
within measurement noise but prepared more slowly, so the preset deliberately
combines the compiled transformer with only the compiled decoder. FP8
transformer linear layers and ReAE memory-layout work remain possible
follow-ups.

Increasing chunk size, changing the SDPA backend, enabling cuDNN benchmarking,
or optimizing adapter format conversions did not show enough potential to be
prioritized.

## Measurement context

- FlashDreams base: `origin/main` at `37e208298d105e54fa5e24edb4fe2b05ed516918`
- SwiftVR checkpoint revision: `743ed2530c550764905400f38eb6cc41af5abc80`
- GPU: one NVIDIA GB300, 256703 MiB
- Driver: 595.71.05
- PyTorch: 2.12.1+cu130
- PyTorch CUDA runtime: 13.0
- cuDNN: 9.20.0
- Precision: BF16
- Attention: PyTorch dense SDPA, 16x16 shifted windows
- DiT overlap: 0
- Input: eight 1280x704 RGB frames
- Output: eight 2560x1408 RGB frames in steady state
- Timing: CUDA events with synchronization; medians reported after warmup
- Compiler measurements used fresh processes and fresh TorchInductor caches

The stage measurements exclude OmniDreams, presentation, encoding, and other
application work. They exercise the same `SwiftVRPipeline` and streaming state
used by the postprocessor. ReAE compilation was validated with matched real
video frames as described below.

## Earlier diagnostic profile

The following stage profile predates the mainline SwiftVR refactor and explains
which compile boundaries were investigated. Use the matched current-main
benchmark below for performance decisions.

| Stage | Eager median | Compiled median | Eager share |
| --- | ---: | ---: | ---: |
| Resize and preprocessing | 0.40 ms | 0.39 ms | 0.3% |
| ReAE encoder | 9.20 ms | 9.25 ms | 7.5% |
| WAN transformer | 81.41 ms | 55.20 ms | 66.3% |
| ReAE decoder | 31.79 ms | 32.06 ms | 25.9% |
| Total per eight frames | 122.77 ms | 96.9 ms | 100% |
| Effective throughput | 65.1 FPS | 82.6 FPS | |

The operator profile of one warmed eager chunk showed:

- 333 `copy_` calls for `[1, 7040, 3072]` tensors consumed 17.94 ms.
- NCHW-to-NHWC and NHWC-to-NCHW kernels consumed 9.60 ms and 5.85 ms.
- Dense 3072-to-14336 and 14336-to-3072 FFN projections were prominent.
- SDPA consumed 2.73 ms and window-index gathers consumed 2.72 ms.
- Input conversion took 0.12 ms; output value/layout conversion and the
  single-chunk concatenation together remained below 1 ms.

These results make transformer fusion, dense projections, and ReAE convolution
layout more important than the attention kernel or postprocessor adapter.

## Investigation order

### 1. Compile transformer blocks

**Status:** Useful only with the compiled decoder on the current mainline stack.

The implementation supports `compile_blocks`, but it defaults to `False`,
including in the `swiftvr-2x` preset. In the current matched benchmark,
transformer-only compilation reduced total latency by 1.8%. Combining it with
the compiled decoder was the fastest tested configuration.

Cold preparation results were:

| Operation | Wall time |
| --- | ---: |
| Compile first steady chunk shape | 25.8 s |
| Compile buffered tail shape | 10.0 s |
| First chunk in a replacement stream | 97 ms |

Recommended experiment:

1. Enable `compile_blocks=True` only for the Interactive Drive SwiftVR preset.
2. Trigger preparation before the first rollout rather than from the first
   postprocessed output.
3. Prewarm both the steady and tail shapes.
4. Keep a stable TorchInductor cache across application launches where the
   deployment environment permits it.
5. Keep the eager preset available because cold compilation remains expensive.

Also test `dynamic=True` as a separate candidate. It may avoid compiling the
tail token length independently, but it must retain the steady-state speedup.

### 2. Compile ReAE encoder and decoder compute

**Status:** Decoder recommended as part of the compiled opt-in preset; encoder
available for investigation but not recommended in combination.

The implementation binds one resident callable for the encoder and one for the
decoder. Causal dictionaries and frame buffers remain explicit inputs and
outputs, so streams share compiled code without sharing temporal state.

Fresh-process results used 24 real 1280x704 frames, five warmup chunks, twenty
measured chunks, fresh TorchInductor caches, and 2560x1408 output:

| Candidate | Median | p90 | FPS | Cold prepare | Peak allocated |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fully eager | 96.47 ms | 96.86 ms | 82.9 | 3.2 s | 18.65 GiB |
| ReAE encoder compiled | 92.31 ms | 92.66 ms | 86.7 | 24.7 s | 18.65 GiB |
| ReAE decoder compiled | 82.96 ms | 83.26 ms | 96.4 | 121.2 s | 20.45 GiB |
| ReAE encoder + decoder compiled | 79.91 ms | 80.57 ms | 100.1 | 145.5 s | 20.45 GiB |
| Transformer compiled | 94.69 ms | 95.34 ms | 84.5 | 30.0 s | 18.65 GiB |
| Transformer + decoder compiled | 79.60 ms | 79.97 ms | 100.5 | 142.0 s | 20.45 GiB |

The registered `swiftvr-2x-compiled` postprocessor also passed an integration
smoke test: 24 inputs produced exactly 24 finite, non-black outputs at
2560x1408, including buffered startup and flush. Preparation took 16.3 seconds
with a warm compiler cache.

Quality comparison between transformer-only and transformer-plus-decoder over
the 24-frame moving clip produced 54.22 dB PSNR, 0.00101 mean absolute error,
0.00195 RMSE, and 0.00108 temporal-delta MAE in `[0, 1]`. Per-frame mean error
stayed between 0.00092 and 0.00115, with no increasing drift. Normal-scale
contact-sheet inspection showed no visible difference; a 10x
absolute-difference view showed low-amplitude changes around edges and texture.

Rejected compile boundaries:

- Compiling 24 individual `_MemoryBlock`, `_TemporalPool`, and `_TemporalGrow`
  modules slowed the eager path to 133.46 ms because graph-launch overhead
  outweighed fusion.
- `mode="reduce-overhead"` failed because CUDA-graph output storage was reused
  and overwritten between block calls.
- Enabling encoder and decoder stage compilation together did not beat the
  selected transformer-plus-decoder path and prepared slightly more slowly.

The reproducible harness is `scripts/benchmark_reae_compile.py`. Use
`--compile-encoder`, `--compile-decoder`, and `--compile-transformer` to isolate
each candidate in a fresh process.

### ReAE memory layout follow-up

**Status:** Deferred experiment.

The format-conversion kernels still suggest that cuDNN repeatedly converts
layouts around the Conv2d-heavy encoder and decoder. A future experiment should
preserve `channels_last` only through Conv2d regions, test `channels_last_3d`
independently for temporal Conv3d, and keep temporal pooling and pixel shuffle
boundaries in their natural contiguous layouts.

### 3. Quantize transformer projections and FFNs to FP8

**Status:** Deferred experiment; potentially high impact and higher quality risk.

The remaining transformer workload is dominated by dense projections across 30
WAN blocks. FlashDreams already provides `QuantizedNonPersistentLinear`, so an
FP8 prototype does not require another dependency.

Recommended experiment:

1. Keep ReAE, normalization, residual accumulation, and initially SDPA in BF16.
2. Start with FP8 E4M3 weights and slice-scaled activations for FFN projections.
3. Add QKV and attention output projections as a separate candidate.
4. Do not prioritize FP8 SDPA: BF16 SDPA is only about 2.2% of current time.
5. Record quantization overhead as part of the measured path.

Validate a short matched clip first, followed by a motion-heavy long rollout.
Generated detail, flicker, and temporal drift matter more than a small PSNR
difference for this experiment.

### 4. Overlap OmniDreams and SwiftVR

**Status:** Deferred architecture change.

Interactive Drive currently generates one model chunk and then synchronously
postprocesses it. A queued postprocessing worker could process chunk N while the
world model generates chunk N+1.

This is most promising when SwiftVR runs on a second GPU. On one GPU, concurrent
dense workloads may contend for the same compute and memory bandwidth and must
be benchmarked rather than assumed faster. The design also adds one chunk of
latency and needs bounded queues, ordered flush behavior, and clean error
propagation.

Do this only if the local SwiftVR optimizations leave end-to-end throughput below
the target.

### 5. Consider newer upstream runtime knobs selectively

**Status:** Low priority.

Newer upstream SwiftVR exposes `torch_compile` and selectable attention
backends. The compile option is valuable. Attention backend selection has a low
ceiling in the current profile because SDPA consumes only 2.73 ms per chunk.

Synchronize individual runtime improvements rather than replacing the adapted
streaming and postprocessor contracts wholesale.

## Experiments not worth prioritizing

### Larger chunks

| Chunk size | Median latency | Effective throughput |
| --- | ---: | ---: |
| 8 | 122.77 ms | 65.1 FPS |
| 16 | 242.65 ms | 65.9 FPS |
| 24 | 363.13 ms | 66.1 FPS |

The throughput difference is approximately 1.5%, while larger chunks increase
latency and buffering. Keep eight frames for Interactive Drive.

### cuDNN benchmark mode

Enabling `torch.backends.cudnn.benchmark` improved eager throughput by less than
1% on the measured system and added several seconds of cold autotuning. It may
remain useful on a different GPU, but it should not be the next optimization.

### Attention window or backend changes

Completely eliminating the measured SDPA time would save only about 2.2% of the
chunk. Smaller attention windows may also change restoration quality. Preserve
the trained 16x16 behavior until larger bottlenecks are addressed.

### Adapter layout conversion

The float-to-uint8 input conversion, restored-output conversion, and generic
single-chunk concatenation together consume less than 1 ms. Removing these
copies may simplify the data path later, but it will not materially change
SwiftVR throughput on the measured system.

## Validation checklist for every candidate

Run baseline and candidate in separate fresh processes and record:

- exact commit and flags;
- compiler-cache state and prewarm policy;
- first-visible frame time;
- median and p90 stage and total chunk latency after at least five warmup chunks;
- aggregate FPS over a long rollout;
- peak CUDA allocated and reserved memory;
- replacement-session startup and first chunk;
- flush latency and exact input/output frame count;
- failed compilations, graph breaks, or fallback kernels.

For quality, use identical inputs and weights and save baseline and candidate
videos. Compare at least:

- maximum and mean absolute error for compile/layout-only changes;
- PSNR or RMSE for deterministic matched-output checks;
- temporal MAE or an amplified frame-difference video;
- worst-frame crops and side-by-side playback;
- a long, motion-heavy Interactive Drive rollout for flicker and drift.

Suggested acceptance criteria:

- at least a 10% steady-state latency reduction for a nontrivial optimization;
- no missing, duplicated, black, or reordered frames;
- no regression in restart/replacement-session behavior;
- no material visual regression on matched clips or long rollouts;
- startup cost documented and absorbed before user-visible generation.

## End-to-end command template

Use the same output resolution and rollout length for every candidate:

```bash
uv run --no-sync flashdreams-run-v2 interactive-drive-omnidreams \
  --mode mp4 \
  --output-path /tmp/interactive-drive-swiftvr-candidate.mp4 \
  --stats-path /tmp/interactive-drive-swiftvr-candidate-stats.json \
  -- --total-blocks 100 --no-ui --width 1280 --height 704 \
  --postprocess-preset swiftvr-2x-compiled
```

Rename the output and stats paths for each candidate and retain the eager run as
the baseline. The existing results and quality artifacts are described in
`BENCHMARK.md`.
