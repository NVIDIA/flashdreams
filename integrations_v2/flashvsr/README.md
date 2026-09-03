<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# FlashVSR

FlashVSR-v1.1 streaming video super-resolution (LR projector + distilled
Wan 2.1 DiT + TC decoder + AdaIN color corrector), packaged as a
[`flashdreams`](../..) integration. The model remains a reusable
`StreamInferencePipeline`; the `v2v` application exposes video input and video
output through the v2 application/session API. It falls back to a bounded Big
Buck Bunny excerpt when no source video is selected.

This is a worked example of the
[Adding a new method](https://nvidia.github.io/flashdreams/main/developer_guides/new_integration.html)
developer-guide flow.

## Applications

FlashVSR binds the reusable
[V2V](../../apps/v2v/README.md) application to the
following model configurations:

| Entry-point slug | Configuration |
| --- | --- |
| `v2v-flashvsr-v1.1-sparse-ratio-2.0` | Stable sparse attention. |
| `v2v-flashvsr-v1.1-sparse-ratio-1.5` | Faster, lower-budget sparse attention. |
| `v2v-flashvsr-v1.1-full-attn` | Dense full attention for context parallelism. |

```bash
uv sync --package flashdreams-flashvsr --inexact
uv run --no-sync flashdreams-run-v2 v2v-flashvsr-v1.1-sparse-ratio-2.0 --output-path artifacts/upscaled.mp4 -- --video-path input.mp4
```

The shared app README documents controls, presentation modes, the
`--video-path` and `--max-chunks` application arguments, the Big Buck Bunny
fallback, and its CPU tests.

## Install

The plugin is registered as a `uv` workspace member in the repo-root
`pyproject.toml`, so a single `uv sync` from the repo root pulls it in:

```bash
uv sync
```

Standalone (outside the workspace) also works:

```bash
uv pip install -e integrations_v2/flashvsr
```

## HuggingFace setup

Checkpoints are auto-downloaded from HuggingFace at first run. Set an
auth token first.

```bash
# huggingface token.
export HF_TOKEN=<your-hf-token>

# (optional) override the cache location.
export HF_HOME=~/.cache/huggingface  # default
```

## Streaming chunk contract

`FlashVSRPipeline.generate(autoregressive_index, cache, input)` processes
**one full FlashVSR chunk** per call. The encoder accepts the four
(raw_T -> padded_T) pairs from `FLASHVSR_CHUNK_FRAME_TARGETS`:

| raw_T | padded_T | when | DiT iters per chunk |
|------:|---------:|------|--------------------:|
| `5`   | `8`      | cold-start (`autoregressive_index == 0`) | 1 |
| `8`   | `8`      | any AR step                              | 1 |
| `13`  | `16`     | cold-start (`autoregressive_index == 0`) | 2 |
| `16`  | `16`     | any AR step                              | 2 |

Cold-start sizes are pad-left replicated inside `FlashVSREncoder` so the
projector's 4-frame causal stride aligns. The DiT runs `T_padded // 8`
internal iterations against per-iter (2-latent-frame) noise slices and
LR-latent token slices; the rolling KV cache holds `kv_ratio + 1` chunks
at attention time (default `kv_ratio = 3` -> 4 chunks).

The FlashVSR post-processor's `chunk_size` config picks the steady-state size
(`8` or `16`); the matching cold-start size (`5` or `13`) is derived
automatically by the model adapter.

## Builder knobs

`build_flashvsr_v1_1` is the single entry point for assembling a
`FlashVSRPipelineConfig` by hand. The most common knobs:

- `input_H`, `input_W`: low-res input dimensions. Output dims are
  `((input_H * scale) // 128 * 128, (input_W * scale) // 128 * 128)`:
  the encoder bicubic-upsamples to `(input_H * scale, input_W * scale)`
  and symmetric-crops to the largest 128-multiple per axis (matching
  upstream's `upscale_then_center_crop` in
  `examples/WanVSR/infer_flashvsr_v1.1_tiny.py`). Inputs need only be
  at least `128 / scale = 64` pixels on each axis at the default
  `scale=2`.
- `scale`: `2` (default) or `4`.
- `sparse_ratio`: block-sparse attention budget multiplier. `2.0`
  (default, "more stable") or `1.5` ("faster" preset).
- `compile_network`: single `torch.compile` switch applied uniformly to
  the DiT, encoder projector, and decoder.
- `use_cuda_graph`: capture the steady-state DiT call into a CUDA graph
  and replay it (Phase 2 of `internal/upsampler/PERF_NOTES.md`). Requires
  `compile_network=True`. Encoder / decoder cuda graphs are always on
  inside the builder. Defaults to `False`; enable it only after validating
  the target resolution.
- `color_corrector_implementation`: `"cuda"` (default; AdaIN-only
  hand-rolled kernel) or `"torch"` (pure-torch wavelet + AdaIN reference).
- `enable_sync_and_profile`: per-AR-step CUDA-event profiling. Adds one
  `cuda.synchronize()` per step.

## Files

| Path | Purpose |
|---|---|
| `config.py` | Pipeline construction and shipped model presets. |
| `apps/v2v/adapter.py` | Thin FlashVSR binding for the shared `v2v` application. |
| `impl/pipeline.py` | `FlashVSRPipeline` + `FlashVSRPipelineConfig` (5-step `generate`; 7 profiler events). |
| `impl/encoder/` | Bicubic upres + `Causal_LQ4x_Proj` LR projector. |
| `impl/transformer/` | `FlashVSRTransformer` + sparse-attention DiT. |
| `impl/decoder/` | TC decoder + AdaIN color corrector wrapper. |
| `impl/postprocess.py` | Stateful FlashVSR video post-processing. |

## Tests

CPU smoke + parity tests live under `integrations_v2/flashvsr/`:

```bash
uv run --extra dev pytest integrations_v2/flashvsr -v
```

The CUDA / weight-gated parity tests (`test_projector_*`,
`test_color_corrector_benchmark.py`) auto-skip when GPU or staged
FlashVSR-v1.1 weights are missing. The DiT-side parity check
(`parity_check/test_dit_parity.py`) and the TC decoder parity check
(`parity_check/test_tcdecoder_parity.py`) live next to upstream's
cloned source tree and are invoked from `parity_check/run.sh` (see
below) so both legacy (upstream) and candidate (flashdreams) sides are
loaded from a single parity-check venv.

Upstream parity benchmark + DiT / TC decoder parity tests (clones
FlashVSR at a pinned commit, applies a local patch that adds
`EventProfiler`-instrumented per-chunk timing, runs both parity tests
against upstream's `diffsynth.models.wan_video_dit.WanModel` +
`diffsynth.pipelines.flashvsr_tiny_long.model_fn_wan_video` and
`examples/WanVSR/utils/TCDecoder.py`, then runs the upstream pipeline
end-to-end):

```bash
bash integrations_v2/flashvsr/tests/parity_check/run.sh
```

See [`integrations_v2/flashvsr/tests/parity_check/README.md`](tests/parity_check/README.md)
for the JSON-stats schema and both parity tests.

## Standalone uplift server

The informal standalone gRPC uplift server can be launched directly with:

```bash
uv run --no-sync --package flashdreams-flashvsr --with grpcio --with Pillow \
    python -m flashvsr.impl.uplift_server --port 50051
```
