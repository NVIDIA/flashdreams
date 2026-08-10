<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# `sana_wm`

FlashDreams SANA-WM integration for the
[SANA-WM_bidirectional](https://huggingface.co/Efficient-Large-Model/SANA-WM_bidirectional)
and
[SANA-WM_streaming](https://huggingface.co/Efficient-Large-Model/SANA-WM_streaming)
camera-controlled world model releases from NVlabs/Sana.

The `sana-wm-bidirectional` and `sana-wm-streaming` runners use FlashDreams
config, runner, pipeline, diffusion-model, scheduler, transformer,
camera-conditioning, VAE decode, and output-writing boundaries. The Stage-1 DiT
module in this package loads the public SANA-WM checkpoints directly. Streaming
uses a FlashDreams-owned config literal instead of vendoring the upstream YAML.

Current scope:

| component | status |
| --- | --- |
| Stage-1 BF16 | FlashDreams DiT execution. |
| Streaming Stage-1 BF16 | Chunked FlashDreams DiT execution with prefix recomputation. |
| Stage-1 FP8 | PyTorch `_scaled_mm` backend. |
| Stage-1 FP4 | Triton quantization plus PyTorch `_scaled_mm`. |
| VAE decode | Direct `diffusers` LTX2 VAE use with upstream-matched tiling. |
| Streaming VAE decode | Direct causal LTX2 VAE use with prefix decode and new-frame emission. |
| LTX-2 refiner | Direct `diffusers` LTX-2 transformer and Gemma connector use. |
| Streaming LTX-2 refiner | Chunk-causal refiner config with FlashDreams prefix refinement. |

## Runner

| slug | description |
| --- | --- |
| `sana-wm-bidirectional` | `SANA-WM_bidirectional` Stage-1 + LTX-2 refiner runner. |
| `sana-wm-streaming` | `SANA-WM_streaming` Stage-1 + streaming LTX-2 refiner/VAE runner. |

The FlashDreams package is named `sana_wm`.

This integration does not use TransformerEngine at runtime. Its FP8/FP4 paths
use Torch/Triton replacement layers; avoid selecting the root `flashdreams`
`dev` extra when testing this package, because that extra contains
TransformerEngine for unrelated core parity tests.

## Setup

Install FlashDreams and the SANA integration into an environment with the
project's GPU runtime dependencies:

```bash
uv sync --package flashdreams-sana-wm --extra dev
```

The examples below use placeholder paths for a first-frame image, prompt,
camera trajectory, and intrinsics. Inputs must follow the same shape
conventions as the public SANA-WM release examples.

## Run

Bidirectional:

```bash
uv run flashdreams-run sana-wm-bidirectional \
    --image-path <path to initial frame PNG> \
    --prompt-path <path to prompt TXT> \
    --camera-path <path to camera trajectory NPY> \
    --intrinsics-path <path to intrinsics NPY> \
    --num-frames 161 \
    --output-dir outputs/sana_wm_bf16
```

Expected output:

```text
outputs/sana_wm_bf16/sana-wm-bidirectional.mp4
```

Streaming:

```bash
uv run flashdreams-run sana-wm-streaming \
    --image-path <path to initial frame PNG> \
    --prompt-path <path to prompt TXT> \
    --camera-path <path to camera trajectory NPY> \
    --intrinsics-path <path to intrinsics NPY> \
    --num-frames 241 \
    --output-dir outputs/sana_wm_streaming_bf16
```

Expected output:

```text
outputs/sana_wm_streaming_bf16/sana-wm-streaming.mp4
```

The streaming runner defaults to 3 latent frames per block and the distilled
schedule `[1000, 960, 889, 727, 0]`. Requested frame counts are snapped to
`8 * --num-frame-per-block * k + 1` before inference.

`--intrinsics-path` is optional. When omitted, intrinsics are derived from the
first-frame size assuming a centered principal point and a horizontal field of
view of `--intrinsics-hfov-deg` (default `90`, matching the demo intrinsics).
Likewise `--camera-path` may be replaced by an `--action` DSL string, which is
repeated or truncated to the requested snapped frame count. Explicit camera
paths are fitted to the same frame count instead of capping the video length. A
minimal run therefore needs only an image and a prompt:

```bash
uv run flashdreams-run sana-wm-bidirectional \
    --image-path my_frame.png \
    --prompt "a scene description; describe the world's own motion" \
    --action "w-100,dw-60,w-101" \
    --num-frames 161 \
    --output-dir outputs/mine
```

For Stage-1-only diagnostics, add `--no-refiner True`.

## BF16, FP8, and FP4

The runner defaults to BF16. Quantized Stage-1 and refiner inference are opt-in
and use Torch-based backends by default.

| option | hardware | backend |
| --- | --- | --- |
| `--stage1-precision fp8` | Hopper or newer (`sm_90+`) | E4M3 `_scaled_mm` |
| `--stage1-precision fp4` | Blackwell (`sm_100+`) | Triton NVFP4 plus `_scaled_mm` |
| `--quant-backend torch-fp8` | Hopper or newer (`sm_90+`) | force FP8 |
| `--quant-backend torch-fp4` | Blackwell (`sm_100+`) | force FP4 |

Stage-1 FP8 and FP4 follow the upstream precision CLI scope: self-attention,
cross-attention, and linearized FFN pointwise projections. This integration uses
Torch/Triton scaled-MM replacements instead of TransformerEngine. FP4 uses
NVFP4 W4A4 GEMMs with tiled 16-wide Hadamard rotation enabled, two-level
per-tensor-plus-block scaling, and 2D 16x16 block scaling for weights, matching
the upstream recipe behavior that keeps Stage-1 camera/action conditioning
coherent.

FP8 smoke:

```bash
uv run flashdreams-run sana-wm-bidirectional \
    --image-path <path to initial frame PNG> \
    --prompt-path <path to prompt TXT> \
    --camera-path <path to camera trajectory NPY> \
    --intrinsics-path <path to intrinsics NPY> \
    --num-frames 161 \
    --output-dir outputs/sana_wm_fp8 \
    --stage1-precision fp8 \
    --refiner-precision fp8
```

FP4 smoke:

```bash
uv run flashdreams-run sana-wm-bidirectional \
    --image-path <path to initial frame PNG> \
    --prompt-path <path to prompt TXT> \
    --camera-path <path to camera trajectory NPY> \
    --intrinsics-path <path to intrinsics NPY> \
    --num-frames 161 \
    --output-dir outputs/sana_wm_fp4 \
    --stage1-precision fp4 \
    --refiner-precision fp4
```

## Development checks

Contributor correctness parity checks live in
[`tests/parity_check/`](tests/parity_check/README.md). Performance benchmarks
live separately in [`tests/performance/`](tests/performance/README.md). Both
workflows compare the FlashDreams runners against the pinned upstream Sana
checkout under matched inputs.

CPU-safe tests cover import, config boundaries, action parsing, intrinsics,
camera conditioning, Stage-1 checkpoint schema, Stage-1 CPU forward shape, VAE
tiling, and low-precision backend selection:

```bash
uv run --package flashdreams-sana-wm --extra dev pytest integrations/sana/tests/test_smoke.py
```

GPU generation checks are heavyweight manual workflows and should stay out of
`ci_cpu`.
