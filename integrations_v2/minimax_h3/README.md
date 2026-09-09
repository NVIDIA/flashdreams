<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# MiniMax H3

Native, video-only MiniMax H3 inference through FlashDreams v2. The three
workflows share FlashDreams' T2V application, synchronized diffusion sampler,
accelerated attention, and scoped checkpoint loader. There is no Diffusers
runtime dependency or fallback pipeline.

## Run

```bash
uv sync --package flashdreams-minimax-h3 --extra dev --inexact
uv run --no-sync flashdreams-run-v2 t2v-minimax-h3-t2va --output-path clip.mp4 -- \
  --prompt "A cat surfing" --duration 5 --steps 30 --seed 42
uv run --no-sync flashdreams-run-v2 t2v-minimax-h3-fl2va --output-path clip.mp4 -- \
  --prompt "The camera moves through the scene" --image-path first.png --last-image-path last.png
uv run --no-sync flashdreams-run-v2 t2v-minimax-h3-ref2va --output-path clip.mp4 -- \
  --prompt "A cinematic scene" --reference image:subject.png --reference audio:reference.wav
```

Runtime options precede `--`; H3 options follow it. Use `-- --help` without
loading any weights. Resolution defaults to 768×768; runtime
`--pixel-width`/`--pixel-height` must be multiples of 32 and have aspect ratio
between 1:4 and 4:1. FPS is fixed at 24. Duration is 5–15 seconds, rounded
up to H3's `17n+5` frame grid, without exceeding 15 seconds. One block generates
the complete video; 30 scheduler points mean 29 joint model predictions.

FL2VA accepts first, last, or both keyframes. REF2VA keeps reference order and
supports images, videos, and audio (at least one visual reference). Audio
conditioning and audio denoising remain active, but generated audio is not
decoded or muxed into the output. `--mode webrtc` uses the shared prompt UI.

LoRA options are `--lora PATH_OR_REPO`, `--lora-weight-name FILE`, and
`--lora-scale NUMBER`. Musubi conversion is retained; a new network is loaded
for each request, so LoRA weights do not accumulate across resets.

## Memory and acceleration

Stage-scoped loading is always enabled: conditioning, denoising, and decoding
do not retain each other's weights. This replaces the old `--low-ram` switch.
Only requested codec subtrees are constructed and loaded. The joint transformer
still needs to fit on one GPU; blockwise CPU offload and distributed inference
are not implemented. A100 80 GB is the initial validation target, not a measured
guarantee that every resolution/duration fits.

Default attention is FlashDreams `OptimizedMultiHeadAttention`, native BF16
SDPA, no quantization, TMA, or duplicated QKV weights. `--attention torch` selects
FlashDreams' reference implementation. `--fuse-qkv` opts into additional fused
weight storage (roughly 11 GiB for the transformer). `--compile` uses the shared
compile helper and is opt-in. These options are **unvalidated performance
candidates**, not published speedups. CUDA graphs and quantization are deferred.

The video codec retains FP32 weights, FP16 decode autocast, reference tiling and
overlap, and precision-preserving Torch attention. Audio references use FP32
encoding. Checkpoint assets are pinned to
`MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe`; `--model-id` accepts
a compatible local snapshot or repository, and `--revision` overrides the pin.
Transformers/Accelerate remain dependencies for headless Qwen3-VL loading; they
do not orchestrate H3 diffusion.

## Migration and checks

The old `flashdreams-run minimax-h3-*` commands and recovery checkpoint files
are not used. Replace them with the three v2 commands above. Existing output
and recovery files are left untouched. No background checkpoint writer exists.

```bash
PYTHONPATH=flashdreams:apps/t2v:integrations_v2 .venv/bin/python -m pytest \
  integrations_v2/minimax_h3/tests -m ci_cpu
```

Native CPU checks require no checkpoint downloads. Optional reference checks
use an already-installed pinned Diffusers oracle and cached checkpoint headers;
they are skipped when unavailable. Diffusers is not installed by this package.
GPU tests are separate (`-m ci_gpu`); real-checkpoint generation/parity must be
requested explicitly. CPU correctness checks do not establish GPU video parity
or a speedup. Stage timings and peak allocated GPU memory are reported through
the shared v2 `--stats-path` output.
