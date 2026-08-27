<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LTX 2.5 synchronized audio and video

A concise FlashDreams V2 application around Diffusers' distilled LTX 2.5
text-to-video pipeline. LTX jointly generates video and stereo audio; this package
returns those tensors through `StepResult` and `AudioOutput`. The V2 runtime owns
threading, pacing, statistics, H.264/AAC encoding, synchronization, and transactional
MP4 publication.

The exact architecture, source/checkpoint pins, validation matrix, and accumulating
results are in [INTEGRATION.md](INTEGRATION.md).

## Requirements

- An NVIDIA GPU with enough memory for the 22B transformer. Model-level CPU offload is
  the default; this host's validation target is a 96 GB RTX PRO 6000 Blackwell.
- Enough host RAM and disk for roughly 154 GB of reconstructed checkpoint components
  (about 134 GB transferred on this validation host).
- A Hugging Face account that has requested and received access through the LTX 2.5
  Diffusers repository gate.
- A host-provided `ffmpeg` executable on `PATH` for MP4 output.

No Lightricks source or weights are redistributed. The adapter pins
`Lightricks/LTX-2.5-Diffusers` to revision
`426936f8b22dc28e4def61e515478b0b7e4a53cc`. LTX 2.5 support is not yet in a
Diffusers release, so the package also pins the upstream Diffusers `main` source to
commit `119c339551f68ea523b9f204120b929e56342421` rather than following a moving branch.

## Install and run

```bash
uv sync --package flashdreams-t2av-ltx25 --group test --inexact

uv run --no-sync flashdreams-run-v2 t2av-ltx25 \
    --output-path ltx25.mp4 \
    --stats-path ltx25-stats.json \
    --pixel-width 768 --pixel-height 512 --fps 24 \
    -- \
    --prompt "A red fox runs through snow while wind moves the pine trees" \
    --num-frames 121 --seed 42 --offload model
```

Arguments before `--` belong to FlashDreams. The application arguments are:

- `--prompt`: required generation prompt.
- `--num-frames`: `8k + 1`, from 1 through 241; defaults to 121.
- `--seed`: non-negative seed; defaults to 42.
- `--device`: CUDA device; defaults to `cuda`.
- `--offload`: `model` (default), `sequential`, or `none`.
- `--local-files-only`: forbid network access and require a warm Hugging Face cache.

The baseline intentionally disables the optional prompt-enhancement model and uses the
tiled convolutional video VAE. Diffusers' LTX 2.5 diffusion decoder is a separate,
deferred quality path rather than a hidden change to this reproducible baseline.

## Test

CPU contracts and the synchronized stand-in MP4:

```bash
uv run --no-sync pytest integrations_v2/t2av_ltx25 -m ci_cpu -v
```

Opt-in real checkpoint smoke:

```bash
T2AV_LTX25_REAL_MODEL_RUN=1 uv run --no-sync pytest \
    integrations_v2/t2av_ltx25 -m ci_gpu -s \
    --basetemp="$HOME/ltx25-test-out"
```

Run the complete prompt, duration, and resolution matrix with one model load:

    uv run --no-sync flashdreams-ltx25-benchmark \
        --output-dir "$HOME/ltx25-gallery" --offload model

The output directory receives an MP4 and runtime stats JSON per case, plus a JSON
manifest with codec, timing, signal, drift, hash, and validation checks,
and a portable `gallery.html` that embeds every clip. Use repeated `--case LABEL`
arguments for a subset; `--help` lists the stable case labels. Existing media is
not replaced unless `--overwrite` is explicit.
