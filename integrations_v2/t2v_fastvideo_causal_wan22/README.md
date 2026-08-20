<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# FastVideo CausalWan 2.2 text-to-video

A prompt goes in, a clip comes out as an MP4 file. The model is FastVideo
CausalWan 2.2 14B, which the `flashdreams-fastvideo-causal-wan22` package
already configures for the v1 runner; this package is the application around it
and holds no model code of its own.

It denoises with two transformers rather than one, a high-noise branch and a
low-noise branch, so it holds two 14B checkpoints and wants a GPU to match.

## Generate a clip

```bash
flashdreams-run-v2 t2v-fastvideo-causal-wan22 --output-path clip.mp4 \
    -- --prompt "A cat surfing" --total-blocks 7 --no-compile
```

Arguments after `--` go to the application, and
`flashdreams-run-v2 t2v-fastvideo-causal-wan22 -- --help` lists them.

`--total-blocks` is how many autoregressive blocks to generate, and the run ends
when the session has generated them. The first block decodes 9 frames and every
block after it 12, at 16 frames per second, so seven blocks is about four and a
half seconds.

`--no-compile` is worth it for a short clip, and doubly so here: compilation is
on in the model's own config and costs minutes per transformer on the first run,
against milliseconds saved a block.

## What it generates

832x480 frames at 16fps, laid out `tchw`, as `[-1, 1]` floats on the GPU. Those
numbers are the checkpoint's, read off the runner config the
`flashdreams-fastvideo-causal-wan22` package ships by
[`T2VApplication`](../../flashdreams/flashdreams/t2v_v2/README.md) rather than
written down here. Something else can be asked for with `--pixel-width`,
`--pixel-height`, and `--fps` before the `--`, each dimension being a multiple
of 8.

Image-to-video is not wired up in the v1 package this wraps, so there is
nothing here for it either.

## First run

Both checkpoints are fetched from Hugging Face the first time, which is a large
download even by the standards of the other models here, so set a token and
expect to wait:

```bash
export HF_TOKEN=<your-hf-token>
```

## Tests

```bash
uv sync --package flashdreams-t2v-fastvideo-causal-wan22 --group test --inexact
uv run --no-sync pytest integrations_v2/t2v_fastvideo_causal_wan22 -m ci_cpu -v
```

Those cover this integration's defaults, and that `--no-compile` reaches both
transformers, against the stand-in model in `flashdreams.t2v_v2.testing`. How an
application of this kind behaves is covered once in `flashdreams/test_v2`, and a
run reaching a file once in the Self-Forcing integration.

The run that uses the real model needs a GPU, and skips unless its environment
variable is set:

```bash
T2V_FASTVIDEO_CAUSAL_WAN22_REAL_MODEL_RUN=1 uv run --no-sync pytest \
    integrations_v2/t2v_fastvideo_causal_wan22 -m ci_gpu -s \
    --basetemp="$HOME/t2v-out"
vlc "$HOME"/t2v-out/*current/clip.mp4
```
