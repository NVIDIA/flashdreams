<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Causal-Forcing text-to-video

A prompt goes in, a clip comes out as an MP4 file. The model is Causal-Forcing
Wan 2.1 1.3B in its chunkwise configuration, which the
`flashdreams-causal-forcing` package already configures for the v1 runner; this
package is the application around it and holds no model code of its own.

## Generate a clip

```bash
flashdreams-run-v2 t2v-causal-forcing --output-path clip.mp4 \
    -- --prompt "A cat surfing" --total-blocks 7 --no-compile
```

Arguments after `--` go to the application, and
`flashdreams-run-v2 t2v-causal-forcing -- --help` lists them.

`--total-blocks` is how many autoregressive blocks to generate, and the run ends
when the session has generated them. The first block decodes 9 frames and every
block after it 12, at 16 frames per second, so seven blocks is about four and a
half seconds.

`--no-compile` is worth it for a short clip. Compilation is on in the model's
own config; it costs minutes on the first run and saves milliseconds a block.

## What it generates

832x480 frames at 16fps, laid out `tchw`, as `[-1, 1]` floats on the GPU. Those
numbers are the checkpoint's, read off the runner config the
`flashdreams-causal-forcing` package ships by
[`T2VApplication`](../../flashdreams/flashdreams/t2v_v2/README.md) rather than
written down here. Something else can be asked for with `--pixel-width`,
`--pixel-height`, and `--fps` before the `--`, each dimension being a multiple
of 8.

The framewise configuration this integration also ships generates one latent
frame a block rather than three. It is not what this application runs, since a
block of one frame is a great deal of overhead per frame.

## First run

The checkpoint is fetched from Hugging Face the first time, which is tens of
gigabytes including the text encoder, so set a token and expect to wait:

```bash
export HF_TOKEN=<your-hf-token>
```

## Tests

```bash
uv sync --package flashdreams-t2v-causal-forcing --group test --inexact
uv run --no-sync pytest integrations_v2/t2v_causal_forcing -m ci_cpu -v
```

Those cover this integration's defaults against the stand-in model in
`flashdreams.t2v_v2.testing`. How an application of this kind behaves is
covered once in `flashdreams/test_v2`, and a run reaching a file once in the
Self-Forcing integration.

The run that uses the real model needs a GPU, and skips unless its environment
variable is set:

```bash
T2V_CAUSAL_FORCING_REAL_MODEL_RUN=1 uv run --no-sync pytest \
    integrations_v2/t2v_causal_forcing -m ci_gpu -s --basetemp="$HOME/t2v-out"
vlc "$HOME"/t2v-out/*current/clip.mp4
```
