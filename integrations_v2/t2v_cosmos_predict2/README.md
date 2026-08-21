<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Cosmos Predict2 text-to-video

A prompt goes in, a clip comes out as an MP4 file. The model is Cosmos Predict2
2B at 720p, which the `flashdreams-cosmos-predict2` package already configures
for the v1 runner; this package is the application around it and holds no model
code of its own.

Like Wan 2.1 and unlike the streaming models here, this one generates its whole
clip in a single block, so a run is one step and the clip is however long the
checkpoint generates. It is also the largest and slowest model here.

## Generate a clip

```bash
flashdreams-run-v2 t2v-cosmos-predict2 --output-path clip.mp4 \
    -- --prompt "A cat surfing" --no-compile
```

Arguments after `--` go to the application, and
`flashdreams-run-v2 t2v-cosmos-predict2 -- --help` lists them.

`--total-blocks` is 1 and has to be: a second block would not continue the
first, so asking for more is refused rather than quietly generating a second
clip. One block decodes 93 frames at 16 frames per second, so a clip is about
six seconds.

`--no-compile` is worth it for a single clip. Compilation is on in the model's
own config; it costs minutes on the first run and saves milliseconds, which is
the wrong trade for one block.

## What it generates

1280x720 frames at 16fps, laid out `tchw`, as `[-1, 1]` floats on the GPU. Those
numbers are the checkpoint's, block count included, read off the runner config
the `flashdreams-cosmos-predict2` package ships by
[`T2VApplication`](../../flashdreams/flashdreams/t2v_v2/README.md) rather than
written down here. Something else can be asked for with `--pixel-width`,
`--pixel-height`, and `--fps` before the `--`, each dimension being a multiple
of 8.

## First run

The checkpoint is fetched from Hugging Face the first time, which is tens of
gigabytes including the text encoder, so set a token and expect to wait:

```bash
export HF_TOKEN=<your-hf-token>
```

## Tests

```bash
uv sync --package flashdreams-t2v-cosmos-predict2 --group test --inexact
uv run --no-sync pytest integrations_v2/t2v_cosmos_predict2 -m ci_cpu -v
```

Those cover this integration's defaults, and that a rollout is refused, against
the stand-in model in `flashdreams.t2v_v2.testing`. How an application of this
kind behaves is covered once in `flashdreams/test_v2`, and a run reaching a file
once in the Self-Forcing integration.

The run that uses the real model needs a GPU, and skips unless its environment
variable is set:

```bash
T2V_COSMOS_PREDICT2_REAL_MODEL_RUN=1 uv run --no-sync pytest \
    integrations_v2/t2v_cosmos_predict2 -m ci_gpu -s --basetemp="$HOME/t2v-out"
vlc "$HOME"/t2v-out/*current/clip.mp4
```
