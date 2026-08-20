<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# FastVideo CausalWan 2.2 text-to-video

A prompt goes in, a clip comes out as an MP4 file. The model is FastVideo
CausalWan 2.2 14B, which the `flashdreams-fastvideo-causal-wan22` package
already configures for the v1 runner; this package is the application around it
and holds no model code of its own.

This is the largest model here. It denoises with two transformers rather than
one, a high-noise branch and a low-noise branch, so it holds two 14B
checkpoints and wants a GPU to match.

## Generate a clip

```bash
flashdreams-run-v2 t2v-fastvideo-causal-wan22 --output-path clip.mp4 --steps 7 \
    -- --prompt "A cat surfing" --no-compile
```

Arguments after `--` go to the application, and
`flashdreams-run-v2 t2v-fastvideo-causal-wan22 -- --help` lists them.

`--steps` is how many autoregressive blocks to generate, and defaults to the
model's `--total-blocks`. The model streams, so a run is as long as it is asked
for: the first block decodes 9 frames and every block after it 12, at 16 frames
per second, so seven blocks is about four and a half seconds.

`--no-compile` is worth it for a short clip, and doubly so here: compilation is
on in the model's own config and costs minutes per transformer on the first run,
against milliseconds saved a block.

## What a run looks like in code

```python
from flashdreams.runtime_v2.batch_runner import run_batch
from flashdreams.runtime_v2.mp4_output_sink import Mp4OutputSink
from t2v_fastvideo_causal_wan22 import FastvideoCausalWan22T2VApplication

app = FastvideoCausalWan22T2VApplication()
app.init(["--prompt", "A cat surfing"])
try:
    run_batch(
        app.create_session(app.session_desc()),
        Mp4OutputSink("clip.mp4"),
        steps=7,
    )
finally:
    app.close()
```

## What it generates

832x480 frames at 16fps, laid out `tchw`, as `[-1, 1]` floats on the GPU. Those
numbers are the checkpoint's and are not written down here: this package is a
factory over
[`T2VApplication`](../../flashdreams/flashdreams/t2v_v2/README.md), which reads
them off the runner config the `flashdreams-fastvideo-causal-wan22` package
already ships. `app.session_desc()` is what they add up to.

Something else can be asked for, with `--pixel-width`, `--pixel-height`, and
`--fps`. A session refuses a description it cannot honour rather than quietly
generating something else, so each dimension has to be a multiple of 8, which
is what one latent covers.

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

Those run against the stand-in model in `flashdreams.t2v_v2.testing`, and cover
what is specific to this integration: that the defaults come off the runner
config, that `--no-compile` reaches both transformers rather than one, and that
a run reaches a file. How a text-to-video application behaves in general is
covered once, in `flashdreams/test_v2`.

The run that uses the real model needs a GPU, and skips unless its environment
variable is set, so asking for it is deliberate:

```bash
T2V_FASTVIDEO_CAUSAL_WAN22_REAL_MODEL_RUN=1 uv run --no-sync pytest \
    integrations_v2/t2v_fastvideo_causal_wan22 -m ci_gpu -s \
    --basetemp="$HOME/t2v-out"
vlc "$HOME"/t2v-out/*/clip.mp4
```

Both go through `flashdreams.t2v_v2.testing.check_t2v_model_impl`,
the shared check a text-to-video integration runs to cover the batch path.
